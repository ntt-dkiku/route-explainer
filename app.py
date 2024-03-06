# standard modules
import os
import pickle
import datetime
from PIL import Image
from typing import List, Union

# useful modules ("pip install" is required)
import numpy as np
import streamlit as st
import pandas as pd
import googlemaps
import langchain
from langchain.globals import set_verbose
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

# our defined modules
import utils.util_app as util_app
from models.solvers.general_solver import GeneralSolver
from models.cf_generator import CFTourGenerator
from models.classifiers.general_classifier import GeneralClassifier
from models.route_explainer import RouteExplainer

# general setting
SEED = 1234
TOUR_NAME         = "static/kyoto_tour"
TOUR_PATH         = TOUR_NAME + ".csv"
TOUR_LATLNG_PATH  = TOUR_NAME + "_latlng.csv"
TOUR_DISTMAT_PATH = TOUR_NAME + "_distmat.pkl"
EXPANDED          = False
DEBUG             = True
ROUTE_EXPLAINER_ICON = np.array(Image.open("static/route_explainer_icon.png"))

# for debug
if DEBUG:
    langchain.debug = True
    set_verbose(True)

def load_tour_list():
    # get lat/lng
    if os.path.isfile(TOUR_LATLNG_PATH):
        df_tour = pd.read_csv(TOUR_LATLNG_PATH)
    else:
        df_tour = pd.read_csv(TOUR_PATH)
        if googleapi_key := st.session_state.googleapi_key:
            gmaps = googlemaps.Client(key=googleapi_key)
            lat_list =[]; lng_list = []
            for destination in df_tour["destination"]:
                geo_result = gmaps.geocode(destination)
                lat_list.append(geo_result[0]["geometry"]["location"]["lat"])
                lng_list.append(geo_result[0]["geometry"]["location"]["lng"])
            # add lat/lng
            df_tour["lat"] = lat_list
            df_tour["lng"] = lng_list
            df_tour.to_csv(TOUR_LATLNG_PATH)
    
    # get the central point
    st.session_state.lat_mean = np.mean(df_tour["lat"])
    st.session_state.lng_mean = np.mean(df_tour["lng"])
    st.session_state.sw = df_tour[["lat", "lng"]].min().tolist()
    st.session_state.ne = df_tour[["lat", "lng"]].max().tolist()
    st.session_state.df_tour = df_tour

    # get the distance matrix
    if os.path.isfile(TOUR_DISTMAT_PATH):
        with open(TOUR_DISTMAT_PATH, "rb") as f:
            distmat = pickle.load(f)
    else:
        if googleapi_key := st.session_state.googleapi_key:
            gmaps = googlemaps.Client(key=googleapi_key)
            distmat = []
            for origin in df_tour["destination"]:
                distrow = []
                for dest in df_tour["destination"]:
                    if origin != dest:
                        dist_result = gmaps.distance_matrix(origin, dest, mode="driving")
                        distrow.append(dist_result["rows"][0]["elements"][0]["duration"]["value"]) # unit: seconds
                    else:
                        distrow.append(0)
                distmat.append(distrow)
            distmat = np.array(distmat)
            with open(TOUR_DISTMAT_PATH, "wb") as f:
                pickle.dump(distmat, f)

    # input features
    def convert_clock2seconds(clock):
        return sum([a*b for a, b in zip([3600, 60], map(int, clock.split(':')))])
    time_windows = []
    for i in range(len(df_tour)):
        time_windows.append([convert_clock2seconds(df_tour["open"][i]), 
                             convert_clock2seconds(df_tour["close"][i])])
    time_windows = np.array(time_windows)
    time_windows -= time_windows[0, 0]
    node_feats = {
        "time_window": time_windows.clip(0), 
        "service_time": df_tour["stay_duration (h)"].to_numpy() * 3600
    }
    st.session_state.node_feats = node_feats
    st.session_state.dist_matrix = distmat
    st.session_state.node_info  = {
        "open": df_tour["open"],
        "close": df_tour["close"],
        "stay": df_tour["stay_duration (h)"]
    }

    # tour list
    if os.path.isfile(TOUR_DISTMAT_PATH) & os.path.isfile(TOUR_LATLNG_PATH):
        st.session_state.tour_list = []
        for i in range(len(df_tour)):
            st.session_state.tour_list.append({
                "name": df_tour["destination"][i],
                "latlng": (df_tour["lat"][i], df_tour["lng"][i]),
                "description": f"<font color='silver'>Hours: {df_tour['open'][i]} - {df_tour['close'][i]}<br>Duration of stay: {df_tour['stay_duration (h)'][i]}h<br>Remarks: {df_tour['remarks'][i]}</font>"
            })

def solve_vrp() -> None:
    if ("node_feats" in st.session_state) and ("dist_matrix" in st.session_state):
        solver = GeneralSolver("tsptw", "ortools", scaling=False)
        classifier = GeneralClassifier("tsptw", "gt(ortools)")
        routes = solver.solve(node_feats=st.session_state.node_feats, 
                              dist_matrix=st.session_state.dist_matrix)
        inputs = classifier.get_inputs(routes, 
                                       0, 
                                       st.session_state.node_feats, 
                                       st.session_state.dist_matrix)
        labels = classifier(inputs)
        st.session_state.routes = routes.copy()
        st.session_state.labels = labels.copy()
        st.session_state.generated_actual_route = True

#----------
# LLM
#----------
def load_route_explainer(llm_type: str) -> None:
    if st.session_state.openai_key:
        # define llm
        llm = ChatOpenAI(model=llm_type, 
                         temperature=0,
                         streaming=True,
                         model_kwargs={"seed": SEED})
        # model_kwargs={"stop": ["\n\n", "Human"]}

        # define RouteExplainer
        cf_generator = CFTourGenerator(cf_solver=GeneralSolver("tsptw", "ortools", scaling=False))
        classifier = GeneralClassifier("tsptw", "gt(ortools)")
        st.session_state.route_explainer = RouteExplainer(llm=llm, 
                                                        cf_generator=cf_generator, 
                                                        classifier=classifier)

#----------
# UI
#----------
# css settings
st.set_page_config(layout="wide")
util_app.apply_responsible_map_css()
util_app.apply_centerize_icon_css()
util_app.apply_red_code_css()
util_app.apply_remove_sidebar_topspace()

#------------------
# side bar setting
#------------------
with st.sidebar:
    #-------
    # Title
    #-------
    icon_col, name_col = st.columns((1,10))
    with icon_col:
        util_app.apply_html('<a href="https://ntt-dkiku.github.io/xai-vrp" target="_blank"><img src="./app/static/route_explainer_icon.png" alt="RouteExplainer" width="30" height="30" style="margin-top: 20px;"></a>')
    with name_col:
        st.title("RouteExplainer")
    
    #----------
    # API keys
    #----------
    st.subheader("API keys")
    openai_key_col1, openai_key_col2 = st.columns((1,10))
    with openai_key_col1:
        util_app.apply_html('<a href="https://openai.com/blog/openai-api" target="_blank"> <img src="./app/static/openai_logo.png" alt="OpenAI API" width="30" height="30"> </a>')
    with openai_key_col2:
        openai_key = st.text_input(label="API keys",
                                   key="openai_key",
                                   placeholder="OpenAI API key",
                                   type="password",
                                   label_visibility="collapsed")
    changed_key = openai_key == os.environ.get('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = openai_key

    google_key_col1, google_key_col2 = st.columns((1, 10))
    with google_key_col1:
        util_app.apply_html('<a href="https://developers.google.com/maps?hl=en" target="_blank"> <img src="./app/static/googlemap_logo.png" alt="GoogleMap API" width="30" height="30"> </a>')
    with google_key_col2:
        st.text_input(label="GoogleMap API key",
                      key="googleapi_key",
                      placeholder="NOT required in this demo",
                      type="password",
                      label_visibility="collapsed")
    
    #----------------
    # Foundation LLM
    #----------------
    st.subheader("Foundation LLM")
    llm_type = st.selectbox("LLM", ["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo"], key="llm_type", label_visibility="collapsed")

    #-----------
    # Tour plan
    #-----------
    st.subheader("Tour plan")
    col1, col2 = st.columns((2, 1))
    with col1:
        # Comming soon: "Taipei Tour (for PAKDD2024)"
        tour_plan = st.selectbox("Tour plan", ["Kyoto Tour"], key="tour_type", label_visibility="collapsed")
    with col2:
        st.button("Generate", on_click=solve_vrp, use_container_width=True)

    # list destinations
    load_tour_list()
    with st.container():
        if "routes" in st.session_state: # rearranage destinations in the route order if a route was derivied
            # re-ordered destinations
            reordered_tour_list = [st.session_state.tour_list[i] for i in st.session_state.routes[0][:-1]] if "routes" in st.session_state else st.session_state.tour_list
            arr_time = datetime.datetime.strptime(st.session_state.node_info["open"][0], "%H:%M")
            for step in range(len(reordered_tour_list)):
                curr = reordered_tour_list[step]
                next = reordered_tour_list[step+1] if step != len(reordered_tour_list) - 1 else reordered_tour_list[0]
                curr_node_id = util_app.find_node_id_by_name(st.session_state.tour_list, curr["name"])
                next_node_id = util_app.find_node_id_by_name(st.session_state.tour_list, next["name"])
                open_time = datetime.datetime.strptime(st.session_state.node_info["open"][curr_node_id], "%H:%M")
                # destination info
                dep_time = max(arr_time, open_time) + datetime.timedelta(hours=st.session_state.node_info["stay"][curr_node_id])
                dep_time_str = dep_time.strftime("%H:%M")
                arr_time_str = arr_time.strftime("%H:%M")
                arr_dep = f"Arr {arr_time_str} - Dep {dep_time_str}" if step != 0 else f"⭐ Dep {dep_time_str}"
                with st.expander(f"{arr_dep} | {curr['name']}", expanded=EXPANDED):
                    st.write(curr["description"], unsafe_allow_html=True)
                # travel time
                travel_time = st.session_state.dist_matrix[curr_node_id][next_node_id].item()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"<center>{util_app.add_time_unit(travel_time)}</center>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<center>|</center>", unsafe_allow_html=True)
                st.write("")
                arr_time = dep_time + datetime.timedelta(seconds=travel_time)
            # return to the origin
            destination = reordered_tour_list[0]
            arr_time_str = arr_time.strftime("%H:%M")
            with st.expander(f"⭐ Arr {arr_time_str} | {destination['name']}", expanded=EXPANDED):
                st.write(destination["description"], unsafe_allow_html=True)
        else: # just list destinations 
            for destination in st.session_state.tour_list:
                with st.expander(destination['name'], expanded=EXPANDED):
                    st.write(destination["description"], unsafe_allow_html=True)

#----------------------
# state initialization
#----------------------
if "count" not in st.session_state:
    st.session_state.count = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "generated_actual_route" not in st.session_state:
    st.session_state.generated_actual_route = False
if "generated_cf_route" not in st.session_state:
    st.session_state.generated_cf_route = False
if "curr_route" not in st.session_state:
    st.session_state.curr_route = "Actual Route" # once the CF route is selected, this will be "Current Route"
if "flag_example" not in st.session_state:
    st.session_state.flag_example = False
if "selected_example" not in st.session_state:
    st.session_state.selected_example = None
if "close_chat" not in st.session_state:
    st.session_state.close_chat = False
if "route_explainer" not in st.session_state or llm_type != st.session_state.curr_llm_type or changed_key:
    load_route_explainer(llm_type)
    st.session_state.curr_llm_type = llm_type

#--------------------------------
# The following is the main page
#--------------------------------

#----------
# Greeding
#----------
if "routes" not in st.session_state:
    util_app.apply_html('<center> <img src="./app/static/route_explainer_icon.png" alt="OpenAI API" width="120" height="120"> </center>')
    greeding = "Hi, I'm RouteExplainer :)<br>Choose a tour and hit the <code>Generate</code> button to generate your initial route!"
    if st.session_state.count == 0:
        util_app.stream_words(greeding, prefix="<center><h4>", suffix="</h4></center>", sleep_time=0.02)
    else:
        util_app.apply_html(f"<center><h4>{greeding}</h4></center>")

#--------------
# chat history
#--------------
def find_last_map(lst: List[Union[str, tuple]]) -> int:
    for i in range(len(lst) - 1, -1, -1):
        if isinstance(lst[i], tuple):
            return i
    return None
last_map_idx = find_last_map(st.session_state.chat_history)
for i, msg in enumerate(st.session_state.chat_history):
    if isinstance(msg, tuple): # if the history type is a tuple of maps
        map1, map2 = (0, 1) if i == last_map_idx else (2, 3)
        actual_route, cf_route = st.columns(2)
        if msg[map1] is not None:
            with actual_route:
                util_app.visualize_actual_route(msg[map1])
        if msg[map2] is not None:
            with cf_route:
                util_app.visualize_cf_route(msg[map2])
    else: # if the history type is string
        if isinstance(msg, AIMessage):
            st.chat_message(msg.type, avatar=ROUTE_EXPLAINER_ICON).write(msg.content)
        else:
            st.chat_message(msg.type).write(msg.content)

# examples
if "cf_routes" not in st.session_state and st.session_state.flag_example:
    def pickup_example(example: str):
        st.session_state.selected_example = example
    
    examples = [
        "Why do we visit Ginkaku-ji Temple from Fushimi-Inari Shrine and why not Kiyomizu-dera Temple?",
        "What if we visit Kinkaku-ji directly from Kyoto Geishinkan, instead of Nijo-jo Castle?",
        "Why was the edge from Kinkaku-ji to Kiyomizu-dera selected and why not the edge from Kinkaku-ji to Hanamikoji Dori?"
    ]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button(examples[0], 
                use_container_width=True,
                on_click=pickup_example,
                args=(examples[0], ))
    with col2:
        st.button(examples[1],
                use_container_width=True,
                on_click=pickup_example,
                args=(examples[1], ))
    with col3:
        st.button(examples[2],
                use_container_width=True,
                on_click=pickup_example,
                args=(examples[2], ))

#----------
# chat box
#----------
def answer(prompt: str):
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)
    if os.environ.get('OPENAI_API_KEY') == "":
        error_msg = "An OpenAI API key has not been set yet :( Please enter a valid key in the side bar!"
        with st.chat_message("assistant", avatar=ROUTE_EXPLAINER_ICON):
            st.write(error_msg)
        st.session_state.chat_history.append(AIMessage(content=error_msg)) 
    elif util_app.validate_openai_api_key(os.environ.get('OPENAI_API_KEY')):
        with st.chat_message("assistant", avatar=ROUTE_EXPLAINER_ICON):
            explanation = st.session_state.route_explainer.generate_explanation(tour_list=st.session_state.tour_list,
                                                                                whynot_question=prompt,
                                                                                actual_routes=st.session_state.routes,
                                                                                actual_labels=st.session_state.labels,
                                                                                node_feats=st.session_state.node_feats,
                                                                                dist_matrix=st.session_state.dist_matrix)
            if len(explanation) > 0:
                st.session_state.chat_history.append(AIMessage(content=explanation))
            if st.session_state.generated_cf_route:
                st.rerun()
    else:
        error_msg = "The input OpenAI API key appears to be invalid :( Please enter a valid key again in the side bar!"
        with st.chat_message("assistant", avatar=ROUTE_EXPLAINER_ICON):
            st.write(error_msg)
        st.session_state.chat_history.append(AIMessage(content=error_msg))

if st.session_state.selected_example is not None:
    example = st.session_state.selected_example
    st.session_state.selected_example = None
    answer(example)
else:
    if "routes" in st.session_state and not st.session_state.close_chat:
        if prompt := st.chat_input(placeholder="Ask a why-not question", key="chat_input"):
            answer(prompt)

#---------------------
# route visualization
#---------------------
if "tour_list" in st.session_state: # if tour info is loaded
    # first message
    if st.session_state.generated_actual_route: # when an actual route is generated
        with st.chat_message("assistant", avatar=ROUTE_EXPLAINER_ICON):
            msg = "Here is your initial route. Please ask me a why and why-not question for a specfic edge!"
            util_app.stream_words(msg, sleep_time=0.01)
        st.session_state.flag_example = True
        st.session_state.chat_history.append(AIMessage(content=msg))

    # visualize the actual & CF routes
    actual_route, cf_route = st.columns(2)
    m = None; m2 = None; m_ = None; m2_ = None
    if st.session_state.generated_actual_route or st.session_state.generated_cf_route:
        m = util_app.initialize_map() # overwrite m
        m_ = util_app.initialize_map()
        if "labels" in st.session_state:
            cf_step = st.session_state.cf_step-1 if st.session_state.generated_cf_route else -1
            util_app.vis_route("routes", st.session_state.labels, m, cf_step, "actual")
            util_app.vis_route("routes", st.session_state.labels, m_, cf_step, "actual", ant_path=False)
        with actual_route:
            util_app.visualize_actual_route(m)
    if st.session_state.generated_cf_route:
        m2 = util_app.initialize_map() # overwrite m2
        m2_ = util_app.initialize_map()
        if "cf_labels" in st.session_state:
            util_app.vis_route("cf_routes", st.session_state.cf_labels, m2, st.session_state.cf_step-1, "cf")
            util_app.vis_route("cf_routes", st.session_state.cf_labels, m2_, st.session_state.cf_step-1, "cf", ant_path=False)
        with cf_route:
            util_app.visualize_cf_route(m2)

    # update states related to maps
    if m is not None:
        st.session_state.chat_history.append((m, m2, m_, m2_))

    # route selection button
    if len(st.session_state.chat_history) > 0:
        last_msg = st.session_state.chat_history[-1]
        if isinstance(last_msg, tuple):
            if (last_msg[0] is not None) and (last_msg[1] is not None):
                col1, col2 = st.columns(2)
                with col1:
                    st.button("Stay this route", on_click=util_app.select_actual_route)
                with col2:
                    st.button("Replace with this route", on_click=util_app.select_cf_route)
                    util_app.change_hover_color("button", "Replace with this route", "#1e90ff")
    
    # for displaying examples
    if st.session_state.generated_actual_route:
        st.session_state.generated_actual_route = False
        st.rerun()

    st.session_state.generated_actual_route = False
    st.session_state.generated_cf_route     = False

# update session count
st.session_state.count += 1

js = f"""
<script>
    function scroll(dummy_var_to_force_repeat_execution){{
        var textAreas = parent.document.querySelectorAll('section.main');
        for (let index = 0; index < textAreas.length; index++) {{
            textAreas[index].scrollTop = textAreas[index].scrollHeight;
        }}
    }}
    scroll({len(st.session_state.chat_history)})
</script>
"""
st.components.v1.html(js, height=0, width=0)