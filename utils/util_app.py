import openai
import time
import base64
import copy
import streamlit as st
import folium
from folium.plugins import FloatImage
from streamlit_folium import folium_static
from langchain.schema import AIMessage

#----------------
# util functions
#----------------
def apply_html(html: str, **kwargs) -> None:
    st.markdown(html, unsafe_allow_html=True, **kwargs)

def stream_words(words: str,
                 prefix: str = "",
                 suffix: str = "",
                 sleep_time: float = 0.02) -> None:
    elem = st.empty()
    _words = ""
    for word in list(words):
        _words += word
        elem.markdown(f"{prefix} {_words} {suffix}", unsafe_allow_html=True)
        time.sleep(sleep_time)

def add_time_unit(time: float) -> str:
    if time < 60:
        return f"{time} sec"
    elif time >= 60 and time < 3599:
        return f"{time/60:.2f} mins"
    else:
        return f"{time/3600:.2f} hours" 

def find_node_id_by_name(data_list, name) -> int:
    for index, item in enumerate(data_list):
        if item.get("name") == name:
            return index
    return -1  # Return -1 if the name is not found

def mark_destination(tour_list, m) -> None:
    for destination in tour_list:
        image_name = f"static/{destination['name'].replace(' ', '-')}.png"
        if destination["name"] == "Ryoanji Temple":
            icon_width = 50; icon_height = 35
        elif destination["name"] == "Ryoanji Temple":
            icon_width = 50; icon_height = 40
        elif destination["name"] == "Kyoto Geishinkan":
            icon_width = 40; icon_height = 40
        elif destination["name"] == "Nijo-jo Castle":
            icon_width = 45; icon_height = 35
        else:
            icon_width = 50; icon_height = 50
        
        folium.Marker(
            location=[destination["latlng"][0], destination["latlng"][1]],
            tooltip=destination["name"],
            icon=folium.features.CustomIcon(icon_image = image_name,
                                            icon_size = (icon_width, icon_height),
                                            icon_anchor = (30, 30),
                                            popup_anchor = (3, 3))
        ).add_to(m)

    # add an indicator of the start/end point to Kyoto Station
    # z_indedx_offset: https://github.com/python-visualization/folium/issues/1281
    destination = tour_list[0]
    folium.Marker(
        location=[destination["latlng"][0]+0.003, destination["latlng"][1]-0.003],
        tooltip="start/end point",
        icon=folium.features.CustomIcon(icon_image = "static/star_emoji.png",
                                        icon_size = (30, 30),
                                        icon_anchor = (30, 30),
                                        popup_anchor = (3, 3)),
        z_index_offset=10000
    ).add_to(m)

    # add legend
    # Ref: https://python-visualization.github.io/folium/latest/user_guide/plugins/float_image.html
    with open("static/legend.png", "rb") as lf:
        # open in binary mode, read bytes, encode, decode obtained bytes as utf-8 string
        b64_content = base64.b64encode(lf.read()).decode("utf-8")
    FloatImage("data:image/png;base64,{}".format(b64_content), bottom=1, left=1).add_to(m)

def initialize_map() -> folium.Map:
    m = folium.Map(location=[st.session_state.lat_mean, st.session_state.lng_mean], tiles="Cartodb Positron")
    m.fit_bounds([st.session_state.sw, st.session_state.ne])
    mark_destination(st.session_state.tour_list, m)
    return m

def vis_route(routes, labels, m, ex_step, route_type, ant_path=True) -> None:
    if ("tour_list" in st.session_state) and (routes in st.session_state):
        tour_list = st.session_state.tour_list
        for j, route in enumerate(st.session_state[routes]): # vehicle loop
            for i in range(len(route)): # edge loop
                if i < len(route) - 1:
                    if i == ex_step:
                        if route_type == "actual":
                            color = "red"
                            popup = "Actual edge"  
                        else:
                            color = "blue"
                            popup = "CF edge"
                    else:
                        if labels[j][i] == 0:
                            color = "#2ca02c" 
                            popup = "Route length priority"
                        else:
                            color = "#9467bd"
                            popup = "Time window priority"
                    origin_id = route[i]
                    dest_id = route[i+1]
                    origin_latlng = tour_list[origin_id]["latlng"]
                    dest_latlng = tour_list[dest_id]["latlng"]
                    line = folium.PolyLine([origin_latlng, dest_latlng], color=color).add_to(m)
                    if ant_path:
                        folium.plugins.AntPath([origin_latlng, dest_latlng], tooltip=popup, color=color).add_to(m)
                    else:
                        folium.plugins.PolyLineTextPath(line, ">  ", offset=11, repeat=True, attributes={"fill": color,
                                                                                                         "font-size": 30}).add_to(m)

def visualize_actual_route(m: folium.Map) -> None:
    with st.columns((0.2, 1, 0.1))[1]:
        st.subheader(f"{st.session_state.curr_route}", divider="red")
        folium_static(m)

def visualize_cf_route(m: folium.Map) -> None:
    with st.columns((0.1, 1, 0.2))[1]:
        st.subheader("CF route", divider="blue")
        folium_static(m)

def select_actual_route():
    route_name =  "the actual route" if st.session_state.curr_route == "Actual Route" else "your current route"
    msg = f"You chose to stay {route_name}. Feel free to ask another why and why-not question for your current route!"
    st.session_state.chat_history.append(AIMessage(content=msg))
    m = initialize_map()
    m_ = initialize_map()
    if "labels" in st.session_state:
        cf_step = st.session_state.cf_step-1 if st.session_state.generated_cf_route else -1
        vis_route("routes", st.session_state.labels, m, cf_step, "actual")
        vis_route("routes", st.session_state.labels, m_, cf_step, "actual", ant_path=False)
    st.session_state.chat_history.append((m, None, m_, None))
    st.session_state.close_chat = False

def select_cf_route():
    msg = "You chose to replace your current route with the CF route. Feel free to ask a why and why-not question for the CF route!"
    st.session_state.chat_history.append(AIMessage(content=msg))
    # replace the actual route & labels with the CF ones
    st.session_state.routes = copy.deepcopy(st.session_state.cf_routes)
    st.session_state.labels = copy.deepcopy(st.session_state.cf_labels)
    st.session_state.curr_route = "Current Route"
    m = initialize_map()
    m_ = initialize_map()
    if "cf_labels" in st.session_state:
        cf_step = st.session_state.cf_step-1 if st.session_state.generated_cf_route else -1
        vis_route("routes", st.session_state.labels, m, cf_step, "actual")
        vis_route("routes", st.session_state.labels, m_, cf_step, "actual", ant_path=False)
    st.session_state.chat_history.append((m, None, m_, None))
    st.session_state.close_chat = False

# ref: https://stackoverflow.com/questions/76522693/how-to-check-the-validity-of-the-openai-key-from-python
def validate_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True

#-----
# CSS
#-----
RESPONSIBLE_MAP = """\
<style>
    [title~="st.iframe"]:not([height="0"]) { 
        width: 100%; 
        aspect-ratio: 1; 
        height: auto;
    }
</style>
"""
def apply_responsible_map_css() -> None:
    apply_html(RESPONSIBLE_MAP)

CENTERIZE_INCON = """
<style>
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    .stButton {
        text-align:center
    }
</style>
"""
def apply_centerize_icon_css() -> None:
    apply_html(CENTERIZE_INCON)

RED_CODE = """\
<style>
    code {
        color: #cc3333;
    }
</style>
"""
def apply_red_code_css() -> None:
    apply_html(RED_CODE)

REMOVE_SIDEBAR_TOPSPACE = """\
<style>
    .st-emotion-cache-6qob1r.eczjsme3 {
        margin-top: -75px;
    }
    .st-emotion-cache-1b9x38r.eczjsme2 {
        margin-top: 75px
    }
</style>
"""
def apply_remove_sidebar_topspace() -> None:
    apply_html(REMOVE_SIDEBAR_TOPSPACE)

#----
# JS
#----
# SET_WORDS_TO_CHATBOX = """\
# <script>
#     function insertText() {{
#         var chatInput = parent.document.querySelector('textarea[data-testid="stChatInput"]');
#         var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
#         nativeInputValueSetter.call(chatInput, "{words}");
#         var event = new Event('input', {{bubbles: true}});
#         chatInput.dispatchEvent(event);
#     }}
#     insertText();
# </script>
# """
# def set_chat_input(words:str) -> None:
#     st.components.v1.html(SET_WORDS_TO_CHATBOX.format(words=words), height=0)

# Ref. https://discuss.streamlit.io/t/issues-with-background-colour-for-buttons/38723/8
# Ref. https://github.com/streamlit/streamlit/issues/6605
CHANGE_HOVER_COLOR = """\
<script>
    var hide_me_list = window.parent.document.querySelectorAll('iframe');
    for (let i = 0; i < hide_me_list.length; i++) {{
        if (hide_me_list[i].height == 0) {{
            hide_me_list[i].parentNode.style.height = 0;
            hide_me_list[i].parentNode.style.marginBottom = '-1rem';
        }};
    }};

    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {{
        var border = 'rgb(250,250,250,.2)';
    }} else {{
        var border = 'rgb(49,51,63,.2)';
    }}
    
    var elements = window.parent.document.querySelectorAll('{widget_type}');
    var fontColor = window.getComputedStyle(elements[0]).color;
    for (var i = 0; i < elements.length; ++i) {{ 
        if (elements[i].innerText == '{widget_label}') {{ 
            elements[i].style.color = fontColor;
            elements[i].style.background = '{background_color}';
            elements[i].onmouseover = function() {{ 
                this.style.color = '{hover_color}';
                this.style.borderColor = '{hover_color}';
            }};
            elements[i].onmouseout = function() {{ 
                this.style.color = fontColor;
                this.style.borderColor = border;
                this.style.background = '{background_color}';
            }};
            elements[i].onclick = function() {{ 
                this.style.color = 'white';
                this.style.borderColor = '{hover_color}';
                this.style.background = '{hover_color}';
            }};
            elements[i].onfocus = function() {{
                this.style.boxShadow = '{hover_color} 0px 0px 0px 0.2rem';
                this.style.borderColor = '{hover_color}';
                this.style.color = '{hover_color}';
            }};
            elements[i].onblur = function() {{
                this.style.boxShadow = 'none';
                this.style.color = fontColor;
                this.style.borderColor = border;
            }};
        }}
    }}
</script>
"""
def change_hover_color(widget_type: str,
                       widget_label: str, 
                       hover_color: str,
                       background_color: str = ""):
    st.components.v1.html(CHANGE_HOVER_COLOR.format(widget_type=widget_type,
                                         widget_label=widget_label,
                                         hover_color=hover_color,
                                         background_color=background_color), height=0)