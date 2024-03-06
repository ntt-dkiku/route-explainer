# templates
import numpy as np
import streamlit as st
from typing import Dict, List
from models.prompts.identify_question import Template4IdentifyQuestion
from models.prompts.generate_explanation import Template4GenerateExplanation
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AIMessage
import utils.util_app as util_app

class StreamingChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        pass

    def on_llm_start(self, *args, **kwargs):
        self.container = st.empty()
        self.text = ""

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.text += token
        self.container.markdown(
            body=self.text,
            unsafe_allow_html=False,
        )

    def on_llm_end(self, response: str, *args, **kwargs):
        self.container.markdown(
            body=response.generations[0][0].text,
            unsafe_allow_html=False,
        )

class RouteExplainer():
    template_identify_question = Template4IdentifyQuestion()
    template_generate_explanation = Template4GenerateExplanation()

    def __init__(self,
                 llm,
                 cf_generator, 
                 classifier) -> None:
        assert cf_generator.problem == classifier.problem, "Problem type of cf_generator and predictor should coincide!"
        self.coord_dim = 2
        self.problem = cf_generator.problem
        self.cf_generator = cf_generator
        self.classifier = classifier
        self.actual_route = None
        self.cf_route = None
        # templates
        self.question_extractor = self.template_identify_question.sandwiches(llm)
        self.explanation_generator = self.template_generate_explanation.sandwiches(llm)

    #----------------
    # whole pipeline
    #----------------
    def generate_explanation(self, 
                             tour_list,
                             whynot_question: str,
                             actual_routes: list,
                             actual_labels: list,
                             node_feats: dict,
                             dist_matrix: np.array) -> str:
        #--------------------------------
        # define why & why-not questions
        #--------------------------------
        route_info_text = self.get_route_info_text(tour_list, actual_routes)
        inputs = self.question_extractor.invoke({
            "whynot_question": whynot_question,
            "route_info": route_info_text
        })
        util_app.stream_words(inputs["summary"] + " " + inputs["intent"])
        st.session_state.chat_history.append(AIMessage(content=inputs["summary"] + inputs["intent"]))
        if not inputs["success"]:
            return ""

        #----------------------
        # validate the CF edge
        #----------------------
        is_cf_edge_feasible, reason = self.validate_cf_edge(node_feats,
                                                            dist_matrix,
                                                            actual_routes[0],
                                                            inputs["cf_step"],
                                                            inputs["cf_visit"]-1)
        # exception
        if not is_cf_edge_feasible:
            util_app.stream_words(reason)
            return reason

        #---------------------
        # generate a cf route
        #---------------------
        cf_routes = self.cf_generator(actual_routes,
                                      vehicle_id=0,
                                      cf_step=inputs["cf_step"],
                                      cf_next_node_id=inputs["cf_visit"]-1,
                                      node_feats=node_feats,
                                      dist_matrix=dist_matrix)
        st.session_state.generated_cf_route = True
        st.session_state.close_chat = True
        st.session_state.cf_step = inputs["cf_step"]

        #--------------------------------------
        # classify the intentions of each edge
        #--------------------------------------
        cf_labels = self.classifier(self.classifier.get_inputs(cf_routes,
                                                               0,
                                                               node_feats,
                                                               dist_matrix))
        st.session_state.cf_routes = cf_routes
        st.session_state.cf_labels = cf_labels

        #-------------------------------------
        # generate a constrastive explanation
        #-------------------------------------
        comparison_results = self.get_comparison_results(question_summary=inputs["summary"],
                                                         tour_list=tour_list,
                                                         actual_routes=actual_routes,
                                                         actual_labels=actual_labels,
                                                         cf_routes=cf_routes,
                                                         cf_labels=cf_labels,
                                                         cf_step=inputs["cf_step"])
        
        explanation = self.explanation_generator.invoke({
            "comparison_results": comparison_results,
            "intent": inputs["intent"]
        }, config={"callbacks": [StreamingChatCallbackHandler()]})

        return explanation
    
    #-------------------------
    # for exctracting inputs
    #-------------------------
    def get_route_info_text(self, tour_list, routes) -> str:
        route_info = ""
        # nodes
        route_info += "Nodes(node id, name): "
        for i, destination in enumerate(tour_list):
            if i != len(tour_list) - 1:
                route_info += f"({i+1}, {destination['name']}), "
            else:
                route_info += f"({i+1}, {destination['name']})\n"

        # routes
        route_info += "Route: "
        for i, node_id in enumerate(routes[0]):
            if i == 0:
                route_info += f"{tour_list[node_id]['name']} "
            else:
                route_info += f"> (step {i}) > {tour_list[node_id]['name']})"
                if i == len(routes[0]) - 1:
                    route_info += "\n"
                else:
                    route_info += " "
        return route_info
    
    #--------------------------
    # for validating a CF edge
    #--------------------------
    def validate_cf_edge(self,
                         node_feats: Dict[str, np.array],
                         dist_matrix: np.array,
                         route: List[int],
                         cf_step: int,
                         cf_visit: int) -> bool:
        # calc current time
        curr_time = node_feats["time_window"][route[0]][0] # start point's open time
        for step in range(1, cf_step):
            curr_node_id = route[step-1]
            next_node_id = route[step]
            curr_time += node_feats["service_time"][curr_node_id] + dist_matrix[curr_node_id][next_node_id]
            curr_time = max(curr_time, node_feats["time_window"][next_node_id][0]) # waiting

        # validate the cf edge
        curr_node_id = route[cf_step-1]
        next_node_id = cf_visit
        next_node_close_time = node_feats["time_window"][next_node_id][1] 
        arrival_time = curr_time + node_feats["service_time"][curr_node_id] + dist_matrix[curr_node_id][next_node_id]
        if next_node_close_time < arrival_time:
            exceed_time = (arrival_time - next_node_close_time)
            return False, f"Oops, your CF edge is infeasible because it does not meet the destination's close time by {util_app.add_time_unit(exceed_time)}."
        else:
            return True, "The CF edge is feasible!"

    #-------------------------------
    # for generating an explanation
    #-------------------------------
    def get_comparison_results(self,
                               tour_list,
                               question_summary,
                               actual_routes: List[List[int]],
                               actual_labels: List[List[int]],
                               cf_routes: List[List[int]],
                               cf_labels: List[List[int]],
                               cf_step: int) -> str:
        comparison_results = "Question:\n" + question_summary + "\n"
        comparison_results += "Actual route:\n" + \
                                self.get_route_info(tour_list, actual_routes[0], actual_labels[0], cf_step-1, "actual") + \
                                self.get_representative_values(actual_routes[0], actual_labels[0], cf_step-1, "actual")
        comparison_results += "CF route:\n" + \
                                self.get_route_info(tour_list, cf_routes[0], cf_labels[0], cf_step-1, "CF") + \
                                self.get_representative_values(cf_routes[0], cf_labels[0], cf_step-1, "CF")
        comparison_results += "Difference between two routes:\n" + self.get_diff(cf_step-1, actual_routes[0], cf_routes[0])
        comparison_results += "Planed desination information:\n" + self.get_node_info()
        return comparison_results

    def get_route_info(self,
                       tour_list,
                       route: List[int],
                       label: List[int], 
                       ex_step: int, 
                       type: str) -> str:
        def get_labelname(label_number):
            return "route_len" if label_number == 0 else "time_window"
        route_info = "- route: "
        for i, node_id in enumerate(route):
            if i == ex_step and i != len(route) - 1:
                if type == "actual":
                    edge_label = {get_labelname(label[i])}
                else:
                    edge_label = "user_preference"
                route_info += f"{tour_list[node_id]['name']} > ({type} edge: {edge_label}) > "
            elif i != len(route) - 1:
                route_info += f"{tour_list[node_id]['name']} > ({get_labelname(label[i])}) > "
            else:
                route_info += f"{tour_list[node_id]['name']}\n"
        return route_info

    def get_representative_values(self, route, labels, ex_step, type) -> str:
        time_window_ratio = self.get_intention_ratio(1, labels, ex_step) * 100
        route_len_ratio = self.get_intention_ratio(0, labels, ex_step) * 100
        return f"- short-term effect (immediate travel time): {self.get_immediate_state(route, ex_step)//60} minutes\n- long-term effect (total travel time): {self.get_route_length(route)//60} minutes\n- missed nodes: {self.get_infeasible_node_name(route)}\n- edge-intention ratio after the {type} edge: time_window {time_window_ratio: .1f}%, route_len {route_len_ratio: .1f}%"

    def get_immediate_state(self, route, ex_step) -> str:
        return st.session_state.dist_matrix[route[ex_step]][route[ex_step+1]]

    def get_route_length(self, route) -> float:
        route_length = 0.0
        for i in range(len(route)-1):
            route_length += st.session_state.dist_matrix[route[i]][route[i+1]]
        return route_length

    def get_infeasible_nodes(self, route) -> int:
        return len(route) - (len(st.session_state.dist_matrix) - 1)

    def get_infeasible_node_name(self, route) -> str:
        if len(route) == len(st.session_state.dist_matrix) - 1:
            return "none"
        else:
            num_nodes = np.arange(len(st.session_state.dist_matrix))
            for node_id in route:
                num_nodes = num_nodes[num_nodes != node_id]
            return ",".join([st.session_state.tour_list[node_id]["name"] for node_id in num_nodes])

    def get_intention_ratio(self, 
                            intention: int, 
                            labels: List[int], 
                            ex_step: int) -> float:
        np_labels = np.array(labels)
        return np.sum(np_labels[ex_step:] == intention) / len(labels[ex_step:])

    def get_diff(self, ex_step, actual_route, cf_route) -> str:
        def get_str(effect: float):
            long_effect_str = "The actual route increases it by" if effect > 0 else "The actual route reduces it by"
            long_effect_str += util_app.add_time_unit(abs(effect))
            return long_effect_str
        
        def get_str2(num_nodes: int, num_missed_nodes):
            if num_nodes < 0:
                num_nodes_str = f"The actual route visits {abs(num_nodes)} more nodes" 
            elif num_nodes == 0:
                if num_missed_nodes == 0:
                    num_nodes_str = f"Both routes missed no node,"
                else:
                    num_nodes_str = f"Both routes missed the same number of nodes ({abs(num_missed_nodes)} node(s))"
            else:
                num_nodes_str = f"The actual route visits {abs(num_nodes)} less nodes" 
            return num_nodes_str

        # short/long-term effects
        short_effect = self.get_immediate_state(actual_route, ex_step) - self.get_immediate_state(cf_route, ex_step)
        long_effect  = self.get_route_length(actual_route) - self.get_route_length(cf_route)
        short_effect_str = get_str(short_effect)
        long_effect_str  = get_str(long_effect)

        # missed nodes
        missed_nodes = self.get_infeasible_nodes(actual_route) - self.get_infeasible_nodes(cf_route)
        missed_nodes_str = get_str2(missed_nodes, self.get_infeasible_nodes(actual_route))

        return f"- short-term effect: {short_effect_str}\n - long-term effect: {long_effect_str}\n- missed nodes: {missed_nodes_str}\n"

    def get_node_info(self) -> str:
        node_info = ""
        for i in range(len(st.session_state.df_tour)):
            node_info += f"- {st.session_state.df_tour['destination'][i]}: {st.session_state.df_tour['remarks'][i]}\n"
        return node_info