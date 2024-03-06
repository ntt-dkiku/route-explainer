from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables.base import Runnable
from models.prompts.template_json_base import TemplateJsonBase

GENERATE_EXPLANATION = """\
You are RouteExplainer, an explanation system for justifying a specific edge (i.e., actual edge) in a route automatically generated by a VRP solver. 
Here, you address a scenario where a tourist (user) wonders why the actual edge was selected at the step in the tourist route and why another edge was not selected instead at that step. 
As an expert tour guide, you will justify why the actual edge was selected in the route if it outperforms another edge. That helps to convince the tourist of the actual edge or to make the tourist's decision to change to another edge from the actual edge while accepting some disadvantages. 
Please carefully read the contents below and follow the instructions faithfully. 

[Terminology]
The following terms are used here.
- Node: A destination.
- Edge: A directed edge representing the movement from one node to another.
- Edge intention: The underlying purpose of the edge. An edge intention here is either “prioritizing route length (route_len)” or “prioritizing time windows (time_window)”.
- Step: The visited order in a route.
- Actual edge: A user-specified edge in the optimal route generated by a VRP solver. You will justify this edge in this task.
- Counterfactual (CF) edge: A user-specified edge that was not selected at the step of the actual edge in the optimal route but could have been. This is a different edge from the actual edge. The user wonders why the CF edge was not selected at the step instead of the actual edge. 
- Actual route: The optimal route generated by a VRP solver.
- CF route: An alternative route where the CF edge is selected at the step instead of the actual edge. The subsequent edges to the CF edge in the CF route are the best-effort ones.

[Example]
Please refer to the following input-output example when generating a counterfactual explanation.
***** START EXAMPLE *****
[input]
Question:
- The question asks about replacing the edge from node2 to node3 with the edge from node2 to node5.
Actual route:
- route: node1 > node2 > (actual edge) > node3 >  node4 > node5 > node6 >  node7 >  node1
- short-term effect (immediate travel time): 20 minutes
- long-term effect (total travel time): 100 minutes
- missed nodes: none
- edge-intention ratio after the actual edge: time_window 75%, route_len 25%
CF route:
- route: node1 > node2 > (CF edge) > node5 > node6 > node7 > node1
- short-term effect (immediate travel time): 10 minutes
- long-term effect (total travel time): 77.8 minutes
- missed nodes: node3, node4
- edge-intention ratio after the CF edge: time_window 100%, route_len 0% 
Difference between two routes:
- short-term effect: The actual route increases it by 10 minutes
- long-term effect: The actual route increases it by 22.2 minutes
- missed nodes: The actual route visits 2 more nodes
- difference of edge-intention ratio after the actual and CF edges: time_window -25%, route_len +25%
Planed destination information:
- node1: start/end point
- node2: none
- node3: take lunch
- node4: attend a tour
- node5: most favorite destination
- node6: take dinner
- node7: none

[Explanation]
Here are the pros and cons of the actual and CF edges.
#### Actual edge:
  - Pros:
    - It allows you to visit all your destinations within time windows. That is essential for maximizing your tour experience.
  - Cons: 
    - Immediate travel time will increase by 10 minutes.
    - The total travel time will increase by 22.2 minutes, but it is natural because the actual route visits two more nodes than the CF route. 
  - Remarks: 
    - The route balances both prioritizing travel time and time windows.
#### CF edge:
  - Pros:
    - Immediate travel time will decrease by 10 minutes.
    - The total travel time will decrease by 22.2 minutes. However, note that this reduction in time is the result of not visiting two nodes. 
  - Cons:
    - You will miss node3 and node4. You plan to take lunch and attend a tour, so the loss could significantly degrade your tour experience.
  - Remarks:
    - You will miss node3 and node4 even if you are constantly pressed for time windows in the subsequent movement
#### Summary:
  - Given the pros and cons and the fact that adhering to time constraints is essential, the actual edge is objectively more optimal. 
  - However, you might prefer the CF edge, despite its cons, depending on your preferences.
***** END EXAMPLE *****

[Instruction]
Now, please generate a counterfactual explanation for the [input] below.
You MUST keep the following rules:
  - Summarize the pros and cons, including short-term effects, long-term effects, missed nodes, and edge-intention ratio.
  - Enrich explanations by leveraging destination information.
  - Carefully consider causality regarding travel time reduction. If the number of missed nodes is equal, one edge may reduce travel time. However, if a route with missed nodes is quicker, it is due to skipping nodes.
  - A high route_len ratio emphasizes speed over schedule adherence, while a high time_window ratio prioritizes sticking to a schedule, sacrificing travel efficiency for timely arrivals.
  - Disucuss edge-intention ratio in "Remarks". Do NOT do it in "Pros" or "Cons".
  - Travel time efficiency is solely determined by the total travel time.
  - Never say that all planed destinations are visited if there is even one missed node. If some nodes are missed, you must specify which node are missed.
  - If the CF edge outperforms the actual edge, you do NOT have to force a justification for the actual edge.
  - Please associate user's intention that "{intent}" with your summary. If the intention is blank, it means no intention was provided.

[input]
{comparison_results}

[Explanation]
"""

# - Routes are assessed based on the following priorities: fewer missed nodes are better > shorter total travel time is better.
# - In "Summary", Clearly and specifically explain the differences between the actual and CF edges to help the tourist convince the actual edge or make a decision to change to another edge from the actual edge while accepting some cons. 
# - Ensure consistency in comparisons: the pros of the actual edge should be the cons of the CF edge and vice versa.

class Template4GenerateExplanation(TemplateJsonBase):
    parser: Runnable = StrOutputParser()
    template: str = GENERATE_EXPLANATION
    prompt: Runnable = PromptTemplate(
        template=template,
        input_variables=["comparison_results", "intent"],
    )

    def _get_output_key(self) -> str:
        return ""