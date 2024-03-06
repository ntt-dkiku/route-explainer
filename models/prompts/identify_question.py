from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.base import Runnable
from models.prompts.template_json_base import TemplateJsonBase

IDENTIFY_QUESTION = """\
Given a route and a question that asks what would happen if we replaced a specific edge in the route with another edge, please extract the step number of the replaced edge (i.e., cf_step) and the node id of the destination of the new edge (i.e., cf_visit) from the question, which is written in natural language.
Please use the following examples as a reference when you answer:
***** START EXAMPLE *****
[route info]
Nodes(node id, name): (1, node1), (2, node2), (3, node3), (4, node4), (5, node5)
Route: node1 > (step1) > node5 > (step2) > node3 > (step3) > node2 > (step4) > node4 > (step 5) > node1
[question] 
Why node3, and why not node2?
[outputs]
```json
{{
    "success": true,
    "summary": "The answer asks about replacing the edge from node5 to node3 with the edge from node6 to node2.",
    "intent": "",
    “process": "The edge from node5 to node3 is at step2 because of "node5 > (step2) > node3". The node id of the destination of the new edge is 2 (node2). Thus, the final answers are cf_step=2 and cf_visit=2.",
    "cf_step": 2,
    "cf_visit": 2,
}}
```

[route info]
Nodes(node id, name): (1, node1), (2, node2), (3, node3), (4, node4), (5, node5)
Route: node1 > (step1) > node5 > (step2) > node3 > (step3) > node2 > (step4) > node4 > (step 5) > node1
[quetsion]
What if we visited node4 instead of node2? We would personally like to visit node4 first.
[outputs]
```json
{{
    "success": true,
    "summary": "The answer asks about replacing the edge from node3 to node2 with the edge from node3 to node4.",
    “intent": "The user would personally like to visit node4 first""
    "process": "The edge from node3 to node2 is at step3 because of "node3 > (step3) > node2". The node id of the destination of the new edge is 4 (node4). Thus, the final answers are cf_step=3 and cf_visit=4.",
    "cf_step": 3,
    "cf_visit": 4,
}}
```
***** END EXAMPLE *****

Given the following route and question, please extract the step number of the replaced edge (i.e., cf_step) and the node id of the destination of the new edge (i.e., cf_visit) from the question.
Please keep the following rules:
- Do not output any sentences outside of JSON format.
- {format_instructions}

[route_info]
{route_info}
[question] 
{whynot_question}
[outputs]
"""

class WhyNotQuestion(BaseModel):
    success: bool = Field(description="Whether cf_step and cf_visit are successfully extracted (True) or not (False).")
    summary:  str = Field(description="Your summary for the given question. If success=False, instead state here what information is missing to extract cf_step/cf_visit and what additional information should be clarified (Additionally, provide an example).")
    intent:   str = Field(description="Your summary for user's intent (if provided). If not provided, this is set to ''.")
    process:  str = Field(description="The thought (reasoning) process in extracting cf_step and cf_visit. if success=False, this is set to ''.")
    cf_step:  int = Field(description="The step number of the replaced edge. if success=False, this is set to -1.")
    cf_visit: int = Field(description="The node id of the destination of the new edge. if success=False, this is set to -1.")

class Template4IdentifyQuestion(TemplateJsonBase):
    parser: Runnable = JsonOutputParser(pydantic_object=WhyNotQuestion)
    template: str = IDENTIFY_QUESTION
    prompt: Runnable = PromptTemplate(
        template=template,
        input_variables=["whynot_question", "route_info"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    def _get_output_key(self) -> str:
        return ""