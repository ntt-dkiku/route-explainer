from abc import ABC, abstractmethod
from typing import Dict, Any 
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import Runnable

class TemplateJsonBase(ABC):
    parser: Runnable
    template: str
    prompt: Runnable

    @abstractmethod
    def _get_output_key(self) -> str:
        raise NotImplementedError

    def get_template(self) -> str:
        return self.template
    
    def extract_value(self, input: Dict[str, Any]) -> Any:
        return input[self._get_output_key()]

    def sandwiches(self, 
                   llm: Runnable,
                   extract_value: bool = False) -> Runnable:
        if extract_value:
            return self.prompt | llm | self.parser | RunnableLambda(self.extract_value)
        else:
            return self.prompt | llm | self.parser