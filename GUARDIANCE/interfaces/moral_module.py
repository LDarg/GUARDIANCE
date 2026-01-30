from abc import ABC, abstractmethod
from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping
from GUARDIANCE.reasoning_unit import ReasoningUnit
import logging

class Moral_Module(ABC):
    def __init__(self, reasoning_unit: ReasoningUnit):
        self.reasoning_unit: ReasoningUnit = reasoning_unit

    """
    Args:
        extracted_data: the data that is extracted from the environment that is normatively relevant 

    Returns:
        the guiding rules that determine which normative requirements the agent needs to conform to
    """
    @abstractmethod
    def guiding_rules(self, extracted_data):
        return self.reasoning_unit.moral_obligations(extracted_data)
    

