from abc import ABC, abstractmethod
from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping
from GUARDIANCE.interfaces.guard import Guard
from GUARDIANCE.interfaces.DMM import DMM
from GUARDIANCE.interfaces.data_processor import Data_Processor
from GUARDIANCE.reasoning_unit import ReasoningUnit
import logging

class Agent_Container(ABC):
    def __init__(self, mat_mapping: MAT_Mapping, dmm: DMM,data_processor: Data_Processor, guard: Guard, reasoining_unit: ReasoningUnit):
        self.mat_mapping: MAT_Mapping = mat_mapping
        self.DMM: DMM = dmm
        self.data_processor: Data_Processor = data_processor
        self.guard: Guard = guard
        self.reasoning_unit: ReasoningUnit = reasoining_unit

    """
    monitor and ensure that the actions proposed by the DMM conform to the guiding rules

      Args:
        extracted_data: information extracted from the observation of the environment

    Returns:
        the action that the DMM selects to exectute
    """
    def take_action(self, observation):
        pass

