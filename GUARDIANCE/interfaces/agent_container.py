from abc import ABC, abstractmethod
from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping
from GUARDIANCE.interfaces.guard import Guard
from GUARDIANCE.interfaces.DMM import DMM
from GUARDIANCE.interfaces.data_processor import Data_Processor
from GUARDIANCE.reasoning_unit import ReasoningUnit
from GUARDIANCE.interfaces.moral_module import Moral_Module
import logging

class Agent_Container(ABC):
    def __init__(self, mat_mapping: MAT_Mapping, dmm: DMM,data_processor: Data_Processor, guard: Guard, reasoning_unit: ReasoningUnit, moral_module: Moral_Module):
        self.mat_mapping: MAT_Mapping = mat_mapping
        self.DMM: DMM = dmm
        
        self.data_processor: Data_Processor = data_processor
        self.guard: Guard = guard
        self.reasoning_unit: ReasoningUnit = reasoning_unit
        self.moral_module: Moral_Module = moral_module

    def take_action(self, observation):
        pass

    