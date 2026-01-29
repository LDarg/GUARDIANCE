from abc import ABC, abstractmethod
from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping
import logging

class Guard(ABC):
    def __init__(self, mat_mapping: MAT_Mapping):
        self.mat_mapping: MAT_Mapping = mat_mapping

    """
    monitor and ensure that the actions proposed by the DMM conform to the guiding rules

      Args:
        action: the action that the DMM wants to execute
        guiding_rules: rules that the agent need to conform to 
        observation: the information of the state of the environment that is normatively relevant 

    Returns:
        the action that the DMM selected if it is conform; another action that is conform elsewise
    """
    @abstractmethod
    def ensure_conformity(self, action, guiding_rules, observation):
        pass
    
    """
    inform the human overseer that the DMM wants to execute an action that is not conform with a binding obligation
    """
    @abstractmethod
    def inform_human(self, action, violated_obligation):
        pass
