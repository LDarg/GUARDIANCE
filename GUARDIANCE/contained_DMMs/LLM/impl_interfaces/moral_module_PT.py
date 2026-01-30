from GUARDIANCE.interfaces.moral_module import Moral_Module
from GUARDIANCE.interfaces.data_processor import Data_Processor
from GUARDIANCE.reasoning_unit import ReasoningUnit
import logging

logger = logging.getLogger(__name__)

class Moral_Module_PT():
    def __init__(self, reasoning_unit:ReasoningUnit, data_processor:Data_Processor):
        self.reasoning_unit = reasoning_unit
        self.data_processor = data_processor
        self.extracted_data_cache = None
        self.guiding_rules_cache = None

    def update_relevant_information(self, observation):
        self.extracted_data_cache = self.data_processor.extract_relevant_information(self.reasoning_unit.reason_theory, observation)

    def update_guiding_rules(self):
        self.guiding_rules_cache = self.reasoning_unit.moral_obligations(self.extracted_data_cache)
    
    def DMM_observation(self):
        return self.data_processor.DMM_observation(self.extracted_data_cache, self.guiding_rules_cache)
    
    def guard_observation(self):
        return self.data_processor.guard_observation(self.extracted_data_cache, self.guiding_rules_cache)
        
    