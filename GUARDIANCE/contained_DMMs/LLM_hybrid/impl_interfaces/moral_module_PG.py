from GUARDIANCE.interfaces.moral_module import Moral_Module
from GUARDIANCE.interfaces.data_processor import Data_Processor
from GUARDIANCE.reasoning_unit import ReasoningUnit
import logging

logger = logging.getLogger(__name__)

class Moral_Module_PG():
    def __init__(self, reasoning_unit:ReasoningUnit, data_processor:Data_Processor):
        self.reasoning_unit = reasoning_unit
        self.data_processor = data_processor
        self.extracted_data_cache = None
        self.guiding_rules_cache = None
        self.static_env_info = None

        # Track if there are morally relevant changes in the envrionment and wether possible changes affect the guiding rules 
        self.reasons_changed = True
        self.rules_changed = True

    def update_relevant_information(self, observation):
        self.reasons_changed = False
        extracted_data = self.data_processor.extract_relevant_information(self.reasoning_unit.reason_theory, observation, self.static_env_info)
        if extracted_data != self.extracted_data_cache:
            self.reasons_changed = True
        self.extracted_data_cache = extracted_data

    def update_guiding_rules(self):
        new_guiding_rules = self.guiding_rules_cache
        if self.reasons_changed:
            new_guiding_rules = self.reasoning_unit.moral_obligations(self.extracted_data_cache)
            if new_guiding_rules != self.guiding_rules_cache:
                self.guiding_rules_cache = new_guiding_rules
                self.rules_changed = True
            else:
                self.rules_changed = False
        else:
            self.rules_changed = False
        self.guiding_rules_cache = new_guiding_rules
    
    def DMM_observation(self):
        return self.data_processor.DMM_observation(self.extracted_data_cache, self.guiding_rules_cache)
    
    def guard_observation(self):
        return self.data_processor.guard_observation(self.extracted_data_cache, self.guiding_rules_cache)
        
    