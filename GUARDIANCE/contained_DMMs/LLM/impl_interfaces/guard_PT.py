from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping
from GUARDIANCE.interfaces.guard import Guard
import logging

class Guard_PT(Guard):
    def __init__(self, mat_mapping:MAT_Mapping):
        self.mat_mapping = mat_mapping

    def ensure_conformity(self, action, guiding_rules, observation):
        MATs = [(rule[0][1],rule[1]) for rule in guiding_rules]
        for MAT in MATs:
            if self.mat_mapping.obligation_violated(action, MAT, observation):  
                self.inform_human(action, MAT)
                action = self.mat_mapping.default_action( MATs, observation)
                return action

        return action
    
    def inform_human(self, action, violated_obligation):
        logging.warning(f"The action {action} violates the obligation {violated_obligation}")
        pass
