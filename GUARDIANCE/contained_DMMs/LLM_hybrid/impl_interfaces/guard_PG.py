from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping
from GUARDIANCE.interfaces.guard import Guard
import logging

logger = logging.getLogger(__name__)
class Guard_PG(Guard):
    def __init__(self, mat_mapping:MAT_Mapping):
        self.mat_mapping = mat_mapping

    def violated_obligation(self, action, guiding_rules, guard_observation):
        MATs = [(rule[0][1],rule[1]) for rule in guiding_rules]
        for MAT in MATs:
            violated_obligation = self.mat_mapping.obligation_violated(action, MAT, guard_observation) 
            if violated_obligation:  
                self.inform_human(action, MAT) 
                return violated_obligation

        return None

    """
    monitor and ensure that the actions proposed by the DMM conform to the guiding rules
    """
    def ensure_conformity(self, action, guiding_rules, observation):
        MATs = [(rule[0][1],rule[1]) for rule in guiding_rules]
        for MAT in MATs:
            violated_obligation = self.mat_mapping.obligation_violated(action, MAT, observation)
            if self.mat_mapping.obligation_violated(action, MAT, observation):  
                self.inform_human(action, MAT)
                self.retrigger(action, violated_obligation)
                action = self.mat_mapping.default_action( MATs, observation)
                return action

        return action
    
    def inform_human(self, action, violated_obligation):
        logging.warning(f"The action {action} violates the obligation {violated_obligation}")
        pass
