from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping
from GUARDIANCE.interfaces.guard import Guard
import logging

logger = logging.getLogger(__name__)
class Guard_PG(Guard):
    def __init__(self, DMM, mat_mapping:MAT_Mapping):
        self.DMM = DMM
        self.mat_mapping = mat_mapping

    def violated_obligation(self, action, guiding_rules, guard_observation):
        MATs = [(rule[0][1],rule[1]) for rule in guiding_rules]
        for MAT in MATs:
            violated_obligation = self.mat_mapping.obligation_violated(action, MAT, guard_observation) #TODO: ensure that function returns the obligation
            if violated_obligation:  
                self.inform_human(action, MAT) #TODO: that something was wrong
                return violated_obligation

        return None

    def ensure_conformity(self, action, guiding_rules, guard_observation, DMM_input):
        pass
    
    def inform_human(self, action, violated_obligation):
        logging.warning(f"The action {action} violates the obligation {violated_obligation}")
        pass

    def retrigger(self, action, violated_obligation, DMM_input):
        pass
        #self.DMM.Rethink_Choice_Of_Action(action=action, obligation=violated_obligation)
