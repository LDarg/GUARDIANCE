from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping
from GUARDIANCE.interfaces.guard import Guard
import logging

class Guard_PT(Guard):
    def __init__(self, DMM, mat_mapping:MAT_Mapping):
        self.DMM = DMM
        self.mat_mapping = mat_mapping
    
    def inform_human(self, action, violated_obligation):
        logging.warning(f"The action {action} violates the obligation {violated_obligation}")
        pass

    def retrigger(self, action, violated_obligation):
        pass
        #self.DMM.Rethink_Choice_Of_Action(action=action, obligation=violated_obligation)
