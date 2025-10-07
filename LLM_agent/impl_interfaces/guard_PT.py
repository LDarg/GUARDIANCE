from preschool.config import Config
from LLM_agent.impl_interfaces.MAT_mapping_PT import MAT_mapping_PT
from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping

class Guard():
    def __init__(self, DMM, mat_mapping:MAT_Mapping):
        self.DMM = DMM
        self.mat_mapping = mat_mapping

    def ensure_conformity(self, action, guiding_rules, observation):
        MATs = [(rule[0][1],rule[1]) for rule in guiding_rules]
        for MAT in MATs:
            violated_obligation = self.mat_mapping.violated_obligation(self, action, MAT, observation)  
            if violated_obligation:  
                self.inform_overseer(action, violated_obligation)
                pass
                #first try to retrigger the DMM
                #LLM needs a memory before this works
                    #self.retrigger(action, guiding_rules, observation, DMM_observation)
                # if retriggering the DMM still does not provide the agent with a useful approach, select a default action and explain to the DMM why it was selected
                if not self.mat_mapping.action_conform_with_MATs(self, action, MATs, observation):  
                    action = self.mat_mapping.default_action(self, MATs, observation)
                    #tell the LLM why the action was selected
        return action

    def check_conformity(self, action, guiding_rules, observation):
        MATs = [(rule[0][1],rule[1]) for rule in guiding_rules]
        if not self.mat_mapping.action_conform_with_MATs(self, action, MATs, observation):  
            return False
        return True
    
    """
    GENERAL: inform the human overseer that the DMM wants to execute an action that is nonconform with a binding obligation
    """
    def inform_overseer(self, action, violated_obligation):
        pass

    def retrigger(self, action, violated_obligation):
        self.DMM.Rethink_Choice_Of_Action(action=action, obligation=violated_obligation)
