from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping
from GUARDIANCE.interfaces.guard import Guard

class Guard_PT(Guard):
    def __init__(self, DMM, mat_mapping:MAT_Mapping):
        self.DMM = DMM
        self.mat_mapping = mat_mapping

    def ensure_conformity(self, action, guiding_rules, observation):
        MATs = [(rule[0][1],rule[1]) for rule in guiding_rules]
        for MAT in MATs:
            if self.mat_mapping.obligation_violated(action, MAT, observation):  
                self.inform_human(action, MAT)
                pass
                #first try to retrigger the DMM
                #LLM needs a memory before this works
                    #action =self.retrigger(action, guiding_rules, observation, DMM_observation)
                # if retriggering the DMM still does not provide the agent with a useful approach, select a default action and explain to the DMM why it was selected
                if self.mat_mapping.obligation_violated(action, MAT, observation):  
                    action = self.mat_mapping.default_action( MATs, observation)
                    #tell the LLM why the action was selected
        return action
    
    def inform_human(self, action, violated_obligation):
        pass

    def retrigger(self, action, violated_obligation):
        self.DMM.Rethink_Choice_Of_Action(action=action, obligation=violated_obligation)
