from abc import ABC, abstractmethod

class Guard(ABC):

    """
    the actions might be from the action space of the environment or decisions made on a higher level of abstraction as decided as part of the reasoning of an LLM-powerd DMM 
    """
    def ensure_conformity(self, action, guiding_rules, observation):
        MATs = [(rule[0][1],rule[1]) for rule in guiding_rules]
        for MAT in MATs:
            if self.mat_mapping.obligation_violated(action, MAT, observation):  
                self.inform_human(action, MAT)
                pass
                #first try to retrigger the DMM
                #LLM needs a memory before this works
                    #action = self.retrigger(action, guiding_rules, observation, DMM_observation)
                # if retriggering the DMM still does not provide the agent with a useful approach, select a default action and explain to the DMM why it was selected
                if self.mat_mapping.obligation_violated(action, MAT, observation):  
                    action = self.mat_mapping.default_action( MATs, observation)
                    #tell the LLM why the action was selected
        return action
    
    """
    inform the human human that the DMM wants to execute an action that is nonconform with a binding obligation
    """
    @abstractmethod
    def inform_human(self, action, violated_obligation):
        pass

    """
    inform the DMM about the noncomformity of its action with a guiding rule and request another action
    """
    @abstractmethod
    def retrigger(self, action, violated_obligation):
        pass