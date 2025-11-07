from abc import ABC, abstractmethod

class Guard(ABC):

    """
    monitor and ensure that the actions proposed by the DMM conform to the guiding rules
    """
    def ensure_conformity(self, action, guiding_rules, observation):
        MATs = [(rule[0][1],rule[1]) for rule in guiding_rules]
        for MAT in MATs:
            violated_obligation = self.mat_mapping.obligation_violated(action, MAT, observation) #TODO: ensure that function returns the obligation
            if self.mat_mapping.obligation_violated(action, MAT, observation):  
                self.inform_human(action, MAT) #TODO: that something was wrong
                self.retrigger(action, violated_obligation)
                #first try to retrigger the DMM
                #LLM needs a memory before this works
                    #action = self.retrigger(action, guiding_rules, observation, DMM_observation)
                # if retriggering the DMM still does not provide the agent with a useful approach, select a default action and explain to the DMM why it was selected
                action = self.mat_mapping.default_action( MATs, observation)
                return action

        return action
    
    """
    inform the human overseer that the DMM wants to execute an action that is not conform with a binding obligation
    """
    @abstractmethod
    def inform_human(self, action, violated_obligation):
        pass
