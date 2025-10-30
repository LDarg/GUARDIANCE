from abc import ABC, abstractmethod

class Guard(ABC):

    """
    monitor and ensure that the actions proposed by the DMM conform to the guiding rules
    """
    def ensure_conformity(self, action, guiding_rules, observation):
        MATs = [(rule[0][1],rule[1]) for rule in guiding_rules]
        for MAT in MATs:
            if self.mat_mapping.obligation_violated(action, MAT, observation):  
                self.inform_human(action, MAT)
                action = self.mat_mapping.default_action( MATs, observation)

        return action
    
    """
    inform the human overseer that the DMM wants to execute an action that is not conform with a binding obligation
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