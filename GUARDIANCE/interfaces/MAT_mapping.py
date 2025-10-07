from abc import ABC, abstractmethod

class MAT_Mapping(ABC):
    @abstractmethod
    def obligation_violated(self, action, MAT, observation):
         pass
         """
         Check if the given action conforms with the MAT in the given observation.

         Args:
            action: The primitive action selected by the DMM for execution.
            MAT : The MAT (macro-action-type) the action needs to be conform with.
            observation: The current state of the environment as perceived by the agent extended with  by the MM. 

         Returns:
            bool: True if action conforms with the MAT in the state, False otherwise.
         """

    @abstractmethod
    def execution_conflicted(self, set_of_rules, extracted_data):
        pass
        """
        determines whehter an action is conform with a MAT rule (function called by the moral module after the agent has selected its action)
        the methods that can be applied here and the type of guarantees that can be given depend on the environment and the action space of the agent
        
         Args:
            set_of_rules: Set of rules whose conclusions are checked whether they are conflicted (can not be fulfilled simultaneously). 
            extracted_data: The information relevant for deciding whether the set of rules is conflicting; extracted by interpretation from the raw observation data by the data processor.

         Returns:
            bool: True if the conclusions in the set of rules are conflicted, False otherwise.
        """

    """
    execute a default action if the DMM fails to decide on an action that is conform with the guiding rules
    if it is not possible to simply calculate a default action that is conform with moral goals, the overall agent could be simply instructed to do nothing (or move out of the way if it is a physical system or call for human advisory etc.) to prevent causing harm
    """
    def default_action(self, MATs, observation):
        pass


