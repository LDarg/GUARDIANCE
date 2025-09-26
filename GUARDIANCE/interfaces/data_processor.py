from abc import ABC, abstractmethod

class Data_Processor(ABC):
    @abstractmethod
    def extract_relevant_information(self, reason_theory, observation):
        pass
    """
    Collect/retrieve/extract data received from the environment that is normatively relevant.
    In particular, interpret raw data to identify morally relevant facts (reasons), which are currently expected by the reasoning unit to be represented in propositional form.
    The function depends on the environment and the reason theory. 

    Depending on the format of the environment's output, the function might need to do some processing/interpretation of the data.
    For example: If the environment outputs raw sensory data, the function might need to do object recognition to identify relevant objects and their properties.

    The extracted information takes the role of W (the background information) in Horty's formalization of reasoning.

    Args:
        reason_theory: Set of rules (represented in a graph structure) which encode normatively relevant reasons known to the agent. 
        observation: The raw observational data from the environment.

    Returns:
        information that is normatively relevant (and thus needed to make a decision on how to act while prioritizing normative requirements)
    """
    
    @abstractmethod
    def groundings_for_rule(self, extracted_data, rule):
        pass
    """
    determines if a proposition holds true based on the extracted data
    note that if a rule is triggered, there is a truthmaker of that rule; i.e. the rule is grounded. hence, all triggered rules have at least one grounding

    Args:
        extracted_data: information extracted by calling extract_relevant_information. 
        rule: a rule that is part of the agent's reason theory 

    Returns:
        A set of rules where each rule consists of a premise and conclusion.
        The input to the function includes additional information identifying the entity 
        whose attributes validate the premise. If no such entity exists, the empty set needs to be returned.
    """

    @abstractmethod
    def filter_and_prepare(self, data, guiding_rules, observation):
        pass
    """
    Receives as input: 
    - the extracted_data: interpreted raw data concerning objects and their properties as this information is needed by the DMM to make a strategy for acting in conformance with its obligations.
    - the observation: raw data based on which the model is able to execute a strategy for fulfilling its deployment purpose. 

    Outputs data in the format which the DMM expects including all information which the DMM needs for determining the next step of the agent.
    Also filters the data to only exclude information that should not be visible to the DMM (for performance reasons or the purpose of preventing hacking).
    """