from itertools import chain, combinations
import networkx as nx
import random
import logging
from GUARDIANCE.interfaces.data_processor import Data_Processor
from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping

logger = logging.getLogger(__name__)

class ReasoningUnit():
    def __init__(self, MAT_mapping: MAT_Mapping, data_processor: Data_Processor):
        super().__init__()
        self.reason_theory = nx.DiGraph()
        self.MAT_mapping = MAT_mapping
        self.data_processor = data_processor

        self.threshold_waiting = 0.8
        self.chosen_scenario = None

        self.conflicted = set()

    @classmethod
    def int_to_subscript(self, number):
        subscript_offset = 8320  
        return ''.join(chr(subscript_offset + int(digit)) for digit in str(number))

    """
    updates the agent's reason theory based on feedback provided by a moral judge
    """
    def update(self, reason): #chosen (proper) scenario:the set of rules that the agent took for guiding its behavior

        #check if agent knows the morally relevant fact 
        if not reason[0] in self.reason_theory:
            self.reason_theory.add_node(reason[0], type='morally relevant fact')
        
        #check if action type is known to the agent as potential moral obligation
        if not reason[1] in self.reason_theory:
            self.reason_theory.add_node(reason[1], type='moral obligation')

        #check if connection between morally relevant fact and moral obligation is known to the agent
        if not self.reason_theory.has_edge(reason[0], reason[1]):
            edge_count = self.reason_theory.number_of_edges()
            self.reason_theory.add_edge(reason[0], reason[1], lower_order=set(), name=f"δ{self.int_to_subscript(edge_count+1)}")

        # correct the order such that the reason is prioritized over all reasons in the chosen scenario
        lower_order = self.reason_theory.get_edge_data(reason[0], reason[1])['lower_order']

        if self.chosen_scenario:
            for rule in self.chosen_scenario:
                if rule != reason:
                    lower_order.add(rule)
                
        nx.set_edge_attributes(self.reason_theory, {(reason[0], reason[1]): {'lower_order': lower_order}})
        self.log_reason_theory()

    """
    log the agent's current reason theory
    """
    def log_reason_theory(self, logger):
        log_lines = []
        log_lines.append("Current Reason Theory:")
        for reason, obligation, priority in self.reason_theory.edges(data=True):
            log_lines.append(f"Reason: {reason}, Obligation: {obligation} with priority order: {priority}")

        # Join all lines into one single string separated by newlines
        log_message = "\n".join(log_lines)

        logger.info(log_message)

    def powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    """
    given the current state of the envrionment and the morally relevant facts, let the agent derive which moral obligation it has according to his current reason theory
    the proper scenarios are the ones for which the agent has no tiebreak; i.e. it chooses one randomly
    """
    def moral_obligations(self, extracted_data):
        proper_scenarios = []
        moral_obligations = []

        # returns all groundings of all triggered rules (all rules in the reason whose antecedence holds true together with all their truthmakers)
        groundings  = self.groundings(extracted_data)

        #check if chosen scenario (course of action followed so far) is still a proper scenario and follow it further if it is
        if self.chosen_scenario in groundings and self.compute_binding(self.chosen_scenario, groundings, extracted_data):
            for rule in self.chosen_scenario:
                moral_obligations.append(rule)
            return moral_obligations

        # only subsets of the triggered_rules can be binding
        for scenario in self.powerset(groundings):
            if self.compute_binding(set(scenario), groundings, extracted_data):
                proper_scenarios.append(set(scenario))

        #Among the proper scenarios, check whether goals should be prioritized or if the DMM is supposed to follow one strategy for fulfilling all goals  
        for scenario in proper_scenarios:
            for rule in set(scenario):
                if self.reason_theory.nodes[rule[0][1]].get('subclass') == 'goal':
                    while True:
                        less_prio_goal = self.trade_off_priorities(rule[0], scenario)
                        if less_prio_goal:
                            rule_to_remove = next((rule for rule in scenario if rule[0][0] == less_prio_goal[0]), None)
                            if rule_to_remove:
                                scenario.remove(rule_to_remove)
                            continue
                        break
                        
        proper_scenario = random.choice(list(proper_scenarios))
        self.chosen_scenario = proper_scenario
        for rule in proper_scenario:
            moral_obligations.append(rule)
        return moral_obligations
    
    
    def trade_off_priorities(self, rule, scenario):
        edge_data = self.reason_theory.get_edge_data(rule[0], rule[1])
        trade_off_prios = edge_data.get('prio_trade_off', [])

        if not trade_off_prios:
            return False

        for other_rule in scenario:
            other_rule = other_rule[0]
            if other_rule in trade_off_prios:
                return other_rule

        for next_rule in trade_off_prios:
            lesser_prio_goal =  self.trade_off_priorities(next_rule, scenario)
            if lesser_prio_goal:
                return lesser_prio_goal

        return False
    
    """
    Returns rules whose antecedences hold true  given the background information and the conclusions of rules in the reason theory.
    The truthness of the propositions is determined based on information that is extracted in the data_processor.
    """
    def groundings(self, extracted_data):
        groundings= set()
        if extracted_data:
            for rule in self.reason_theory.edges():
                for grounding in self.data_processor.groundings_for_rule(extracted_data, rule):
                    groundings.add(grounding)
        return groundings
    
    """
    Compute whether a subset of grounded rules constitutes a binding scenario.
    """
    def compute_binding(self, grounded_rules_scenario, triggered_grounded_rules, extracted_data):

        triggerd_rules_not_in_scenario = set(triggered_grounded_rules) - set(grounded_rules_scenario)
        

        # If there are rules within the scenario that are conflicting, the scenario is not binding
        scenario_conflicted = self.MAT_mapping.execution_conflicted(
            grounded_rules_scenario, extracted_data
        )
        if scenario_conflicted:
            return False
        
        # For each triggered rule not included in the scenario, check whether it conflicts
        # with at least one rule in the scenario. If a triggered rule can be jointly executed
        # with the scenario rules, the scenario is not binding.
        unconflicted_triggered_rules = self.unconflicted(
            grounded_rules_scenario, triggerd_rules_not_in_scenario, extracted_data
        )
        if unconflicted_triggered_rules:
            return False
        
        # Check priority relations:
        # If a conflicting external rule has higher priority than a scenario rule,
        # there must exist a scenario rule that defeats it; otherwise, the scenario
        # is not binding.
        defeated_rule_in_scenario = self.defeated(
            grounded_rules_scenario, triggerd_rules_not_in_scenario
        )
        if defeated_rule_in_scenario:
            return False

        return True


    def unconflicted(self, grounded_rules_scenario, difference, extracted_data): 

        conflicted_rules = set()

        for rule in difference:
            if self.MAT_mapping.execution_conflicted(grounded_rules_scenario.union({rule}), extracted_data):
                conflicted_rules.add(rule)

        return difference - conflicted_rules

    def defeated(self, grounded_rules_scenario, triggered_rules_not_in_scenario):

        for rule in triggered_rules_not_in_scenario:
            for second_rule in grounded_rules_scenario:
                 lower_order_rules_1 = self.reason_theory.get_edge_data(rule[0][0], rule[0][1])['lower_order']
                 lower_order_rules_1 = self._ensure_edge_tuple(lower_order_rules_1)
                 if (second_rule[0][0],second_rule[0][1]) in lower_order_rules_1:
                    for third_rule in grounded_rules_scenario:
                        lower_order_rules_2 = self.reason_theory.get_edge_data(third_rule[0][0], third_rule[0][1])['lower_order']
                        lower_order_rules_2 = self._ensure_edge_tuple(lower_order_rules_2)
                        if (rule[0][0], rule[0][1]) in lower_order_rules_2:
                           return False 
                    return True
        return False

    def _ensure_edge_tuple(self, lo):
        if not lo:
            return ()
        if isinstance(lo[0], str):
            return (lo,)
        return lo