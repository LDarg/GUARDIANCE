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
        groundings  = self.groundings(self.reason_theory, extracted_data)

        # only subsets of the triggered_rules can be binding
        for scenario in self.powerset(groundings):
            if self.compute_binding(set(scenario), groundings, extracted_data):
                proper_scenarios.append(set(scenario))

        #if there are several proper scenarios, let the agent choose one randomly; (buridan's ass)
        proper_scenario = random.choice(list(proper_scenarios))
        self.chosen_scenario = proper_scenario
        for rule in proper_scenario:
            moral_obligations.append(rule)
        return moral_obligations
    
    """
    Returns rules whose antecedences hold true  given the background information and the conclusions of rules in the reason theory.
    The truthness of the propositions is determined based on information that is extracted in the data_processor.
    """
    def groundings(self, reason_theory, extracted_data):
        groundings= set()
        if extracted_data:
            for rule in reason_theory.edges():
                for grounding in self.data_processor.groundings_for_rule(extracted_data, rule):
                    groundings.add(grounding)
        return groundings
    
    """
    compute the binding rules for a subset of the agent's rules
    """
    def compute_binding(self, grounded_rules_scenario, triggered_grounded_rules, extracted_data):

        triggerd_rules_not_in_scenario = set(triggered_grounded_rules) - set(grounded_rules_scenario)
        

        # prüft, ob regeln in dem scenario conflicted sind, wenn ja, ist das scenario nicht binding
        scenario_conflicted = self.MAT_mapping.execution_conflicted(grounded_rules_scenario, extracted_data)
        if scenario_conflicted:
            return False
        
        # prüft für jede regel, die triggered ist, aber nicht in dem scenario, ob die regel conflicted ist mit einer regel in dem scenario;
        # wenn das nicht der fall ist, dann ist das scenario nicht binding (es werden regeln nicht berücksichtigt, die zusammen mit denen in dem scenario ausgeführt werden können)
        unconflicted_triggered_rules = self.unconflicted(grounded_rules_scenario, triggerd_rules_not_in_scenario, extracted_data)
        if unconflicted_triggered_rules:
            return False
        
        # hier steht schon fest, dass alle regeln in triggerd_rules_not_in_scenario conflicted mit mindestens einer regel in dem scenario sind
        # prüft, ob eine regel in triggerd_rules_not_in_scenario eine höhere priorität hat als eine regel in dem scenario; wenn ja, dann muss es auch eine regel in dem scenario geben, die eine höhere priorität hat als die regel in triggerd_rules_not_in_scenario; sonst ist das scenario nicht binding
        defeated_rule_in_scenario = self.defeated(grounded_rules_scenario, triggerd_rules_not_in_scenario)
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

        #wenn es eine regel r1 (triggered und nicht Teil des scenarios) gibt, die conflicted ist mit einer regel r2 in dem scenario und eine höhere priorität hat als r2, dann muss es eine weitere regel r3 in dem scenario geben, die eine höhere priorität hat als die regel r1
        for rule in triggered_rules_not_in_scenario:
            for second_rule in grounded_rules_scenario:
                 if (second_rule[0][0],second_rule[0][1]) in self.reason_theory.get_edge_data(rule[0][0], rule[0][1])['lower_order']:
                     if not any((rule[0][0], rule[0][1]) in self.reason_theory.get_edge_data(third_rule[0][0], third_rule[0][1])['lower_order'] for third_rule in grounded_rules_scenario):
                         return True
        return False

