from GUARDIANCE.interfaces.agent_container import Agent_Container
from GUARDIANCE.reasoning_unit import ReasoningUnit
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.MAT_mapping_PG import MAT_mapping_PG
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.data_processor_PG import Data_Processor_PG
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.guard_PG import Guard_PG
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.moral_module_PG import Moral_Module_PG
import logging
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.LLM_hybrid import LLM_hybrid

logger = logging.getLogger(__name__)


class Agent_Container_PG(Agent_Container):
    def __init__(self):
        self.mat_mapping = MAT_mapping_PG()
        self.data_processor = Data_Processor_PG()
        self.DMM = LLM_hybrid()
        self.guard = Guard_PG(self.mat_mapping)
        reasoning_unit = ReasoningUnit(self.mat_mapping, self.data_processor)
        self.moral_module = Moral_Module_PG(reasoning_unit, self.data_processor)

        self.static_env_info = None

        # Normatively relevant information used for applying a reason theory to ensure normative conformity
        self.normative_reasons = None
        self.guiding_rules = set()

    def take_action(self, observation):
        rl_obs, observation = observation

        extracted_data = self.data_processor.extract_relevant_information(
            self.moral_module.reasoning_unit.reason_theory,
            observation,
            self.static_env_info,
        )

        if self.guiding_rules is None:
            self.guiding_rules = self.moral_module.guiding_rules(extracted_data)

        current_reasons = extracted_data["children"] | extracted_data["happenings"]
        reasons_changed = current_reasons != self.normative_reasons

        if reasons_changed:
            self.normative_reasons = current_reasons
            new_guiding_rules = self.moral_module.guiding_rules(extracted_data)

            # Check whether the situation changed such that new rules are guiding for conforming with normative requirements
            if new_guiding_rules != self.guiding_rules:
                self.guiding_rules = new_guiding_rules
                rules_changed = True
            else:
                rules_changed = False
        else:
            rules_changed = False

        DMM_observation = self.data_processor.DMM_observation(
            extracted_data,
            self.guiding_rules,
        )

        DMM_input = {
            "reasons_changed": rules_changed,
            "rl_obs": rl_obs,
            "DMM_observation": DMM_observation,
            "guiding_rules": self.guiding_rules,
        }

        action = self.DMM.take_action(DMM_input)

        # Call the guard to check whether the action the agent wants to execute is conform with the binding rules; possibly initiate course correction
        guard_observation = self.data_processor.guard_observation(extracted_data, self.guiding_rules)
        violated_obligation = self.guard.violated_obligation(action, self.guiding_rules, guard_observation)
        if violated_obligation:
            self.DMM.add_feedback(violated_obligation, action)
            action = self.DMM.retrigger(rl_obs)
            violated_obligation = self.guard.violated_obligation(action, self.guiding_rules, guard_observation)
            if violated_obligation:
                MATs = [(rule[0][1],rule[1]) for rule in self.guiding_rules]
                action = self.mat_mapping.default_action(MATs, guard_observation)
            self.course_correction = True

        return action
