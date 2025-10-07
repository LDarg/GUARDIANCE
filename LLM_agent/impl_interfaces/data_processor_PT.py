from collections import defaultdict
from GUARDIANCE.interfaces.data_processor import Data_Processor
import copy

"""
A data_processor for the text version of the text-version of the preschool setting navigated by an LLM agent.
"""
class Data_Processor_PT(Data_Processor):
    def __init__(self):
        self.extracted_information = None
    
    """
    Input from the environment is already morally relevant facts in propositional form together with the location where they are situated.
    The function just retrieves the propositions along with this information.
    Return the reason as first element and the entities for which the reason holds (th grounding of the reason) as second element.
    """
    def extract_relevant_information(self, reason_theory, observation):
        reasons = [reason[0] for reason in list(reason_theory.edges())]
        relevant_data = defaultdict(dict)
        relevant_data["children"] = {
            item[0]: {
                "child_id": item[0],
                "description": item[1].replace(" ", "_"),
                "zone_name": item[2],
                "zone_id": item[3]
            }
            for item in observation["children"]
            if item[1].replace(" ", "_") in reasons
        }
        relevant_data["zones"] = {
            item[0]: {
                "zone_id": item[0],
                "description": item[1].replace(" ", "_"),
                "zone_name": item[2]
            }
            for item in observation["zones"]
            if item[1].replace(" ", "_") in reasons
        }
        relevant_data["agent_zone"] = observation["agent_zone"]
        relevant_data["stations_zones"] = observation["stations_zones"]
        relevant_data["stations_zones"] = observation["zone_ids"]

        self.extracted_information = relevant_data
        
        return relevant_data
    
    def groundings_for_rule(self, extracted_data, rule):
        groundings = set()
        truthmakers = [item for item in extracted_data["children"].items() if item[1].get("description") == rule[0]] + \
                      [item for item in extracted_data["zones"].items() if item[1].get("description") ==  rule[0]]
        for truthmaker in truthmakers:
            resolution = rule[1]
            groundings.add(((truthmaker[1]["description"], resolution), truthmaker[0]))
        return groundings
    
    def filter_and_prepare(self, data, guiding_rules, observation):

        child_conditions = [{"child_id": str(data["children"][rule[1]]["child_id"]),
                             "reason": rule[0][0],
                             "required_MAT": rule[0][1],
                             "zone_name": data["children"][rule[1]]["zone_name"],
                             "zone_id": str(data["children"][rule[1]]["zone_id"])}
                            for rule in guiding_rules if rule[0][0] in [child["description"] for child in data["children"].values()]]
        
        happenings = [{"zone_id": str(data["zones"][rule[1]]["zone_id"]),
                          "reason": rule[0][0],
                          "required_MAT": rule[0][1],
                          "zone_name": data["zones"][rule[1]]["zone_name"]}
                       for rule in guiding_rules if rule[0][0] in [zone["description"] for zone in data["zones"].values()]]

        DMM_input = {"child_conditions": child_conditions, "happenings": happenings}
        DMM_input["stations_zones"] = data["stations_zones"]
        DMM_input["agent_zone"] = {"zone_id": str(data["agent_zone"])}
        DMM_input["zone_ids"] = [{"zone_id": str(zone_id)} for zone_id in observation["zone_ids"]]
        
        return DMM_input
    
    """
    all data is passed to the DMM; no filtering is applied.
    """
    def DMM_observation(self, data, guiding_rules):
        
        #TODO: check if this makes sense; then, data does not need to be passed 
        
        child_conditions = [{"child_id": str(data["children"][rule[1]]["child_id"]),
                             "reason": rule[0][0],
                             "required_MAT": rule[0][1],
                             "zone_name": data["children"][rule[1]]["zone_name"],
                             "zone_id": str(data["children"][rule[1]]["zone_id"])}
                            for rule in guiding_rules if rule[0][0] in [child["description"] for child in data["children"].values()]]
        
        happenings = [{"zone_id": str(data["zones"][rule[1]]["zone_id"]),
                          "reason": rule[0][0],
                          "required_MAT": rule[0][1],
                          "zone_name": data["zones"][rule[1]]["zone_name"]}
                       for rule in guiding_rules if rule[0][0] in [zone["description"] for zone in data["zones"].values()]]

        DMM_observation = {"child_conditions": child_conditions, "happenings": happenings}
        DMM_observation["stations_zones"] = data["stations_zones"]
        DMM_observation["agent_zone"] = {"zone_id": str(data["agent_zone"])}
        DMM_observation["zone_ids"] = [{"zone_id": str(zone_id)} for zone_id in data["zone_ids"]]
        
        return DMM_observation
    
    """
    GENERAL: data relevant for selecting a default action is selected and transformed such that it can be processed by the guard
    only instrumentally relvant data can be filtered out while normatively relevant data that was hidden from the DMM (like the existence of a zone) could be taken in 

    PRESCHOOL: similar to the input of the DMM; station zones are filtered out, because they are only relevant for the deployment prupose, not for behaving compliant with normative requirements
    """

    def guard_observation(self, data, guiding_rules):
        data = copy.deepcopy(self.extracted_information)

        child_conditions = [{"child_id": str(data["children"][rule[1]]["child_id"]),
                             "zone_id": str(data["children"][rule[1]]["zone_id"]),
                            }
                            for rule in guiding_rules if rule[0][0] in [child["description"] for child in data["children"].values()]]
        
        happenings = [{"zone_id": str(data["zones"][rule[1]]["zone_id"]),
                        "required_MAT": rule[0][1],
                        "zone_name": data["zones"][rule[1]]["zone_name"]}
                       for rule in guiding_rules if rule[0][0] in [zone["description"] for zone in data["zones"].values()]]
        guard_observation = {"child_conditions": child_conditions, "happenings": happenings}
        guard_observation["agent_zone"] = {"zone_id": str(data["agent_zone"])}
        guard_observation["zone_ids"] = [{"zone_id": str(zone_id)} for zone_id in data["zone_ids"]]

        return guard_observation
