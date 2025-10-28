from collections import defaultdict
from GUARDIANCE.interfaces.data_processor import Data_Processor
import copy

"""
A data_processor for the text version of the text-version of the preschool setting navigated by an LLM agent.
"""
class Data_Processor_PG(Data_Processor):
    def __init__(self):
        self.extracted_information = None
        self.former_reasons = None
    
    """
    Input from the environment is already morally relevant facts in propositional form together with the location where they are situated.
    The function just retrieves the propositions along with this information.
    Return the reason as first element and the entities for which the reason holds (th grounding of the reason) as second element.
    """
    def extract_relevant_information(self, reason_theory, observation, static_env_info):
        reasons = [reason[0] for reason in list(reason_theory.edges())]
        relevant_data = defaultdict(dict)
        relevant_data["children"] = {
            item[0]: {
                "child_id": item[0],
                "description": item[1].replace(" ", "_"),
                "coordinate": item[2],
            }
            for item in observation["children"]
            if item[1].replace(" ", "_") in reasons
        }
        if observation["happenings"]:
            pass
        relevant_data["happenings"] = {
            item[0]: {
                "zone_id": item[0],
                "description": item[1].replace(" ", "_"),
                "coordinates": item[2]
            }
            for item in observation["happenings"]
            if item[1].replace(" ", "_") in reasons
        }

        relevant_data["agent_coordinate"] = observation["agent_coordinate"]
        relevant_data["station_coordinates"] = observation["station_coordinates"]
        relevant_data["zones"] = static_env_info["zones"]

        self.extracted_information = relevant_data
        
        return relevant_data
    
    def groundings_for_rule(self, extracted_data, rule):
        groundings = set()
        truthmakers = [item for item in extracted_data["children"].items() if item[1].get("description") == rule[0]] + \
                      [item for item in extracted_data["happenings"].items() if item[1].get("description") == rule[0]] 
        for truthmaker in truthmakers:
            resolution = rule[1]
            groundings.add(((truthmaker[1]["description"], resolution), truthmaker[0]))
        return groundings
    
    """
    all data is passed to the DMM; no filtering is applied.
    """
    def DMM_observation(self, data, guiding_rules):
        
        child_conditions = [{"child_id": str(data["children"][rule[1]]["child_id"]),
                             "reason": rule[0][0],
                             "required_MAT": rule[0][1],
                             "coordinate": {"x": data["children"][rule[1]]["coordinate"][0], "y": data["children"][rule[1]]["coordinate"][1]}
                             }
                            for rule in guiding_rules if rule[0][0] in [child["description"] for child in data["children"].values()]]
        
        happenings = [{"zone_id": str(data["zones"][rule[1]]["zone_id"]),
                  "reason": rule[0][0],
                  "required_MAT": rule[0][1]}
                  for rule in guiding_rules if rule[0][0] in [happening["description"] for happening in data["happenings"].values()]]
                   #for rule in guiding_rules if rule[0][0] in [zone["description"] for zone in data["zones"].values()]]

        normative_reasons= {"child_conditions": child_conditions, "happenings": happenings}
        DMM_input = normative_reasons
        if normative_reasons != self.former_reasons:
            DMM_input["reasons_changed"] = True
        else :
            DMM_input["reasons_changed"] = False
        self.former_reasons = normative_reasons

        #test = [zone for zone in data["zones"].values()]

        DMM_input["zones"] = [
            {
                "zone_id": str( zone["zone_id"]),
                "coordinates": [
                    {"x": coord[0], "y": coord[1]} for coord in zone["coordinates"] 
                ],
            }
            for zone in data["zones"].values()
        ]
        DMM_input["station_coordinates"] = [{"x": coordinate[0], "y": coordinate[1]} for coordinate in data["station_coordinates"]]
        DMM_input["agent_coordinate"] = {"x": int(data["agent_coordinate"][0]), "y": int(data["agent_coordinate"][1])}
        DMM_input["reasons_changed"] = data.get("reasons_changed", False)
        
        return DMM_input
    
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
        
        happenings = [{"zone_id": str(data["happenings"][rule[1]]["zone_id"]),
                        "required_MAT": rule[0][1],
                        "zone_name": data["happenings"][rule[1]]["zone_name"]}
                       for rule in guiding_rules if rule[0][0] in [zone["description"] for zone in data["zones"].values()]]
        guard_observation = {"child_conditions": child_conditions, "happenings": happenings}
        guard_observation["agent_zone"] = {"zone_id": str(data["agent_zone"])}
        guard_observation["zone_ids"] = [{"zone_id": str(zone_id)} for zone_id in data["zone_ids"]]

        return guard_observation
