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
    All data is passed to the DMM; no filtering is applied.
    """
    def DMM_observation(self, extracted_data, guiding_rules):
        
        child_conditions = [{"child_id": str(extracted_data["children"][rule[1]]["child_id"]),
                             "reason": rule[0][0],
                             "required_MAT": rule[0][1],
                             "coordinate": {"x": extracted_data["children"][rule[1]]["coordinate"][0], "y": extracted_data["children"][rule[1]]["coordinate"][1]}
                             }
                            for rule in guiding_rules if rule[0][0] in [child["description"] for child in extracted_data["children"].values()]]
        
        happenings = [{"zone_id": str(extracted_data["happenings"][rule[1]]["zone_id"]),
              "reason": rule[0][0],
              "required_MAT": rule[0][1]}
              for rule in guiding_rules if rule[0][0] in [happening["description"] for happening in extracted_data["happenings"].values()]]
        
       #happenings = [{"zone_id": str(extracted_data["zones"][rule[1]]["zone_id"]),
       #           "reason": rule[0][0],
       #           "required_MAT": rule[0][1]}
        #          for rule in guiding_rules if rule[0][0] in [happening["description"] for happening in extracted_data["happenings"].values()]]
        #           #for rule in guiding_rules if rule[0][0] in [zone["description"] for zone in extracted_data["zones"].values()]]

        normative_reasons= {"child_conditions": child_conditions, "happenings": happenings}
        DMM_input = normative_reasons

        DMM_input["zones"] = [
            {
                "zone_id": str( zone["zone_id"]),
                "coordinates": [
                    {"x": coord[0], "y": coord[1]} for coord in zone["coordinates"] 
                ],
            }
            for zone in extracted_data["zones"].values()
        ]
        DMM_input["station_coordinates"] = [{"x": coordinate[0], "y": coordinate[1]} for coordinate in extracted_data["station_coordinates"]]
        DMM_input["agent_coordinate"] = {"x": int(extracted_data["agent_coordinate"][0]), "y": int(extracted_data["agent_coordinate"][1])}
        DMM_input["reasons_changed"] = extracted_data.get("reasons_changed", False)
        
        return DMM_input
    
    """
    Similar to the input of the DMM; learning are filtered out, because they are irrelevant for behaving compliant with normative requirements.
    """

    def guard_observation(self, extracted_data, guiding_rules):
        extracted_data = copy.deepcopy(self.extracted_information)

        child_conditions = [{"child_id": extracted_data["children"][rule[1]]["child_id"],
                             "reason": rule[0][0],
                             "required_MAT": rule[0][1],
                             "coordinate": [extracted_data["children"][rule[1]]["coordinate"][0], extracted_data["children"][rule[1]]["coordinate"][1]]
                             }
                            for rule in guiding_rules if rule[0][0] in [child["description"] for child in extracted_data["children"].values()]]
        
        happenings = [{"zone_id": extracted_data["zones"][rule[1]]["zone_id"],
                  "reason": rule[0][0],
                  "required_MAT": rule[0][1]}
                  for rule in guiding_rules if rule[0][0] in [happening["description"] for happening in extracted_data["happenings"].values()]]
        
        guard_observation = {"child_conditions": child_conditions, "happenings": happenings}
        guard_observation["agent_coordinate"] = extracted_data["agent_coordinate"]
        guard_observation["zones"] = [
            {
                "zone_id": zone["zone_id"],
                "coordinates": zone["coordinates"] 
            }
            for zone in extracted_data["zones"].values()
        ]

        return guard_observation
