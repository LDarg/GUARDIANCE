from preschool.config import Config
from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping

config = Config()

"""
An MAT_mapping for the text-version of the preschool setting navigated by an LLM agent.
"""
class MAT_mapping_PT(MAT_Mapping):
    def __init__(self, config):
        # config is needed for retrieving the infomration which action is required for which child condition 
        self.config = config

    """
    PRESCHOOL: determining the set of allowed actions for one MAT rule-based hardcoded according to the 1 to 1  mapping
    makes in particular the assumption that for every condition there is at most one child with that condition???
    """
    def action_conform_with_MAT(self, action, MAT, state):
        if MAT["type"] == "moral_goal":
                if state["agent_zone"] == MAT["zone"]:
                    if action == ("help", config.resolutions[MAT["description"]]):
                        return True
                else:
                    if action == ("move", MAT["zone"]):
                        return True
                return False
        elif MAT["type"] == "moral_constraint":
            if action  == ("move", MAT["zone"]):
                return False
            elif state["agent_zone"] == MAT["zone"]:
                if action[0] != "move":
                    return False
        return True
    
    """
    To determine conflicts (in Horty’s formalization of reasoning), a conformity check of MATs against the first elements of chains of primitive actions takes place here.
    """
    def execution_conflicted(self, set_of_rules, extracted_data):
        forbidden_zones = []
        for rule in set_of_rules:
            if rule[0][1] == "Stay_out_of_the_zone":
                forbidden_zones += [extracted_data["zones"][id]["zone_name"] for id in extracted_data["zones"] if id == rule[1]]
        if 'A' in forbidden_zones and 'B' in forbidden_zones:
            return True
        
        required_action = None
        for rule in set_of_rules:
            if rule[0][1] != "Stay_out_of_the_zone":
                child_zone = extracted_data["children"][rule[1]]["zone_name"] 
                if child_zone in forbidden_zones:
                    return True
                elif child_zone != extracted_data["agent_zone"]:
                    required_action_rule = ("move", child_zone)
                else:
                    required_action_rule = ("help", rule[0][1])

                if required_action and required_action_rule != required_action:
                    return True
                else:
                    required_action = required_action_rule
        return False
    
                    

            


    


