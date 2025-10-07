from preschool.config import Config
from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping
import uuid

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
    def obligation_violated(self, action, MAT, observation):

        if MAT[0] == "Stay_out_of_the_zone":
            #agent plans to not move out of forbidden zone
            if observation["agent_zone"]["zone_id"] == MAT[1]:
                other_zones = [elem for elem in observation["zone_ids"] if elem != MAT[1]]
                if not any(action == ("move", other_zone) for other_zone in other_zones):
                    return True
            #agent plans to move into the forbidden zone
            if action == ("move", MAT[1]):
                return True
        else:
            child_zone = next(child_condition["zone_id"] for child_condition in observation["child_conditions"] if uuid.UUID(child_condition["child_id"]) == MAT[1])
            #agent does not move to the zone where the child is
            if child_zone:
                if observation["agent_zone"]["zone_id"] != child_zone and action != ("move", uuid.UUID(child_zone)):
                    return True
                elif observation["agent_zone"]["zone_id"] == child_zone and action != ("help", MAT[1], MAT[0]):
                    return True
            
        return False
    
    """
    rules for what needs to be done given a certain set of MATs (encode the correct next primitive action for every situation possible)
    of course this is only possible if there is no epistemic or normative uncertainty and the action space is overseeable which is normally not the case
    """
    def default_action(self, MATs, observation):
        # help the child in need
        child_in_need  = next(
            (MAT[1] for MAT in MATs if MAT[0] != "Stay_out_of_the_zone"),
            None
        )
        if child_in_need:
            child_zone = next(child_condition["zone_id"] for child_condition in observation["child_conditions"] if child_condition["child_id"] == child_in_need)
            if observation["agent_zone"]["zone_id"] != child_zone:
                return ("move", uuid.UUID(child_zone))
         # get out of the forbidden zone
        else:
            forbidden_zone = next(
                (MAT[1] for MAT in MATs if MAT[0] == "Stay_out_of_the_zone"),
                None
            )
            other_zone = next(zone for zone in observation["zone_ids"] if zone != forbidden_zone)
            return ("move", uuid.UUID(other_zone))
    
    """
    To determine conflicts (in Horty’s formalization of reasoning), a conformity check of MATs against the first elements of chains of primitive actions takes place here.
    """
    def execution_conflicted(self, set_of_rules, extracted_data):
        forbidden_zones = []
        for rule in set_of_rules:
            if rule[0][1] == "Stay_out_of_the_zone":
                forbidden_zones += [zone for zone in extracted_data["zones"] if zone == rule[1]]
        if len(forbidden_zones) >= 2:
            return True
        
        required_action = None
        for rule in set_of_rules:
            if rule[0][1] != "Stay_out_of_the_zone":
                child_zone_id = extracted_data["children"][rule[1]]["zone_id"] 
                if child_zone_id in forbidden_zones:
                    return True
                elif child_zone_id != extracted_data["agent_zone"]:
                    required_action_rule = ("move", child_zone_id)
                else:
                    required_action_rule = ("help", rule[0][1])

                if required_action and required_action_rule != required_action:
                    return True
                else:
                    required_action = required_action_rule
        return False
    
    
                    

            


    


