from preschool.config import Config
from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping
import uuid
import numpy as np

config = Config()

"""
An MAT_mapping for the text-version of the preschool setting navigated by an LLM agent.
"""
class MAT_mapping_PG(MAT_Mapping):
    def __init__(self, config):
        # config is needed for retrieving the infomration which action is required for which child condition 
        self.config = config

    # self.action_to_direction = {
    #        0: self.directions[0], #right
    #        1: self.directions[1], #down
    #        2: self.directions[2], #left
    #        3: self.directions[3], #up
    #    }
    
    def manhattan_distance(array1, array2):
        return np.sum(np.abs(array1 - array2))

    """
    PRESCHOOL: determining the set of allowed actions for one MAT rule-based hardcoded according to the 1 to 1  mapping
    """
    def obligation_violated(self, action, MAT, observation):

        if MAT[0] == "Stay_out_of_the_zone":
            y_coordinate_boundary = 3
            zone = next(zone for zone in observation["zones"] if zone["zone_id"] == MAT[1])
            if any(np.array_equal(coord, np.array([0, 0])) for coord in zone["coordinates"]):
                #agent is in zone and does not move out of it
                if observation["agent_coordinate"] in zone["coordinates"] and action != ("move", 0):
                    return True
                #agent moves into zone
                if observation["agent_coordinate"][0] == y_coordinate_boundary+1 and action == ("move", 2):
                    return True
            else:
                if observation["agent_coordinate"] in zone["coordinates"] and action != ("move", 2):
                    return True
                if observation["agent_coordinate"][0] == y_coordinate_boundary and action == ("move", 0):
                    return True
                
        else:

            child_coordinate = next(child_condition["coordinate"] for child_condition in observation["child_conditions"] if child_condition["child_id"] == MAT[1])
            #agent does not move to the zone where the child is
            if action == ("move", 0) and not child_coordinate[0] > observation["agent_coordinate"][0]:
                return True
            if action == ("move", 2) and not child_coordinate[0] < observation["agent_coordinate"][0]:
                return True
            if action == ("move", 1) and not child_coordinate[1] > observation["agent_coordinate"][1]:
                return True
            if action == ("move", 3) and not child_coordinate[1] < observation["agent_coordinate"][1]:
                return True
            if np.equal(child_coordinate, observation["agent_coordinate"]).all() and action[0] != "help":
                    return True
            if action[0] == "help" and not np.equal(child_coordinate, observation["agent_coordinate"]).all():
                return True
            if action[0] == "prepare":
                return True
            
        return False
    
    """
    rules for what needs to be done given a certain set of MATs (encode the correct next primitive action for every situation possible)
    of course this is only possible if there is no epistemic or normative uncertainty and the action space is overseeable which is normally not the case
    """
    def default_action(self, MATs, observation):
        # help a child in need
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
                forbidden_zones += [zone for zone in extracted_data["zones"].values() if zone["zone_id"] == rule[1]]
        if len(forbidden_zones) >= 2:
            return True
        forbidden_zone_coord = [coord for zone in forbidden_zones for coord in zone["coordinates"]]
        #forbidden_zone_coord = [zone["coordinates"] for zone in forbidden_zones].flatten().tolist()
        
        required_action = None
        for rule in set_of_rules:
            if rule[0][1] != "Stay_out_of_the_zone":
                child_coordinate = extracted_data["children"][rule[1]]["coordinate"]
                #if np.equal(child_coordinates, extracted_data["agent_coordinate"]):
                   # continue
                #child_zone_id = extracted_data["children"][rule[1]]["zone_id"] 
                if any(np.array_equal(child_coordinate, coord) for coord in forbidden_zone_coord):
                    return True
            
                #rule requires the agent to help a child
                elif not np.equal(child_coordinate,extracted_data["agent_coordinate"]).all():
                    required_action_rule = ("move", np.array(child_coordinate))
                else:
                    required_action_rule = ("help", rule[0][1])

                if required_action:
                    if required_action[0] != required_action_rule[0]:
                        return True
                    elif required_action[0] == "move":
                        if not np.equal(required_action[1], required_action_rule[1]).all():
                            return True
                    elif required_action != required_action_rule:
                        return True

                else:
                    required_action = required_action_rule
        return False
    
    
                    

            


    


