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
            # check if the agent is in the left zone
            if any(np.array_equal(coord, np.array([0, 0])) for coord in zone["coordinates"]):
                #check if the agent does not move out of it
                if any(np.array_equal(observation["agent_coordinate"], coord) for coord in zone["coordinates"]) and action != ("move", 0):
                    return True
                #check if the agent wants to enter the zone
                if observation["agent_coordinate"][0] == y_coordinate_boundary+1 and action == ("move", 2):
                    return True
            # the agent is in the right zone
            else:
                # check if the agent does not move out of it
                if any(np.array_equal(observation["agent_coordinate"], coord) for coord in zone["coordinates"]) and action != ("move", 2):
                    return True
                # check if the agent wants to enter the zone
                if observation["agent_coordinate"][0] == y_coordinate_boundary and action == ("move", 0):
                    return True
                
        else:
            child_coordinate = next(child_condition["coordinate"] for child_condition in observation["child_conditions"] if child_condition["child_id"] == MAT[1])
            #agent does not move to the position of the child
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
        """
        Determine default action based on MATs (Moral Action Types).
        Priority 1: Exit forbidden zones if currently in one
        Priority 2: Move towards children that need help
        """
        # Helper function to get agent's current zone
        def get_agent_current_zone():
            agent_coord = np.array(observation["agent_coordinate"])
            for zone in observation["zones"]:
                for coord in zone["coordinates"]:
                    if np.array_equal(agent_coord, np.array(coord)):
                        return zone["zone_id"]
            return None
        
        # Helper function to calculate direction from current position to target
        def calculate_direction(current_pos, target_pos):
            """
            Calculate the optimal direction to move towards target.
            Returns direction integer: 0=right, 1=down, 2=left, 3=up
            """
            current = np.array(current_pos)
            target = np.array(target_pos)
            diff = target - current
            
            # Choose direction that reduces Manhattan distance most
            # Prioritize horizontal movement, then vertical
            if diff[0] > 0:  # Target is to the right
                return 0  # right
            elif diff[0] < 0:  # Target is to the left
                return 2  # left
            elif diff[1] > 0:  # Target is down
                return 1  # down
            elif diff[1] < 0:  # Target is up
                return 3  # up
            else:
                return None  # Already at target
        
        # Helper function to find closest allowed zone
        def find_closest_allowed_zone(forbidden_zone_ids):
            agent_coord = np.array(observation["agent_coordinate"])
            min_distance = float('inf')
            closest_zone_coord = None
            
            for zone in observation["zones"]:
                if zone["zone_id"] not in forbidden_zone_ids:
                    for coord in zone["coordinates"]:
                        coord_array = np.array(coord)
                        distance = np.sum(np.abs(agent_coord - coord_array))  # Manhattan distance
                        if distance < min_distance:
                            min_distance = distance
                            closest_zone_coord = coord_array
            
            return closest_zone_coord
        
        # Helper function to find closest child
        def find_closest_child(child_ids):
            agent_coord = np.array(observation["agent_coordinate"])
            min_distance = float('inf')
            closest_child_coord = None
            closest_child_id = None
            
            for child_condition in observation["child_conditions"]:
                if child_condition["child_id"] in child_ids:
                    child_coord = np.array(child_condition["coordinate"])
                    distance = np.sum(np.abs(agent_coord - child_coord))  # Manhattan distance
                    if distance < min_distance:
                        min_distance = distance
                        closest_child_coord = child_coord
                        closest_child_id = child_condition["child_id"]
            
            return closest_child_coord, closest_child_id
        
        # Extract forbidden zones and children in need from MATs
        forbidden_zone_ids = [MAT[1] for MAT in MATs if MAT[0] == "Stay_out_of_the_zone"]
        children_in_need = [MAT[1] for MAT in MATs if MAT[0] != "Stay_out_of_the_zone"]
        
        current_zone = get_agent_current_zone()
        agent_coord = np.array(observation["agent_coordinate"])
        
        # Priority 1: Exit forbidden zone if currently in one
        if current_zone in forbidden_zone_ids:
            closest_allowed_coord = find_closest_allowed_zone(forbidden_zone_ids)
            if closest_allowed_coord is not None:
                direction = calculate_direction(agent_coord, closest_allowed_coord)
                if direction is not None:
                    return ("move", direction)
        
        # Priority 2: Move towards children that need help
        if children_in_need:
            closest_child_coord, closest_child_id = find_closest_child(children_in_need)
            if closest_child_coord is not None:
                # If already at child's location, help them
                if np.array_equal(agent_coord, closest_child_coord):
                    # Find the required MAT for this child
                    required_mat = next(MAT[0] for MAT in MATs if MAT[1] == closest_child_id)
                    return ("help", closest_child_id, required_mat)
                else:
                    # Move towards the child
                    direction = calculate_direction(agent_coord, closest_child_coord)
                    if direction is not None:
                        return ("move", direction)
        
        # No action needed if no MATs apply
        return None
    
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
