
class Zone_PG {
  zone_id string
  coordinates Coordinate[] @description(#"
    the coordinates that define the zone
  "#)
}

class Child_Condition_PG {
  child_id string @description(#"
    the identifier of the child
  "#)
  reason string @description(#"
    description why the child needs help
  "#)
  required_MAT string @description(#"
    the action through which you help the child
  "#)
  coordinate string @description(#"
    the coordinate of the child's position
  "#)
}

class Happening {
  zone Zone_PG @description(#"
    the zone at which the event is happening
  "#)
  reason string @description(#"
    description of what is happening in the zone
  "#)
  required_MAT string @description(#"
    what is required to be conform with what is demanded due to the happening
  "#)
}

class Output_PG {
  type "move" | "prepare" |"help" @description(#"
    the type of action you want to take (move, prepare or help)
  "#)
  identifier string? @description(#"
    the identifier of the child you want to help (needed iff type is help)
  "#)
  help string? @description(#"
    the action through which you help a child (needed iff type is help)
  "#)
  target_position_coordinates int[]? @description(#"
    the coordinates, you want to move to by calling the navigation agent (needed iff type is move)
  "#)
}

class Coordinate {
  x int
  y int
}

function Take_Action_PG(agent_coordinate: Coordinate, stations_coordinates: Coordinate[], zones: Zone_PG[], child_conditions: Child_Condition_PG[]?, happenings: Happening[]?) -> Output_PG {
    client CustomGPT4o
    prompt #"
      {{ _.role("system") }}
      You are a robot that is supposed to prepare learning station in preschool.
      You can either move to a specific zone by calling a navigation agent that gets you there, prepare a learning station or you can help a child.
      In order to prepare a learning you must be in the same zone as the learning station.
      Your goal is to prepare all learning stations as fast as possible.
      But you should prioritize fulfilling additional obligations.
      There are two types of obligations: helping children and keeping constraints that may arise from happenings in zones.
      In order to help a child you must be at the same coordinates as the child. Otherwise, in order to help the child, you first have to move there. 
      Happenings in zones may require you to not move to a specific zone (or to exit the zone if you are already in the zone).
      These are the zones defined through their coordinates:
      {{ zones }}
  
      {{ _.role("user") }}
      You are currently at position {{ agent_coordinate }}.
      The following are the coordinates of the learning stations:
      {{ stations_coordinates }}
      The following lists (if any) the additional obligations (children that need help as well as the happenings in the zones):
      {{ child_conditions }}
      {{ happenings }}
      Double check that your next action does not violate any of the obligations if there are any.
      Also double check that your strategy is as efficient as possible if you concentrate on preparing learning stations (meaning you should try to minimize unnecessary movements considering your current position).
      Also double check that if you choose to move, you specify the zone you want to move to. You can not simply "move out of a zone". You always have to move TO a specific zone.
      Choose you next action and explicitly state you reasoning behind your decision. 

    {{ ctx.output_format }}«
    "#
}


test take_action_pg_basic {
  functions [Take_Action_PG]
  args {
    agent_coordinate {"x": 1, "y": 1}
    stations_coordinates [
      {"x": 3, "y": 2},
      {"x": 5, "y": 4},
      {"x": 2, "y": 6}
    ]
    zones [
      {
        "zone_id": "reading_corner",
        "coordinates": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}]
      },
      {
        "zone_id": "craft_area", 
        "coordinates": [{"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 6, "y": 4}]
      },
      {
        "zone_id": "play_zone",
        "coordinates": [{"x": 1, "y": 6}, {"x": 2, "y": 6}, {"x": 3, "y": 6}]
      }
    ]
    child_conditions null
    happenings null
  }
}

test take_action_pg_child_help {
  functions [Take_Action_PG]
  args {
    agent_coordinate {"x": 1, "y": 1}
    stations_coordinates [
      {"x": 3, "y": 2},
      {"x": 5, "y": 4}
    ]
    zones [
      {
        "zone_id": "reading_corner",
        "coordinates": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}]
      }
    ]
    child_conditions [
      {
        "child_id": "Emma_5",
        "reason": "Child is crying and needs comfort",
        "required_MAT": "provide_emotional_support",
        "coordinate": "2,3"
      }
    ]
    happenings null
  }
}

test take_action_pg_zone_happening {
  functions [Take_Action_PG]
  args {
    agent_coordinate {"x": 4, "y": 4}
    stations_coordinates [
      {"x": 3, "y": 2},
      {"x": 5, "y": 4}
    ]
    zones [
      {
        "zone_id": "craft_area", 
        "coordinates": [{"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 6, "y": 4}]
      },
      {
        "zone_id": "reading_corner",
        "coordinates": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}]
      }
    ]
    child_conditions null
    happenings [
      {
        "zone": {
          "zone_id": "craft_area",
          "coordinates": [{"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 6, "y": 4}]
        },
        "reason": "Spilled water on floor - slip hazard",
        "required_MAT": "avoid_zone_until_cleaned"
      }
    ]
  }
}

test take_action_pg_complex {
  functions [Take_Action_PG]
  args {
    agent_coordinate {"x": 1, "y": 1}
    stations_coordinates [
      {"x": 3, "y": 2},
      {"x": 5, "y": 4},
      {"x": 2, "y": 6}
    ]
    zones [
      {
        "zone_id": "reading_corner",
        "coordinates": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}]
      },
      {
        "zone_id": "craft_area", 
        "coordinates": [{"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 6, "y": 4}]
      },
      {
        "zone_id": "quiet_zone",
        "coordinates": [{"x": 1, "y": 6}, {"x": 2, "y": 6}, {"x": 3, "y": 6}]
      }
    ]
    child_conditions [
      {
        "child_id": "Liam_4",
        "reason": "Child fell down and is hurt",
        "required_MAT": "provide_first_aid",
        "coordinate": "6,5"
      }
    ]
    happenings [
      {
        "zone": {
          "zone_id": "quiet_zone",
          "coordinates": [{"x": 1, "y": 6}, {"x": 2, "y": 6}, {"x": 3, "y": 6}]
        },
        "reason": "Naptime in progress - noise restrictions",
        "required_MAT": "maintain_quiet_environment"
      }
    ]
  }
}