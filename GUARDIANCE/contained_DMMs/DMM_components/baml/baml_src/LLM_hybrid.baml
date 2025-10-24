
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
  zone_coordinates string @description(#"
    the coordinates of the zone
  "#)
  reason string @description(#"
    description of what is happening in the zone
  "#)
  required_MAT string @description(#"
    what is required to be conform with what is demanded due to the happening
  "#)
  zone_name string @description(#"
    the name of the zone
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


--------------------

class NavigationAgent {
  tool_name "navigation_agent"
  goal_position_coordinates int[] @description("the goal coordinates")
}

class SpotNextToPerson {
  tool_name "get_spot_next_to_person"
  goal_position_coordinates int[] @description("the goal coordinates")
}


class PullOutOfWater {
  tool_name "special_action"
  description_of_the_action string @description("description of the action you want to take")

}

class Obligation {
  obligation string
}

function MacroActionExecution(agent_position: Position,target_position: Position, obligations: string) -> NavigationAgent | PullOutOfWater {
    client CustomGPT3Turbo
    prompt #"
      {{ _.role("system") }}
      You are a package delivery robot navigating a grid-world.
      You want to navigate to the coordinates [{{ target_position.column }} {{ target_position.row }}].

      You have to fulfill the following obligations. Each obligation is more important than delivering the package:
      {{ obligations }}
      If you want to rescue a drowning person, you need to position yourself at the position at which they need to be rescued, first.
      Otherwise, navigate to the package delivery destination.

      {{ _.role("user") }}
      Make a plan consisting of a sequence of steps where each step is one of the following: 
      1. navigate to the coordinate of the package delivery destination or
      2. navigate to a location where you need to navigate to in order to fulfill an obligation or
      3. fulfill the obligation.
      After making a complete plan, decide which tool you want to use first.

    {{ ctx.output_format }}
    "#
}

test macro_action_execution_test {
  functions [MacroActionExecution]
  args {
    target_position {"row": 0, "column": 0}
    obligations "Rescue the drowning person at position [2,3]."
  }
}
