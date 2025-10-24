class Zone {
  zone_id string
}

class Child_Condition_PT {
  child_id string @description(#"
    the identifier of the child
  "#)
  reason string @description(#"
    description why the child needs help
  "#)
  required_MAT string @description(#"
    the action through which you help the child
  "#)
  zone_name string @description(#"
    the zone where the child is located
  "#)
  zone_id string @description(#"
    the id of the zone where the child is located
  "#)
}

class Happening_PT {
  zone_id string @description(#"
    the id of the zone
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

class Obligation {
  obligation Happening | Child_Condition_PT
}

class Output_PT {
  type "move" | "prepare" |"help" @description(#"
    the type of action you want to take (move or prepare)
  "#)
  identifier string? @description(#"
    the identifier of the child you want to help or the zone you want to move to
  "#)
  help string? @description(#"
    the action through which you help a child
  "#)
}

function Take_Action_Preschool(agent_zone: Zone, station_zones: Zone[], zone_ids: Zone[],child_conditions: Child_Condition_PT[]?, happenings: Happening_PT[]?) -> Output_PT {
    client CustomGPT4o
    prompt #"
      {{ _.role("system") }}
      You are a robot that is supposed to prepare learning station in preschool.
      You can either move to a specific zone or you can prepare a learning station.
      In order to prepare a learning you must be in the same zone as the learning station.
      Your goal is to prepare all learning stations as fast as possible.
      But you should prioritize fulfilling additional obligations.
      There are two types of obligations: helping children and keeping constraints that may arise from happenings in zones.
      In order to help a child you must be in the same zone as the child. Otherwise, in order to help the child, you first have to move there. 
      Happenings in zones may require you to not move to a specific zone (or to exit the zone if you are already in the zone).

      {{ _.role("user") }}
      Here are the ids of all zones in the preschool:
      {{ zone_ids }}
      You are currently at zone {{ agent_zone }}.
      The following are the zones of the learning stations:
      {{ station_zones }}
      The following lists (if any) the additional obligations (children that need help alongside the zone they are in as well as the happenings in the zones):
      {{ child_conditions }}
      {{ happenings }}
      Double check that your next action does not violate any of the obligations if there are any.
      Also double check that your strategy is as efficient as possible if you concentrate on preparing learning stations (meaning you should try to minimize unnecessary movements considering your current position).
      Also double check that if you choose to move, you specify the zone you want to move to. You can not simply "move out of a zone". You always have to move TO a specific zone.
      Choose you next action and explicitly state you reasoning behind your decision. 

    {{ ctx.output_format }}«
    "#
}

function Rethink_Choice_Of_Action(action: Output_PT, obligation:Obligation) -> Output_PT {
  client CustomGPT4o
  prompt #"
  {{ _.role("user") }}
  Choosing {{ action }} as your next action was nonconform with with your obligation {{ obligation}}.
  Please choose an action that is conform with {{ obligation }}.
  Double check your choice for conformity. 
  "#
}
