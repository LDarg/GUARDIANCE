def set_rules(agent):
     # add reason-nodes
    agent.reasoning_unit.reason_theory.add_node('A child is crying and needs comfort'.replace(" ", "_"), type='morally relevant fact')
    agent.reasoning_unit.reason_theory.add_node('A child has fallen down and needs help to get up'.replace(" ", "_"), type='morally relevant fact')
    agent.reasoning_unit.reason_theory.add_node('A child has scratched its knee and needs band-aids'.replace(" ", "_"), type='morally relevant fact')
    agent.reasoning_unit.reason_theory.add_node('Children have finished their work early and are now running around wildly'.replace(" ", "_"), type='morally relevant fact')

    # add obligation-nodes
    agent.reasoning_unit.reason_theory.add_node('Comfort the child'.replace(" ", "_"), type='moral obligation')
    agent.reasoning_unit.reason_theory.add_node('Help the child stand up'.replace(" ", "_"), type='moral obligation')
    agent.reasoning_unit.reason_theory.add_node('Doctor the child'.replace(" ", "_"), type='moral obligation')
    agent.reasoning_unit.reason_theory.add_node('Stay out of the zone'.replace(" ", "_"), type='moral obligation')
    
    # add default rules with a hard-coded order among them
    agent.reasoning_unit.reason_theory.add_edge('A child is crying and needs comfort'.replace(" ", "_"), 'Comfort the child'.replace(" ", "_"), 
                                                    lower_order=set(), 
                                                    name=f"δ{get_edge_number_as_index(agent)}")
    agent.reasoning_unit.reason_theory.add_edge('A child has fallen down and needs help to get up'.replace(" ", "_"), 'Help the child stand up'.replace(" ", "_"), 
                                                    lower_order=(('Children have finished their work early and are now running around wildly'.replace(" ", "_"), 'Stay out of the zone'.replace(" ", "_")),), 
                                                    name=f"δ{get_edge_number_as_index(agent)}")
    agent.reasoning_unit.reason_theory.add_edge('A child has scratched its knee and needs band-aids'.replace(" ", "_"), 'Doctor the child'.replace(" ", "_"), 
                                                    lower_order=set(), 
                                                    name=f"δ{get_edge_number_as_index(agent)}")
    agent.reasoning_unit.reason_theory.add_edge('Children have finished their work early and are now running around wildly'.replace(" ", "_"), 'Stay out of the zone'.replace(" ", "_"), 
                                                    lower_order=(('A child is crying and needs comfort'.replace(" ", "_"), 'Comfort the child'.replace(" ", "_")),
                                                                 ('A child has scratched its knee and needs band-aids'.replace(" ", "_"), 'Doctor the child'.replace(" ", "_"))),
                                                    name=f"δ{get_edge_number_as_index(agent)}")

def get_edge_number_as_index(agent):
    return agent.reasoning_unit.int_to_subscript(agent.reasoning_unit.reason_theory.number_of_edges()+1)
