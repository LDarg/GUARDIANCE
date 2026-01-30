from GUARDIANCE.interfaces.moral_module import Moral_Module
from GUARDIANCE.reasoning_unit import ReasoningUnit
import logging

logger = logging.getLogger(__name__)

class Moral_Module_PG(Moral_Module):
    def __init__(self, reasoning_unit:ReasoningUnit):
        self.reasoning_unit = reasoning_unit

    def guiding_rules(self, extracted_data):
        return self.reasoning_unit.moral_obligations(extracted_data)