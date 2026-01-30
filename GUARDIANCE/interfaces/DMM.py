from abc import ABC, abstractmethod

class DMM(ABC):
    @abstractmethod
    def take_action(self, DMM_observation):
        pass

