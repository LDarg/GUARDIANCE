from abc import ABC, abstractmethod

class DMM(ABC):
    """
    monitor and ensure that the actions proposed by the DMM conform to the guiding rules

      Args:
        extracted_data: information extracted from the observation of the environment

    Returns:
        the action that the DMM selects to exectute
    """
    def take_action(self, DMM_observation):
        pass

