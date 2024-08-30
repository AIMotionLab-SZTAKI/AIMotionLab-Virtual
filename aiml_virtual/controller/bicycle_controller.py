"""
This module houses the BicycleController, which is a dummy controller for testing.
"""

from aiml_virtual.controller import controller


class BicycleController(controller.Controller):
    """
    Dummy controller to be used with the Bicycle class: it always turns the wheels with an even torque.
    """
    def __init__(self, torque: float = 0.01):
        """
        Constructor that sets the target torque for the bicycle.

        Args:
            torque (float): The torque for the motors of the wheels to hold.
        """
        super().__init__()
        self.torque: float = torque  #: the (constant) torque for the actuators

    def compute_control(self) -> float:
        """
        Overrides (implements) superclass' compture_control, ensuring that BycicleController is a concrete class.

        Returns:
            float: The torque to be used for the motors.
        """
        return self.torque
