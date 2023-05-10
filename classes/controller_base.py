import numpy as np

class ControllerBase:

    def __init__(self, mass, inertia, gravity):
        
        self.mass = mass
        self.inertia = inertia
        self.gravity = gravity

    def compute_control(self, state, setpoint, time) -> np.array:
        # if it's a drone controller, return four floats that are the controls for the four drone actuators
        # if it's a car controller, return two floats: d, delta. They will be converted to actuator controls by the car class.

        raise NotImplementedError("compute_control method must be implemented")


class DummyDroneController(ControllerBase):

    def __init__(self, mass, inertia, gravity):
        super().__init__(mass, inertia, gravity)

    
    def compute_control(self, state, setpoint, time) -> np.array:
        return np.array((0, 0, 0, 0))

class DummyCarController(ControllerBase):

    def __init__(self, mass, inertia, gravity):
        super().__init__(mass, inertia, gravity)
    

    def compute_control(self, state, setpoint, time) -> np.array:
        d = 0
        delta = 0
        return np.array((d, delta))