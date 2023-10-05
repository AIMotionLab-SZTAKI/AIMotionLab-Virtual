import mujoco
from aiml_virtual.object.moving_object import MovingObject

class Bicycle(MovingObject):

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name_in_xml):
        
        super().__init__(model, name_in_xml)

        self.actr = data.actuator(name_in_xml + "_actr")
        self.ctrl = self.actr.ctrl
        self.sensor_velocimeter = data.sensor(name_in_xml + "_velocimeter").data
    
    def update(self, i, control_step):
        
        if self.controllers is not None:
            ctrl = self.controllers[0].compute_control(i)

            self.ctrl[0] = ctrl


from aiml_virtual.controller import ControllerBase

class BicycleController(ControllerBase):

    def __init__(self):
        pass
    
    def compute_control(self, i):
        
        return 0.1