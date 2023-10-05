import numpy as np

class TrajectoryBase:

    def __init__(self):
        # initialize the output dictionary of evaluate here, and only update it in evaluate()
        # this way, it won't have to be recreated every time

        self.output = {}

    def evaluate(self, state, i, time, control_step) -> dict:

        raise NotImplementedError("evaluate method must be implemented")


class DummyDroneTrajectory(TrajectoryBase):

    def __init__(self):
        super().__init__()
        
        # self.output needs to be updated and returned in evaluate()
        # the keys of the dictionary can be changed, this is only an example
        self.output = {
            "load_mass" : 0.0,
            "target_pos" : None,
            "target_rpy" : np.zeros(3),
            "target_vel" : np.zeros(3),
            "target_acc" : None,
            "target_ang_vel": np.zeros(3),
            "target_quat" : None,
            "target_quat_vel" : None,
            "target_pos_load" : None
        }
    

    def evaluate(self, state, i, time, control_step):

        return self.output

class DummyCarTrajectory(TrajectoryBase):

    def __init__(self):
        super().__init__()

        # self.output needs to be updated and returned in evaluate()
        # the keys of the dictionary can be changed, this is only an example
        self.output = {
            "target_pos" : None,
            "target_rpy" : np.zeros(3),
            "target_vel" : np.zeros(3),
            "target_acc" : None,
            "target_quat" : None,
            "target_quat_vel" : None,
        }
    
    def evaluate(self, state, i, time, control_step) -> dict:
        return self.output
