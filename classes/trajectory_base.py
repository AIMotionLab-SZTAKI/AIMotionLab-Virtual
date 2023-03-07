

class TrajectoryBase:

    def __init__(self):
        # initialize the output dictionary of evaluate here, and only update it in evaluate()
        # this way, it won't have to be recreated every time

        self.output = {}

    def evaluate(self, state, i, time, control_step) -> dict:

        raise NotImplementedError("evaluate method must be implemented")