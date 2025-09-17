from skyc_utils.trajectory import Trajectory, Pose, TrajectoryType
from skyc_utils.skyc import Skyc

if __name__ == '__main__':
    traj = Trajectory(degree=7, traj_type=TrajectoryType.POLY4D)
    traj.add_goto(Pose(0.0, 0.0, 0.5, 0.0), 3)  # takeoff
    traj.add_goto(Pose(1.0, -0.5, 1.0, 3.1415), 5)  # half a turn in place
    traj.add_goto(Pose(-1.0, 0.5, 0.3, 0.0), 5)
    traj.add_goto(Pose(0.0, 0.0, 0.0, 0.0), 3)  # land
    skyc = Skyc()
    skyc.add_drone(traj)
    skyc.write("skyc_example")