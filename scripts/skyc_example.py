from skyc_utils import skyc_maker, skyc_inspector

if __name__ == '__main__':
    traj = skyc_maker.Trajectory(degree=7)
    traj.add_goto(skyc_maker.XYZYaw(0.0, 0.0, 0.5, 0.0), 3)  # takeoff
    traj.add_goto(skyc_maker.XYZYaw(1.0, -0.5, 1.0, 180.0), 5)  # half a turn in place
    traj.add_goto(skyc_maker.XYZYaw(-1.0, 0.5, 0.3, 0.0), 5)
    traj.add_goto(skyc_maker.XYZYaw(0.0, 0.0, 0.0, 0.0), 3)  # land
    skyc_maker.write_skyc([traj])
    skyc_inspector.inspect("skyc_example.skyc")