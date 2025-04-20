from skyc_utils import skyc_maker, skyc_inspector

if __name__ == '__main__':
    traj = skyc_maker.Trajectory(degree=7)

    traj.set_start(skyc_maker.XYZYaw(-1, 1, 1, 0.0))
    traj.add_goto(skyc_maker.XYZYaw(-1, -1, 1, 0.0), 3)
    traj.add_goto(skyc_maker.XYZYaw(0, -1, 1, 0.0), 3)
    traj.add_goto(skyc_maker.XYZYaw(0, 1, 0.4, 0.0), 3)

    traj.add_goto(skyc_maker.XYZYaw(-1, 1, 1, 0.0), 3)

    traj.add_goto(skyc_maker.XYZYaw(-1, -1, 1, 0.0), 3)
    traj.add_goto(skyc_maker.XYZYaw(0, -1, 1, 0.0), 3)
    traj.add_goto(skyc_maker.XYZYaw(0, 1, 0.4, 0.0), 3)
    traj.add_goto(skyc_maker.XYZYaw(-1, 1, 1, 0.0), 3)

    skyc_maker.write_skyc([traj])
    #skyc_inspector.inspect('skyc_example.skyc')