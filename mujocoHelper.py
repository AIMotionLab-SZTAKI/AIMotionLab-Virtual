import mujoco
import math

def update_drone(data, droneID, position, orientation):
    """
    Update the position and orientation of a drone
    first drone's position is data.qpos[:3], orientation is data.qpos[3:7]
    second drone's position is data.qpos[7:10], orientation is data.qpos[10:14]
    and so on
    droneID should be 0 for the first drone defined in the xml, 1 for the second etc.
    """
    startIdx = droneID * 7
    if startIdx + 6 >= data.qpos.size:
        print("Drone id: " + str(droneID) + " out of bounds of data.qpos. (Not enough drones defined in the xml)")
        return
    data.qpos[startIdx:startIdx + 3] = position
    data.qpos[startIdx + 3:startIdx + 7] = orientation


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

def update_follow_cam(qpos, droneID, cam):
    """
    updates the position and orientation of the camera that follows the drone from behind
    qpos is the array in which the position and orientation of all the drones are stored
    """

    startIdx = droneID * 7
    position = qpos[startIdx:startIdx + 3]
    orientation = qpos[startIdx + 3:startIdx + 7]

    roll_x, pitch_y, yaw_z = euler_from_quaternion(orientation[0], orientation[1], orientation[2], orientation[3])

    cam.lookat = position
    cam.azimuth = -math.degrees(roll_x)
    cam.elevation = -math.degrees(pitch_y) - 20