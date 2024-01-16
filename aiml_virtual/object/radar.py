from aiml_virtual.util.mujoco_helper import move_point_on_sphere
import math

class Radar:

    """ NOT A MUJOCO OBJECT, ONLY A CONTAINER FOR DATA & CALCULATIONS """

    def __init__(self, pos, a, exp, res, rres, height_scale=1.0, tilt=0.0, color="0.5 0.5 0.5 0.5") -> None:
        
        self.pos = pos # center of the radar field
        self.a = a # size of the radar field (radius = 2 * a)
        self.exp = exp # shape of the radar field
        self.res = res # sample size of the 2D teardrop function
        self.rres = rres # how many times to rotate the tear drop to get the circular shape
        self.tilt = tilt # how much the teardrop is tilted before being rotated by 360°
        self.color = color # color of the radar field
        self.height_scale = height_scale

    @staticmethod
    def is_point_inside_lobe(point, radar_center, a, exponent, height_scale, tilt):

        p = move_point_on_sphere(point - radar_center, -tilt, 0.0) + radar_center

        #plt.plot(p[0], p[2], marker='x')

        d = math.sqrt((radar_center[0] - p[0])**2 + (radar_center[1] - p[1])**2)

        if d <= 2 * a:

            z_lim = height_scale * a * math.sin(math.acos((a - d) / a)) * math.sin(math.acos((a - d) / a) / 2.0)**exponent

            if p[2] < radar_center[2] + z_lim and p[2] > (radar_center[2] - z_lim):
                return True
        
        return False
    
    def sees_drone(self, drone):

        drone_pos = drone.get_state()["pos"]

        return Radar.is_point_inside_lobe(drone_pos, self.pos, self.a, self.exp, self.height_scale, self.tilt)
    
    def sees_point(self, point):
        
        return Radar.is_point_inside_lobe(point, self.pos, self.a, self.exp, self.height_scale, self.tilt)