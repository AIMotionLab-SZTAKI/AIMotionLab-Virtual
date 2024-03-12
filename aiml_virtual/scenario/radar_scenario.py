import numpy as np
import os
from aiml_virtual.object import Radar
from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.object.drone import DRONE_TYPES



def parentheses_contents(string: str):
    stack = []
    for i, c in enumerate(string):
        if c == '[':
            stack.append(i)
        elif c == ']' and stack:
            start = stack.pop()
            yield (len(stack), string[start + 1: i])

class DroneParams:

    def __init__(self, size: np.array, position_idx: int, safe_sphere_radius: float) -> None:
        self.size = size
        self.position_idx = position_idx
        self.safe_sphere_radius = safe_sphere_radius


class RadarScenario:

    radar_stl_resolution = 50
    radar_stl_rot_resolution = 60

    def __init__(self, sim_volume_size=np.array((0.0, 0.0, 0.0)), mountain_height=0.0, height_map_name="",
                 target_point_list=np.array(((0., 0., 0.), (0., 0., 0.))),
                 drone_param_list=[DroneParams(np.array((0.1, 0.1, 0.1)), 0, 0.0)],
                 radar_list=[Radar(np.array((0.0, 0.0, 0.0)), 1.0, 1.0, 50, 60)]) -> None:
        
        self.sim_volume_size = sim_volume_size
        self.mountain_height = mountain_height
        self.height_map_name = height_map_name
        self.target_point_list = target_point_list
        self.drone_param_list = drone_param_list
        self.radar_list = radar_list
        self.drone_names = []

    @staticmethod
    def parse_config_file(full_filename: str) -> "RadarScenario":

        file = open(full_filename,'r')

        line = file.readline().split('#')[0].strip()

        split_line = line.split(' ')

        volume_x = float(split_line[1])
        volume_y = float(split_line[3])
        volume_z = float(split_line[5])

        volume_size = np.array((volume_x, volume_y, volume_z))

        line = file.readline().split('#')[0].strip()

        split_line = line.split(' ')

        mountain_height = float(split_line[1])

        line = file.readline().split('#')[0].strip()
        
        height_map_filename = line

        line = file.readline().split('#')[0].strip()[1:-2]

        split_line = line.split("] ")

        target_point_list = []

        for s in split_line:
            point = np.fromstring(s[1:], sep=' ')
            target_point_list += [point]
        

        line = file.readline().split('#')[0].strip()

        drone_param_list = []
        
        for l in list(parentheses_contents(line)):
            if l[0] == 1:
                split_dp = l[1][1:-1].split('] [')

                size = np.fromstring(split_dp[0], sep=' ')
                position_idx = int(split_dp[1])
                safe_sphere_radius = float(split_dp[2])

                drone_param_list += [DroneParams(size, position_idx, safe_sphere_radius)]


        line = file.readline().split('#')[0].strip()

        radar_list = []

        for l in list(parentheses_contents(line)):

            if l[0] == 1:
                split_r = l[1][1:-1].split('] [')

                pos = np.fromstring(split_r[0], sep=' ')
                a = float(split_r[1])
                exp = float(split_r[2])
                height_scale = float(split_r[3])
                tilt = float(split_r[4])    

                res = RadarScenario.radar_stl_resolution
                rres = RadarScenario.radar_stl_rot_resolution

                radar_list += [Radar(pos, a, exp, res, rres, height_scale, tilt, display_lobe=True)]


        return RadarScenario(volume_size, mountain_height, height_map_filename,
                             target_point_list, drone_param_list, radar_list)
    

    def generate_xml(self, xml_base_file_name, xml_path, display_radar_lobe=True):

        save_filename = "built_scene.xml"

        drone_colors = ["0.2 0.6 0.85 1.0", "0.2 0.85 0.6 1.0", "0.85 0.2 0.6 1.0", "0.85 0.6 0.2 1.0"]

        scene = SceneXmlGenerator(xml_base_file_name)

        terrain_hfield_filename = os.path.join("heightmaps", self.height_map_name)
        terrain_size = self.sim_volume_size / 2.
        terrain_size[2] = self.mountain_height

        scene.add_terrain(terrain_hfield_filename, size=np.array2string(terrain_size)[1:-1])

        for i, dp in enumerate(self.drone_param_list):

            pos = np.array2string(self.target_point_list[dp.position_idx])[1:-1]
            size = dp.size
            safe_sphere_radius = str(dp.safe_sphere_radius)

            self.drone_names += [scene.add_drone(pos, "1 0 0 0", drone_colors[i], type=DRONE_TYPES.BUMBLEBEE, safety_sphere_size=safe_sphere_radius)]


        for radar in self.radar_list:

            name = scene.add_radar_field(np.array2string(radar.pos)[1:-1], radar.color, radar.a, radar.exp, radar.rres,
                                        radar.res, radar.height_scale, radar.tilt, sampling="curv", display_lobe=display_radar_lobe)

            radar.set_name(name)
        
        save_path = os.path.join(xml_path, save_filename)

        scene.save_xml(save_path)

        return os.path.abspath(save_path), self.drone_names