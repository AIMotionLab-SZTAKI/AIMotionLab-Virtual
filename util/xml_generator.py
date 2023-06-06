import xml.etree.ElementTree as ET

import util.mujoco_helper as mh
import math

from classes.payload import PAYLOAD_TYPES
from classes.drone import DRONE_TYPES, BUMBLEBEE_PROP, CRAZYFLIE_PROP 
from classes.car import F1T_PROP

import os



PROP_COLOR = "0.1 0.1 0.1 1.0"
PROP_LARGE_COLOR = "0.1 0.02 0.5 1.0"

SITE_NAME_END = "_cog"


ROD_LENGTH = 0.4

class SceneXmlGenerator:

    def __init__(self, base_scene_filename):
        self.root = ET.Element("mujoco")
        ET.SubElement(self.root, "include", file=base_scene_filename)
        self.worldbody = ET.SubElement(self.root, "worldbody")
        self.contact = ET.SubElement(self.root, "contact")
        self.actuator = ET.SubElement(self.root, "actuator")
        self.sensor = ET.SubElement(self.root, "sensor")
        self.parking_lot = None
        self.airport = None
        self.hospital = None
        self.post_office = None
        self.sztaki = None

        self.__virtcrazyflie_cntr = 0
        self.__virtbumblebee_cntr = 0
        self.__virtbumblebee_hooked_cntr = 0
        self.__realcrazyflie_cntr = 0
        self.__realbumblebee_cntr = 0
        self.__realbumblebee_hooked_cntr = 0
        self.__virtfleet1tenth_cntr = 0
        self.__realfleet1tenth_cntr = 0
        
        self.__load_cntr = 0
        self.__mocap_load_cntr = 0


    def add_airport(self, pos, quat=None):
        if self.airport is None:

            tag = "geom"
            name = "airport"
            size = "0.105 0.105 .05"
            type = "plane"
            material = "mat-airport"

            if quat is None:
                self.airport = ET.SubElement(self.worldbody, tag, name=name, pos=pos, size=size, type=type, material=material)
            else:
                self.airport = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat, size=size, type=type, material=material)
            return self.airport
        else:
            print("[SceneXmlGenerator] Airport already added")


    def add_parking_lot(self, pos, quat=None):
        if self.parking_lot is None:

            tag = "geom"
            name = "parking_lot"
            size = "0.105 0.115 .05"
            type = "plane"
            material = "mat-parking_lot"

            if quat is None:
                self.parking_lot = ET.SubElement(self.worldbody, tag, name=name, pos=pos, size=size, type=type, material=material)
            else:
                self.parking_lot = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat, size=size, type=type, material=material)
            return self.parking_lot
        else:
            print("[SceneXmlGenerator] Parking lot already added")
    

    def add_pole(self, name, pos, quat=None):
        tag = "body"
        if quat is None:
            pole = ET.SubElement(self.worldbody, tag, name=name, pos=pos)
        else:
            pole = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat)
        
        ET.SubElement(pole, "geom", {"class": "pole_top"})
        ET.SubElement(pole, "geom", {"class": "pole_bottom1"})
        ET.SubElement(pole, "geom", {"class": "pole_bottom2"})

        return pole


    def add_drone(self, pos, quat, color, is_virtual = True, type="crazyflie", is_hooked = False, hook_dof = 1):

        name = None

        if is_hooked and type != "bumblebee":
            print("[SceneXmlGenerator] Error: Hooked drone can only be bumblebee...")
            return None


        if is_virtual:

            if type == "bumblebee":

                if is_hooked:
                    name = "virtbumblebee_hooked_" + str(self.__virtbumblebee_hooked_cntr)

                    drone = self.__add_bumblebee(name, pos, quat, color, True, hook_dof)

                    self.__virtbumblebee_hooked_cntr += 1
                    return name
                
                else:

                    name = "virtbumblebee_" + str(self.__virtbumblebee_cntr)
                    drone = self.__add_bumblebee(name, pos, quat, color)

                    self.__virtbumblebee_cntr += 1
                    return name

            elif type == "crazyflie":
                name = "virtcrazyflie_" + str(self.__virtcrazyflie_cntr)

                drone = self.__add_crazyflie(name, pos, quat, color)

                self.__virtcrazyflie_cntr += 1
                return name
                
            
            else:
                print("[SceneXmlGenerator] Error: unknown drone type: " + str(type))
                return None
        
        else:
            if type == "bumblebee":

                if is_hooked:

                    name = "realbumblebee_hooked_" + str(self.__realbumblebee_hooked_cntr)

                    drone = self.__add_mocap_bumblebee(name, pos, quat, color)
                    self.__add_mocap_hook_to_drone(drone, pos, name)
                    #self.__add_hook_to_drone(drone, name)

                    self.__realbumblebee_hooked_cntr += 1
                    return name

                else:

                    name = "realbumblebee_" + str(self.__virtbumblebee_cntr)
                    drone = self.__add_mocap_bumblebee(name, pos, quat, color)

                    self.__realbumblebee_cntr += 1
                    return name

            elif type == "crazyflie":
                name = "realcrazyflie_" + str(self.__realcrazyflie_cntr)

                drone = self.__add_mocap_crazyflie(name, pos, quat, color)

                self.__realcrazyflie_cntr += 1
                return name
            
            else:
                print("[SceneXmlGenerator] Error: unknown drone type: " + str(type))
                return None

        
    
    def __add_crazyflie(self, name, pos, quat, color):
        site_name = name + SITE_NAME_END
    
        drone = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat)
        ET.SubElement(drone, "inertial", pos="0 0 0", diaginertia="1.4e-5 1.4e-5 2.17e-5", mass="0.028")
        ET.SubElement(drone, "joint", name=name, type="free")
        #ET.SubElement(drone, "geom", name=name, type="mesh", pos="0 0 0", mesh="drone", rgba=color)

        ET.SubElement(drone, "geom", name=name + "_body", type="mesh", mesh="drone_body", rgba=color)
        ET.SubElement(drone, "geom", name=name + "_4_motormounts", type="mesh", mesh="drone_4_motormounts", rgba=color)
        ET.SubElement(drone, "geom", name=name + "_4_motors", type="mesh", mesh="drone_4_motors", rgba=color)

        ET.SubElement(drone, "site", name=site_name, pos="0 0 0")

        prop_name = name + "_prop1"
        mass = "0.00001"
        #
        pos = CRAZYFLIE_PROP.OFFSET.value + " -" + CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET_Z.value
        prop1_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop1_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop1_body, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop", mass=mass, pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")
        ET.SubElement(self.actuator, "general", joint=prop_name, name=name + "_actr1", gear=" 0 0 1 0 0 " + CRAZYFLIE_PROP.MOTOR_PARAM.value, ctrllimited="true", ctrlrange="0 " + CRAZYFLIE_PROP.MAX_THRUST.value)

        prop_name = name + "_prop2"
        pos = "-" + CRAZYFLIE_PROP.OFFSET.value + " -" + CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET_Z.value
        prop2_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop2_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop2_body, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop", mass=mass, pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")
        ET.SubElement(self.actuator, "general", joint=prop_name, name=name + "_actr2", gear=" 0 0 1 0 0 -" + CRAZYFLIE_PROP.MOTOR_PARAM.value, ctrllimited="true", ctrlrange="0 " + CRAZYFLIE_PROP.MAX_THRUST.value)

        prop_name = name + "_prop3"
        pos = "-" + CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET_Z.value
        prop3_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop3_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop3_body, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop", mass=mass, pos=pos, rgba=PROP_COLOR, euler="0 0 0.785")
        ET.SubElement(self.actuator, "general", joint=prop_name, name=name + "_actr3", gear=" 0 0 1 0 0 " + CRAZYFLIE_PROP.MOTOR_PARAM.value, ctrllimited="true", ctrlrange="0 " + CRAZYFLIE_PROP.MAX_THRUST.value)

        prop_name = name + "_prop4"
        
        pos = CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET_Z.value
        prop4_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop4_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop4_body, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop", mass=mass, pos=pos, rgba=PROP_COLOR, euler="0 0 0.785")
        ET.SubElement(self.actuator, "general", joint=prop_name, name=name + "_actr4", gear=" 0 0 1 0 0 -" + CRAZYFLIE_PROP.MOTOR_PARAM.value, ctrllimited="true", ctrlrange="0 " + CRAZYFLIE_PROP.MAX_THRUST.value)

        ET.SubElement(self.sensor, "gyro", site=site_name, name=name + "_gyro")
        ET.SubElement(self.sensor, "velocimeter", site=site_name, name=name + "_velocimeter")
        ET.SubElement(self.sensor, "accelerometer", site=site_name, name=name + "_accelerometer")
        ET.SubElement(self.sensor, "framepos", objtype="site", objname=site_name, name=name + "_posimeter")
        ET.SubElement(self.sensor, "framequat", objtype="site", objname=site_name, name=name + "_orimeter")
        ET.SubElement(self.sensor, "frameangacc ", objtype="site", objname=site_name, name=name + "_ang_accelerometer")

        return drone
    
    def __add_mocap_crazyflie(self, name, pos, quat, color):
        
        drone = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(drone, "geom", name=name + "_body", type="mesh", mesh="drone_body", rgba=color)
        ET.SubElement(drone, "geom", name=name + "_4_motormounts", type="mesh", mesh="drone_4_motormounts", rgba=color)
        ET.SubElement(drone, "geom", name=name + "_4_motors", type="mesh", mesh="drone_4_motors", rgba=color)

        prop_name = name + "_prop1"
        pos = CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop", pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")

        prop_name = name + "_prop2"
        pos = "-" + CRAZYFLIE_PROP.OFFSET.value + " -" + CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop", pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")

        prop_name = name + "_prop3"
        pos = "-" + CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop", pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")

        prop_name = name + "_prop4"
        pos = CRAZYFLIE_PROP.OFFSET.value + " -" + CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop", pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")


    
    def __add_bumblebee(self, name, pos, quat, color, is_hooked=False, hook_dof = 1):

        drone = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat)
        ET.SubElement(drone, "inertial", pos="0 0 0", diaginertia="1.5e-3 1.45e-3 2.66e-3", mass="0.605")
        ET.SubElement(drone, "joint", name=name, type="free")
        
        # need to rotate the body mesh to match optitrack orientation
        quat_mesh = mh.quaternion_from_euler(0, 0, math.radians(270))
        quat_mesh_str = str(quat_mesh[0]) + " " + str(quat_mesh[1]) + " " + str(quat_mesh[2]) + " " + str(quat_mesh[3])

        #drone_body = ET.SubElement(drone, "body", name=name + "_body", pos="0 0 0")
        ET.SubElement(drone, "geom", name=name + "_body", pos="0.0132 0 0", type="mesh", quat=quat_mesh_str, mesh="drone_body_large", rgba=color)
        #ET.SubElement(drone_body, "geom", name=name + "_4_motormounts", type="mesh", mesh="drone_4_motormounts_large", rgba=color, mass="0.0001")
        #ET.SubElement(drone_body, "geom", name=name + "_4_motors", type="mesh", mesh="drone_4_motors_large", rgba=color, mass="0.0001")

        site_name = name + SITE_NAME_END
        ET.SubElement(drone, "site", name=site_name, pos="0 0 0")

        if is_hooked:
            self.__add_hook_to_drone(drone, name, hook_dof)

        prop_name = name + "_prop1"
        mass = "0.00001"
        pos = BUMBLEBEE_PROP.OFFSET_X2.value + " -" + BUMBLEBEE_PROP.OFFSET_Y.value + " " + BUMBLEBEE_PROP.OFFSET_Z.value
        prop1_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop1_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop1_body, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop_large", mass=mass, pos=pos, rgba=PROP_LARGE_COLOR)
        ET.SubElement(drone, "site", name=prop_name, pos=pos)
        ET.SubElement(self.actuator, "general", site=prop_name, name=name + "_actr1", gear=" 0 0 1 0 0 " + BUMBLEBEE_PROP.MOTOR_PARAM.value, ctrllimited="true", ctrlrange="0 " + BUMBLEBEE_PROP.MAX_THRUST.value)

        prop_name = name + "_prop2"
        pos = "-" + BUMBLEBEE_PROP.OFFSET_X1.value + " -" + BUMBLEBEE_PROP.OFFSET_Y.value + " " + BUMBLEBEE_PROP.OFFSET_Z.value
        prop2_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop2_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop2_body, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop_large", mass=mass, pos=pos, rgba=PROP_LARGE_COLOR)
        ET.SubElement(drone, "site", name=prop_name, pos=pos)
        ET.SubElement(self.actuator, "general", site=prop_name, name=name + "_actr2", gear=" 0 0 1 0 0 -" + BUMBLEBEE_PROP.MOTOR_PARAM.value, ctrllimited="true", ctrlrange="0 " + BUMBLEBEE_PROP.MAX_THRUST.value)

        prop_name = name + "_prop3"
        pos = "-" + BUMBLEBEE_PROP.OFFSET_X1.value + " " + BUMBLEBEE_PROP.OFFSET_Y.value + " " + BUMBLEBEE_PROP.OFFSET_Z.value
        prop3_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop3_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop3_body, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop_large", mass=mass, pos=pos, rgba=PROP_LARGE_COLOR)
        ET.SubElement(drone, "site", name=prop_name, pos=pos)
        ET.SubElement(self.actuator, "general", site=prop_name, name=name + "_actr3", gear=" 0 0 1 0 0 " + BUMBLEBEE_PROP.MOTOR_PARAM.value, ctrllimited="true", ctrlrange="0 " + BUMBLEBEE_PROP.MAX_THRUST.value)

        prop_name = name + "_prop4"
        pos = BUMBLEBEE_PROP.OFFSET_X2.value + " " + BUMBLEBEE_PROP.OFFSET_Y.value + " " + BUMBLEBEE_PROP.OFFSET_Z.value
        prop4_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop4_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop4_body, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop_large", mass=mass, pos=pos, rgba=PROP_LARGE_COLOR)
        ET.SubElement(drone, "site", name=prop_name, pos=pos)
        ET.SubElement(self.actuator, "general", site=prop_name, name=name + "_actr4", gear=" 0 0 1 0 0 -" + BUMBLEBEE_PROP.MOTOR_PARAM.value, ctrllimited="true", ctrlrange="0 " + BUMBLEBEE_PROP.MAX_THRUST.value)

        ET.SubElement(self.sensor, "gyro", site=site_name, name=name + "_gyro")
        ET.SubElement(self.sensor, "velocimeter", site=site_name, name=name + "_velocimeter")
        ET.SubElement(self.sensor, "accelerometer", site=site_name, name=name + "_accelerometer")
        ET.SubElement(self.sensor, "framepos", objtype="site", objname=site_name, name=name + "_posimeter")
        ET.SubElement(self.sensor, "framequat", objtype="site", objname=site_name, name=name + "_orimeter")
        ET.SubElement(self.sensor, "frameangacc ", objtype="site", objname=site_name, name=name + "_ang_accelerometer")

        return drone


    def __add_mocap_bumblebee(self, name, pos, quat, color):
        
        drone = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat, mocap="true")

        # need to rotate the body mesh to match optitrack orientation
        quat_mesh = mh.quaternion_from_euler(0, 0, math.radians(270))
        quat_mesh_str = str(quat_mesh[0]) + " " + str(quat_mesh[1]) + " " + str(quat_mesh[2]) + " " + str(quat_mesh[3])
        ET.SubElement(drone, "geom", name=name + "_body", pos="0.0132 0 0", type="mesh", quat=quat_mesh_str, mesh="drone_body_large", rgba=color)
        ET.SubElement(drone, "geom", type="box", size="0.0475 0.025 0.025", pos="0.01 0 -0.02", rgba=color)

        prop_name = name + "_prop1"
        pos_m = "-" + BUMBLEBEE_PROP.OFFSET_X1.value + " " + BUMBLEBEE_PROP.OFFSET_Y.value + " " + BUMBLEBEE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos_m)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop_large", pos=pos_m, rgba=PROP_COLOR)

        prop_name = name + "_prop2"
        pos_m = BUMBLEBEE_PROP.OFFSET_X2.value + " -" + BUMBLEBEE_PROP.OFFSET_Y.value + " " + BUMBLEBEE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos_m)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop_large", pos=pos_m, rgba=PROP_COLOR)

        prop_name = name + "_prop3"
        pos_m = BUMBLEBEE_PROP.OFFSET_X2.value + " " + BUMBLEBEE_PROP.OFFSET_Y.value + " " + BUMBLEBEE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos_m)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop_large", pos=pos_m, rgba=PROP_COLOR)

        prop_name = name + "_prop4"
        pos_m = "-" + BUMBLEBEE_PROP.OFFSET_X1.value + " -" + BUMBLEBEE_PROP.OFFSET_Y.value + " " + BUMBLEBEE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos_m)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop_large", pos=pos_m, rgba=PROP_COLOR)

        return drone


    def __add_hook_to_drone(self, drone, drone_name, hook_dof = 1):
        
        rod = ET.SubElement(drone, "body", name=drone_name + "_rod", pos="0 0 0")
        ET.SubElement(rod, "geom", type="cylinder", fromto="0 0 0  0 0 -0.4", size="0.005", mass="0.00")
        site_name = drone_name + "_rod_end"
        ET.SubElement(rod, "site", name=site_name, pos="0 0 -0.4", type="sphere", size="0.002")
        ET.SubElement(rod, "joint", name=drone_name + "_hook_y", axis="0 1 0", pos="0 0 0", damping="0.001")
        if hook_dof == 2:
            ET.SubElement(rod, "joint", name=drone_name + "_hook_x", axis="1 0 0", pos="0 0 0", damping="0.001")
            ET.SubElement(self.sensor, "jointvel", joint=drone_name + "_hook_x", name=drone_name + "_hook_jointvel_x")
            ET.SubElement(self.sensor, "jointpos", joint=drone_name + "_hook_x", name=drone_name + "_hook_jointpos_x")
        elif hook_dof != 1:
            print("Too many or not enough degrees of freedom for hook joint. 1 degree of freedom assumed.")
        hook = ET.SubElement(rod, "body", name=drone_name + "_hook", pos="0 0 -0.4", euler="0 3.141592 -1.57")
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01299 0.0475", euler="-1.0472 0 0", size="0.005 0.018", mass="0.0001")
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.02598 0.07", euler="0 0 0", size="0.005 0.018", mass="0.0001")
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01299 0.0925", euler="1.0472 0 0", size="0.005 0.018", mass="0.0001")
        ET.SubElement(hook, "geom", type="capsule", pos="0 -0.01299 0.0925", euler="2.0944 0 0", size="0.005 0.018", mass="0.0001")
        
        ET.SubElement(self.sensor, "jointvel", joint=drone_name + "_hook_y", name=drone_name + "_hook_jointvel_y")
        ET.SubElement(self.sensor, "jointpos", joint=drone_name + "_hook_y", name=drone_name + "_hook_jointpos_y")

        ET.SubElement(self.sensor, "framepos", objtype="site", objname=site_name, name=drone_name + "_hook_pos")
        ET.SubElement(self.sensor, "framelinvel", objtype="site", objname=site_name, name=drone_name + "_hook_vel")
        ET.SubElement(self.sensor, "framequat", objtype="site", objname=site_name, name=drone_name + "_hook_quat")
        ET.SubElement(self.sensor, "frameangvel", objtype="site", objname=site_name, name=drone_name + "_hook_angvel")
    

    def __add_mocap_hook_to_drone(self, drone, drone_pos, drone_name, hook_dof=1):

        splt = drone_pos.split()

        pos_z = float(splt[2])
        pos_z -= ROD_LENGTH
        hook_pos = splt[0] + " " + splt[1] + " " + str(pos_z)
        
        hook = ET.SubElement(self.worldbody, "body", name=drone_name + "_hook", pos=drone_pos, mocap="true")
        #ET.SubElement(hook, "joint", name=drone_name + "_hook_y", axis="0 1 0", pos="0 0 0", damping="0.001")
        ET.SubElement(hook, "geom", type="cylinder", fromto="0 0 0  0 0 -0.4", size="0.0025")
        #hook = ET.SubElement(self.worldbody, "body", name=drone_name + "_hook", pos=hook_pos, euler="0 3.141592 -1.57", mocap="true")
        
        pos_z = 0.02 - ROD_LENGTH
        hpos = "0 0 " + str(pos_z)
        ET.SubElement(hook, "geom", type="capsule", pos=hpos, size="0.002 0.02")
        pos_z = -0.0475 - ROD_LENGTH
        hpos = "-0.01299 0 " + str(pos_z)
        ET.SubElement(hook, "geom", type="capsule", pos=hpos, euler="0 -1.0472 0", size="0.005 0.018")
        pos_z = -0.025 - ROD_LENGTH
        hpos = "-0.02598 0 " + str(pos_z)
        ET.SubElement(hook, "geom", type="capsule", pos=hpos, euler="0 0 0", size="0.005 0.018")
        pos_z = -0.005 - ROD_LENGTH
        hpos = "-0.01299 0 " + str(pos_z)
        ET.SubElement(hook, "geom", type="capsule", pos=hpos, euler="0 1.0472 0", size="0.005 0.018")
        pos_z = -0.0475 - ROD_LENGTH
        hpos = "0.01299 0 " + str(pos_z)
        ET.SubElement(hook, "geom", type="capsule", pos=hpos, euler="0 1.0472 0", size="0.005 0.018")



    def add_hospital(self, pos, quat=None):
        name = "hospital"
        if self.hospital is None:
            tag = "body"
            if quat is None:
                self.hospital = ET.SubElement(self.worldbody, tag, name=name, pos=pos)
            else:
                self.hospital = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat)

            ET.SubElement(self.hospital, "geom", name=name, type="box", pos="0 0 0.445", size="0.1275 0.13 0.445", material="mat-hospital")

            return self.hospital
        else:
            print("[SceneXmlGenerator] Hospital already added")


    def add_post_office(self, pos, quat=None):
        name = "post_office"
        if self.post_office is None:
            tag = "body"
            if quat is None:
                self.post_office = ET.SubElement(self.worldbody, tag, name=name, pos=pos)
            else:
                self.post_office = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat)

            ET.SubElement(self.post_office, "geom", name=name, type="box", pos="0 0 0.205", size="0.1275 0.1275 0.205", material="mat-post_office")

            return self.post_office
        else:
            print("[SceneXmlGenerator] Post office already added")


    def add_landing_zone(self, name, pos, quat=None):
        tag = "body"
        if quat is None:
            landing_zone = ET.SubElement(self.worldbody, tag, name=name, pos=pos)
        else:
            landing_zone = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat)
        
        ET.SubElement(landing_zone, "geom", {"class" : "landing_zone"})

        return landing_zone


    def add_sztaki(self, pos, quat):
        if self.sztaki is None:
            name = "sztaki"
            
            self.sztaki = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat)

            ET.SubElement(self.sztaki, "geom", name=name, type="box", pos="0 0 0.0925", size="0.105 0.105 0.0925", rgba="0.8 0.8 0.8 1.0", material="mat-sztaki")

            return self.sztaki

        else:
            print("[SceneXmlGenerator] Sztaki already added")
    
    def add_load(self, pos, size, mass, quat, color, type=PAYLOAD_TYPES.Box.value, is_mocap=False):

        if is_mocap:
            name = "loadmocap_" + str(self.__mocap_load_cntr)
            self.__mocap_load_cntr += 1
            load = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat, mocap="true")
            
            if type == PAYLOAD_TYPES.Box.value:
                box_pos = "0 0 " + size.split()[2]
                ET.SubElement(load, "geom", type="box", size=size, pos=box_pos, rgba=color)
                hook_pos = "0 0 " + str(2 * float(size.split()[2]))
            elif type == PAYLOAD_TYPES.Teardrop.value:
                ET.SubElement(load, "geom", type="mesh", mesh="payload_simplified", pos="0 0 0.0405", rgba=color, euler="1.57 0 0")
                hook_pos = "0 0 0.05"
        
        else:
            name = "load_" + str(self.__load_cntr)
            self.__load_cntr += 1
            
            load = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat)

            if type == PAYLOAD_TYPES.Box.value:
                box_pos = "0 0 " + size.split()[2]
                ET.SubElement(load, "geom", name=name, type="box", size=size, pos=box_pos, mass=mass, rgba=color)
                hook_pos = "0 0 " + str(2 * float(size.split()[2]))
            elif type == PAYLOAD_TYPES.Teardrop.value:
                ET.SubElement(load, "geom", name=name, type="mesh", mesh="payload_simplified", pos="0 0 0.0405", mass=mass, rgba=color, euler="1.57 0 0")
                hook_pos = "0 0 0.05"

            ET.SubElement(load, "joint", name=name, type="free")
            
            ET.SubElement(self.sensor, "framepos", objtype="body", objname=name, name=name + "_posimeter")
            ET.SubElement(self.sensor, "framequat", objtype="body", objname=name, name=name + "_orimeter")

        hook = ET.SubElement(load, "body", name=name + "_hook", pos=hook_pos, euler="0 0 3")

        hook_mass = "0.0001"
        ET.SubElement(hook, "geom", type="capsule", pos="0 0 0.02", size="0.002 0.02", mass=hook_mass)

        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01173 0.04565", euler="-1.12200 0 0", size="0.004 0.01562", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01061 0.04439", euler="-1.17810 0 0", size="0.004 0.01378", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.02561 0.05939", euler="-0.39270 0 0", size="0.004 0.01378", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.02561 0.08061", euler="0.39270 0 0", size="0.004 0.01378", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01061 0.09561", euler="1.17810 0 0", size="0.004 0.01378", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 -0.01061 0.09561", euler="1.96350 0 0", size="0.004 0.01378", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 -0.02561 0.08061", euler="2.74889 0 0", size="0.004 0.01378", mass=hook_mass)

        return name
    """
    def add_mocap_load(self, pos, size, quat, color, type=PAYLOAD_TYPES.Box.value):

        name = "loadmocap_" + str(self.__mocap_load_cntr)
        self.__mocap_load_cntr += 1


        load = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat, mocap="true")
        if type == PAYLOAD_TYPES.Box.value:
            box_pos = "0 0 " + size.split()[2]
            ET.SubElement(load, "geom", type="box", size=size, pos=box_pos, rgba=color)
            hook_pos = "0 0 " + str(2 * float(size.split()[2]))
        elif type == PAYLOAD_TYPES.Teardrop.value:
            ET.SubElement(load, "geom", type="mesh", mesh="payload_simplified", pos="0 0 0.04", rgba=color, euler="1.57 0 0")
            hook_pos = "0 0 0.05"
            

        hook = ET.SubElement(load, "body", name=name + "_hook", pos=hook_pos, euler="0 0 3")

        ET.SubElement(hook, "geom", type="capsule", pos="0 0 0.02", size="0.002 0.02")

        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01173 0.04565", euler="-1.12200 0 0", size="0.004 0.01562")
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01061 0.04439", euler="-1.17810 0 0", size="0.004 0.01378")
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.02561 0.05939", euler="-0.39270 0 0", size="0.004 0.01378")
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.02561 0.08061", euler="0.39270 0 0", size="0.004 0.01378")
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01061 0.09561", euler="1.17810 0 0", size="0.004 0.01378")
        ET.SubElement(hook, "geom", type="capsule", pos="0 -0.01061 0.09561", euler="1.96350 0 0", size="0.004 0.01378")
        ET.SubElement(hook, "geom", type="capsule", pos="0 -0.02561 0.08061", euler="2.74889 0 0", size="0.004 0.01378")

        return name
"""



    def add_car(self, pos, quat, color, is_virtual, has_rod=False, type="fleet1tenth"):

        name = None

        if is_virtual and type == "fleet1tenth":
            name = "virtfleet1tenth_" + str(self.__virtfleet1tenth_cntr)
            self.__add_fleet1tenth(pos, quat, name, color, has_rod)
            self.__virtfleet1tenth_cntr += 1
        
        elif not is_virtual and type == "fleet1tenth":
            name = "realfleet1tenth_" + str(self.__realfleet1tenth_cntr)
            self.__add_mocap_fleet1tenth(pos, quat, name, color, has_rod)
            self.__realfleet1tenth_cntr += 1
        
        else:
            print("[SceneXmlGenerator] Unknown car type")
            return None
        
        return name
    
    def __add_fleet1tenth(self, pos, quat, name, color, has_rod):
        site_name = name + SITE_NAME_END

        posxyz = str.split(pos)
        pos = posxyz[0] + " " + posxyz[1] + " " + F1T_PROP.WHEEL_RADIUS.value
        
        car = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat)

        ET.SubElement(car, "inertial", pos="0 0 0", diaginertia=".05 .05 .08", mass="3.0")
        ET.SubElement(car, "joint", name=name, type="free")
        ET.SubElement(car, "site", name=site_name, pos="0 0 0")

        self.__add_fleet1tenth_body(car, name, color, has_rod)

        armature = "0.05"
        armature_steer = "0.001"
        fric_steer = "0.2"
        damp_steer = "0.2"
        damping = "0.00001"
        frictionloss = "0.01"

        steer_range = "-0.6 0.6"


        wheelfl = ET.SubElement(car, "body", name=name + "_wheelfl", quat="1 0 0 0" )
        ET.SubElement(wheelfl, "joint", name=name + "_wheelfl_steer", type="hinge", pos="0.16113 .10016 0", limited="true", frictionloss=fric_steer, damping=damp_steer, armature=armature_steer, range=steer_range, axis="0 0 1")
        ET.SubElement(wheelfl, "joint", name=name + "_wheelfl", type="hinge", pos="0.16113 .122385 0", axis="0 1 0", frictionloss=frictionloss, damping=damping, armature=armature, limited="false")

        ET.SubElement(wheelfl, "geom", name=name + "_wheelfl", type="cylinder", size=F1T_PROP.WHEEL_SIZE.value, pos="0.16113 .122385 0", mass="0.1", material="material_check", euler="1.571 0 0")

        wheelrl = ET.SubElement(car, "body", name=name + "_wheelrl", quat="1 0 0 0" )
        ET.SubElement(wheelrl, "joint", name=name + "_wheelrl", type="hinge", pos="-0.16113 .122385 0", axis="0 1 0", frictionloss=frictionloss, damping=damping, armature=armature, limited="false")

        ET.SubElement(wheelrl, "geom", name=name + "_wheelrl", type="cylinder", size=F1T_PROP.WHEEL_SIZE.value, pos="-0.16113 .122385 0", mass="0.1", material="material_check", euler="1.571 0 0")

        wheelfr = ET.SubElement(car, "body", name=name + "_wheelfr", quat="1 0 0 0" )
        ET.SubElement(wheelfr, "joint", name=name + "_wheelfr_steer", type="hinge", pos="0.16113 -.10016 0", limited="true", frictionloss=fric_steer, damping=damp_steer, armature=armature_steer, range=steer_range, axis="0 0 1")
        ET.SubElement(wheelfr, "joint", name=name + "_wheelfr", type="hinge", pos="0.16113 -.122385 0", axis="0 1 0", frictionloss=frictionloss, damping=damping, armature=armature, limited="false")

        ET.SubElement(wheelfr, "geom", name=name + "_wheelfr", type="cylinder", size=F1T_PROP.WHEEL_SIZE.value, pos="0.16113 -.122385 0", mass="0.1", material="material_check", euler="1.571 0 0")

        wheelrr = ET.SubElement(car, "body", name=name + "_wheelrr", quat="1 0 0 0" )
        ET.SubElement(wheelrr, "joint", name=name + "_wheelrr", type="hinge", pos="-0.16113 -.122385 0", axis="0 1 0", frictionloss=frictionloss, damping=damping, armature=armature, limited="false")

        ET.SubElement(wheelrr, "geom", name=name + "_wheelrr", type="cylinder", size=F1T_PROP.WHEEL_SIZE.value, pos="-0.16113 -.122385 0", mass="0.1", material="material_check", euler="1.571 0 0")

        friction = "2.5 2.5 .009 .0001 .0001"

        ET.SubElement(self.contact, "pair", geom1=name + "_wheelfl", geom2="roundabout", condim="6", friction=friction)
        ET.SubElement(self.contact, "pair", geom1=name + "_wheelfr", geom2="roundabout", condim="6", friction=friction)
        ET.SubElement(self.contact, "pair", geom1=name + "_wheelrl", geom2="roundabout", condim="6", friction=friction)
        ET.SubElement(self.contact, "pair", geom1=name + "_wheelrr", geom2="roundabout", condim="6", friction=friction)

        ET.SubElement(self.actuator, "motor", name=name + "_wheelfl_actr", joint=name + "_wheelfl")
        ET.SubElement(self.actuator, "motor", name=name + "_wheelfr_actr", joint=name + "_wheelfr")
        ET.SubElement(self.actuator, "motor", name=name + "_wheelrl_actr", joint=name + "_wheelrl")
        ET.SubElement(self.actuator, "motor", name=name + "_wheelrr_actr", joint=name + "_wheelrr")

        kp = "15"
        ET.SubElement(self.actuator, "position", forcelimited="true", forcerange="-5 5", name=name + "_wheelfl_actr_steer", joint=name + "_wheelfl_steer", kp=kp)
        ET.SubElement(self.actuator, "position", forcelimited="true", forcerange="-5 5", name=name + "_wheelfr_actr_steer", joint=name + "_wheelfr_steer", kp=kp)


        ET.SubElement(self.sensor, "gyro", site=site_name, name=name + "_gyro")
        ET.SubElement(self.sensor, "velocimeter", site=site_name, name=name + "_velocimeter")
        ET.SubElement(self.sensor, "framepos", objtype="site", objname=site_name, name=name + "_posimeter")
        ET.SubElement(self.sensor, "framequat", objtype="site", objname=site_name, name=name + "_orimeter")


    def __add_mocap_fleet1tenth(self, pos, quat, name, color, has_rod):

        posxyz = str.split(pos)
        pos = posxyz[0] + " " + posxyz[1] + " " + F1T_PROP.WHEEL_RADIUS.value
        
        car = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat, mocap="true")

        self.__add_fleet1tenth_body(car, name, color, has_rod)

        ET.SubElement(car, "geom", name=name + "_wheelfl", type="cylinder", size=F1T_PROP.WHEEL_SIZE.value, pos="0.16113 .122385 0", rgba="0.1 0.1 0.1 1.0", euler="1.571 0 0")

        ET.SubElement(car, "geom", name=name + "_wheelrl", type="cylinder", size=F1T_PROP.WHEEL_SIZE.value, pos="-0.16113 .122385 0", rgba="0.1 0.1 0.1 1.0", euler="1.571 0 0")

        ET.SubElement(car, "geom", name=name + "_wheelfr", type="cylinder", size=F1T_PROP.WHEEL_SIZE.value, pos="0.16113 -.122385 0", rgba="0.1 0.1 0.1 1.0", euler="1.571 0 0")

        ET.SubElement(car, "geom", name=name + "_wheelrr", type="cylinder", size=F1T_PROP.WHEEL_SIZE.value, pos="-0.16113 -.122385 0", rgba="0.1 0.1 0.1 1.0", euler="1.571 0 0")

    
    def __add_fleet1tenth_body(self, car, name, color, has_rod):
        ET.SubElement(car, "geom", name=name + "_chassis_b", type="box", size=".10113 .1016 .02", pos= "-.06 0 0", rgba=color)
        ET.SubElement(car, "geom", name=name + "_chassis_f", type="box", size=".06 .07 .02", pos=".10113 0 0", rgba=color)
        ET.SubElement(car, "geom", name=name + "_front", type="box", size=".052388 .02 .02", pos=".2135 0 0", rgba=color)
        ET.SubElement(car, "geom", name=name + "_back", type="box", size=".052388 .02 .02", pos="-.2135 0 0", rgba=color)
        ET.SubElement(car, "geom", name=name + "_front_bumper", type="box", size=".005 .09 .02", pos=".265888 0 0.02", rgba=color)
        ET.SubElement(car, "geom", name=name + "_back_bumper", type="box", size=".005 .08 .02", pos="-.265888 0 0.02", rgba=color)
        ET.SubElement(car, "geom", name=name + "_number", type="cylinder", size=".01984 .03", pos=".12 0 .05", rgba="0.1 0.1 0.1 1.0")
        ET.SubElement(car, "geom", name=name + "_camera", type="box", size=".012 .06 0.02", pos=".18 0 .08")
        ET.SubElement(car, "geom", name=name + "_camera_holder", type="box", size=".012 .008 .02", pos=".18 0 .04")
        ET.SubElement(car, "geom", name=name + "_circuits", type="box", size=".08 .06 .03", pos="-.05 0 .05", rgba=color)
        ET.SubElement(car, "geom", name=name + "_antennal", type="box", size=".007 .004 .06", pos="-.16 -.01 .105", euler="0.2 0 0", rgba=".1 .1 .1 1.0")
        ET.SubElement(car, "geom", name=name + "_antennar", type="box", size=".007 .004 .06", pos="-.16 .01 .105", euler="-0.2 0 0", rgba=".1 .1 .1 1.0")
        ET.SubElement(car, "geom", name=name + "_antenna_holder", type="box", size=".008 .008 .02", pos="-.16 0 .04", rgba=".1 .1 .1 1.0")

        if has_rod:
            ET.SubElement(car, "geom", name=name + "_rod", type="cylinder", size="0.01 0.47625", pos="-.2135 0 0.47625", rgba="0.3 0.3 0.3 1.0")


    def save_xml(self, file_name):
        
        tree = ET.ElementTree(self.root)
        #ET.indent(tree, space="\t", level=0) # uncomment this if python version >= 3.9
        tree.write(file_name)
        print()
        print("[SceneXmlGenerator] Scene xml file saved at: " + os.path.normpath(file_name))


"""
scene = SceneXmlGenerator("scene.xml")

scene.add_airport(pos="0.5 -1.2 0.0025")
scene.add_parking_lot(pos="-0.5 1.2 0.0025")

scene.add_drone(name="drone0", pos="0 -1 0.2", color="0.1 0.9 0.1 1")
scene.add_drone(name="drone1", pos="0 1 0.2", color="0.1 0.9 0.1 1")

scene.add_pole(name="pole1", pos="0.25 0.25 0")
scene.add_pole(name="pole2", pos="-0.25 0.25 0")
scene.add_pole(name="pole3", pos="0.3 -0.3 0", quat="0.924 0 0 0.383")
scene.add_pole(name="pole4", pos="-0.3 -0.3 0", quat="0.924 0 0 0.383")

scene.add_hospital(pos="-1 1 0")
scene.add_post_office(pos="1 1.255 0")

scene.add_landing_zone(name="landing_zone1", pos="-1.2 -0.7 0", quat="0.924 0 0 0.383")
scene.add_landing_zone(name="landing_zone2", pos="-1 -0.9 0", quat="0.924 0 0 0.383")
scene.add_landing_zone(name="landing_zone3", pos="-0.8 -1.1 0", quat="0.924 0 0 0.383")
scene.add_landing_zone(name="landing_zone4", pos="-0.6 -1.3 0", quat="0.924 0 0 0.383")

scene.add_sztaki(pos="0 0 0", quat="1 0 0 0")

scene.save_xml("first_xml.xml")
"""