import xml.etree.ElementTree as ET

import aiml_virtual.util.mujoco_helper as mh
import math

from aiml_virtual.object.payload import PAYLOAD_TYPES
from aiml_virtual.object.drone import DRONE_TYPES, BUMBLEBEE_PROP, CRAZYFLIE_PROP 
from aiml_virtual.object.car import F1T_PROP

import os



PROP_COLOR = "0.1 0.1 0.1 1.0"
PROP_LARGE_COLOR = "0.1 0.02 0.5 1.0"

SITE_NAME_END = "_cog"

ROD_LENGTH = float(BUMBLEBEE_PROP.ROD_LENGTH.value)

class SceneXmlGenerator:

    def __init__(self, base_scene_filename):

        self.root = ET.Element("mujoco")
        ET.SubElement(self.root, "include", file=base_scene_filename)
        self.asset = ET.SubElement(self.root, "asset")
        self.worldbody = ET.SubElement(self.root, "worldbody")
        self.contact = ET.SubElement(self.root, "contact")
        self.actuator = ET.SubElement(self.root, "actuator")
        self.sensor = ET.SubElement(self.root, "sensor")

        self.parking_lot = None
        self.airport = None
        self.hospital = None
        self.post_office = None
        self.sztaki = None

        self._bicycle_counter = 0

        self._pole_cntr = 0
        self._virtcrazyflie_cntr = 0
        self._virtbumblebee_cntr = 0
        self._virtbumblebee_hooked_cntr = 0
        self._realcrazyflie_cntr = 0
        self._realbumblebee_cntr = 0
        self._realbumblebee_hooked_cntr = 0
        self._virtfleet1tenth_cntr = 0
        self._realfleet1tenth_cntr = 0
        
        self._box_payload_cntr = 0
        self._teardrop_payload_cntr = 0
        self._mocap_box_payload_cntr = 0
        self._mocap_teardrop_payload_cntr = 0

        self.radar_cntr = 0

        self._mocap_drone_names = []
        self._mocap_payload_names = []

    def add_bicycle(self, pos, quat, color):

        name = "Bicycle_" + str(self._bicycle_counter)
        self._bicycle_counter += 1

        site_name = name + "_cog"

        bicycle = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat)

        ET.SubElement(bicycle, "inertial", pos="0 0 0", diaginertia=".01 .01 .01", mass="1.0")
        ET.SubElement(bicycle, "joint", name = name, type="free")
        ET.SubElement(bicycle, "site", name=site_name, pos="0 0 0")

        ET.SubElement(bicycle, "geom", name=name + "_crossbar", type="box", size=".06 .015 .02", pos="0 0 0", rgba=color)

        front_wheel_name = name + "_wheelf"
        wheelf = ET.SubElement(bicycle, "body", name=front_wheel_name)

        ET.SubElement(wheelf, "joint", name=front_wheel_name, type="hinge", pos="0.1 0 0",
                      axis="0 1 0", frictionloss="0.001", damping="0.00001", armature="0.01")
        ET.SubElement(wheelf, "geom", name=front_wheel_name, type="cylinder", size="0.04 0.015",
                      pos="0.1 0 0", euler="1.571 0 0", material="material_check")

        rear_wheel_name = name + "_wheelr"
        wheelr = ET.SubElement(bicycle, "body", name=rear_wheel_name)

        ET.SubElement(wheelr, "joint", name=rear_wheel_name, type="hinge", pos="-0.1 0 0",
                      axis="0 1 0", frictionloss="0.001", damping="0.00001", armature="0.01")
        ET.SubElement(wheelr, "geom", name=rear_wheel_name, type="cylinder", size="0.04 0.015",
                      pos="-0.1 0 0", euler="1.571 0 0", material="material_check")
        
        ET.SubElement(self.actuator, "motor", name=name + "_actr", joint=rear_wheel_name)
        ET.SubElement(self.sensor, "velocimeter", site=site_name, name=name + "_velocimeter")


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
    

    def add_pole(self, pos, quat=None):
        name = "pole_" + str(self._pole_cntr)
        self._pole_cntr += 1
        tag = "body"
        if quat is None:
            pole = ET.SubElement(self.worldbody, tag, name=name, pos=pos)
        else:
            pole = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat)
        
        ET.SubElement(pole, "geom", {"class": "pole_top"})
        ET.SubElement(pole, "geom", {"class": "pole_bottom1"})
        ET.SubElement(pole, "geom", {"class": "pole_bottom2"})

        return pole


    def add_drone(self, pos, quat, color, type: DRONE_TYPES, hook_dof = 1):

        if type == DRONE_TYPES.BUMBLEBEE_HOOKED:
            name = "BumblebeeHooked_" + str(self._virtbumblebee_hooked_cntr)

            drone = self._add_bumblebee(name, pos, quat, color, True, hook_dof)

            self._virtbumblebee_hooked_cntr += 1
            return name
        
        elif type == DRONE_TYPES.BUMBLEBEE:

            name = "Bumblebee_" + str(self._virtbumblebee_cntr)
            drone = self._add_bumblebee(name, pos, quat, color)

            self._virtbumblebee_cntr += 1
            return name

        elif type == DRONE_TYPES.CRAZYFLIE:
            name = "Crazyflie_" + str(self._virtcrazyflie_cntr)

            drone = self._add_crazyflie(name, pos, quat, color)

            self._virtcrazyflie_cntr += 1
            return name
            
        
        else:
            print("[SceneXmlGenerator] Error: unknown drone type: " + str(type))
            return None
        

    @staticmethod
    def print_elements(root, tabs=""):

        for v in root:
            if "name" in v.attrib:
                print(tabs + str(v.attrib["name"]))

            tbs = tabs + "\t"

            SceneXmlGenerator.print_elements(v, tbs)


    def add_mocap_drone(self, pos, quat, color, type: DRONE_TYPES, index=-1):
        
        if type == DRONE_TYPES.CRAZYFLIE:

            if index < 0:
                name = "DroneMocap_crazyflie_" + str(self._realcrazyflie_cntr)
                while name in self._mocap_drone_names:
                    self._realcrazyflie_cntr += 1
                    name = "DroneMocap_crazyflie_" + str(self._realcrazyflie_cntr)
            
            else:
                name = "DroneMocap_crazyflie_" + str(index)
                if name in self._mocap_drone_names:
                    print("[SceneXmlGenerator] this mocap crazyflie index already exists: " + str(index))
                    return

            self._add_mocap_crazyflie(name, pos, quat, color)
            self._mocap_drone_names += [name]
        
        elif type == DRONE_TYPES.BUMBLEBEE:

            if index < 0:
                name = "DroneMocap_bumblebee_" + str(self._realbumblebee_cntr)
                while name in self._mocap_drone_names:
                    self._realbumblebee_cntr += 1
                    name = "DroneMocap_bumblebee_" + str(self._realbumblebee_cntr)
                    
            else:
                name = "DroneMocap_bumblebee_" + str(index)
                if name in self._mocap_drone_names:
                    print("[SceneXmlGenerator] this mocap bumblebee index already exists: " + str(index))
                    return

            self._add_mocap_bumblebee(name, pos, quat, color)
            self._mocap_drone_names += [name]

        elif type == DRONE_TYPES.BUMBLEBEE_HOOKED:

            if index < 0:
                name = "DroneMocapHooked_bumblebee_" + str(self._realbumblebee_hooked_cntr)
                while name in self._mocap_drone_names:
                    self._realbumblebee_hooked_cntr += 1
                    name = "DroneMocapHooked_bumblebee_" + str(self._realbumblebee_hooked_cntr)

            else:
                name = "DroneMocapHooked_bumblebee_" + str(index)
                if name in self._mocap_drone_names:
                    print("[SceneXmlGenerator] this mocap hooked bumblebee index already exists: " + str(index))
                    return

            self._add_mocap_bumblebee(name, pos, quat, color)
            self.add_mocap_hook(pos, name)
            self._mocap_drone_names += [name]
        
        return name


        
    
    def _add_crazyflie(self, name, pos, quat, color):

        
        mass = CRAZYFLIE_PROP.MASS.value
        diaginertia = CRAZYFLIE_PROP.DIAGINERTIA.value
        Lx1 = CRAZYFLIE_PROP.OFFSET.value
        Lx2 = CRAZYFLIE_PROP.OFFSET.value
        Ly = CRAZYFLIE_PROP.OFFSET.value
        Lz = CRAZYFLIE_PROP.OFFSET_Z.value
        motor_param = CRAZYFLIE_PROP.MOTOR_PARAM.value
        max_thrust = CRAZYFLIE_PROP.MAX_THRUST.value

        drone = self._add_drone_common_parts(name, pos, quat, PROP_COLOR, mass, diaginertia, Lx1, Lx2, Ly, Lz, motor_param, max_thrust, "crazyflie")

        ET.SubElement(drone, "geom", name=name + "_body", type="mesh", mesh="crazyflie_body", rgba=color)
        ET.SubElement(drone, "geom", name=name + "_4_motormounts", type="mesh", mesh="crazyflie_4_motormounts", rgba=color)
        ET.SubElement(drone, "geom", name=name + "_4_motors", type="mesh", mesh="crazyflie_4_motors", rgba=color)


        return drone

    def _add_drone_common_parts(self, name, pos, quat, propeller_color, mass, diaginertia, Lx1, Lx2, Ly, Lz, motor_param, max_thrust, mesh_prefix):

        drone = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat)
        ET.SubElement(drone, "inertial", pos="0 0 0", diaginertia=diaginertia, mass=mass)
        ET.SubElement(drone, "joint", name=name, type="free")
        #drone_body = ET.SubElement(drone, "body", name=name + "_body", pos="0 0 0")

        site_name = name + SITE_NAME_END
        ET.SubElement(drone, "site", name=site_name, pos="0 0 0", size="0.005")

        prop_site_size = "0.0001"

        prop_name = name + "_prop1"
        mass = "0.00001"
        pos = Lx2 + " -" + Ly + " " + Lz
        prop1_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop1_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop1_body, "geom", name=prop_name, type="mesh", mesh=mesh_prefix + "_ccw_prop", mass=mass, pos=pos, rgba=propeller_color)
        ET.SubElement(drone, "site", name=prop_name, pos=pos, size=prop_site_size)
        ET.SubElement(self.actuator, "general", site=prop_name, name=name + "_actr1", gear=" 0 0 1 0 0 " + motor_param, ctrllimited="true", ctrlrange="0 " + max_thrust)

        prop_name = name + "_prop2"
        pos = "-" + Lx1 + " -" + Ly + " " + Lz
        prop2_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop2_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop2_body, "geom", name=prop_name, type="mesh", mesh=mesh_prefix + "_cw_prop", mass=mass, pos=pos, rgba=propeller_color)
        ET.SubElement(drone, "site", name=prop_name, pos=pos, size=prop_site_size)
        ET.SubElement(self.actuator, "general", site=prop_name, name=name + "_actr2", gear=" 0 0 1 0 0 -" + motor_param, ctrllimited="true", ctrlrange="0 " + max_thrust)

        prop_name = name + "_prop3"
        pos = "-" + Lx1 + " " + Ly + " " + Lz
        prop3_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop3_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop3_body, "geom", name=prop_name, type="mesh", mesh=mesh_prefix + "_ccw_prop", mass=mass, pos=pos, rgba=propeller_color)
        ET.SubElement(drone, "site", name=prop_name, pos=pos, size=prop_site_size)
        ET.SubElement(self.actuator, "general", site=prop_name, name=name + "_actr3", gear=" 0 0 1 0 0 " + motor_param, ctrllimited="true", ctrlrange="0 " + max_thrust)

        prop_name = name + "_prop4"
        pos = Lx2 + " " + Ly + " " + Lz
        prop4_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop4_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop4_body, "geom", name=prop_name, type="mesh", mesh=mesh_prefix + "_cw_prop", mass=mass, pos=pos, rgba=propeller_color)
        ET.SubElement(drone, "site", name=prop_name, pos=pos, size=prop_site_size)
        ET.SubElement(self.actuator, "general", site=prop_name, name=name + "_actr4", gear=" 0 0 1 0 0 -" + motor_param, ctrllimited="true", ctrlrange="0 " + max_thrust)

        ET.SubElement(self.sensor, "gyro", noise="0.0027", site=site_name, name=name + "_gyro")
        ET.SubElement(self.sensor, "velocimeter", noise="0.00078", site=site_name, name=name + "_velocimeter")
        ET.SubElement(self.sensor, "accelerometer", site=site_name, name=name + "_accelerometer")
        ET.SubElement(self.sensor, "framepos", noise="0.00014", objtype="site", objname=site_name, name=name + "_posimeter")
        ET.SubElement(self.sensor, "framequat", noise="0.00026", objtype="site", objname=site_name, name=name + "_orimeter")
        ET.SubElement(self.sensor, "frameangacc ", objtype="site", objname=site_name, name=name + "_ang_accelerometer")

        return drone
    
    def _add_mocap_crazyflie(self, name, pos, quat, color):
        
        drone = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(drone, "geom", name=name + "_body", type="mesh", mesh="crazyflie_body", rgba=color)
        ET.SubElement(drone, "geom", name=name + "_4_motormounts", type="mesh", mesh="crazyflie_4_motormounts", rgba=color)
        ET.SubElement(drone, "geom", name=name + "_4_motors", type="mesh", mesh="crazyflie_4_motors", rgba=color)

        prop_name = name + "_prop1"
        pos = CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="crazyflie_ccw_prop", pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")

        prop_name = name + "_prop2"
        pos = "-" + CRAZYFLIE_PROP.OFFSET.value + " -" + CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="crazyflie_ccw_prop", pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")

        prop_name = name + "_prop3"
        pos = "-" + CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="crazyflie_cw_prop", pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")

        prop_name = name + "_prop4"
        pos = CRAZYFLIE_PROP.OFFSET.value + " -" + CRAZYFLIE_PROP.OFFSET.value + " " + CRAZYFLIE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="crazyflie_cw_prop", pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")


    
    def _add_bumblebee(self, name, pos, quat, color, is_hooked=False, hook_dof = 1):

        mass = BUMBLEBEE_PROP.MASS.value
        diaginertia = BUMBLEBEE_PROP.DIAGINERTIA.value
        Lx1 = BUMBLEBEE_PROP.OFFSET_X1.value
        Lx2 = BUMBLEBEE_PROP.OFFSET_X2.value
        Ly = BUMBLEBEE_PROP.OFFSET_Y.value
        Lz = BUMBLEBEE_PROP.OFFSET_Z.value
        motor_param = BUMBLEBEE_PROP.MOTOR_PARAM.value
        max_thrust = BUMBLEBEE_PROP.MAX_THRUST.value
        

        drone = self._add_drone_common_parts(name, pos, quat, PROP_LARGE_COLOR, mass, diaginertia, Lx1, Lx2, Ly, Lz, motor_param, max_thrust, "bumblebee")

        # need to rotate the body mesh to match optitrack orientation
        quat_mesh = mh.quaternion_from_euler(0, 0, math.radians(270))
        quat_mesh_str = str(quat_mesh[0]) + " " + str(quat_mesh[1]) + " " + str(quat_mesh[2]) + " " + str(quat_mesh[3])

        ET.SubElement(drone, "geom", name=name + "_body", pos="0.0132 0 0", type="mesh", quat=quat_mesh_str, mesh="bumblebee_body", rgba=color)
        if is_hooked:
            self._add_hook_to_drone(drone, name, hook_dof)

        return drone

    def _add_mocap_bumblebee(self, name, pos, quat, color):
        
        drone = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat, mocap="true")
        #ET.SubElement(drone, "site", pos="0 0 -.485", size="0.01")

        # need to rotate the body mesh to match optitrack orientation
        quat_mesh = mh.quaternion_from_euler(0, 0, math.radians(270))
        quat_mesh_str = str(quat_mesh[0]) + " " + str(quat_mesh[1]) + " " + str(quat_mesh[2]) + " " + str(quat_mesh[3])
        ET.SubElement(drone, "geom", name=name + "_body", pos="0.0132 0 0", type="mesh", quat=quat_mesh_str, mesh="bumblebee_body", rgba=color)
        ET.SubElement(drone, "geom", type="box", size="0.0475 0.02 0.02", pos="0.01 0 -0.02", rgba=color)
        ET.SubElement(drone, "geom", type="box", size="0.015 0.015 0.015", pos="-0.015 0 -0.055", rgba=".1 .1 .1 1.0")

        prop_name = name + "_prop1"
        pos_m = "-" + BUMBLEBEE_PROP.OFFSET_X1.value + " " + BUMBLEBEE_PROP.OFFSET_Y.value + " " + BUMBLEBEE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos_m)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="bumblebee_ccw_prop", pos=pos_m, rgba=PROP_COLOR)

        prop_name = name + "_prop2"
        pos_m = BUMBLEBEE_PROP.OFFSET_X2.value + " -" + BUMBLEBEE_PROP.OFFSET_Y.value + " " + BUMBLEBEE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos_m)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="bumblebee_ccw_prop", pos=pos_m, rgba=PROP_COLOR)

        prop_name = name + "_prop3"
        pos_m = BUMBLEBEE_PROP.OFFSET_X2.value + " " + BUMBLEBEE_PROP.OFFSET_Y.value + " " + BUMBLEBEE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos_m)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="bumblebee_cw_prop", pos=pos_m, rgba=PROP_COLOR)

        prop_name = name + "_prop4"
        pos_m = "-" + BUMBLEBEE_PROP.OFFSET_X1.value + " -" + BUMBLEBEE_PROP.OFFSET_Y.value + " " + BUMBLEBEE_PROP.OFFSET_Z.value
        prop = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop, "joint", name=prop_name, axis="0 0 1", pos=pos_m)
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="bumblebee_cw_prop", pos=pos_m, rgba=PROP_COLOR)

        return drone


    def _add_hook_to_drone(self, drone, drone_name, hook_dof = 1):
        
        hook_structure = ET.SubElement(drone, "body", name=drone_name + "_hookstructure", pos="0 0 0")
        site_name = drone_name + "_rod_end"
        ET.SubElement(hook_structure, "site", name=site_name, pos="0 0 -" + str(ROD_LENGTH), type="sphere", size="0.002")
        ET.SubElement(hook_structure, "joint", name=drone_name + "_hook_y", axis="0 1 0", pos="0 0 0", damping="0.001")
        if hook_dof == 2:
            ET.SubElement(hook_structure, "joint", name=drone_name + "_hook_x", axis="1 0 0", pos="0 0 0", damping="0.001")
            ET.SubElement(self.sensor, "jointvel", joint=drone_name + "_hook_x", name=drone_name + "_hook_jointvel_x")
            ET.SubElement(self.sensor, "jointpos", joint=drone_name + "_hook_x", name=drone_name + "_hook_jointpos_x")
        elif hook_dof != 1:
            print("Too many or not enough degrees of freedom for hook joint. 1 degree of freedom assumed.")
        
        self._add_hook_structure(hook_structure, drone_name)
            
        
        ET.SubElement(self.sensor, "jointvel", joint=drone_name + "_hook_y", name=drone_name + "_hook_jointvel_y")
        ET.SubElement(self.sensor, "jointpos", joint=drone_name + "_hook_y", name=drone_name + "_hook_jointpos_y")

        ET.SubElement(self.sensor, "framepos", objtype="site", objname=site_name, name=drone_name + "_hook_pos")
        ET.SubElement(self.sensor, "framelinvel", objtype="site", objname=site_name, name=drone_name + "_hook_vel")
        ET.SubElement(self.sensor, "framequat", objtype="site", objname=site_name, name=drone_name + "_hook_quat")
        ET.SubElement(self.sensor, "frameangvel", objtype="site", objname=site_name, name=drone_name + "_hook_angvel")
    

    def add_mocap_hook(self, drone_pos, drone_name):

        splt = drone_pos.split()

        pos_z = float(splt[2])
        pos_z -= ROD_LENGTH
        hook_pos = splt[0] + " " + splt[1] + " " + str(pos_z)

        name_tail = drone_name.split("_")[-1]
        
        name = "HookMocap" + "_" + name_tail

        hook = ET.SubElement(self.worldbody, "body", name=name, pos=hook_pos, mocap="true")

        hook_structure_body = ET.SubElement(hook, "body", pos="0 0 " + str(ROD_LENGTH))

        self._add_hook_structure(hook_structure_body, name)




    def _add_hook_structure(self, hook_structure_body, name_base):
        
        ET.SubElement(hook_structure_body, "geom", name=name_base + "_rod", type="cylinder", fromto="0 0 0  0 0 -" + str(ROD_LENGTH), size="0.005", mass="0.0")
        hookbody = ET.SubElement(hook_structure_body, "body", name=name_base + "_hook", pos="0 0 -" + str(ROD_LENGTH), euler="0 3.141592 0")

        ET.SubElement(hookbody, "geom", type="capsule", pos="0 0 0.02", size="0.003 0.02", mass="0.01")
        ET.SubElement(hookbody, "geom", type="capsule", pos="0 0.01299 0.0475", euler="-1.0472 0 0", size="0.005 0.018", mass="0.0001")
        ET.SubElement(hookbody, "geom", type="capsule", pos="0 0.02598 0.07", euler="0 0 0", size="0.005 0.018", mass="0.0001")
        ET.SubElement(hookbody, "geom", type="capsule", pos="0 0.01299 0.0925", euler="1.0472 0 0", size="0.005 0.018", mass="0.0001")
        ET.SubElement(hookbody, "geom", type="capsule", pos="0 -0.01299 0.0925", euler="2.0944 0 0", size="0.005 0.018", mass="0.0001")



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
    
    def add_payload(self, pos, size, mass, quat, color, type=PAYLOAD_TYPES.Box):

        if type == PAYLOAD_TYPES.Box:
            name = "BoxPayload_" + str(self._box_payload_cntr)
            self._box_payload_cntr += 1
            box_pos = "0 0 " + size.split()[2]
            load = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat)
            ET.SubElement(load, "geom", name=name, type="box", size=size, pos=box_pos, mass=mass, rgba=color)
            #ET.SubElement(load, "inertial", pos="0 0 0", diaginertia="5e-4 5e-4 5e-4", mass=mass)
            hook_pos = "0 0 " + str(2 * float(size.split()[2]))
        elif type == PAYLOAD_TYPES.Teardrop:
            name = "TeardropPayload_" + str(self._teardrop_payload_cntr)
            self._teardrop_payload_cntr += 1
            load = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat)
            ET.SubElement(load, "geom", name=name, type="mesh", mesh="payload_simplified", pos="0 0 0.0405", mass=mass, rgba=color, euler="1.57 0 0")
            hook_pos = "0 0 0.05"

        self._add_hook_to_payload(load, name, hook_pos)

        ET.SubElement(load, "joint", name=name, type="free")
        
        ET.SubElement(self.sensor, "framepos", objtype="body", objname=name, name=name + "_posimeter")
        ET.SubElement(self.sensor, "framequat", objtype="body", objname=name, name=name + "_orimeter")


        return name
    
    def add_mocap_payload(self, pos, size, quat, color, type=PAYLOAD_TYPES.Box, index=-1):
        
        if type == PAYLOAD_TYPES.Box:
            if index < 0:
                name = "PayloadMocap_box_" + str(self._mocap_box_payload_cntr)
                while name in self._mocap_payload_names:
                    self._mocap_box_payload_cntr += 1
                    name = "PayloadMocap_box_" + str(self._mocap_box_payload_cntr)
            
            else:
                name = "PayloadMocap_box_" + str(index)
                if name in self._mocap_payload_names:
                    print("[SceneXmlGenerator] this mocap box payload index already exists: " + str(index))
                    return

            load = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat, mocap="true")
                

            box_pos = "0 0 " + size.split()[2]
            ET.SubElement(load, "geom", type="box", size=size, pos=box_pos, rgba=color)
            hook_pos = "0 0 " + str(2 * float(size.split()[2]))
            self._add_hook_to_payload(load, name, hook_pos)

            self._mocap_payload_names += [name]

            return name

        elif type == PAYLOAD_TYPES.Teardrop:
            if index < 0:
                name = "PayloadMocap_teardrop_" + str(self._mocap_teardrop_payload_cntr)
                while name in self._mocap_payload_names:
                    self._mocap_teardrop_payload_cntr += 1
                    name = "PayloadMocap_teardrop_" + str(self._mocap_teardrop_payload_cntr)
            
            else:
                name = "PayloadMocap_teardrop_" + str(index)
                if name in self._mocap_payload_names:
                    print("[SceneXmlGenerator] this mocap teardrop payload index already exists: " + str(index))
                    return

            load = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat, mocap="true")

            ET.SubElement(load, "geom", type="mesh", mesh="payload_simplified", pos="0 0 0.0405", rgba=color, euler="1.57 0 0")
            hook_pos = "0 0 0.05"
            self._add_hook_to_payload(load, name, hook_pos)
            
            self._mocap_payload_names += [name]
            
            return name
        
        else:
            print("[SceneXmlGenerator] unknown payload type.")
            return
        


    def _add_hook_to_payload(self, payload, name, hook_pos):

        hook = ET.SubElement(payload, "body", name=name + "_hook", pos=hook_pos, euler="0 0 -1.57")

        hook_mass = "0.0001"

        ET.SubElement(hook, "geom", type="capsule", pos="0 0 0.025", size="0.002 0.027", mass=hook_mass)

        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01173 0.05565", euler="-1.12200 0 0", size="0.004 0.01562", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01061 0.05439", euler="-1.17810 0 0", size="0.004 0.01378", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.02561 0.06939", euler="-0.39270 0 0", size="0.004 0.01378", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.02561 0.09061", euler="0.39270 0 0", size="0.004 0.01378", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01061 0.10561", euler="1.17810 0 0", size="0.004 0.01378", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 -0.01061 0.10561", euler="1.96350 0 0", size="0.004 0.01378", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 -0.02561 0.09061", euler="2.74889 0 0", size="0.004 0.01378", mass=hook_mass)


    def add_car(self, pos, quat, color, is_virtual, has_rod=False, type="fleet1tenth"):

        name = None

        if is_virtual and type == "fleet1tenth":
            name = "Fleet1Tenth_" + str(self._virtfleet1tenth_cntr)
            self._add_fleet1tenth(pos, quat, name, color, has_rod)
            self._virtfleet1tenth_cntr += 1
        
        elif not is_virtual and type == "fleet1tenth":
            name = "CarMocap_fleet1tenth_" + str(self._realfleet1tenth_cntr)
            self._add_mocap_fleet1tenth(pos, quat, name, color, has_rod)
            self._realfleet1tenth_cntr += 1
        
        else:
            print("[SceneXmlGenerator] Unknown car type")
            return None
        
        return name
    
    def _add_fleet1tenth(self, pos, quat, name, color, has_rod):
        site_name = name + SITE_NAME_END

        posxyz = str.split(pos)
        pos = posxyz[0] + " " + posxyz[1] + " " + F1T_PROP.WHEEL_RADIUS.value
        
        car = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat)

        ET.SubElement(car, "inertial", pos="0 0 0", diaginertia=".05 .05 .08", mass="3.0")
        ET.SubElement(car, "joint", name=name, type="free")
        ET.SubElement(car, "site", name=site_name, pos="0 0 0")

        self._add_fleet1tenth_body(car, name, color, has_rod)

        armature = "0.05"
        armature_steer = "0.001"
        fric_steer = "0.2"
        damp_steer = "0.2"
        damping = "0.00001"
        frictionloss = "0.01"

        steer_range = "-0.6 0.6"


        wheelfl = ET.SubElement(car, "body", name=name + "_wheelfl" )
        ET.SubElement(wheelfl, "joint", name=name + "_wheelfl_steer", type="hinge", pos="0.16113 .10016 0", limited="true", frictionloss=fric_steer, damping=damp_steer, armature=armature_steer, range=steer_range, axis="0 0 1")
        ET.SubElement(wheelfl, "joint", name=name + "_wheelfl", type="hinge", pos="0.16113 .122385 0", axis="0 1 0", frictionloss=frictionloss, damping=damping, armature=armature, limited="false")

        ET.SubElement(wheelfl, "geom", name=name + "_wheelfl", type="cylinder", size=F1T_PROP.WHEEL_SIZE.value, pos="0.16113 .122385 0", mass="0.1", material="material_check", euler="1.571 0 0")

        wheelrl = ET.SubElement(car, "body", name=name + "_wheelrl" )
        ET.SubElement(wheelrl, "joint", name=name + "_wheelrl", type="hinge", pos="-0.16113 .122385 0", axis="0 1 0", frictionloss=frictionloss, damping=damping, armature=armature, limited="false")

        ET.SubElement(wheelrl, "geom", name=name + "_wheelrl", type="cylinder", size=F1T_PROP.WHEEL_SIZE.value, pos="-0.16113 .122385 0", mass="0.1", material="material_check", euler="1.571 0 0")

        wheelfr = ET.SubElement(car, "body", name=name + "_wheelfr" )
        ET.SubElement(wheelfr, "joint", name=name + "_wheelfr_steer", type="hinge", pos="0.16113 -.10016 0", limited="true", frictionloss=fric_steer, damping=damp_steer, armature=armature_steer, range=steer_range, axis="0 0 1")
        ET.SubElement(wheelfr, "joint", name=name + "_wheelfr", type="hinge", pos="0.16113 -.122385 0", axis="0 1 0", frictionloss=frictionloss, damping=damping, armature=armature, limited="false")

        ET.SubElement(wheelfr, "geom", name=name + "_wheelfr", type="cylinder", size=F1T_PROP.WHEEL_SIZE.value, pos="0.16113 -.122385 0", mass="0.1", material="material_check", euler="1.571 0 0")

        wheelrr = ET.SubElement(car, "body", name=name + "_wheelrr" )
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


    def _add_mocap_fleet1tenth(self, pos, quat, name, color, has_rod):

        posxyz = str.split(pos)
        pos = posxyz[0] + " " + posxyz[1] + " " + F1T_PROP.WHEEL_RADIUS.value
        
        car = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat, mocap="true")

        self._add_fleet1tenth_body(car, name, color, has_rod)

        ET.SubElement(car, "geom", name=name + "_wheelfl", type="cylinder", size=F1T_PROP.WHEEL_SIZE.value, pos="0.16113 .122385 0", rgba="0.1 0.1 0.1 1.0", euler="1.571 0 0")

        ET.SubElement(car, "geom", name=name + "_wheelrl", type="cylinder", size=F1T_PROP.WHEEL_SIZE.value, pos="-0.16113 .122385 0", rgba="0.1 0.1 0.1 1.0", euler="1.571 0 0")

        ET.SubElement(car, "geom", name=name + "_wheelfr", type="cylinder", size=F1T_PROP.WHEEL_SIZE.value, pos="0.16113 -.122385 0", rgba="0.1 0.1 0.1 1.0", euler="1.571 0 0")

        ET.SubElement(car, "geom", name=name + "_wheelrr", type="cylinder", size=F1T_PROP.WHEEL_SIZE.value, pos="-0.16113 -.122385 0", rgba="0.1 0.1 0.1 1.0", euler="1.571 0 0")

    
    def _add_fleet1tenth_body(self, car, name, color, has_rod):
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
            ET.SubElement(car, "geom", name=name + "_rod", type="cylinder", size="0.02 0.5225", pos="-.175 0 0.5225", rgba="0.3 0.3 0.3 1.0", euler="0 0.1 0")
    

    def add_radar_pyramids(self, pos, number_of_pyramids=18, scale="1 1 1", color="0.5 0.5 0.5 0.2"):
        
        name = "radar_" + str(self.radar_cntr)
        mesh_name = "pyramid_tobe_rotated_" + str(self.radar_cntr)

        angle = 2 * math.pi / number_of_pyramids
        
        h = 1.0

        aper2 = math.tan(angle / 2) * h

        z_dim = 0.5

        v0 = str(aper2) + " 1 -" + str(z_dim)
        v1 = str(aper2) + " 1 " + str(z_dim)
        v2 = "-" + str(aper2) + " 1 " + str(z_dim)
        v3 = "-" + str(aper2) + " 1 -" + str(z_dim)
        v4 = "0 0 0"

        vertices = v0 + "  " + v1 + "  " + v2 + "  " + v3 + "  " + v4

        pyramid_mesh = ET.SubElement(self.asset, "mesh", name=mesh_name, vertex=vertices, scale=scale)

        body = ET.SubElement(self.worldbody, "body", name=name, pos=pos)

        angle_rotated = 0.0
        for i in range(number_of_pyramids):
            euler = "0 0 " + str(angle_rotated)
            ET.SubElement(body, "geom", type="mesh", mesh=mesh_name, rgba=color, euler=euler)

            angle_rotated += angle

        self.radar_cntr += 1

        return name
    
    def add_radar_field(self, pos, color="0.5 0.5 0.5 0.2"):

        name = "radar_" + str(self.radar_cntr)

        body = ET.SubElement(self.worldbody, "body", name=name, pos=pos)

        ET.SubElement(self.asset, "material", name=name + "_mat", rgba=color, specular="0.0", shininess="0.0")

        #ET.SubElement(body, "geom", type="mesh", mesh="radar_field", material="matte_tp_red_mat")
        ET.SubElement(body, "geom", type="mesh", mesh="radar_field", contype="0", conaffinity="0", material=name + "_mat")

        self.radar_cntr += 1

        return name



    def save_xml(self, file_name):
        
        tree = ET.ElementTree(self.root)
        #ET.indent(tree, space="\t", level=0) # uncomment this if python version >= 3.9
        tree.write(file_name)
        print()
        print("[SceneXmlGenerator] Scene xml file saved at: " + os.path.normpath(file_name))

