import xml.etree.ElementTree as ET

import util.mujoco_helper as mh
import math



PROP_OFFS = "0.047"
PROP_OFFS_Z = "0.032"

PROP_OFFS_X1_LARGE = "0.074"
PROP_OFFS_X2_LARGE = "0.091"
PROP_OFFS_Y_LARGE = "0.087"
PROP_OFFS_Z_LARGE = "0.036"

PROP_COLOR = "0.1 0.1 0.1 1.0"
PROP_LARGE_COLOR = "0.1 0.02 0.5 1.0"

SITE_NAME_END = "_cog"


ROD_LENGTH = 0.4

class SceneXmlGenerator:

    def __init__(self, base_scene_filename):
        self.root = ET.Element("mujoco")
        ET.SubElement(self.root, "include", file=base_scene_filename)
        self.worldbody = ET.SubElement(self.root, "worldbody")
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
        
        self.__load_cntr = 0


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


    def add_drone(self, pos, quat, color, is_virtual = True, type="crazyflie", is_hooked = False):

        if is_hooked and type != "bumblebee":
            print("[SceneXmlGenerator] Error: Hooked drone can only be bumblebee...")
            return None


        if is_virtual:

            if type == "bumblebee":

                if is_hooked:
                    name = "virtbumblebee_hooked_" + str(self.__virtbumblebee_hooked_cntr)

                    drone = self.__add_bumblebee(name, pos, quat, color, True)

                    self.__virtbumblebee_hooked_cntr += 1
                    return
                
                else:

                    name = "virtbumblebee_" + str(self.__virtbumblebee_cntr)
                    drone = self.__add_bumblebee(name, pos, quat, color)

                    self.__virtbumblebee_cntr += 1
                    return

            elif type == "crazyflie":
                name = "virtcrazyflie_" + str(self.__virtcrazyflie_cntr)

                drone = self.__add_crazyflie(name, pos, quat, color)

                self.__virtcrazyflie_cntr += 1
                return
                
            
            else:
                print("[SceneXmlGenerator] Error: unknown drone type: " + str(type))
                return
        
        else:
            if type == "bumblebee":

                if is_hooked:

                    name = "realbumblebee_hooked_" + str(self.__realbumblebee_hooked_cntr)

                    drone = self.__add_mocap_bumblebee(name, pos, quat, color)
                    self.__add_mocap_hook_to_drone(drone, pos, name)

                    self.__realbumblebee_hooked_cntr += 1
                    return

                else:

                    name = "realbumblebee_" + str(self.__virtbumblebee_cntr)
                    drone = self.__add_mocap_bumblebee(name, pos, quat, color)

                    self.__realbumblebee_cntr += 1
                    return

            elif type == "crazyflie":
                name = "realcrazyflie_" + str(self.__realcrazyflie_cntr)

                drone = self.__add_mocap_crazyflie(name, pos, quat, color)

                self.__realcrazyflie_cntr += 1
                return
            
            else:
                print("[SceneXmlGenerator] Error: unknown drone type: " + str(type))
                return

        
    
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
        pos = PROP_OFFS + " " + PROP_OFFS + " " + PROP_OFFS_Z
        prop1_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop1_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop1_body, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop", mass=mass, pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")

        prop_name = name + "_prop2"
        pos = "-" + PROP_OFFS + " -" + PROP_OFFS + " " + PROP_OFFS_Z
        prop2_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop2_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop2_body, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop", mass=mass, pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")

        prop_name = name + "_prop3"
        pos = "-" + PROP_OFFS + " " + PROP_OFFS + " " + PROP_OFFS_Z
        prop3_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop3_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop3_body, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop", mass=mass, pos=pos, rgba=PROP_COLOR, euler="0 0 0.785")

        prop_name = name + "_prop4"
        pos = PROP_OFFS + " -" + PROP_OFFS + " " + PROP_OFFS_Z
        prop4_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop4_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop4_body, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop", mass=mass, pos=pos, rgba=PROP_COLOR, euler="0 0 0.785")

        
        ET.SubElement(self.actuator, "general", site=site_name, name=name + "_actr0", gear=" 0 0 1 0 0 0", ctrllimited="true", ctrlrange="0 0.64")
        ET.SubElement(self.actuator, "general", site=site_name, name=name + "_actr1", gear=" 0 0 0 1 0 0", ctrllimited="true", ctrlrange="-0.01 0.01")
        ET.SubElement(self.actuator, "general", site=site_name, name=name + "_actr2", gear=" 0 0 0 0 1 0", ctrllimited="true", ctrlrange="-0.01 0.01")
        ET.SubElement(self.actuator, "general", site=site_name, name=name + "_actr3", gear=" 0 0 0 0 0 1", ctrllimited="true", ctrlrange="-0.01 0.01")

        ET.SubElement(self.sensor, "gyro", site=site_name, name=name + "_sensor0")

        return drone
    
    def __add_mocap_crazyflie(self, name, pos, quat, color):
        
        drone = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(drone, "geom", name=name + "_body", type="mesh", mesh="drone_body", rgba=color)
        ET.SubElement(drone, "geom", name=name + "_4_motormounts", type="mesh", mesh="drone_4_motormounts", rgba=color)
        ET.SubElement(drone, "geom", name=name + "_4_motors", type="mesh", mesh="drone_4_motors", rgba=color)

        prop_name = name + "_prop1"
        pos = PROP_OFFS + " " + PROP_OFFS + " " + PROP_OFFS_Z
        prop = ET.SubElement(self.worldbody, "body", name=prop_name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop", pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")

        prop_name = name + "_prop2"
        pos = "-" + PROP_OFFS + " -" + PROP_OFFS + " " + PROP_OFFS_Z
        prop = ET.SubElement(self.worldbody, "body", name=prop_name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop", pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")

        prop_name = name + "_prop3"
        pos = "-" + PROP_OFFS + " " + PROP_OFFS + " " + PROP_OFFS_Z
        prop = ET.SubElement(self.worldbody, "body", name=prop_name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop", pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")

        prop_name = name + "_prop4"
        pos = PROP_OFFS + " -" + PROP_OFFS + " " + PROP_OFFS_Z
        prop = ET.SubElement(self.worldbody, "body", name=prop_name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop", pos=pos, rgba=PROP_COLOR, euler="0 0 -0.785")


    
    def __add_bumblebee(self, name, pos, quat, color, is_hooked=False):

        drone = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat)
        ET.SubElement(drone, "inertial", pos="0 0 0", diaginertia="1.5e-3 1.45e-3 2.66e-3", mass="0.407")
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
            self.__add_hook_to_drone(drone, name)

        prop_name = name + "_prop1"
        mass = "0.00001"
        pos = "-" + PROP_OFFS_X1_LARGE + " " + PROP_OFFS_Y_LARGE + " " + PROP_OFFS_Z_LARGE
        prop1_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop1_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop1_body, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop_large", mass=mass, pos=pos, rgba=PROP_LARGE_COLOR)

        prop_name = name + "_prop2"
        pos = PROP_OFFS_X2_LARGE + " -" + PROP_OFFS_Y_LARGE + " " + PROP_OFFS_Z_LARGE
        prop2_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop2_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop2_body, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop_large", mass=mass, pos=pos, rgba=PROP_LARGE_COLOR)

        prop_name = name + "_prop3"
        pos = PROP_OFFS_X2_LARGE + " " + PROP_OFFS_Y_LARGE + " " + PROP_OFFS_Z_LARGE
        prop3_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop3_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop3_body, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop_large", mass=mass, pos=pos, rgba=PROP_LARGE_COLOR)

        prop_name = name + "_prop4"
        pos = "-" + PROP_OFFS_X1_LARGE + " -" + PROP_OFFS_Y_LARGE + " " + PROP_OFFS_Z_LARGE
        prop4_body = ET.SubElement(drone, "body", name=prop_name)
        ET.SubElement(prop4_body, "joint", name=prop_name, axis="0 0 1", pos=pos)
        ET.SubElement(prop4_body, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop_large", mass=mass, pos=pos, rgba=PROP_LARGE_COLOR)


        ET.SubElement(self.actuator, "general", site=site_name, name=name + "_actr0", gear=" 0 0 1 0 0 0", ctrllimited="true", ctrlrange="0 67.2")
        ET.SubElement(self.actuator, "general", site=site_name, name=name + "_actr1", gear=" 0 0 0 1 0 0", ctrllimited="true", ctrlrange="-6 6")
        ET.SubElement(self.actuator, "general", site=site_name, name=name + "_actr2", gear=" 0 0 0 0 1 0", ctrllimited="true", ctrlrange="-6 6")
        ET.SubElement(self.actuator, "general", site=site_name, name=name + "_actr3", gear=" 0 0 0 0 0 1", ctrllimited="true", ctrlrange="-1.5 1.5")

        ET.SubElement(self.sensor, "gyro", site=site_name, name=name + "_sensor0")

        return drone


    def __add_mocap_bumblebee(self, name, pos, quat, color):
        
        drone = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat, mocap="true")

        # need to rotate the body mesh to match optitrack orientation
        quat_mesh = mh.quaternion_from_euler(0, 0, math.radians(270))
        quat_mesh_str = str(quat_mesh[0]) + " " + str(quat_mesh[1]) + " " + str(quat_mesh[2]) + " " + str(quat_mesh[3])
        ET.SubElement(drone, "geom", name=name + "_body", pos="0.0132 0 0", type="mesh", quat=quat_mesh_str, mesh="drone_body_large", rgba=color)

        prop_name = name + "_prop1"
        pos_m = "-" + PROP_OFFS_X1_LARGE + " " + PROP_OFFS_Y_LARGE + " " + PROP_OFFS_Z_LARGE
        prop = ET.SubElement(self.worldbody, "body", name=prop_name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop_large", pos=pos_m, rgba=PROP_COLOR)

        prop_name = name + "_prop2"
        pos_m = PROP_OFFS_X2_LARGE + " -" + PROP_OFFS_Y_LARGE + " " + PROP_OFFS_Z_LARGE
        prop = ET.SubElement(self.worldbody, "body", name=prop_name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_ccw_prop_large", pos=pos_m, rgba=PROP_COLOR)

        prop_name = name + "_prop3"
        pos_m = PROP_OFFS_X2_LARGE + " " + PROP_OFFS_Y_LARGE + " " + PROP_OFFS_Z_LARGE
        prop = ET.SubElement(self.worldbody, "body", name=prop_name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop_large", pos=pos_m, rgba=PROP_COLOR)

        prop_name = name + "_prop4"
        pos_m = "-" + PROP_OFFS_X1_LARGE + " -" + PROP_OFFS_Y_LARGE + " " + PROP_OFFS_Z_LARGE
        prop = ET.SubElement(self.worldbody, "body", name=prop_name, pos=pos, quat=quat, mocap="true")
        ET.SubElement(prop, "geom", name=prop_name, type="mesh", mesh="drone_cw_prop_large", pos=pos_m, rgba=PROP_COLOR)


    def __add_hook_to_drone(self, drone, drone_name):
        
        rod = ET.SubElement(drone, "body", name=drone_name + "_rod", pos="0 0 0")
        ET.SubElement(rod, "geom", type="cylinder", fromto="0 0 0  0 0 -0.4", size="0.0025", mass="0.00")
        ET.SubElement(rod, "site", name=drone_name + "_rod_end", pos="0 0 -0.4", type="sphere", size="0.002")
        ET.SubElement(rod, "joint", name=drone_name + "_hook", axis="0 1 0", pos="0 0 0", damping="0.001")
        hook = ET.SubElement(rod, "body", name=drone_name + "_hook", pos="0 0 -0.4", euler="0 3.141592 -1.57")
        ET.SubElement(hook, "geom", type="capsule", pos="0 0 0.02", size="0.002 0.02", mass="0.02")
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01299 0.0475", euler="-1.0472 0 0", size="0.005 0.018", mass="0.0001")
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.02598 0.07", euler="0 0 0", size="0.005 0.018", mass="0.0001")
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01299 0.0925", euler="1.0472 0 0", size="0.005 0.018", mass="0.0001")
        ET.SubElement(hook, "geom", type="capsule", pos="0 -0.01299 0.0925", euler="2.0944 0 0", size="0.005 0.018", mass="0.0001")
    

    def __add_mocap_hook_to_drone(self, drone, drone_pos, drone_name):


        splt = drone_pos.split()

        pos_z = float(splt[2])
        pos_z -= ROD_LENGTH
        #hook_pos = splt[0] + " " + splt[1] + " " + str(pos_z)
        
        hook = ET.SubElement(self.worldbody, "body", name=drone_name + "_hook", pos=drone_pos, mocap="true")
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
    
    def add_load(self, pos, size, mass, quat, color):

        name = "load_" + str(self.__load_cntr)
        self.__load_cntr += 1

        box_pos = "0 0 " + size.split()[2]

        load = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat)
        ET.SubElement(load, "geom", type="box", size=size, mass=mass, pos=box_pos, rgba=color)
        ET.SubElement(load, "joint", type="free")

        hook_pos = "0 0 " + str(2 * float(size.split()[2]))
        hook = ET.SubElement(load, "body", name=name + "_hook", pos=hook_pos)

        hook_mass = "0.0001"
        ET.SubElement(hook, "geom", type="capsule", pos="0 0 0.02", size="0.002 0.02", mass=hook_mass)

        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01173 0.04565", euler="-1.12200 0 0", size="0.004 0.01562", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01061 0.04439", euler="-1.17810 0 0", size="0.004 0.01378", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.02561 0.05939", euler="-0.39270 0 0", size="0.004 0.01378", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.02561 0.08061", euler="0.39270 0 0", size="0.004 0.01378", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 0.01061 0.09561", euler="1.17810 0 0", size="0.004 0.01378", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 -0.01061 0.09561", euler="1.96350 0 0", size="0.004 0.01378", mass=hook_mass)
        ET.SubElement(hook, "geom", type="capsule", pos="0 -0.02561 0.08061", euler="2.74889 0 0", size="0.004 0.01378", mass=hook_mass)


    def add_mocapcar(self, pos, quat):
        
        car = ET.SubElement(self.worldbody, "body", name="car0", pos=pos, quat=quat, mocap="true")

        ET.SubElement(car, "geom", name="car0", type="box", size=".1 .13 .02", pos= "-.07 0 0", rgba=".2 .2 .9 1.0")
        ET.SubElement(car, "geom", name="car0_front", type="box", size=".07 .07 .02", pos=".13 0 0", rgba=".2 .2 .9 1.0")

        wheelfl = ET.SubElement(car, "body", name="car0_wheelfl", pos="0.2 .12 0", quat="1 0 0 0" )
        ET.SubElement(wheelfl, "joint", name="car0_wheelfl_steer", type="hinge", axis="0 0 1")
        ET.SubElement(wheelfl, "joint", name="car0_wheelfl", type="hinge", axis="0 1 0")

        ET.SubElement(wheelfl, "geom", name="car0_wheelfl", type="cylinder", size=".06 .02", material="material_check", euler="1.571 0 0")

        wheelrl = ET.SubElement(car, "body", name="car0_wheelrl", pos="-0.2 .12 0", quat="1 0 0 0" )
        ET.SubElement(wheelrl, "joint", name="car0_wheelrl_steer", type="hinge", axis="0 0 1")
        ET.SubElement(wheelrl, "joint", name="car0_wheelrl", type="hinge", axis="0 1 0")

        ET.SubElement(wheelrl, "geom", name="car0_wheelrl", type="cylinder", size=".06 .02", material="material_check", euler="1.571 0 0")

        wheelfr = ET.SubElement(car, "body", name="car0_wheelfr", pos="0.2 -.12 0", quat="1 0 0 0" )
        #ET.SubElement(wheelfr, "joint", name="car0_wheelfr_steer", type="hinge", axis="0 0 1")
        ET.SubElement(wheelfr, "joint", name="car0_wheelfr", type="hinge", axis="0 1 0")

        ET.SubElement(wheelfr, "geom", name="car0_wheelfr", type="cylinder", size=".06 .02", material="material_check", euler="1.571 0 0")

        wheelrr = ET.SubElement(car, "body", name="car0_wheelrr", pos="-0.2 -.12 0", quat="1 0 0 0" )
        #ET.SubElement(wheelrr, "joint", name="car0_wheelrr_steer", type="hinge", axis="0 0 1")
        ET.SubElement(wheelrr, "joint", name="car0_wheelrr", type="hinge", axis="0 1 0")

        ET.SubElement(wheelrr, "geom", name="car0_wheelrr", type="cylinder", size=".06 .02", material="material_check", euler="1.571 0 0")

    def save_xml(self, file_name):
        
        tree = ET.ElementTree(self.root)
        #ET.indent(tree, space="\t", level=0) # uncomment this if python version >= 3.9
        tree.write(file_name)


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