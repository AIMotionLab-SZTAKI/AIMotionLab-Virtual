from pickle import FALSE
import mujoco
import numpy as np


class Drone:

    def __init__(self, data: mujoco.MjData, name_in_xml, name_in_motive, is_virtual, trajectories, controllers, parameters):

        self.data = data
        self.is_virtual = is_virtual

        self.name_in_xml = name_in_xml
        self.name_in_motive = name_in_motive

        self.trajectories = trajectories
        self.controllers = controllers
        self.parameters = parameters

        free_joint = self.data.joint(self.name_in_xml)

        self.qpos = free_joint.qpos

        self.prop1_qpos = self.data.joint(self.name_in_xml + "_prop1").qpos
        self.prop2_qpos = self.data.joint(self.name_in_xml + "_prop2").qpos
        self.prop3_qpos = self.data.joint(self.name_in_xml + "_prop3").qpos
        self.prop4_qpos = self.data.joint(self.name_in_xml + "_prop4").qpos

        self.ctrl0 = self.data.actuator(self.name_in_xml + "_actr0").ctrl
        self.ctrl1 = self.data.actuator(self.name_in_xml + "_actr1").ctrl
        self.ctrl2 = self.data.actuator(self.name_in_xml + "_actr2").ctrl
        self.ctrl3 = self.data.actuator(self.name_in_xml + "_actr3").ctrl

        # make a copy now, so that mj_step() does not affect these
        self.prop1_angle = self.prop1_qpos[0]
        self.prop2_angle = self.prop2_qpos[0]
        self.prop3_angle = self.prop3_qpos[0]
        self.prop4_angle = self.prop4_qpos[0]

        self.top_body_xquat = self.data.body("virtbumblebee_hooked_0").xquat

        self.qvel = free_joint.qvel

        self.sensor_data = self.data.sensor("virtbumblebee_hooked_0_sensor0").data

    def get_qpos(self):
        return self.qpos
    
    def set_qpos(self, position, orientation):
        """
        orientation should be quaternion
        """
        self.qpos[:7] = np.append(position, orientation)
    
    def get_ctrl(self):
        return np.concatenate((self.ctrl0, self.ctrl1, self.ctrl2, self.ctrl3))
    
    def set_ctrl(self, ctrl):
        self.ctrl0[0] = ctrl[0]
        self.ctrl1[0] = ctrl[1]
        self.ctrl2[0] = ctrl[2]
        self.ctrl3[0] = ctrl[3]

    def get_top_body_xquat(self):
        return self.top_body_xquat

    def get_qvel(self):
        return self.qvel

    def get_sensor_data(self):
        return self.sensor_data

    def print_prop_angles(self):
        print("prop1: " + str(self.prop1_qpos))
        print("prop2: " + str(self.prop2_qpos))
        print("prop3: " + str(self.prop3_qpos))
        print("prop4: " + str(self.prop4_qpos))
    
    def spin_propellers(self, angle_step):
        #print("angle step: " + str(angle_step))
                
        self.prop1_angle -= angle_step + 0.005
        self.prop2_angle -= angle_step - 0.002
        self.prop3_angle += angle_step + 0.005
        self.prop4_angle += angle_step - 0.003

        self.prop1_qpos[0] = self.prop1_angle
        self.prop2_qpos[0] = self.prop2_angle
        self.prop3_qpos[0] = self.prop3_angle
        self.prop4_qpos[0] = self.prop4_angle
    
    def stop_propellers(self):
        self.prop1_qpos[0] = self.prop1_angle
        self.prop2_qpos[0] = self.prop2_angle
        self.prop3_qpos[0] = self.prop3_angle
        self.prop4_qpos[0] = self.prop4_angle



    def print_names(self):
        print("name in xml:      " + self.name_in_xml)
        if not self.is_virtual:
            print("name in motive:   " + self.name_in_motive)
    
    def print_info(self):
        self.print_names()
        print("Is virtual:       " + str(self.is_virtual))
    
    def get_label(self):
        return self.name_in_xml

    @staticmethod
    def parse_drones(data, joint_names):
        """
        Create a list of Drone objects from mujoco's MjData following a naming convention
        found in naming_convention_in_xml.txt
        """

        virtdrones = []
        realdrones = []
        icf = 0
        ibb = 0

        for _name in joint_names:

            _name_cut = _name[:len(_name) - 1]

            if _name.startswith("virtbumblebee_hooked") and not _name.endswith("hook") and not _name_cut.endswith("prop"):
                # this joint must be a drone
                hook = Drone.find_hook_for_drone(joint_names, _name)
                if hook:
                    d = DroneHooked(data, name_in_xml=_name,
                                    hook_name_in_xml=hook,
                                    name_in_motive=None,
                                    is_virtual=True,
                                    trajectories=None,
                                    controller=None,
                                    parameters=None)

                    virtdrones += [d]

                else:
                    print("Error: did not find hook joint for this drone: " +
                          _name + " ... Ignoring drone.")
            
            elif _name.startswith("virtbumblebee") and not _name.endswith("hook") and not _name_cut.endswith("prop"):
                # this joint must be a drone

                d = Drone(data, _name, None, True, None, None, None)
                virtdrones += [d]

            elif _name.startswith("virtcrazyflie") and not _name_cut.endswith("prop"):

                d = Drone(data, _name, None, True, None, None, None)
                virtdrones += [d]

            elif _name.startswith("realbumblebee_hooked") and not _name.endswith("hook") and not _name_cut.endswith("prop"):
                hook = Drone.find_hook_for_drone(joint_names, _name)
                if hook:
                    d = DroneHooked(data, name_in_xml=_name,
                                    hook_name_in_xml=hook,
                                    name_in_motive="bb" + str(ibb + 1),
                                    is_virtual=False,
                                    trajectories=None,
                                    controller=None,
                                    parameters=None)

                    realdrones += [d]
                    ibb += 1

                else:
                    print("Error: did not find hook joint for this drone: " +
                          _name + " ... Ignoring drone.")
            
            
            elif _name.startswith("realbumblebee") and not _name.endswith("hook") and not _name_cut.endswith("prop"):

                d = Drone(data, _name, "bb" + str(ibb + 1), False, None, None, None)
                realdrones += [d]
                ibb += 1

            elif _name.startswith("realcrazyflie") and not _name_cut.endswith("prop"):

                d = Drone(data, _name, "cf" + str(icf + 1), False, None, None, None)
                realdrones += [d]
                icf += 1
                


        print()
        print(str(len(virtdrones) + len(realdrones)) + " drones found in xml.")
        print()
        return virtdrones, realdrones

    @staticmethod
    def find_hook_for_drone(joint_names, drone_name):
        for jnt in joint_names:
            if drone_name + "_hook" == jnt:
                return jnt
        
        return None
    
    @staticmethod
    def get_drone_by_name_in_motive(drones, name: str):
        for i in range(len(drones)):
            if drones[i].name_in_motive == name:
                return drones[i]
        
        return None
    
    @staticmethod
    def get_drone_names_motive(drones):
        names = []
        for d in drones:
            if not d.is_virtual:
                names += [d.name_in_motive]
        
        return names
    
    @staticmethod
    def set_drone_names_motive(drones, names):
        
        #if len(drones) != len(names):
        #    print("[Drone.set_drone_names()] Error: too many or not enough drone names provided")
        #    return
        j = 0
        for i in range(len(drones)):
            if not drones[i].is_virtual:
                drones[i].name_in_motive = names[j]
                j += 1
    

    @staticmethod
    def get_drone_labels(drones):
        labels = []
        for d in drones:
            if not d.is_virtual:
                labels += [d.get_label()]
            
        return labels



class DroneHooked(Drone):

    def __init__(self, data: mujoco.MjData, name_in_xml, hook_name_in_xml, name_in_motive, is_virtual, trajectories, controller, parameters):
        super().__init__(data, name_in_xml, name_in_motive, is_virtual,
                         trajectories, controller, parameters)
        self.hook_name_in_xml = hook_name_in_xml

        

    #def get_qpos(self):
        #drone_qpos = self.data.joint(self.name_in_xml).qpos
        #return np.append(drone_qpos, self.data.joint(self.hook_name_in_xml).qpos)
    
    def get_hook_qpos(self):
        return self.data.joint(self.hook_name_in_xml).qpos[0]

    def get_hook_qvel(self):
        return self.data.joint(self.hook_name_in_xml).qvel[0]

    def print_names(self):
        super().print_names()
        print("hook name in xml: " + self.hook_name_in_xml)
    
    def get_label(self):
        """ extend this method later
        """
        return self.name_in_xml
    
