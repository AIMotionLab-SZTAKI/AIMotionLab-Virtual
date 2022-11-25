from pickle import FALSE
import mujoco
import numpy as np


class Drone:

    def __init__(self, data: mujoco.MjData, name_in_xml, name_in_motive, is_virtual, trajectories, controller, parameters):

        self.data = data
        self.is_virtual = is_virtual

        self.name_in_xml = name_in_xml
        self.name_in_motive = name_in_motive

        self.trajectories = trajectories
        self.controller = controller
        self.parameters = parameters

    def get_qpos(self):
        return self.data.joint(self.name_in_xml).qpos
    
    def set_qpos(self, position, orientation):
        """
        orientation should be quaternion
        """
        self.data.joint(self.name_in_xml).qpos = np.append(position, orientation)


    def print_names(self):
        print("name in xml:      " + self.name_in_xml)
        print("name in motive:   " + self.name_in_motive)
    
    def print_info(self):
        self.print_names()
        print("Is virtual:       " + str(self.is_virtual))
    
    def get_label(self):
        return self.name_in_xml

    @staticmethod
    def parse_drones(data: mujoco.MjData, joint_names: list[str]):
        """
        Create a list of Drone objects from mujoco's MjData following a naming convention
        found in naming_convention_in_xml.txt
        """

        drones = []
        i = 0

        for _name in joint_names:

            if _name.startswith("virtdrone_hooked") and not _name.endswith("hook"):
                # this joint must be a drone
                hook = Drone.find_hook_for_drone(joint_names, _name)
                if hook:
                    d = DroneHooked(data, name_in_xml=_name,
                                    hook_name_in_xml=hook,
                                    name_in_motive="cf" + str(i + 1),
                                    is_virtual=True,
                                    trajectories=None,
                                    controller=None,
                                    parameters=None)

                    drones += [d]
                    i += 1

                else:
                    print("Error: did not find hook joint for this drone: " +
                          _name + " ... Ignoring drone.")

            elif _name.startswith("virtdrone") and not _name.endswith("hook"):

                d = Drone(data, _name, "cf" + str(i + 1), True, None, None, None)
                drones += [d]
                i += 1
            
            elif _name.startswith("realdrone_hooked") and not _name.endswith("hook"):
                hook = Drone.find_hook_for_drone(joint_names, _name)
                if hook:
                    d = DroneHooked(data, name_in_xml=_name,
                                    hook_name_in_xml=hook,
                                    name_in_motive="cf" + str(i + 1),
                                    is_virtual=False,
                                    trajectories=None,
                                    controller=None,
                                    parameters=None)

                    drones += [d]
                    i += 1

                else:
                    print("Error: did not find hook joint for this drone: " +
                          _name + " ... Ignoring drone.")
            
            
            elif _name.startswith("realdrone") and not _name.endswith("hook"):

                d = Drone(data, _name, "cf" + str(i + 1), False, None, None, None)
                drones += [d]
                i += 1
                


        print()
        print(str(len(drones)) + " drones found in xml.")
        print()
        return drones

    @staticmethod
    def find_hook_for_drone(joint_names: list[str], drone_name):
        for jnt in joint_names:
            if drone_name + "_hook" == jnt:
                return jnt
        
        return None
    
    @staticmethod
    def get_drone_by_name_in_motive(drones, name: str):
        for d in drones:
            if d.name_in_motive == name:
                return d
        
        return None
    
    @staticmethod
    def get_drone_names_motive(drones):
        names = []
        for d in drones:
            names += [d.name_in_motive]
        
        return names
    
    @staticmethod
    def set_drone_names_motive(drones, names):
        
        if len(drones) != len(names):
            print("[Drone.set_drone_names()] Error: too many or not enough drone names provided")
            return
        
        for i in range(len(drones)):
            drones[i].name_in_motive = names[i]
    

    @staticmethod
    def get_drone_labels(drones):
        labels = []
        for d in drones:
            labels += [d.get_label()]
            
        return labels



class DroneHooked(Drone):

    def __init__(self, data: mujoco.MjData, name_in_xml, hook_name_in_xml, name_in_motive, is_virtual, trajectories, controller, parameters):
        super().__init__(data, name_in_xml, name_in_motive, is_virtual,
                         trajectories, controller, parameters)
        self.hook_name_in_xml = hook_name_in_xml

    def get_qpos(self):
        drone_qpos = self.data.joint(self.name_in_xml).qpos
        return np.append(drone_qpos, self.data.joint(self.hook_name_in_xml).qpos)

    def print_names(self):
        super().print_names()
        print("hook name in xml: " + self.hook_name_in_xml)
    
    def get_label(self):
        """ extend this method later
        """
        return self.name_in_xml
    
