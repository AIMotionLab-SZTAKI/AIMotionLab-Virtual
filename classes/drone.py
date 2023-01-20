from pickle import FALSE
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import util.mujoco_helper as mh
import math
from enum import Enum
from classes.moving_object import MovingObject

class SPIN_DIR(Enum):
    CLOCKWISE = 1
    COUNTER_CLOCKWISE = -1


class Drone(MovingObject):

    def __init__(self, data: mujoco.MjData, name_in_xml, trajectory, controllers, parameters = {"mass" : 0.1}):

        self.data = data

        self.name_in_xml = name_in_xml

        self.trajectory = trajectory
        self.controllers = controllers
        self.parameters = parameters
        self.mass = parameters["mass"]

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
        # needed to fake spinning propellers
        self.prop1_angle = self.prop1_qpos[0]
        self.prop2_angle = self.prop2_qpos[0]
        self.prop3_angle = self.prop3_qpos[0]
        self.prop4_angle = self.prop4_qpos[0]

        self.top_body_xquat = self.data.body(self.name_in_xml).xquat

        self.qvel = free_joint.qvel

        self.sensor_data = self.data.sensor(self.name_in_xml + "_sensor0").data

    
    def update(self, i):

        if self.trajectory is not None:

            pos = self.get_qpos()[:3]
            vel = self.get_qvel()[:3]

            alpha = None
            dalpha = None

            controller_input = self.trajectory.evaluate(i, self.data.time)

            ctrl = self.compute_control(controller_input)
            
            if ctrl is not None:
                self.set_ctrl(ctrl)
            #else:
            #    print("[Drone] Error: ctrl was None")
    

    
    def compute_control(self, input_dict):

        target_pos = input_dict["target_pos"]
        target_vel = input_dict["target_vel"]

        pos = self.get_qpos()[:3]
        quat = self.get_top_body_xquat()
        vel = self.get_qvel()[:3]
        ang_vel = self.get_sensor_data()


        if input_dict["controller_name"] == "geom_pos":

            target_rpy = input_dict["target_rpy"]
            ctrl = self.controllers["geom"].compute_pos_control(pos, quat, vel, ang_vel, target_pos,
                                                               target_vel=target_vel, target_rpy=target_rpy)
        
        elif input_dict["controller_name"] == "geom_att":
            target_quat = input_dict["target_quat"]
            target_acc = input_dict["target_acc"]
            target_quat_vel = input_dict["target_quat_vel"]
            ctrl = self.controllers["geom"].compute_att_control(pos, quat, vel, ang_vel, target_pos, target_vel, target_acc,
                                                                target_quat=target_quat, target_quat_vel=target_quat_vel)
        
        else:
            print("[Drone] Error: unknown controller")
            return None
        
        return ctrl

    
    def set_trajectory(self, trajectory):
        self.trajectory = trajectory
    
    def set_controllers(self, controllers):
        if isinstance(controllers, dict):
            self.controllers = controllers
            
        else:
            print("[Drone] Error: controllers must be a dictionary")
    
    def set_mass(self, mass):
        self.mass = mass
    
    def get_mass(self):
        return self.mass

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
    
      
    def fake_propeller_spin(self, control_step , speed = 10):

            if self.get_qpos()[2] > 0.10:
                self.spin_propellers(speed * control_step)
            else:
                self.stop_propellers()
    
    def stop_propellers(self):
        self.prop1_qpos[0] = self.prop1_angle
        self.prop2_qpos[0] = self.prop2_angle
        self.prop3_qpos[0] = self.prop3_angle
        self.prop4_qpos[0] = self.prop4_angle

    def print_names(self):
        print("name in xml:      " + self.name_in_xml)
    
    def print_info(self):
        print("Virtual")
        self.print_names()
    
    def get_name_in_xml(self):
        return self.name_in_xml

    @staticmethod
    def parse_drones(data, joint_names):
        """
        Create a list of Drone instances from mujoco's MjData following a naming convention
        found in naming_convention_in_xml.txt
        """

        virtdrones = []
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
                                    trajectory=None,
                                    controller=None)

                    virtdrones += [d]

                else:
                    print("Error: did not find hook joint for this drone: " +
                          _name + " ... Ignoring drone.")
            
            elif _name.startswith("virtbumblebee") and not _name.endswith("hook") and not _name_cut.endswith("prop"):
                # this joint must be a drone

                d = Drone(data, _name, None, None)
                virtdrones += [d]

            elif _name.startswith("virtcrazyflie") and not _name_cut.endswith("prop"):

                d = Drone(data, _name, None, None)
                virtdrones += [d]

        #print()
        #print(str(len(virtdrones)) + " virtual drone(s) found in xml.")
        #print()
        return virtdrones

    @staticmethod
    def find_hook_for_drone(names, drone_name):
        for n_ in names:
            if drone_name + "_hook" == n_:
                return n_
        
        return None

    


################################## DroneHooked ##################################
class DroneHooked(Drone):

    def __init__(self, data: mujoco.MjData, name_in_xml, hook_name_in_xml, trajectory, controller, parameters = {"mass" : 0.1}):
        super().__init__(data, name_in_xml, trajectory, controller, parameters)
        self.hook_name_in_xml = hook_name_in_xml

        self.hook_qpos = self.data.joint(self.hook_name_in_xml).qpos
        self.hook_qvel = self.data.joint(self.hook_name_in_xml).qvel

        self.load_mass = 0.0
        self.rod_length = 0.4

    #def get_qpos(self):
        #drone_qpos = self.data.joint(self.name_in_xml).qpos
        #return np.append(drone_qpos, self.data.joint(self.hook_name_in_xml).qpos)
    
    def update(self, i):

        if self.trajectory is not None:
            pos = self.get_qpos()[:3]
            vel = self.get_qvel()[:3]

            alpha = self.get_hook_qpos()
            dalpha = self.get_hook_qvel()

            controller_input = self.trajectory.evaluate(i, self.data.time)
            
            self.set_load_mass(controller_input["load_mass"])

            ctrl = self.compute_control(controller_input)
            
            if ctrl is not None:
                self.set_ctrl(ctrl)
            #else:
            #    print("[DroneHooked] Error: ctrl was None")
    
    def compute_control(self, input_dict):

        target_pos = input_dict["target_pos"]
        target_vel = input_dict["target_vel"]
        target_rpy = input_dict["target_rpy"]
        target_pos_load = input_dict["target_pos_load"]

        pos = self.get_qpos()[:3]
        quat = self.get_top_body_xquat()
        vel = self.get_qvel()[:3]
        ang_vel = self.get_sensor_data()


        if input_dict["controller_name"] == "geom_pos":

            ctrl = self.controllers["geom"].compute_pos_control(pos, quat, vel, ang_vel, target_pos,
                                                               target_vel=target_vel, target_rpy=target_rpy)

        elif input_dict["controller_name"] == "lqr":

            alpha = self.get_hook_qpos()
            dalpha = self.get_hook_qvel()

            pos_ = pos.copy()
            vel_ = vel.copy()
            R_plane = np.array([[np.cos(target_rpy[2]), -np.sin(target_rpy[2])],
                                [np.sin(target_rpy[2]), np.cos(target_rpy[2])]])
            pos_[0:2] = R_plane.T @ pos_[0:2]
            vel_[0:2] = R_plane.T @ vel_[0:2]
            hook_pos = pos_ + self.rod_length * np.array([-np.sin(alpha), 0, -np.cos(alpha)])
            hook_vel = vel_ + self.rod_length * dalpha * np.array([-np.cos(alpha), 0, np.sin(alpha)])
            hook_pos = np.take(hook_pos, [0, 2])
            hook_vel = np.take(hook_vel, [0, 2])

            phi_Q = Rotation.from_quat(np.roll(quat, -1)).as_euler('xyz')[1]
            dphi_Q = ang_vel[1]

            ctrl = self.controllers["geom"].compute_pos_control(pos, quat, vel, ang_vel, target_pos,
                                                               target_vel=target_vel, target_rpy=target_rpy)

            ctrl_lqr = self.controllers["lqr"].compute_control(hook_pos,
                                                                hook_vel,
                                                                alpha,
                                                                dalpha,
                                                                phi_Q,
                                                                dphi_Q,
                                                                target_pos_load)

            ctrl[0] = ctrl_lqr[0]
            ctrl[2] = ctrl_lqr[2]
        
        else:
            print("[DroneHooked] Error: unknown controller")
            return None
        
        return ctrl
    
    def get_hook_qpos(self):
        return self.hook_qpos[0]
    
    def set_hook_qpos(self, q):
        self.hook_qpos[0] = q

    def get_hook_qvel(self):
        return self.hook_qvel[0]

    def print_names(self):
        super().print_names()
        print("hook name in xml: " + self.hook_name_in_xml)
    
    def get_name_in_xml(self):
        """ extend this method later
        """
        return self.name_in_xml

    def set_load_mass(self, load_mass):
        self.load_mass = load_mass

        for k, v in self.controllers.items():
            self.controllers[k].mass = self.mass + self.load_mass

################################## DroneMocap ##################################
class DroneMocap:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, drone_mocapid, prop_mocapids, name_in_motive, name_in_xml):
        self.data = data
        self.name_in_xml = name_in_xml
        self.name_in_motive = name_in_motive
        self.mocapid = drone_mocapid
        self.prop_mocapids = prop_mocapids

        self.prop1 = PropellerMocap(model, data, name_in_xml + "_prop1", drone_mocapid, SPIN_DIR.COUNTER_CLOCKWISE)
        self.prop2 = PropellerMocap(model, data, name_in_xml + "_prop2", drone_mocapid, SPIN_DIR.COUNTER_CLOCKWISE)
        self.prop3 = PropellerMocap(model, data, name_in_xml + "_prop3", drone_mocapid, SPIN_DIR.CLOCKWISE)
        self.prop4 = PropellerMocap(model, data, name_in_xml + "_prop4", drone_mocapid, SPIN_DIR.CLOCKWISE)

        #self.prop1.print_data()
        #print()
        #self.prop2.print_data()
        #print()
        #self.prop3.print_data()
        #print()
        #self.prop4.print_data()
        #print()

    #def set_pos(self, pos):
    #    self.data.mocap_pos[self.mocapid] = pos


    #def set_quat(self, quat):
    #    self.data.mocap_quat[self.mocapid] = quat

        #for i in range(len(self.prop_mocapids)):
        #    self.data.mocap_quat[self.prop_mocapids[i]] = quat

    def get_pos(self):
        return self.data.mocap_pos[self.mocapid]

    def get_quat(self):
        return self.data.mocap_quat[self.mocapid]

    def get_qpos(self):
        return np.append(self.data.mocap_pos[self.mocapid], self.data.mocap_quat[self.mocapid])
    
    def set_qpos(self, pos, quat):
        """To match the simulated (non-mocap) Drone function names
           This drone does not have a qpos in MjData, because it's a mocap body
           and does not have any joints.
        """
        self.data.mocap_pos[self.mocapid] = pos
        self.data.mocap_quat[self.mocapid] = quat
        # gotta update the propellers too, otherwise they get left behind
        self.update_propellers()
    
    def get_name_in_xml(self):
        return self.name_in_xml
    
    def print_names(self):
        print("name in xml:      " + self.name_in_xml)
        print("name in motive:   " + self.name_in_motive)

    def print_info(self):
        print("Mocap")
        self.print_names()
    
    def spin_propellers(self, control_step, spin_speed):
        self.prop1.spin(control_step, spin_speed)
        self.prop2.spin(control_step, spin_speed)
        self.prop3.spin(control_step, spin_speed)
        self.prop4.spin(control_step, spin_speed)


    def update_propellers(self):
        self.prop1.update()
        self.prop2.update()
        self.prop3.update()
        self.prop4.update()


    @staticmethod
    def get_drone_names_motive(drones):
        names = []
        for d in drones:
            names += [d.name_in_motive]
        
        return names
    
    @staticmethod
    def set_drone_names_motive(drones, names):
        
        #if len(drones) != len(names):
        #    print("[Drone.set_drone_names()] Error: too many or not enough drone names provided")
        #    return
        #j = 0
        for i in range(len(drones)):
            drones[i].name_in_motive = names[i]
            #j += 1
    

    @staticmethod
    def get_drone_names_in_xml(drones):
        labels = []
        for d in drones:
            labels += [d.get_name_in_xml()]
            
        return labels
    

    @staticmethod
    def get_drone_by_name_in_motive(drones, name: str):
        for i in range(len(drones)):
            if drones[i].name_in_motive == name:
                return drones[i]
        
        return None

    @staticmethod
    def parse_mocap_drones(data, model, body_names):

        realdrones = []
        icf = 0
        ibb = 0
        for _name in body_names:

            _name_cut = _name[:len(_name) - 1]

            if _name.startswith("realbumblebee_hooked") and not _name.endswith("hook") and not _name_cut.endswith("prop"):
                hook = DroneMocap.find_mocap_hook_for_drone(body_names, _name)
                if hook:

                    prop_mocapids = []
                    drone_mocapid = model.body(_name).mocapid[0]
                    prop_mocapids += [model.body(_name + "_prop1").mocapid[0]]
                    prop_mocapids += [model.body(_name + "_prop2").mocapid[0]]
                    prop_mocapids += [model.body(_name + "_prop3").mocapid[0]]
                    prop_mocapids += [model.body(_name + "_prop4").mocapid[0]]

                    d = DroneMocapHooked(model, data, drone_mocapid, prop_mocapids, "bb" + str(ibb + 1), _name, hook)

                    realdrones += [d]
                    ibb += 1

                else:
                    print("Error: did not find hook body for this drone: " +
                          _name + " ... Ignoring drone.")

            elif _name.startswith("realbumblebee") and not _name.endswith("hook") and not _name_cut.endswith("prop"):

                prop_mocapids = []
                drone_mocapid = model.body(_name).mocapid[0]
                prop_mocapids += [model.body(_name + "_prop1").mocapid[0]]
                prop_mocapids += [model.body(_name + "_prop2").mocapid[0]]
                prop_mocapids += [model.body(_name + "_prop3").mocapid[0]]
                prop_mocapids += [model.body(_name + "_prop4").mocapid[0]]

                d = DroneMocap(model, data, drone_mocapid, prop_mocapids, "bb" + str(ibb + 1), _name)
                realdrones += [d]
                ibb += 1

            elif _name.startswith("realcrazyflie") and not _name_cut.endswith("prop"):

                prop_mocapids = []
                drone_mocapid = model.body(_name).mocapid[0]
                prop_mocapids += [model.body(_name + "_prop1").mocapid[0]]
                prop_mocapids += [model.body(_name + "_prop2").mocapid[0]]
                prop_mocapids += [model.body(_name + "_prop3").mocapid[0]]
                prop_mocapids += [model.body(_name + "_prop4").mocapid[0]]

                d = DroneMocap(model, data, drone_mocapid, prop_mocapids, "cf" + str(icf + 1), _name)
                realdrones += [d]
                icf += 1


        #print()
        #print(str(len(realdrones)) + " mocap drone(s) found in xml.")
        #print()
        return realdrones
    

    @staticmethod
    def find_mocap_hook_for_drone(names, drone_name):
        for n_ in names:
            if drone_name + "_hook" == n_:
                return n_
        
        return None


################################## DroneMocapHooked ##################################
class DroneMocapHooked(DroneMocap):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, drone_mocapid, prop_mocapids, name_in_motive, name_in_xml, hook_name_in_xml):
        super().__init__(model, data, drone_mocapid, prop_mocapids, name_in_motive, name_in_xml)

        self.hook_name_in_xml = hook_name_in_xml

        self.hook_mocapid = model.body(name_in_xml + "_hook").mocapid[0]

        self.hook_position = data.mocap_pos[self.hook_mocapid]
        self.hook_rotation = data.mocap_quat[self.hook_mocapid]
    
    def get_hook_pos(self):
        return self.hook_position
    
    def __set_hook_pos(self, pos):
        self.hook_position[0] = pos[0]
        self.hook_position[1] = pos[1]
        self.hook_position[2] = pos[2]
    
    def get_hook_quat(self):
        return self.hook_rotation
    
    def __set_hook_quat(self, quat):
        self.hook_rotation[0] = quat[0]
        self.hook_rotation[1] = quat[1]
        self.hook_rotation[2] = quat[2]
        self.hook_rotation[3] = quat[3]

    def set_qpos(self, pos, quat):
        self.__set_hook_pos(pos)
        self.__set_hook_quat(quat)
        return super().set_qpos(pos, quat)


    def print_names(self):
        super().print_names()
        print("hook name in xml: " + self.hook_name_in_xml)



################################## PropellerMocap ##################################
class PropellerMocap():
    def __init__(self, model, data, name_in_xml, drone_mocap_id, spin_direction = SPIN_DIR.CLOCKWISE):

        self.set_spin_direction(spin_direction)
        
        self.name_in_xml = name_in_xml
        self.mocapid = model.body(name_in_xml).mocapid[0]

        #print(model.geom(name_in_xml))
        
        self.position = data.mocap_pos[self.mocapid]
        self.rotation = data.mocap_quat[self.mocapid]
        self.OFFSET = model.geom(name_in_xml).pos

        self.drone_pos = data.mocap_pos[drone_mocap_id]
        self.drone_rot = data.mocap_quat[drone_mocap_id]

        self.spin_angle = 0.0

        self.spinned = True


    def print_data(self):
        print("name in xml ", self.name_in_xml)
        print("mocapid     ", self.mocapid)
        print("position    ", self.position)
        print("rotation    ", self.rotation)
        print("offset      ", self.OFFSET)
        print("drone pos   ", self.drone_pos)
        print("drone rot   ", self.drone_rot)
    
    def set_spin_direction(self, spin_dir: SPIN_DIR):
        self.__spin_direction = spin_dir
    
    def get_spin_direction(self):
        return self.__spin_direction
    
    def update(self):
        #if self.spinned:
        #    self.spin_angle += (spin_speed * control_step * self.__spin_direction.value)


        # combine the orientation and the spin quaternion
        quat = mh.quaternion_from_euler(0, 0, self.spin_angle)
        quat = mh.quaternion_multiply(quat, self.drone_rot)

        # set new rotation
        self.rotation[0] = quat[0]
        self.rotation[1] = quat[1]
        self.rotation[2] = quat[2]
        self.rotation[3] = quat[3]
        
        # compensate for the shift caused by the spin
        # as the origin of the propeller coordinate frame is the same as the drone's
        new_offs = mh.qv_mult(quat, self.OFFSET)

        o = mh.qv_mult(self.drone_rot, self.OFFSET)

        self.position[0] = self.drone_pos[0] - new_offs[0] + o[0]
        self.position[1] = self.drone_pos[1] - new_offs[1] + o[1]
        self.position[2] = self.drone_pos[2] - new_offs[2] + o[2]
    
    def spin(self, control_step, spin_speed):
        if self.spinned:
            self.spin_angle += (spin_speed * control_step * self.__spin_direction.value)
        
        self.update()
