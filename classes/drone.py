import math
from pickle import FALSE
from re import M
import mujoco
import numpy as np
from enum import Enum
from classes.moving_object import MovingObject, MovingMocapObject
from util import mujoco_helper
from scipy.spatial.transform import Rotation

class SPIN_DIR(Enum):
    CLOCKWISE = 1
    COUNTER_CLOCKWISE = -1

class CRAZYFLIE_PROP(Enum):
    OFFSET = "0.047"
    OFFSET_Z = "0.032"
    MOTOR_PARAM = "0.02514"
    MAX_THRUST = "0.16"

class BUMBLEBEE_PROP(Enum):
    OFFSET_X1 = "0.074"
    OFFSET_X2 = "0.091"
    OFFSET_Y = "0.087"
    OFFSET_Z = "0.036"
    MOTOR_PARAM = "0.5954"
    MAX_THRUST = "15"

class DRONE_TYPES(Enum):
    CRAZYFLIE = 0
    BUMBLEBEE = 1


class Drone(MovingObject):

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name_in_xml):

        super().__init__(model, name_in_xml)

        self.data = data

        if "crazyflie" in name_in_xml:
            self.type = DRONE_TYPES.CRAZYFLIE
        elif "bumblebee" in name_in_xml:
            self.type = DRONE_TYPES.BUMBLEBEE

        free_joint = self.data.joint(self.name_in_xml)

        self.qpos = free_joint.qpos

        self.prop1_qpos = self.data.joint(self.name_in_xml + "_prop1").qpos
        self.prop2_qpos = self.data.joint(self.name_in_xml + "_prop2").qpos
        self.prop3_qpos = self.data.joint(self.name_in_xml + "_prop3").qpos
        self.prop4_qpos = self.data.joint(self.name_in_xml + "_prop4").qpos
        
        self.prop1_joint = self.data.joint(self.name_in_xml + "_prop1")
        self.prop2_joint = self.data.joint(self.name_in_xml + "_prop2")
        self.prop3_joint = self.data.joint(self.name_in_xml + "_prop3")
        self.prop4_joint = self.data.joint(self.name_in_xml + "_prop4")

        self.ctrl0 = self.data.actuator(self.name_in_xml + "_actr1").ctrl
        self.ctrl1 = self.data.actuator(self.name_in_xml + "_actr2").ctrl
        self.ctrl2 = self.data.actuator(self.name_in_xml + "_actr3").ctrl
        self.ctrl3 = self.data.actuator(self.name_in_xml + "_actr4").ctrl

        # make a copy now, so that mj_step() does not affect these
        # needed to fake spinning propellers
        self.prop1_angle = self.prop1_qpos[0]
        self.prop2_angle = self.prop2_qpos[0]
        self.prop3_angle = self.prop3_qpos[0]
        self.prop4_angle = self.prop4_qpos[0]

        self.top_body_xquat = self.data.body(self.name_in_xml).xquat

        self.prop1_joint_pos = self.model.joint(self.name_in_xml + "_prop1").pos

        self.qvel = free_joint.qvel
        self.qacc = free_joint.qacc
        self.qfrc_passive = free_joint.qfrc_passive
        self.qfrc_applied = free_joint.qfrc_applied

        self.sensor_gyro = self.data.sensor(self.name_in_xml + "_gyro").data
        self.sensor_velocimeter = self.data.sensor(self.name_in_xml + "_velocimeter").data
        self.sensor_accelerometer = self.data.sensor(self.name_in_xml + "_accelerometer").data
        self.sensor_posimeter = self.data.sensor(self.name_in_xml + "_posimeter").data
        self.sensor_orimeter = self.data.sensor(self.name_in_xml + "_orimeter").data
        self.sensor_ang_accelerometer = self.data.sensor(self.name_in_xml + "_ang_accelerometer").data

        self.state = {
            "pos" : self.sensor_posimeter,
            "vel" : self.sensor_velocimeter,
            "acc" : self.sensor_accelerometer,
            "quat" : self.sensor_orimeter,
            "ang_vel" : self.sensor_gyro,
            "ang_acc" : self.sensor_ang_accelerometer
        }

        if self.type == DRONE_TYPES.CRAZYFLIE:
            self.Lx1 = float(CRAZYFLIE_PROP.OFFSET.value)
            self.Lx2 = float(CRAZYFLIE_PROP.OFFSET.value)
            self.Ly = float(CRAZYFLIE_PROP.OFFSET.value)
            self.motor_param = float(CRAZYFLIE_PROP.MOTOR_PARAM.value)
        
        elif self.type == DRONE_TYPES.BUMBLEBEE:
            self.Lx1 = float(BUMBLEBEE_PROP.OFFSET_X1.value)
            self.Lx2 = float(BUMBLEBEE_PROP.OFFSET_X2.value)
            self.Ly = float(BUMBLEBEE_PROP.OFFSET_Y.value)
            self.motor_param = float(BUMBLEBEE_PROP.MOTOR_PARAM.value)


        self.input_mtx = np.array([[1/4, -1/(4*self.Ly), -1/(4*self.Lx2),  1 / (4*self.motor_param)],
                                  [1/4, -1/(4*self.Ly),   1/(4*self.Lx1), -1 / (4*self.motor_param)],
                                  [1/4,  1/(4*self.Ly),   1/(4*self.Lx1),  1 / (4*self.motor_param)],
                                  [1/4,  1/(4*self.Ly),  -1/(4*self.Lx2), -1 / (4*self.motor_param)]])
    
    def get_state(self):

        return self.state

    
    def update(self, i, control_step):

        #self.fake_propeller_spin(0.02)
        self.prop1_joint.qvel = -100 * math.pi
        self.prop2_joint.qvel = -100 * math.pi
        self.prop3_joint.qvel =  100 * math.pi
        self.prop4_joint.qvel =  100 * math.pi


        if self.trajectory is not None:

            state = self.get_state()
            setpoint = self.trajectory.evaluate(state, i, self.data.time, control_step)

            self.update_controller_type(state, setpoint, self.data.time, i)

            if self.controller is not None:
                ctrl = self.controller.compute_control(state, setpoint, self.data.time)
            
            if ctrl is not None:
                motor_thrusts = self.input_mtx @ ctrl
                self.set_ctrl(motor_thrusts)
            #else:
            #    print("[Drone] Error: ctrl was None")
    

    def set_trajectory(self, trajectory):
        self.trajectory = trajectory
    
    def set_controllers(self, controllers):
        self.controllers = controllers
    
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

    def get_sensor_gyro(self):
        return self.sensor_gyro

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
    def find_hook_for_drone(names, drone_name):
        hook_names = []
        for n_ in names:
            if drone_name + "_hook_x" == n_:
                hook_names += [n_]
            elif drone_name + "_hook_y" == n_:
                hook_names += [n_]
        
        return hook_names

    


################################## DroneHooked ##################################
class DroneHooked(Drone):

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name_in_xml):
        super().__init__(model, data, name_in_xml)
        
        joint_names = mujoco_helper.get_joint_name_list(model)

        hook_joint_names_in_xml = Drone.find_hook_for_drone(joint_names, name_in_xml)

        self.hook_dof = len(hook_joint_names_in_xml)

        self.hook_qpos_y = self.data.joint(self.name_in_xml + "_hook_y").qpos
        self.hook_qvel_y = self.data.joint(self.name_in_xml + "_hook_y").qvel
        self.sensor_hook_jointpos_y = self.data.sensor(self.name_in_xml + "_hook_jointpos_y").data
        self.sensor_hook_jointvel_y = self.data.sensor(self.name_in_xml + "_hook_jointvel_y").data
        self.state["joint_ang"] = np.array(self.sensor_hook_jointpos_y)
        self.state["joint_ang_vel"] = np.array(self.sensor_hook_jointvel_y)
        if self.hook_dof == 2:
            self.hook_qpos_x = self.data.joint(self.name_in_xml + "_hook_x").qpos
            self.hook_qvel_x = self.data.joint(self.name_in_xml + "_hook_x").qvel
            self.sensor_hook_jointvel_x = self.data.sensor(self.name_in_xml + "_hook_jointvel_x").data
            self.sensor_hook_jointpos_x = self.data.sensor(self.name_in_xml + "_hook_jointpos_x").data
            self.sensor_hook_pos = self.data.sensor(self.name_in_xml + "_hook_pos").data
            self.sensor_hook_vel = self.data.sensor(self.name_in_xml + "_hook_vel").data
            self.sensor_hook_quat = self.data.sensor(self.name_in_xml + "_hook_quat").data
            self.sensor_hook_ang_vel = self.data.sensor(self.name_in_xml + "_hook_angvel").data
            self.state["joint_ang"] = np.array((self.sensor_hook_jointpos_y, self.sensor_hook_jointpos_x))
            self.state["joint_ang_vel"] = np.array((self.sensor_hook_jointvel_y, self.sensor_hook_jointvel_x))
            self.state["load_pos"] = np.array(self.sensor_hook_pos)
            self.state["load_vel"] = np.array(self.sensor_hook_vel)
            # if np.linalg.norm(self.sensor_hook_quat) < 1e-4:
            #     self.sensor_hook_quat = np.array([0, 0, 0, 1])
            self.state["pole_eul"] = np.zeros(2) #Rotation.from_quat(np.roll(np.array(self.sensor_hook_quat), -1)).as_euler('XYZ')[0:2]
            self.state["pole_ang_vel"] = np.array(self.sensor_hook_ang_vel)[0:2]

        self.load_mass = 0.0
        self.rod_length = 0.4

    
    def get_state(self):
        #super().get_state()
        if self.hook_dof == 1:
            self.state["joint_ang"] = np.array(self.sensor_hook_jointpos_y[0])
            self.state["joint_ang_vel"] = np.array(self.sensor_hook_jointvel_y[0])
        elif self.hook_dof == 2:
            self.state["joint_ang"] = np.array((self.sensor_hook_jointpos_y[0], self.sensor_hook_jointpos_x[0]))
            self.state["joint_ang_vel"] = np.array((self.sensor_hook_jointvel_y[0], self.sensor_hook_jointvel_x[0]))
            self.state["load_pos"] = np.array(self.sensor_hook_pos)
            self.state["load_vel"] = np.array(self.sensor_hook_vel)
            self.state["pole_eul"] = Rotation.from_quat(np.roll(np.array(self.sensor_hook_quat), -1)).as_euler('XYZ')[0:2]
            self.state["pole_ang_vel"] = np.array(self.sensor_hook_ang_vel)[0:2]
        return self.state
    
    def update(self, i, control_step):
        self.fake_propeller_spin(0.02)

        if self.trajectory is not None:

            state = self.get_state()
            setpoint = self.trajectory.evaluate(state, i, self.data.time, control_step)
            
            self.update_controller_type(state, setpoint, self.data.time, i)

            if self.controller is not None:
                ctrl = self.controller.compute_control(state, setpoint, self.data.time)
            
            if ctrl is not None:
                
                motor_thrusts = self.input_mtx @ ctrl
                self.set_ctrl(motor_thrusts)
            #else:
            #    print("[DroneHooked] Error: ctrl was None")
    
    
    def get_hook_qpos(self):
        if self.hook_dof == 1:
            return self.hook_qpos_y[0]
        elif self.hook_dof == 2:
            return [self.hook_qpos_x[0], self.hook_qpos_y[0]]
        
    
    def set_hook_qpos(self, q):
        if self.hook_dof == 1:
            self.hook_qpos_x[0] = q
        else:
            self.hook_qpos_x[0] = q[0]
            self.hook_qpos_y[0] = q[1]

    def get_hook_qvel(self):
        if self.hook_dof == 1:
            return self.hook_qvel_y[0]
        elif self.hook_dof == 2:
            return [self.hook_qvel_x[0], self.hook_qvel_y[0]]

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
class DroneMocap(MovingMocapObject):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, drone_mocapid, name_in_xml, name_in_motive):

        super().__init__(name_in_xml, name_in_motive)
        self.data = data
        self.name_in_xml = name_in_xml
        self.name_in_motive = name_in_motive
        self.mocapid = drone_mocapid
        #print(name_in_xml)

        self.prop1_jnt = data.joint(name_in_xml + "_prop1")
        self.prop2_jnt = data.joint(name_in_xml + "_prop2")
        self.prop3_jnt = data.joint(name_in_xml + "_prop3")
        self.prop4_jnt = data.joint(name_in_xml + "_prop4")

        self.set_propeller_speed(21.6)
        

    def get_pos(self):
        return self.data.mocap_pos[self.mocapid]

    def get_quat(self):
        return self.data.mocap_quat[self.mocapid]

    def get_qpos(self):
        return np.append(self.data.mocap_pos[self.mocapid], self.data.mocap_quat[self.mocapid])
    
    def set_propeller_speed(self, speed):
        self.prop1_jnt.qvel[0] = -speed
        self.prop2_jnt.qvel[0] = -speed
        self.prop3_jnt.qvel[0] = speed
        self.prop4_jnt.qvel[0] = speed

    
    def update(self, pos, quat):
        self.set_propeller_speed(21.6)
        self.data.mocap_pos[self.mocapid] = pos
        self.data.mocap_quat[self.mocapid] = quat
    
    def get_name_in_xml(self):
        return self.name_in_xml
    
    def print_names(self):
        print("name in xml:      " + self.name_in_xml)
        print("name in motive:   " + self.name_in_motive)

    def print_info(self):
        print("Mocap")
        self.print_names()

    

    @staticmethod
    def find_mocap_hook_for_drone(names, drone_name):
        for n_ in names:
            if drone_name + "_hook" == n_:
                return n_
        
        return None


################################## DroneMocapHooked ##################################
class DroneMocapHooked(DroneMocap):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, drone_mocapid, name_in_xml, name_in_motive):
        super().__init__(model, data, drone_mocapid, name_in_xml, name_in_motive)

        self.hook_name_in_xml = "HookMocap_" + name_in_xml

    def update(self, pos, quat):
        return super().update(pos, quat)


    def print_names(self):
        super().print_names()
        print("hook name in xml: " + self.hook_name_in_xml)


class HookMocap(MovingMocapObject):

    def __init__(self, model, data, hook_mocapid, name_in_xml, name_in_motive) -> None:
        super().__init__(name_in_xml, name_in_motive)
        self.data = data
        self.name_in_xml = name_in_xml
        self.name_in_motive = name_in_motive
        self.mocapid = hook_mocapid
    

    def update(self, pos, quat):
        pos_ = pos.copy()
        pos_[2] = pos_[2] + .03
        self.data.mocap_pos[self.mocapid] = pos_
        self.data.mocap_quat[self.mocapid] = quat
        
    def get_qpos(self):
        return np.append(self.data.mocap_pos[self.mocapid], self.data.mocap_quat[self.mocapid])
