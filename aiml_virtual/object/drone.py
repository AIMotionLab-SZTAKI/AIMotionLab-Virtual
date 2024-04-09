import math
import mujoco
import numpy as np
from enum import Enum
from aiml_virtual.object.moving_object import MovingObject, MocapObject
from aiml_virtual.util import mujoco_helper
from scipy.spatial.transform import Rotation

class SPIN_DIR(Enum):
    CLOCKWISE = 1
    COUNTER_CLOCKWISE = -1

class CRAZYFLIE_PROP(Enum):
    #OFFSET = "0.047"
    OFFSET = "0.03275"
    OFFSET_Z = "0.0223"
    MOTOR_PARAM = "0.02514"
    MAX_THRUST = "0.16"
    MASS = "0.028"
    DIAGINERTIA = "1.4e-5 1.4e-5 2.17e-5"
    COG = "0.0 0.0 0.0"

class BUMBLEBEE_PROP(Enum):
    OFFSET_X1 = "0.074"
    OFFSET_X2 = "0.091"
    OFFSET_Y = "0.087"
    OFFSET_Z = "0.036"
    MOTOR_PARAM = "0.5954"
    MAX_THRUST = "15"
    ROD_LENGTH = ".4"
    HOOK_ROTATION_OS = ".01"
    MASS = "0.605"
    DIAGINERTIA = "1.5e-3 1.45e-3 2.66e-3"
    COG = "0.0085 0.0 0.0"

class DRONE_TYPES(Enum):
    CRAZYFLIE = 0
    BUMBLEBEE = 1
    BUMBLEBEE_HOOKED = 2


class Drone(MovingObject):

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name_in_xml):

        super().__init__(model, name_in_xml)

        self.data = data

        free_joint = self.data.joint(self.name_in_xml)
        self.xquat = self.data.body(self.name_in_xml).xquat

        self.qpos = free_joint.qpos
        self.mass = model.body(self.name_in_xml).mass

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

        self.sphere_geom = self.model.geom(self.name_in_xml + "_sphere")
        self.initial_sphere_color = self.sphere_geom.rgba[:3].copy()
        try:
            self.safety_sphere_mocapid = self.model.body(self.name_in_xml + "_safety_sphere").mocapid[0]
            self.safety_sphere_geom = self.model.geom(self.name_in_xml + "_safety_sphere")
            self.initial_safety_sphere_color = self.safety_sphere_geom.rgba[:3].copy()
        except:
            #print("[Drone] No safety sphere found.")
            pass

        self.state = {
            "pos" : self.sensor_posimeter,
            "vel" : self.sensor_velocimeter,
            "acc" : self.sensor_accelerometer,
            "quat" : self.sensor_orimeter,
            "ang_vel" : self.sensor_gyro,
            "ang_acc" : self.sensor_ang_accelerometer
        }
        
        self.ctrl_input = np.zeros(4)
        
    
    def _create_input_matrix(self, Lx1, Lx2, Ly, motor_param):

        self.input_mtx = np.array([[1/4, -1/(4*Ly), -1/(4*Lx2),  1 / (4*motor_param)],
                                  [1/4, -1/(4*Ly),   1/(4*Lx1), -1 / (4*motor_param)],
                                  [1/4,  1/(4*Ly),   1/(4*Lx1),  1 / (4*motor_param)],
                                  [1/4,  1/(4*Ly),  -1/(4*Lx2), -1 / (4*motor_param)]])

    
    def get_state(self):

        return self.state
    
    def get_state_copy(self):
        state = {
            "pos" : self.sensor_posimeter.copy(),
            "vel" : self.sensor_velocimeter.copy(),
            "acc" : self.sensor_accelerometer.copy(),
            "quat" : self.sensor_orimeter.copy(),
            "ang_vel" : self.sensor_gyro.copy(),
            "ang_acc" : self.sensor_ang_accelerometer.copy()
        }
        return state

    
    def update(self, i, control_step):

        self.spin_propellers()


        if self.trajectory is not None:

            state = self.get_state()
            setpoint = self.trajectory.evaluate(state, i, self.data.time, control_step)

            self.update_controller_type(state, setpoint, self.data.time, i)

            if self.controller is not None:
                ctrl = self.controller.compute_control(state, setpoint, self.data.time)
                self.ctrl_input = ctrl
            
                if ctrl is not None:
                    motor_thrusts = self.input_mtx @ ctrl
                    self.set_ctrl(motor_thrusts)
            #else:
            #    print("[Drone] Error: ctrl was None")
    
    
    def get_mass(self):
        return self.mass

    def get_qpos(self):
        return self.qpos
    
    def set_qpos(self, position, orientation=np.array((1.0, 0.0, 0.0, 0.0))):
        """
        orientation should be quaternion
        """
        self.qpos[:7] = np.append(position, orientation)
    
    def get_motor_thrusts(self):
        return np.concatenate((self.ctrl0, self.ctrl1, self.ctrl2, self.ctrl3))
    
    def get_ctrl_input(self):
        return self.ctrl_input
    
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
    
    def spin_propellers(self):
        #print("angle step: " + str(angle_step))
                
        self.prop1_angle += self.ctrl0[0]
        self.prop2_angle -= self.ctrl1[0]
        self.prop3_angle += self.ctrl2[0]
        self.prop4_angle -= self.ctrl3[0]

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
    
    def print_info(self):
        print("Virtual")
        self.print_names()
    
    def get_name_in_xml(self):
        return self.name_in_xml

    def set_sphere_color(self, new_rgb):
        self.sphere_geom.rgba[0] = new_rgb[0]
        self.sphere_geom.rgba[1] = new_rgb[1]
        self.sphere_geom.rgba[2] = new_rgb[2]
    
    def reset_sphere_color(self):
        self.sphere_geom.rgba[:3] = self.initial_sphere_color

    def set_sphere_alpha(self, new_alpha):
        self.sphere_geom.rgba[3] = new_alpha
    
    def toggle_sphere_alpha(self):
        if self.sphere_geom.rgba[3] > 0.01:
            self.sphere_geom.rgba[3] = 0.0
        else:
            self.sphere_geom.rgba[3] = 0.3

    def get_sphere_size(self):

        return self.sphere_geom.size

    def set_sphere_size(self, new_size):

        s = max(0.25, new_size)
        
        self.sphere_geom.size = s
    
    def scale_sphere(self, simulator):
        
        drone0_pos = self.state["pos"]

        elev_rad = math.radians(simulator.activeCam.elevation)
        azim_rad = math.radians(simulator.activeCam.azimuth)

        a = simulator.activeCam.distance * math.cos(elev_rad)
        dz = simulator.activeCam.distance * math.sin(elev_rad)

        dx = a * math.cos(azim_rad)
        dy = a * math.sin(azim_rad)

        c = simulator.activeCam.lookat - np.array((dx, dy, dz))

        d_cs = math.sqrt((c[0] - drone0_pos[0])**2 + (c[1] - drone0_pos[1])**2 + (c[2] - drone0_pos[2])**2)

        self.set_sphere_size(d_cs / 100.0)
    
    def set_safety_sphere_pos(self, new_pos):

        self.data.mocap_pos[self.safety_sphere_mocapid] = new_pos

    def set_safety_sphere_color(self, rgb):
        self.safety_sphere_geom.rgba[:3] = rgb

    def reset_safety_sphere_color(self):
        self.safety_sphere_geom.rgba[:3] = self.initial_safety_sphere_color

    def toggle_safety_sphere_alpha(self):
        if self.safety_sphere_geom.rgba[3] > 0.01:
            self.safety_sphere_geom.rgba[3] = 0.0
        else:
            self.safety_sphere_geom.rgba[3] = 0.2


    @staticmethod
    def find_hook_for_drone(names, drone_name):
        hook_names = []
        for n_ in names:
            if drone_name + "_hook_x" == n_:
                hook_names += [n_]
            elif drone_name + "_hook_y" == n_:
                hook_names += [n_]
        
        return hook_names


class Crazyflie(Drone):

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name_in_xml):
        super().__init__(model, data, name_in_xml)

        self.Lx1 = float(CRAZYFLIE_PROP.OFFSET.value)
        self.Lx2 = float(CRAZYFLIE_PROP.OFFSET.value)
        self.Ly = float(CRAZYFLIE_PROP.OFFSET.value)
        self.motor_param = float(CRAZYFLIE_PROP.MOTOR_PARAM.value)

        self._create_input_matrix(self.Lx1, self.Lx2, self.Ly, self.motor_param)

            
    def spin_propellers(self):
        #print("angle step: " + str(angle_step))

        if self.sensor_posimeter[2] > 0.015:
            
            self.prop1_angle += self.ctrl0[0] * 100
            self.prop2_angle -= self.ctrl1[0] * 100
            self.prop3_angle += self.ctrl2[0] * 100
            self.prop4_angle -= self.ctrl3[0] * 100

        self.prop1_qpos[0] = self.prop1_angle
        self.prop2_qpos[0] = self.prop2_angle
        self.prop3_qpos[0] = self.prop3_angle
        self.prop4_qpos[0] = self.prop4_angle



class Bumblebee(Drone):

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name_in_xml):
        super().__init__(model, data, name_in_xml)
        self.Lx1 = float(BUMBLEBEE_PROP.OFFSET_X1.value)
        self.Lx2 = float(BUMBLEBEE_PROP.OFFSET_X2.value)
        self.Ly = float(BUMBLEBEE_PROP.OFFSET_Y.value)
        self.motor_param = float(BUMBLEBEE_PROP.MOTOR_PARAM.value)

        self._create_input_matrix(self.Lx1, self.Lx2, self.Ly, self.motor_param)

    def spin_propellers(self):
        #print("angle step: " + str(angle_step))

        if self.sensor_posimeter[2] > 0.1:
            
            self.prop1_angle += self.ctrl0[0] * 100
            self.prop2_angle -= self.ctrl1[0] * 100
            self.prop3_angle += self.ctrl2[0] * 100
            self.prop4_angle -= self.ctrl3[0] * 100

        self.prop1_qpos[0] = self.prop1_angle
        self.prop2_qpos[0] = self.prop2_angle
        self.prop3_qpos[0] = self.prop3_angle
        self.prop4_qpos[0] = self.prop4_angle





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
        self.sensor_hook_pos = self.data.sensor(self.name_in_xml + "_hook_pos").data
        self.sensor_hook_vel = self.data.sensor(self.name_in_xml + "_hook_vel").data
        self.sensor_hook_ang_vel = self.data.sensor(self.name_in_xml + "_hook_angvel").data
        self.sensor_hook_quat = self.data.sensor(self.name_in_xml + "_hook_quat").data
        self.state["joint_ang"] = np.array(self.sensor_hook_jointpos_y)
        self.state["joint_ang_vel"] = np.array(self.sensor_hook_jointvel_y)
        self.state["load_pos"] = np.array(self.sensor_hook_pos)
        self.state["load_vel"] = np.array(self.sensor_hook_vel)
        self.state["pole_eul"] = np.zeros(2) #Rotation.from_quat(np.roll(np.array(self.sensor_hook_quat), -1)).as_euler('XYZ')[0:2]
        self.state["pole_ang_vel"] = np.array(self.sensor_hook_ang_vel)[0:2]

        self._rod_length = model.geom(name_in_xml + "_rod").size[1] * 2

        if self.hook_dof == 2:
            self.hook_qpos_x = self.data.joint(self.name_in_xml + "_hook_x").qpos
            self.hook_qvel_x = self.data.joint(self.name_in_xml + "_hook_x").qvel
            self.sensor_hook_jointvel_x = self.data.sensor(self.name_in_xml + "_hook_jointvel_x").data
            self.sensor_hook_jointpos_x = self.data.sensor(self.name_in_xml + "_hook_jointpos_x").data
            self.state["joint_ang"] = np.array((self.sensor_hook_jointpos_y, self.sensor_hook_jointpos_x))
            self.state["joint_ang_vel"] = np.array((self.sensor_hook_jointvel_y, self.sensor_hook_jointvel_x))
            # if np.linalg.norm(self.sensor_hook_quat) < 1e-4:
            #     self.sensor_hook_quat = np.array([0, 0, 0, 1])

    
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
        self.state["pole_eul"] = Rotation.from_quat(np.roll(np.array(self.sensor_hook_quat), -1)).as_euler('xyz')[0:2]
        self.state["pole_ang_vel"] = np.array(self.sensor_hook_ang_vel)[0:2]
        return self.state
    
    def update(self, i, control_step):
        self.spin_propellers()

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

    def get_rod_length(self):
        return self._rod_length

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



class BumblebeeHooked(DroneHooked):

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name_in_xml):
        super().__init__(model, data, name_in_xml)

        self.Lx1 = float(BUMBLEBEE_PROP.OFFSET_X1.value)
        self.Lx2 = float(BUMBLEBEE_PROP.OFFSET_X2.value)
        self.Ly = float(BUMBLEBEE_PROP.OFFSET_Y.value)
        self.motor_param = float(BUMBLEBEE_PROP.MOTOR_PARAM.value)

        rod_geom = model.geom(name_in_xml + "_rod")

        self.rod_length = rod_geom.size[1] * 2

        #print(self.rod_length)
        
        self._create_input_matrix(self.Lx1, self.Lx2, self.Ly, self.motor_param)




################################## DroneMocap ##################################
class DroneMocap(MocapObject):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, mocapid, name_in_xml, name_in_motive):

        super().__init__(model, data, mocapid, name_in_xml, name_in_motive)

        self.prop1_jnt = data.joint(name_in_xml + "_prop1")
        self.prop2_jnt = data.joint(name_in_xml + "_prop2")
        self.prop3_jnt = data.joint(name_in_xml + "_prop3")
        self.prop4_jnt = data.joint(name_in_xml + "_prop4")

        if "bumblebee" in name_in_xml:
            self.propeller_spin_threshold = 0.15
        elif "crazyflie" in name_in_xml:
            self.propeller_spin_threshold = 0.03
        else:
            self.propeller_spin_threshold = 0.1

        self.set_propeller_speed(121.6)

    def get_pos(self):
        return self.data.mocap_pos[self.mocapid]

    def get_quat(self):
        return self.data.mocap_quat[self.mocapid]

    
    def set_propeller_speed(self, speed):
        self.prop1_jnt.qvel[0] = -speed
        self.prop2_jnt.qvel[0] = -speed
        self.prop3_jnt.qvel[0] = speed
        self.prop4_jnt.qvel[0] = speed

    
    def update(self, pos, quat):
        if pos[2] > self.propeller_spin_threshold:
            self.set_propeller_speed(51.6)
        else:
            self.set_propeller_speed(0.0)
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
        
        #rod_geom = model.geom(name_in_xml + "_rod")

        #self.rod_length = rod_geom.size[1] * 2

        #print(self.rod_length)

    def update(self, pos, quat):
        return super().update(pos, quat)


    def print_names(self):
        super().print_names()
        print("hook name in xml: " + self.hook_name_in_xml)


class HookMocap(MocapObject):

    def __init__(self, model, data, mocapid, name_in_xml, name_in_motive) -> None:
        super().__init__(model, data, mocapid, name_in_xml, name_in_motive)
    

    def update(self, pos, quat):
        pos_ = pos.copy()
        pos_[2] = pos_[2] + .03
        quat_rot = quat.copy()
        #quat_rot = mujoco_helper.quaternion_multiply(quat, np.array((0.71, 0.0, 0.0, 0.71)))
        self.data.mocap_pos[self.mocapid] = pos_
        self.data.mocap_quat[self.mocapid] = quat_rot
        
    def get_qpos(self):
        return np.append(self.data.mocap_pos[self.mocapid], self.data.mocap_quat[self.mocapid])
