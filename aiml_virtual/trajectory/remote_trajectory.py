import socket

from _thread import *
import threading
import time

from aiml_virtual.trajectory.trajectory_base import TrajectoryBase
from aiml_virtual.object.moving_object import MovingObject
from aiml_virtual.trajectory.skyc_traj_eval import get_traj_data, get_traj_data_from_json, proc_json_trajectory, evaluate_trajectory
from aiml_virtual.object.drone import Crazyflie, Bumblebee, BumblebeeHooked
from aiml_virtual.object.car import Fleet1Tenth

import os
import numpy as np
import json

DRONE_IDS = {
    "02": "Bumblebee_0",
    "04": "Crazyflie_0",
    "06": "Crazyflie_1",
    "07": "Crazyflie_2",
    "08": "Crazyflie_3",
    "09": "Crazyflie_4",
}


def get_crazyflies(list_of_vehicles):
    cfs = []
    for v in list_of_vehicles:
        if isinstance(v, Crazyflie):
            cfs += [v]

    return cfs

def get_bumblebees(list_of_vehicles):
    bbs = []
    for v in list_of_vehicles:
        if isinstance(v, Bumblebee):
            bbs += [v]

    return bbs

def get_bumblebeehookeds(list_of_vehicles):
    bbhs = []
    for v in list_of_vehicles:
        if isinstance(v, BumblebeeHooked):
            bbhs += [v]

    return bbhs

def get_f1tenths(list_of_vehicles):
    f1ts = []
    for v in list_of_vehicles:
        if isinstance(v, Fleet1Tenth):
            f1ts += [v]

    return f1ts


class TestServer():

    def __init__(self) -> None:

        abs_path = os.path.dirname(os.path.abspath(__file__))
        test_message0_address = os.path.join(abs_path, "..", "..", "server_test_message00.txt")
        self.test_message0 = open(test_message0_address, 'r').read()
        test_message1_address = os.path.join(abs_path, "..", "..", "server_test_message01.txt")
        self.test_message1 = open(test_message1_address, 'r').read()
        self.host = ""
        self.port = 12345

        
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.host, self.port))
        print("socket bound to port", self.port)

        # put the socket into listening mode
        self.s.listen(5)
        print("socket is listening")
    

    def run(self):
        
        # establish connection with client
        c, addr = self.s.accept()

        # lock acquired by client
        print('Connected to :', addr[0], ':', addr[1])


        i = 0
        j = 1
        k = 2

        time.sleep(3)

        while True:

            if i == 1:
                message = self.test_message0

                c.send(message.encode("utf-8"))
                time.sleep(5)
            
            if i == 2:
                message = "start"

                c.send(message.encode("utf-8"))
                time.sleep(10)
            
            if i == 3:
                message = self.test_message1

                c.send(message.encode("utf-8"))
                time.sleep(5)
            
            
            if i == 4:
                message = "start"

                c.send(message.encode("utf-8"))
                time.sleep(8)

            
            i += 1

        self.s.close()




class TrajectoryDistributor():

    def __init__(self, list_of_vehicles, skyc_save_directory:str, skyc_play_mode=False) -> None:

        if not os.path.exists(skyc_save_directory):
            os.mkdir(skyc_save_directory)

        self.list_of_vehicles = list_of_vehicles

        self._list_of_crazyflies = get_crazyflies(list_of_vehicles)
        self._current_cf_idx = 0

        self.latest_skyc_filename_filename = "latest_skyc_filename.txt"
        
        #self.host = "192.168.2.77"
        #self.port = 7002
        self.host = ""
        self.port = 0

        self.vehicles_waiting_for_start = []

        self.id_to_vehicle = {}
        self._vehicle_has_teleported = {}

        self.skyc_save_directory = skyc_save_directory

        if skyc_play_mode:
            skyc_filename_filename = os.path.join(self.skyc_save_directory, self.latest_skyc_filename_filename)
            with open(skyc_filename_filename, 'r') as f:
                skyc_filename = f.read()
            trajectories = get_traj_data(skyc_filename)
            
            print()
            print("[TrajectoryDistributor] Vehicles waiting for start:")
            i = 0
            for t in trajectories:

                if i < len(self._list_of_crazyflies):
                    cf = self._list_of_crazyflies[i]
                    if cf is not None:
                        cf.trajectory.update_trajectory_data(t)
                        self.vehicles_waiting_for_start += [cf]
                        print(cf.name_in_xml)
                    
                    i += 1
                else:
                    print("[TrajectoryDistributor] Not enough crazyflies for SKYC file.")


    

    def connect(self, host, port):
        self.host = host
        self.port = port

        try:
            self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        
            self.s.connect((self.host, self.port))
            return True
        except Exception as e:
            print(e)
            return False
    
    def start_background_thread(self):
        start_new_thread(self.receiver, (self.s,))
    
    
    def receiver(self, s):

        BUFF_SIZE = 65536
        
        while True:
    
            # data received from client
            data = s.recv(BUFF_SIZE)
            
            if not data:
                print("[TrajectoryDistributor] Good bye!")
                
                break
    
            else:
                
                while not data.endswith(b"EOF") and not data.endswith(b"SKYC"):
                    data += s.recv(BUFF_SIZE)
                
                s.sendall(b"ACK")
                
                if data.endswith(b"SKYC"):
                    data = data[:-4]

                    skyc_filename = os.path.join(self.skyc_save_directory, "skyc_file_" + time.strftime("%Y%m%d-%H%M%S") + ".skyc")

                    with open(skyc_filename, 'wb') as writer:
                        writer.write(data)

                    print()
                    print("[TrajectoryDistributor] SKYC file received, saved at:\n" + os.path.abspath(skyc_filename))
                    print()

                    latest_filename_filename = os.path.join(self.skyc_save_directory, self.latest_skyc_filename_filename)

                    with open(latest_filename_filename, 'w') as f:
                        f.write(os.path.abspath(skyc_filename))
                    

                else:
                    
                    data = data.decode("utf-8").strip()
                    
                    split_data = data.split('_')

                    id = split_data[0]
                    cmd = split_data[2]

                    if id not in self.id_to_vehicle:

                        if id == "02" or id == "03":
                            # this is a bumblebee
                            pass
                        else:
                            if self._current_cf_idx < len(self._list_of_crazyflies):
                                self.id_to_vehicle[id] = self._list_of_crazyflies[self._current_cf_idx]
                                self._vehicle_has_teleported[id] = False
                                #print("added ", id, " : ", self._list_of_crazyflies[self._current_cf_idx].name_in_xml)
                                self._current_cf_idx += 1

                            else:
                                print("[TrajectoryDistributor] No more crazyflies in simulation.")
                    
                

                    if cmd == "upload":
                        print(id + " upload")
                        cf = self.id_to_vehicle.get(id)

                        if cf is not None:
                            trajectory_data = json.loads(split_data[3])
                            cf.trajectory.update_trajectory_data(trajectory_data)

                            if not self._vehicle_has_teleported[id]:
                                p, v = evaluate_trajectory(trajectory_data, 0.0)
                                cf.set_qpos(p)
                                self._vehicle_has_teleported[id] = True

                    elif cmd == "takeoff":
                        print(id + " takeoff")
                        cf = self.id_to_vehicle.get(id)

                        if cf is not None:
                            cf.trajectory.set_target_z(float(split_data[3]))


                    elif cmd == "land":
                        print(id + " land")
                        cf = self.id_to_vehicle.get(id)

                        if cf is not None:
                            cf.trajectory.clear_trajectory_data()
                            cf.trajectory.set_target_z(0.0)
                        
                    
                    elif cmd == "start":

                        if split_data[3] == "absolute":
                            cf = self.id_to_vehicle.get(id)
                            if cf is not None:
                                cf.trajectory.start()
                            print(id + " start absolute")

                        elif split_data[3] == "relative":
                            print(id + " start relative")
                            print("[TrajectoryDistributor] Relative trajectory not yet implemented.")
                    
                    elif cmd == "show":
                        print("cmd: show")
                        
                        for v in self.vehicles_waiting_for_start:
                            v.trajectory.start()
                
        s.close()

class RemoteDroneTrajectory(TrajectoryBase):

    def __init__(self, can_execute: bool = True, directory: str = None, init_pos=np.zeros(3)):
        super().__init__()
        
        self.data_lock = threading.Lock()

        self.trajectory_data = None

        if directory is not None:
            self.trajectory_data = get_traj_data_from_json(directory)

        self.output = {
            "load_mass" : 0.0,
            "target_pos" : init_pos,
            "target_rpy" : np.zeros(3),
            "target_vel" : np.zeros(3),
            "target_acc" : None,
            "target_ang_vel": np.zeros(3),
            "target_quat" : None,
            "target_quat_vel" : None,
            "target_pos_load" : None
        }

        self.current_time = 0.0
        self.start_time = 0.0
        self.prev_yaw = 0.0

        self._can_execute = can_execute
    

    def start(self):
        self.start_time = self.current_time
        self._can_execute = True
    

    def update_trajectory_data(self, trajectory_data: dict):
        self.data_lock.acquire()
        self.trajectory_data = proc_json_trajectory(trajectory_data)
        self.data_lock.release()
        self._can_execute = False

    
    def clear_trajectory_data(self):
        self.data_lock.acquire()
        self.trajectory_data = None
        self.data_lock.release()

    def set_target_z(self, target_z: float):
        self.output["target_pos"][2] = target_z

    def evaluate(self, state, i, time, control_step) -> dict:
        
        self.current_time = time
        self.data_lock.acquire()
        if self._can_execute:
            time = self.current_time - self.start_time
            target_pos, target_vel = self.evaluate_trajectory(time)
            self.output["target_pos"] = target_pos
            self.output["target_vel"] = target_vel
        else:
            target_pos, target_vel = self.evaluate_trajectory(0.0)
            self.output["target_pos"] = target_pos
            self.output["target_vel"] = target_vel

        self.data_lock.release()

        return self.output


    def evaluate_trajectory(self, time) -> (np.array, np.array):
        """ evaluate the trajectory at a point in time 
        returns:
        position and velocity

        """
        if self.trajectory_data is not None:
            return evaluate_trajectory(self.trajectory_data, time)
        else:
            return (self.output["target_pos"], self.output["target_vel"])
    

    def print_data(self):
        print(self.trajectory_data)