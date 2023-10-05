import socket

from _thread import *
import threading
import time

from aiml_virtual.trajectory.trajectory_base import TrajectoryBase
from aiml_virtual.object.moving_object import MovingObject
from aiml_virtual.trajectory.skyc_traj_eval import get_traj_data, proc_json_trajectory, evaluate_trajectory

import os
import numpy as np
import json

DRONE_IDS = {
    "02": "Bumblebee_0",
    "04": "Crazyflie_0",
    "07": "Crazyflie_1"
}


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

    def __init__(self, list_of_vehicles) -> None:

        self.list_of_vehicles = list_of_vehicles
        
        #self.host = "192.168.2.77"
        #self.port = 7002
        self.host = ""
        self.port = 0
    

    def connect(self, host, port):
        self.host = host
        self.port = port

        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    
        self.s.connect((self.host, self.port))
    
    def start_background_thread(self):
        start_new_thread(self.receiver, (self.s,))



    def print_current_trajectories(self):
        if self.data is not None:
            print(self.data)
    
    
    def receiver(self, s):
        
        while True:
    
            # data received from client
            data = s.recv(1024).decode('utf-8').strip()
            if not data:
                print('Bye')
                
                break
    
            else:
                
                first_uscore = data.find('_')

                if data[first_uscore + 1:].startswith("CMDSTART"):
                    while not data.endswith("EOF"):
                        data += s.recv(1024).decode('utf-8').strip()
                        
                    #print(data)

                else:
                    print("[TrajectoryDistributor] Unknown message Header")
                    
                split_data = data.split('_')

                if split_data[2] == "upload":
                    print(split_data[0] + " upload")
                    cf = MovingObject.get_object_by_name_in_xml(self.list_of_vehicles, DRONE_IDS[split_data[0]])

                    if cf is not None:
                        trajectory_data = json.loads(split_data[3])
                        cf.trajectory.update_trajectory_data(trajectory_data)

                elif split_data[2] == "takeoff":
                    print(split_data[0] + " takeoff")
                    cf = MovingObject.get_object_by_name_in_xml(self.list_of_vehicles, DRONE_IDS[split_data[0]])

                    cf.trajectory.set_target_z(float(split_data[3]))


                elif split_data[2] == "land":
                    print(split_data[0] + " land")
                    cf = MovingObject.get_object_by_name_in_xml(self.list_of_vehicles, DRONE_IDS[split_data[0]])

                    cf.trajectory.clear_trajectory_data()
                    cf.trajectory.set_target_z(0.0)
                    
                
                elif split_data[2] == "start":

                    if split_data[3] == "absolute":
                        cf = MovingObject.get_object_by_name_in_xml(self.list_of_vehicles, DRONE_IDS[split_data[0]])
                        cf.trajectory.start()
                        print(split_data[0] + " start absolute")

                    elif split_data[3] == "relative":
                        print(split_data[0] + " start relative")
                        print("[TrajectoryDistributor] Relative trajectory not yet implemented.")
                
        s.close()


class RemoteDroneTrajectory(TrajectoryBase):

    def __init__(self, can_execute: bool = True, directory: str = None, init_pos=np.zeros(3)):
        super().__init__()

        
        self.data_lock = threading.Lock()

        self.trajectory_data = None

        if directory is not None:
            self.trajectory_data = get_traj_data(directory)

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

        self.current_time = i * control_step
        
        self.data_lock.acquire()
        if (self.trajectory_data is not None):
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
    

    def print_data(self):
        print(self._data)