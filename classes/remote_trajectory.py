import socket

from _thread import *
import threading
import time

from classes.trajectory_base import TrajectoryBase
from classes.moving_object import MovingObject
from classes.skyc_traj_eval import get_traj_data, proc_json_trajectory, evaluate_trajectory

import os
import numpy as np
import json


class TestServer():

    def __init__(self) -> None:

        abs_path = os.path.dirname(os.path.abspath(__file__))
        test_message_address = os.path.join(abs_path, "..", "server_test_message00.txt")
        self.test_message = open(test_message_address, 'r').read()
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


        while True:
    
            time.sleep(3)

            #t1 = "cf0 trajectory " + str(i)
            #t2 = "cf1 trajectory " + str(j)
            #t3 = "cf2 trajectory " + str(k)

            #i += 1
            #j += 1
            #k += 1

            #if i > 2:
            #    i = 0
            #if j > 2:
            #    j = 0
            #if k > 2:
            #    k = 0

            #message = t1 + '\n' + t2 + '\n' + t3

            if i == 1:
                message = self.test_message

                c.send(message.encode("utf-8"))
                time.sleep(2)
            
            if i == 2:
                message = "start"

                c.send(message.encode("utf-8"))
            
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
            data = s.recv(16384).decode('utf-8')
            if not data:
                print('Bye')
                
                break
    
            else:
                #print("data from server:\n" + data)
                print()
                print("data arrived from server")
                print()
                
                if data == "start":
                    cf = MovingObject.get_object_by_name_in_xml(self.list_of_vehicles, "Crazyflie_0")
                    cf.trajectory.start()

                    print("cmd: " + data)

                elif (not data[2:].startswith("_CMDSTART")) or (not data.endswith("_EOF")):
                    print("data in multiple pieces")

                else:
                    usc1 = data.find('_')
                    id = data[:usc1]
                    print("id: " + str(id))

                    cmdidx = data.find("CMDSTART") + 8
                    uscl = data[cmdidx + 1:].find('_')
                    cmd = data[cmdidx + 1 : cmdidx + uscl + 1]
                    print("cmd: " + str(cmd))

                    if cmd == "upload":
                        trajectory_data = data[cmdidx + uscl + 2 : -4]
                        
                        trajectory_json = json.loads(trajectory_data)
                        cf = MovingObject.get_object_by_name_in_xml(self.list_of_vehicles, "Crazyflie_0")
                        cf.trajectory.update_trajectory_data(trajectory_json)
                        
                
                    
    
        # connection closed
        s.close()


class RemoteDroneTrajectory(TrajectoryBase):

    def __init__(self, can_execute: bool = True, directory: str = None):
        super().__init__()

        
        self.data_lock = threading.Lock()

        self.trajectory_data = None

        if directory is not None:
            self.trajectory_data = get_traj_data(directory)

        self.output = {
            "load_mass" : 0.0,
            "target_pos" : np.zeros(3),
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

        self._can_execute = can_execute

    def start(self):
        self.start_time = self.current_time
        self._can_execute = True
    

    def update_trajectory_data(self, trajectory_data: dict):
        self.data_lock.acquire()
        self.trajectory_data = proc_json_trajectory(trajectory_data)
        self.data_lock.release()

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


    def evaluate_trajectory(self, time):
        if self.trajectory_data is not None:
            return evaluate_trajectory(self.trajectory_data, time)
    

    def print_data(self):
        print(self._data)