import socket

from _thread import *
import threading
import time

from classes.trajectory_base import TrajectoryBase
from classes.moving_object import MovingObject
from classes.skyc_traj import get_traj_data

import os
import numpy as np


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

            message = self.test_message

            c.send(message.encode('utf-8'))

        self.s.close()
    



class TrajectoryDistributor():

    def __init__(self, list_of_vehicles) -> None:

        self.list_of_vehicles = list_of_vehicles
        
        #self.host = "192.168.2.77"
        #self.port = 7002
        self.host = ""
        self.port = 0
    
        self.data = None

        self.has_new_data = False
    

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
                #self.data = data

                #splt = data.split('\n')

                #for s_ in splt:
                    #split_segment = s_.split(' ')

                    #if split_segment[0].startswith("cf"):
                        
                    #    object_name = "Crazyflie_" + split_segment[0][2:]
                    #    moving_object = MovingObject.get_object_by_name_in_xml(self.list_of_vehicles, object_name)

                    #    moving_object.trajectory.update_trajectory_data(split_segment[-1])
                
                if (not data[2:].startswith("_CMDSTART")) or (not data.endswith("_EOF")):
                    print("data in multiple pieces")

                else:
                    usc1 = data.find('_')
                    id = data[:usc1]
                    print("id: " + str(id))

                    cmdidx = data.find("CMDSTART") + 8
                    uscl = data[cmdidx + 1:].find('_')
                    cmd = data[cmdidx + 1 : cmdidx + uscl + 1]
                    print("cmd: " + str(cmd))


                    
                    self.has_new_data = True
    
        # connection closed
        s.close()


class RemoteDroneTrajectory(TrajectoryBase):

    def __init__(self):
        super().__init__()
        
        self.data_lock = threading.Lock()

        self.output = {
            "load_mass" : 0.0,
            "target_pos" : None,
            "target_rpy" : np.zeros(3),
            "target_vel" : np.zeros(3),
            "target_acc" : None,
            "target_ang_vel": np.zeros(3),
            "target_quat" : None,
            "target_quat_vel" : None,
            "target_pos_load" : None
        }

        self._data = None
    
    def evaluate(self, state, i, time, control_step) -> dict:
        return self.output

    def update_trajectory_data(self, new_data):
        self.data_lock.acquire()
        self._data = new_data
        self.data_lock.release()
    

    def print_data(self):
        print(self._data)