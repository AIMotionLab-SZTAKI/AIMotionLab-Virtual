import mujoco
import numpy as np
import util.mujoco_helper as mh
import math
from enum import Enum


class MovingObject:
    """ Base class for any moving vehicle or object
    """
    def __init__(self, name_in_xml) -> None:
        self.name_in_xml = name_in_xml

    def update(self, i):
        # must implement this method
        raise NotImplementedError("Derived class must implement update()")


class MovingMocapObject:
    """ Base class for any mocap vehicle or object
    """

    def __init__(self, name_in_xml, name_in_motive) -> None:
        self.name_in_xml = name_in_xml
        self.name_in_motive = name_in_motive
    
    def get_name_in_xml(self):
        return self.name_in_xml
    

    def print_names(self):
        print("name in xml:      " + self.name_in_xml)
        print("name in motive:   " + self.name_in_motive)


    def print_info(self):
        print("Mocap")
        self.print_names()
    

    @staticmethod
    def get_object_names_motive(objects):
        names = []
        for d in objects:
            names += [d.name_in_motive]
        
        return names
    

    @staticmethod
    def set_object_names_motive(objects, names):
        
        #if len(objects) != len(names):
        #    print("[MovingMocapObject.set_object_names_motive()] Error: too many or not enough object names provided")
        #    return
        #j = 0
        for i in range(len(objects)):
            objects[i].name_in_motive = names[i]
            #j += 1
    
    @staticmethod
    def get_object_names_in_xml(objects):
        labels = []
        for o in objects:
            labels += [o.get_name_in_xml()]
            
        return labels
    

    @staticmethod
    def get_object_by_name_in_motive(objects, name: str):
        for i in range(len(objects)):
            if objects[i].name_in_motive == name:
                return objects[i]
        
        return None