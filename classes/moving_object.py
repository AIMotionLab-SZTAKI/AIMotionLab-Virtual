import numpy as np

class MovingObject:
    """ Base class for any moving vehicle or object
    """
    def __init__(self, model, name_in_xml) -> None:
        self.name_in_xml = name_in_xml
        self.model = model

        self.mass = model.body(self.name_in_xml).mass
        self.inertia = model.body(self.name_in_xml).inertia

        self.trajectory = None
        self.controllers = None
        self.controller = None

        self.update_controller_type_method = None

        self.sensors = []
    
    def set_trajectory(self, trajectory):
        self.trajectory = trajectory
    
    def set_controllers(self, controllers):
        self.controllers = controllers

    def update(self, i, control_step):
        # must implement this method
        raise NotImplementedError("Derived class must implement update()")

    def update_controller_type(self, state, setpoint, time, i):

        if len(self.controllers) == 1:
            self.controller = self.controllers[0]
            return

        if self.update_controller_type_method is not None:
            idx = self.update_controller_type_method(state, setpoint, time, i)
            self.controller = self.controllers[idx]
            return
        
        print("update controller type method is None")

    
    def set_update_controller_type_method(self, method):

        if callable(method):
            self.update_controller_type_method = method
        
        else:
            raise TypeError("passed method is not callable")
        
    def get_state(self):
        state = []
        for i in range(len(self.sensors)):
            state += [self.sensors[i].data]

        return state



class MocapObject:
    """ Base class for any mocap vehicle or object
    """

    def __init__(self, model, data, mocapid, name_in_xml, name_in_motive) -> None:

        self.model = model
        self.data = data
        self.mocapid = mocapid
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


    def get_qpos(self):
        return np.append(self.data.mocap_pos[self.mocapid], self.data.mocap_quat[self.mocapid])
    

    @staticmethod
    def get_object_names_motive(objects):
        names = []
        for d in objects:
            names += [d.name_in_motive]
        
        return names
    

    @staticmethod
    def set_object_names_motive(objects, names):
        
        #if len(objects) != len(names):
        #    print("[MocapObject.set_object_names_motive()] Error: too many or not enough object names provided")
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