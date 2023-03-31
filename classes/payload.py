from classes.moving_object import MovingMocapObject
from util import mujoco_helper
from enum import Enum
import numpy as np


class PAYLOAD_TYPES(Enum):
    Box = "Box"
    Teardrop = "Teardrop"

class PayloadMocap(MovingMocapObject):

    def __init__(self, model, data, mocapid, name_in_xml, name_in_motive) -> None:
        super().__init__(name_in_xml, name_in_motive)

        self.data = data
        self.mocapid = mocapid
    
    
    def update(self, pos, quat):

        self.data.mocap_pos[self.mocapid] = pos
        self.data.mocap_quat[self.mocapid] = quat
    
    def get_qpos(self):
        return np.append(self.data.mocap_pos[self.mocapid], self.data.mocap_quat[self.mocapid])

    @staticmethod
    def parse(data, model):
        payloads = []
        plc = 1

        body_names = mujoco_helper.get_body_name_list(model)

        for name in body_names:
            if name.startswith("loadmocap") and not name.endswith("hook"):
                
                mocapid = model.body(name).mocapid[0]
                c = PayloadMocap(model, data, mocapid, name, "loadmocap" + str(plc))
                
                payloads += [c]
                plc += 1
        
        return payloads
