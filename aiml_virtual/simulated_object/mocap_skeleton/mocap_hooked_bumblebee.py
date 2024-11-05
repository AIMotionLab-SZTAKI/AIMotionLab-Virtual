import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation

from aiml_virtual.simulated_object.mocap_object import mocap_object
from aiml_virtual.mocap import mocap_source
from aiml_virtual.simulated_object.mocap_object.mocap_drone import mocap_bumblebee
from aiml_virtual.simulated_object.mocap_skeleton import mocap_skeleton

from aiml_virtual.utils import utils_general
warning = utils_general.warning

class MocapHookedBumblebee2DOF(mocap_skeleton.MocapSkeleton):

    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return ["MocapHookedBumblebee2DOF"]

    def __init__(self, source: Optional[mocap_source.MocapSource] = None, bumblebee_mocap_name: str = "bb3",
                 hook_mocap_name: str = "hook12"):
        super().__init__(source)
        self.bumblebee: mocap_bumblebee.MocapBumblebee =  mocap_bumblebee.MocapBumblebee(source, bumblebee_mocap_name)
        self.mocap_objects.append(self.bumblebee)
        self.hook: mocap_object.MocapHook = mocap_object.MocapHook(source, hook_mocap_name)
        self.mocap_objects.append(self.hook)

    def update(self) -> None:
        if self.source is not None:
            mocap_frame = self.source.data
            if self.bumblebee.mocap_name in mocap_frame and self.hook.mocap_name in mocap_frame:
                bb_pos, bb_quat = mocap_frame[self.bumblebee.mocap_name]
                hook_pos, hook_quat = mocap_frame[self.hook.mocap_name]
                self.data.mocap_pos[self.bumblebee.mocapid] = bb_pos
                self.data.mocap_quat[self.bumblebee.mocapid] = bb_quat
                self.data.mocap_quat[self.hook.mocapid] = hook_quat
                offset_drone_frame = np.array([0.03, 0, -0.03])
                offset_world_frame = Rotation.from_quat(np.roll(bb_quat, -1)).as_matrix() @ offset_drone_frame
                self.data.mocap_pos[self.hook.mocapid] = bb_pos + offset_world_frame
        else:
            warning(f"Obj {self.name} Mocap is none.")
            return