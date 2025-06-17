import numpy as np
from typing import Optional, cast, Type
from scipy.spatial.transform import Rotation

from aiml_virtual.simulated_object.mocap_object import mocap_object
from aiml_virtual.mocap.mocap_source import MocapSource
from aiml_virtual.simulated_object.mocap_object.mocap_drone.mocap_bumblebee import MocapBumblebee
from aiml_virtual.simulated_object.mocap_skeleton import mocap_skeleton

from aiml_virtual.utils import utils_general
warning = utils_general.warning

class MocapHookedBumblebee2DOF(mocap_skeleton.MocapSkeleton):
    """
    Implementation of a mocap skeleton: is made up of a MocapBumblebee and a MocapHook.
    The orientation of the hook is read from optitrack, but its position is determined by the drone's pose.
    """

    def __init__(self, source: Optional[MocapSource] = None, mocap_name: Optional[str] = None):
        super().__init__(source, mocap_name)
        self.bumblebee: MocapBumblebee =  cast(MocapBumblebee, self.mocap_objects[0])  #: The top-level 'owner' body.
        self.hook: mocap_object.MocapHook = cast(mocap_object.MocapHook, self.mocap_objects[1])  #: The hook's rigid body

    def update(self) -> None:
        """
        Overrides MocapObject's update: instead of merely reading the data from the mocap system and applying an offset,
        it does this for the drone, and then calculates the position of the hook so that the end of the rod is attached
        to the bumblebee's bottom.
        """
        if self.source is not None:
            mocap_frame = self.source.data
            self.bumblebee.spin_propellers()
            if self.bumblebee.mocap_name in mocap_frame and self.hook.mocap_name in mocap_frame:
                bb_pos, bb_quat = mocap_frame[self.bumblebee.mocap_name]
                bb_offset = self.bumblebee.offset
                bb_rotmat = Rotation.from_quat(np.roll(bb_quat, -1)).as_matrix()
                bb_pos += bb_rotmat @ bb_offset
                hook_pos, hook_quat = mocap_frame[self.hook.mocap_name]
                self.data.mocap_pos[self.bumblebee.mocapid] = bb_pos
                self.data.mocap_quat[self.bumblebee.mocapid] = bb_quat
                self.data.mocap_quat[self.hook.mocapid] = hook_quat
                hook_offset_drone_frame = np.array([0.03, 0, -0.03])
                hook_offset_world_frame = bb_rotmat @ hook_offset_drone_frame
                self.data.mocap_pos[self.hook.mocapid] = bb_pos + hook_offset_world_frame
        else:
            warning(f"Obj {self.name} Mocap is None.")
            return