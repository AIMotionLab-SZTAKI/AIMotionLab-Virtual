import numpy as np
from typing import Optional, cast, Type

from aiml_virtual.simulated_object.mocap_object.mocap_object import MocapObject
from aiml_virtual.mocap.mocap_source import MocapSource
from aiml_virtual.simulated_object.mocap_object.mocap_car import MocapCar, MocapTrailer
from aiml_virtual.simulated_object.mocap_skeleton import mocap_skeleton
from aiml_virtual.utils import utils_general
from aiml_virtual.simulated_object.dynamic_object.controlled_object.car import TRAILER
from scipy.spatial.transform import Rotation
warning = utils_general.warning

class MocapHitchedCar(mocap_skeleton.MocapSkeleton):
    """
    Implementation of a mocap skeleton: is madeup of a MocapCar and a MocapTrailer.
    The orientation of the trailer is read from optitrack, but its position is constrained by the car's position.
    """
    configurations: dict[str, list[tuple[str, Type[MocapObject]]]] = {
        #: The recognized combinations for MocapHitchedCar objects: bb3 with hook12
        "JoeBush1": [("JoeBush1", MocapCar), ("trailer", MocapTrailer)]
    }

    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return "MocapHitchedCar"

    def __init__(self, source: Optional[MocapSource] = None, mocap_name: Optional[str] = None):
        super().__init__(source, mocap_name)
        self.car: MocapCar = cast(MocapCar, self.mocap_objects[0])  #: The top-level 'owner' body.
        self.trailer: MocapTrailer = cast(MocapTrailer, self.mocap_objects[1])  #: The trailer's rigid body

    def update(self) -> None:
        """
        Overrides MocapObject's update: instead of merely reading the data from te mocap system and applying an offset,
        it calculates the position of the trailer in relation to the car so that the trailer and drawbar's attachment
        point is a drawbar's length of distance away from the car's hitch.
        """
        if self.source is not None:
            mocap_frame = self.source.data
            if self.car.mocap_name in mocap_frame and self.trailer.mocap_name in mocap_frame:
                car_pos, car_quat = mocap_frame[self.car.mocap_name]
                car_rot = Rotation.from_quat(np.roll(car_quat, -1))
                # account for the car's offset in the model vs optitrack, which would normally occur in the car's update
                car_offset_world_frame = car_rot.apply(self.car.offset)
                car_pos = car_pos + car_offset_world_frame
                trailer_pos, trailer_quat = mocap_frame[self.trailer.mocap_name]
                trailer_rot = Rotation.from_quat(np.roll(trailer_quat, -1))
                # account for the trailer's offset in the model vs opitrack as well
                trailer_offset_world_frame = trailer_rot.apply(self.trailer.offset)
                trailer_pos = trailer_pos + trailer_offset_world_frame
                trailer_pos[2] = car_pos[2] + 0.04 # account for the height difference between the car and trailer
                # the hitch is at the back of the car. Note that TRAILER.CAR_COG_TO_HITCH is a bit off, but I don't
                # want to touch it as it may affect the transportation demo
                hitch_pos = car_pos + car_rot.apply([-TRAILER.CAR_COG_TO_HITCH+0.03, 0, 0])
                # the trailer's attachment point is at the front of the trailer
                trailer_front_pos = trailer_pos + trailer_rot.apply([TRAILER.TRAILER_COG_TO_HITCH, 0, 0])
                rod_vec = hitch_pos - trailer_front_pos  # vector between attachment points

                # The constraint we need to add is the following:
                # The distance between the hitch points on the car and the trailer must be the same as the length
                # of the drawbar. This must be maintained while keeping the orientation of the trailer and the car.
                ideal_rod_vec = rod_vec/np.linalg.norm(rod_vec) * TRAILER.DRAWBAR_LENGTH # vector between attachment points if the bar is the correct length
                ideal_trailer_front_pos = hitch_pos + ideal_rod_vec # trailer-drawbar connection point if the bar is the correct length
                ideal_trailer_pos = ideal_trailer_front_pos - trailer_rot.apply([TRAILER.TRAILER_COG_TO_HITCH, 0, 0])  # trailer COG if the bar is the correct length

                # these are the values we need to set:
                self.data.mocap_pos[self.car.mocapid] = car_pos
                self.data.mocap_quat[self.car.mocapid] = car_quat
                self.data.mocap_pos[self.trailer.mocapid] = ideal_trailer_pos
                self.data.mocap_quat[self.trailer.mocapid] = trailer_quat

        else:
            warning(f"Obj {self.name} Mocap is None.")
            return