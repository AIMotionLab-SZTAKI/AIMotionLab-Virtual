"""
Module that contains the class responsible for saving and storing frames for a video.
"""
# TODO: figure out a way where this is much faster and/or uses less memory
import mujoco
import mujoco.viewer
import cv2
from typing import Optional


class VideoRenderer:
    """
    Wrapper class around mujoco.Renderer and cv2.VideoWriter.
    """
    def __init__(self, model: mujoco.MjModel, fps: float, height: int, width: int,
                 codec_format: str = "mp4v", output_file: str = "simulator.mp4"):
        self.mujocoRenderer: mujoco.Renderer = mujoco.Renderer(model, height, width)  #: Internal framre renderer.

        codec = cv2.VideoWriter_fourcc(*codec_format)
        self.writer: cv2.VideoWriter = cv2.VideoWriter(output_file, codec, fps, (width, height))

    def render_frame(self, data: Optional[mujoco.MjData], cam: Optional[mujoco.MjvCamera],
                     opt: Optional[mujoco.MjvOption]) -> None:
        """
        Calculates and saves a frame to the mp4 file.

        Args:
            data (Optional[mujoco.MjData]): The data representing the current state of the simulation.
            cam (Optional[mujoco.MjvCamera]): The camera from which to view the simulation.
            opt (Optional[mujoco.MjvOption]): The options of the model.
        """
        if data is not None and cam is not None and opt is not None:
            self.mujocoRenderer.update_scene(data, cam, opt)
            frame_rgb = self.mujocoRenderer.render()
            self.writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))