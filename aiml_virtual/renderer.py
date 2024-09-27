"""
Module that contains the class responsible for saving and storing frames for a video.
"""

import mujoco
import mujoco.viewer
import sys
import cv2
import numpy as np
from typing import Optional


class Renderer:
    """
    Wrapper class around mujoco.Renderer and cv2.VideoWriter.
    """
    def __init__(self, model: mujoco.MjModel, fps: float, height: int, width: int,
                 codec_format: str):
        self.mujocoRenderer: mujoco.Renderer = mujoco.Renderer(model, height, width)  #: Internal framre renderer.
        self.frames: list[np.ndarray] = []  #: The frames that will make up the video eventually.
        self.fps: float = fps  #: The frame rate of the eventual video.
        self.codec_format: str = codec_format  #: The video codec, e.g.: mp4v for mp4.
        self.height: int = height  #: The vertical resolution.
        self.width: int = width  #: The horizontal resolution.

    def render_frame(self, data: Optional[mujoco.MjData], cam: Optional[mujoco.MjvCamera],
                     opt: Optional[mujoco.MjvOption]) -> None:
        """
        Calculates and saves a frame to its internal list.

        Args:
            data (Optional[mujoco.MjData]): The data representing the current state of the simulation.
            cam (Optional[mujoco.MjvCamera]): The camera from which to view the simulation.
            opt (Optional[mujoco.MjvOption]): The options of the model.
        """
        if data is not None and cam is not None and opt is not None:
            self.mujocoRenderer.update_scene(data, cam, opt)
            frame_rgb = self.mujocoRenderer.render()
            self.frames.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def write(self, name: str) -> None:
        """
        Writes a video file from the stored frames under the provided name.

        Args:
            name (str): The name of the eventual video file.
        """
        if len(self.frames) == 0:
            return
        codec = cv2.VideoWriter_fourcc(*self.codec_format)
        writer = cv2.VideoWriter(name, codec, self.fps, (self.width, self.height))
        for i, frame in enumerate(self.frames):
            writer.write(frame)
            progress = int(i*100/len(self.frames))  #: the progress in %
            leftover = 99 - progress
            progress_bar = "["+"#"*progress+"_"*leftover+"]"  #: a string looking like this: [###___]
            sys.stdout.write(f"\rSaving {name}: {progress_bar}")  #: carriage return \r makes it replace previous print
        print("\n")
        writer.release()