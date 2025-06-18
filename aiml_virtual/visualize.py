"""
This module contains classes relating to the display of the simulator, and the glfw window handling.
The visualization happens in two phases: abstract visualization and openGL rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any
import glfw
import platform
import mujoco
import math
import cv2
import numpy as np
import os  # needed for video conversion
import subprocess  # needed for video conversion

if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time

from aiml_virtual.utils import utils_general
if TYPE_CHECKING:
    from aiml_virtual.simulator import Simulator
warning = utils_general.warning

class Display:
    """
    Class responsible for handing the OpenGL rendering phase and display of the simulation (as opposed to the
    abstract visualization).
    """
    def __init__(self, model: mujoco.MjModel, width: int, height: int, title: str):
        if not glfw.init():
            warning("Could not create glfw window.")
            glfw.terminate()
            return
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)  # careful! resizing may mess up the video capture!
        self.window: Any = glfw.create_window(width, height, title, None, None)  #: The handle for the glfw window.
        if not self.window:
            warning("Could not create glfw window.")
            glfw.terminate()
            return
        glfw.make_context_current(self.window)  # The context needs to be made current before handling the window.
        self.mjrContext = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)  #: Context for rendering.
        w, h = glfw.get_framebuffer_size(self.window)
        self.viewport: mujoco.MjrRect = mujoco.MjrRect(0, 0, w, h)  #: OpenGL rectangle where the rendering happens

    def render_scene(self, scene: mujoco.MjvScene) -> None:
        """
        Renders the provided MjvScene to the glfw framebuffer.
        """
        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)
        mujoco.mjr_render(self.viewport, scene, self.mjrContext)  # openGL rendering

    def render_recording_symbol(self) -> None:
        """
        Renders a red rectangle to the top left of the display to indicate a recording is taking place.
        """
        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)
        recording_symbol = mujoco.MjrRect(10, self.viewport.height - 30, 20, 20)
        mujoco.mjr_rectangle(recording_symbol, r=1.0, g=0.0, b=0.0, a=1.0)

    def display(self) -> None:
        """
        Display the latest rendered frame.
        """
        glfw.make_context_current(self.window)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def get_frame(self) -> np.ndarray:
        """
        Generate a frame from the latest rendered pixels.
        """
        rgb = np.empty(self.viewport.width * self.viewport.height * 3, dtype=np.uint8)
        depth = np.zeros(self.viewport.height * self.viewport.width)
        mujoco.mjr_readPixels(rgb, depth, self.viewport, self.mjrContext)
        rgb = np.reshape(rgb, (self.viewport.height, self.viewport.width, 3))
        rgb = np.flip(rgb, 0)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

class Visualizer:
    """
    Class responsible for handling the glfw window of the simulation and rendering mujoco data.
    A lot of the initialization under here is based on sample C code from this link:
    https://mujoco.readthedocs.io/en/stable/programming/visualization.html

    The visualizer's job is to render frames from which a video can be produced, and the frames may be displayed in a
    window. A video can be saved even when there is no display.

    .. todo:
        Make the video frame saving use multiprocessing.

    .. todo:
        When the window is dragged, the simulation lags and then rushes to catch up. This is a bug.
    """
    DEFAULT_WIDTH: int = 1536  #: The width of the window unless specified otherwise.
    DEFAULT_HEIGHT: int = 864  #: The height of the window unless specified otherwise.
    MAX_GEOM: int = 1000  #: Size of allocated geom buffer for the mjvScene struct.
    SCROLL_DISTANCE_STEP: float = 0.2  #: How much closer/farther we get to the lookat point in a mouse scroll.

    def mouse_dxdy(self, window: Any) -> tuple[int, int]:
        """
        Calculate x and y distance between the cursor's current position and the previous position.

        Args:
            window (Any): The glfw window for the Display

        Returns:
            tuple[int, int]: The x and y distances since this method was last called.
        """
        x, y = glfw.get_cursor_pos(window)
        dx = x - self.prev_x
        dy = y - self.prev_y
        self.prev_x, self.prev_y = x, y
        return dx, dy

    def handle_mouse_button(self, window: Any, button: int, action: int, mods: int) -> None:
        """
        Callback for mouse button events. Sets whether the mouse buttons are pressed down or not, which
        will modify the behavior of the mouse movement callback. Also saves the previous position of the
        cursor.

        Args:
            window (Any): The glfw window for the Display
            button (int): The keycode for the mouse button pressed.
            action (int): The code for press/release/etc.
            mods (int): The code for shift/alt/etc.
        """
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            self.prev_x, self.prev_y = glfw.get_cursor_pos(window)
            self.mouse_left_btn_down = True

        elif button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
            self.mouse_left_btn_down = False

        if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
            self.prev_x, self.prev_y = glfw.get_cursor_pos(window)
            self.mouse_right_btn_down = True

        elif button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.RELEASE:
            self.mouse_right_btn_down = False

    def handle_scroll(self, window: Any, x: float, y: float) -> None:
        """
        Callback for the mouse scroll. Zooms the camera.
        """
        self.mjvCamera.distance -= Visualizer.SCROLL_DISTANCE_STEP * y

    def handle_keypress(self, window: Any, key: int, scancode: int, action: int, mods: int) -> None:
        """
        Callback for key presses. Calls the appropriate method from the keybinds upon button press.

        Args:
            window (Any): The glfw window for the Display
            key (int): The keycode for the button pressed.
            action (int): The code for press/release/etc.
            mods (int): The code for shift/alt/etc.
            scancode (int): Code for buttons that don't have a glfw code?
        """
        if action == glfw.PRESS and key in self.keybinds:
            self.keybinds[key]()

    def handle_mouse_movement(self, window: Any, x: float, y: float) -> None:
        """
        Callback for mouse movement.

        Args:
            window (Any): The glfw window for the Display
            x (float): The x position of the mouse.
            y (float): The y position of the mouse.
        """
        if self.mouse_left_btn_down: # Rotate camera about the lookat point
            dx, dy = self.mouse_dxdy(window)
            scale = 0.1
            self.mjvCamera.azimuth -= dx * scale
            self.mjvCamera.elevation -= dy * scale

        if self.mouse_right_btn_down: # Move the point that the camera is looking at
            dx, dy = self.mouse_dxdy(window)
            scale = 0.005
            angle = math.radians(self.mjvCamera.azimuth + 90)
            dx3d = math.cos(angle) * dx * scale
            dy3d = math.sin(angle) * dx * scale
            self.mjvCamera.lookat[0] += dx3d
            self.mjvCamera.lookat[1] += dy3d

            # vertical axis is Z in 3D, so 3rd element in the lookat array
            self.mjvCamera.lookat[2] += dy * scale

    def __init__(self, simulator, fps: float, with_display: bool = True, codec_format: str = "mp4v",
                 output_file: str = "simulator.mp4",
                 width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        self.width: int = width  #: width of the rendered frame in pixels
        self.height: int = height #: height of the rendered frame in pixels

        self.simulator: Simulator = simulator  #: Reference to the simulator which the display will animate.
        self.fps: float = fps  #: The frame rate for the recording.
        self.model: mujoco.MjModel = simulator.scene.model  #: The mujoco model.
        self.data: mujoco.MjData = simulator.data  #: The mujoco data.

        self.mjvScene = mujoco.MjvScene(self.model, maxgeom=Visualizer.MAX_GEOM)  #: MjvScene for rendering.
        self.mjvCamera: mujoco.MjvCamera = mujoco.MjvCamera()  #: The camera through which the scene is viewed.
        self.mjvOption: mujoco.MjvOption = mujoco.MjvOption()  #: The visual options for rendering.
        self.mjvCamera.azimuth = 180
        self.mjvCamera.elevation = -20
        self.mjvCamera.lookat = [0, 0, 0.5]
        self.mjvCamera.distance = 5

        self.codec_format: str = codec_format  #: Codec format for the cv2 video writer.
        self.output_file: str = output_file  #: Name of the resulting mp4 file if a recording is started.
        self.writer: Optional[cv2.VideoWriter] = None  #: The object responsible for saving the video.
        self.recording: bool = False  #: Whether the frames are currently being saved.

        self.with_display: bool = with_display #: Whether a window will be displayed for the simulation.
        if with_display:
            self.prev_x: int = 0  #: Mouse x position when the last click happened.
            self.prev_y: int = 0  #: Mouse y position when the last click happened.
            self.mouse_left_btn_down: bool = False  #: Whether the left mouse button is pressed.
            self.mouse_right_btn_down: bool = False  #: Whether the right mouse button is pressed.
            self.keybinds: dict[int, callable] = {
                glfw.KEY_SPACE: self.simulator.pause_physics,
                glfw.KEY_R: self.toggle_record
            } #: Dictionary to save keybinds and their callbacks.
            self.display: Display = Display(self.model, self.width, self.height, title="simulator") #: The window handler.
            # Save callbacks: mouse movement, clicks, scroll and keybinds.
            glfw.set_scroll_callback(self.display.window, self.handle_scroll)
            glfw.set_mouse_button_callback(self.display.window, self.handle_mouse_button)
            glfw.set_cursor_pos_callback(self.display.window, self.handle_mouse_movement)
            glfw.set_key_callback(self.display.window, self.handle_keypress)
        else:
            self.renderer: mujoco.Renderer = mujoco.Renderer(self.model, self.height, self.width) #: The renderer used to generate frames without display.


    def write(self) -> None:
        """
        Saves the latest frame to a mp4 file. If there is a display for the visualizator, it will use the frames
        that appear in the window. When there isn't one, it will use a mujoco renderer to generate them.
        """
        # We don't initialize the writer in the constructor, because it's often unnecessary.
        if self.writer is None:
            codec = cv2.VideoWriter_fourcc(*self.codec_format)
            self.writer = cv2.VideoWriter(self.output_file, codec, self.fps, (self.width, self.height))
        if self.with_display:
            self.writer.write(self.display.get_frame())
        else:
            self.renderer.update_scene(self.data, self.mjvCamera, self.mjvOption) # not sure if necessary
            frame = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
            self.writer.write(frame)

    def toggle_record(self) -> None:
        """
        Starts/stops the recording, and adds the visualize process to the simulator if it's not already present.
        """
        if "visualize" not in self.simulator.processes:
            self.simulator.add_process("visualize", self.visualize, self.fps)
        self.recording = not self.recording

    def visualize(self, speed: float = 1.0) -> None:
        """
        Updates the scene for rendering. If there is a display for the simulator, also instructs the display to
        populate the window with the rendered image, then waits in order to not let the display run ahead of the wall
        clock.

        Args:
            speed (float): The relative speed of the simulation compared to the wall clock.
        """
        mujoco.mjv_updateScene(self.model, self.data, self.mjvOption, pert=None, cam=self.mjvCamera,
                               catmask=mujoco.mjtCatBit.mjCAT_ALL,
                               scn=self.mjvScene)  # abstract visualization
        if self.with_display:
            self.display.render_scene(self.mjvScene)
            if self.recording:
                self.write()
                self.display.render_recording_symbol()
            self.display.display()  # this is the point where the new pixels actually appear on screen

        else:
            if self.recording:
                self.write()

    def close(self) -> None:
        """
        Frees up the resources used by the Display.
        """
        if self.writer is not None:
            self.writer.release()
            # if ffmpeg is installed, convert the video to H.264 codec that is more versatile
            try:
                # check if it is installed by running bash command
                ffmpeg_installed = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                if ffmpeg_installed.returncode == 0:
                    # if installed: convert and save video
                    os.system("ffmpeg -i simulator.mp4 -vcodec libx264 simulator_x264.mp4")
                else:
                    print("FFmpeg is installed but there was an issue running it.")
            except FileNotFoundError:
                print("FFmpeg is not installed, skipping file conversion.")
            except Exception as e:
                print(f"Unexpected error: {e}")
        if self.with_display:
            glfw.terminate()