import math
import mujoco
import mujoco.viewer
from typing import Optional, Callable
from contextlib import contextmanager
import platform
import glfw
if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time


from aiml_virtual import scene
from aiml_virtual.simulated_object import simulated_object

Scene = scene.Scene
SimulatedObject = simulated_object.SimulatedObject


class PausableTime:
    """
    Class that mimics the behaviour of time.time(), but shows the time since it was last reset (or initialized), and
    it can be paused and resumed like a stopwatch.
    """
    def __init__(self):
        self.last_started: float = time.time()  #: when resume was last called
        self.sum_time: float = 0.0  #: the amount of time the clock ran the last time it was paused
        self.ticking: bool = True  #: whether the clock is ticking

    def pause(self) -> None:
        """
        Pauses the timer, which freezes the time it displays to its current state.
        """
        if self.ticking:
            self.sum_time += time.time() - self.last_started
            self.ticking = False

    def resume(self) -> None:
        """
        Resumes time measurement, which starts updating the displayed time again.
        """
        if not self.ticking:
            self.last_started = time.time()
            self.ticking = True

    def __call__(self, *args, **kwargs) -> float:
        """
        Shows the cummulated time the clock was running since it was last reset.

        Returns:
            float: The cummulated measured time.
        """
        if self.ticking:
            return self.sum_time + (time.time() - self.last_started)
        else:
            return self.sum_time

    def reset(self) -> None:
        """
        Resets the measured time to 0, and starts the clock ticking if it was paused.

        .. note::
            This is the same as __init__, but in theory it's not good practice to call __init__ by hand.
        """
        self.last_started: float = time.time()
        self.sum_time = 0.0
        self.ticking: bool = True


class Simulator:
    def __init__(self, scene: Scene, control_freq: float = 100, target_fps: int = 50):
        self.scene: Scene = scene
        self.model: mujoco.MjModel = scene.model
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self.simulated_objects: list[SimulatedObject] = self.scene.simulated_objects
        self.viewer: Optional[mujoco.viewer.Handle] = None
        self.opt: mujoco.MjOption = self.model.opt
        self.mj_step_count: int = 0  #
        self.physics_step: float = self.opt.timestep
        self.processes: list[tuple[Callable, float]] = []
        self.time = PausableTime()
        self.add_process(self.update_objects, control_freq)
        self.add_process(self.sync, target_fps)
        self.callback_dictionary: dict[int, callable] = {
            glfw.KEY_SPACE: self.toggle_pause
        }

    @property
    def paused(self):
        return not self.time.ticking

    def toggle_pause(self):
        if self.paused:
            self.time.resume()
        else:
            self.time.pause()

    @contextmanager
    def launch_viewer(self) -> 'Simulator':
        self.bind_scene()
        try:
            mujoco.mj_step(self.model, self.data)  # TODO: look up: I think we need a 0th step?
            self.time.reset()
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False,
                                                       key_callback=self.handle_keypress)
            yield self
        finally:
            if self.viewer:
                self.viewer.close()
                self.viewer = None

    def bind_scene(self) -> None:
        for obj in self.simulated_objects:
            obj.bind_to_data(self.data)

    def add_process(self, method: Callable, frequency: float) -> None:
        """Register a process to be run at a specified frequency.

        Args:
            method (function): The method of the Simulator class to run.
            frequency (float): The frequency (in Hz) at which to run the method.
        """
        # the method we're registering shall be called after interval number of physics loops, for example, if
        # the physics step is 1ms, and the control frequency is 100Hz, then the method calculating control inputs
        # must be called every 10th physics loop
        interval = max(1, math.ceil((1 / frequency) / self.physics_step))
        self.processes.append((method, interval))

    def step(self) -> None:
        # this is the function that steps the mujoco simulator, and calls all the other processes
        for method, interval in self.processes:  # each process must be registered
            if self.mj_step_count % interval == 0:
                method()  # maybe arguments? we'll see
        # We may wish to do some optimization here: if each process step time (interval) is an integer multiple of
        # the process that's closest to it in frequency, then we can save some time by calling mj_step with an extra
        # argument nstep. This nstep may be the interval of the fastest process.
        # For example, if the physics is 1000Hz, the control is 100Hz, and the display is 50Hz, then we can call the
        # physics engine for 10 steps at every loop, call the control every loop and the display every other loop
        # TODO: think through whether this needs to be reconciled with data.time and whether it needs to be moved inside
        #  the if condition under here
        dt = self.data.time - self.time()
        if dt > 0:  # if the simulation needs to wait in order to not run ahead
             time.sleep(dt)
        if not self.paused:
            mujoco.mj_step(self.model, self.data)
            self.mj_step_count += 1

    def update_objects(self) -> None:
        for obj in self.simulated_objects:
            obj.update(self.mj_step_count, self.physics_step)

    def sync(self) -> None:
        # print(f"time: {self.data.time}")
        self.viewer.sync()

    def handle_keypress(self, keycode) -> None:
        self.callback_dictionary[keycode]()

