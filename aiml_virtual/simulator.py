"""
Module that contains the class handling simulation.
"""
from __future__ import annotations
import math
import mujoco
import mujoco.viewer
from typing import Optional, Callable
from contextlib import contextmanager
import platform
import glfw
from functools import partial
from dataclasses import dataclass
import os

if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time

from aiml_virtual.utils import utils_general
warning = utils_general.warning
from aiml_virtual import scene
from aiml_virtual.simulated_object import simulated_object
from aiml_virtual.visualize import Visualizer

Scene = scene.Scene
SimulatedObject = simulated_object.SimulatedObject

@dataclass
class Event: # TODO: COMMENTS AND DOCSTRINGS REGARDING EVENTS
    t: float
    func: Callable[[], None]

class Simulator:
    """
    Class that uses the scene and the mujoco package to run the mujoco simulation and display the results.
    The way the user can drive the Simulator is by ticking it (calling tick()). When ticking, the simulator checks
    its processes, and decides which ones need to be called depending on their frequency and whether they are stopped.
    The user can add any number of processes, but some are provided by default such as the physics process which steps
    the mujoco simulation.
    """

    class Process:
        """
        Class for containing all the data for a process that is tied to the Simulator. I.e. a process that may only
        run when the simulator ticks.
        """
        def __init__(self, func: Callable[[], None], frequency: float, **kwargs):
            self.func: Callable[[], None] = partial(func, **kwargs)  #: The function that gets called when the process gets its turn.
            self.target_frequency: float = frequency  #: The target frequency of the process
            self.paused: bool = False  #: Whether the process should run when it gets the next chance.
            self.actual_frequency: float = frequency  #: How often the process actually runs in wall clock time.
            self.last_called = time.time()  #: When the process was last called, in wall clock time.

        def pause(self) -> None:
            """
            Pauses the process so that when it gets the chance to run next (based on its frequency and the Simulator
            tick), it will skip.
            """
            self.paused = True

        def resume(self) -> None:
            """
            Resumes the process so that when it gets the chance to run next (based on its frequency and the Simulator
            tick), it will run.
            """
            self.paused = False

        def toggle(self) -> bool:
            """
            Starts the process if it was stopped, pauses it if it was running.

            Returns:
                bool: Whether the process is on, *after* this operation.
            """
            self.paused = not self.paused
            return not self.paused

        def __call__(self) -> None:
            """
            Calls the underyling function, or raises a warning if the process was called despite being paused.
            """
            if not self.paused:
                self.func()
                self.actual_frequency = 1/(time.time()-self.last_called)
                self.last_called = time.time()
            else:
                warning("Calling paused process.")

    def __init__(self, scene: Scene):
        self.scene: Scene = scene  #: The scene corresponding to the mujoco model.
        self.data: mujoco.MjData = mujoco.MjData(self.model)  #: The data corresponding to the model.
        self.tick_count: int = 0  #: The number of times self.tick was called. Still ticks when sim is stopped.
        self.processes: dict[str, Simulator.Process] = {}  #: The dictionary of simulator processes and their names
        self.start_time: float = time.time()  #: The time when the simulation starts.
        self.visualizer: Optional[Visualizer] = None  #: Responsible for rendering, display and video creation
        self.events: list[Event] = []

    @staticmethod
    def is_headless():
        return not any([
            os.environ.get("DISPLAY"),
            os.environ.get("WAYLAND_DISPLAY"),
            os.environ.get("XDG_SESSION_TYPE") in ("x11", "wayland"),
        ])

    def add_event(self, event: Event) -> None:
        self.events.append(event)
        self.events.sort(key=lambda e: e.t)

    def handle_events(self) -> None:
        if self.data is not None and len(self.events) > 0:
            while len(self.events) > 0 and self.sim_time > self.events[0].t:
                event = self.events.pop(0)
                event.func()

    def add_process(self, name: str, func: Callable[[], None], frequency: float, *args, **kwargs) -> None:
        """
        Register a process to be run at a specified frequency. The function will be ran after a given number of
        physics steps to match the frequency, rounded down. If additional keyword arguments are passed to this method,
        they will be used as keyword arguments for the function of the process. This may be used if the process is
        something that requires arguments. If no additional keyword arguments are used, then func must take no
        arguments (and not return anything either).

        Args:
            name (str): The name of the process in the dictionary, to allow named access later.
            func (func: Callable): The function that will run when the process gets its turn.
            frequency (float): The frequency (in Hz) at which to run the method.
        """
        # the method we're registering shall be called after interval number of physics loops, for example, if
        # the physics step is 1ms, and the control frequency is 100Hz, then the method calculating control inputs
        # must be called every 10th physics loop
        if name in self.processes.keys():
            warning(f"Process {name} already present.")
        else:
            self.processes[name] = Simulator.Process(func, frequency, **kwargs)

    def initialize_processes(self) -> None:
        """
        Adds the necessary processes to the process dictionary. Should be called after all the simulated objects
        are set, since each object will have a process associated with it, where it can update.
        """
        self.add_process("physics", self.mj_step, 1 / self.timestep)
        self.add_process("handle_events", self.handle_events, 1/self.timestep)
        for obj in self.simulated_objects:
            if obj.update_frequency > 0:
                self.add_process(obj.name, obj.update, obj.update_frequency)

    @property
    def wallclock_time(self) -> float:
        """
        Property to grab the real-life (wall clock) time since the simulation started (time.time() with an offset).
        """
        return time.time() - self.start_time

    @property
    def sim_time(self) -> float:
        """
        Property to grab the time that has passed in the simulation.
        """
        return self.data.time

    @property
    def timestep(self) -> float:
        """
        Property to grab the timestep of a physics iteration from the model, which is currently the same as the internal
        tick. This supposes that the physics loop is the most often called process.
        """
        return self.opt.timestep

    @property
    def opt(self) -> mujoco.MjOption:
        """
        Property to grab the options from the model.
        """
        return self.model.opt

    @property
    def simulated_objects(self) -> list[SimulatedObject]:
        """
        Property to grab the list of objects in the scene.
        """
        return self.scene.simulated_objects

    @property
    def model(self) -> mujoco.MjModel:
        """
        Property to grab the mujoco model from the scene.
        """
        return self.scene.model

    def pause_physics(self) -> None:
        """
        Pauses the physics loop if it's running; resumes it if it's paused. If a process must be paused whenever the
        simulation is paused, this is where that can be done.
        """
        self.processes["physics"].toggle()
        for obj in self.simulated_objects:
            if obj.name in self.processes:
                self.processes[obj.name].toggle()

    # TODO: fix type hint error
    @contextmanager
    def launch(self, with_display: bool = True, fps: float = 50, speed: float = 1.0) -> 'Simulator':
        """
        A context handler to wrap the initialization and cleanup of a simulation run.
        When used with a display, it can be used like so:

        Args:
            with_display (bool): Whether to display the simulation in a window as it's being calculated.
            fps (float): The fps of the display animation (if present).
            speed (float): The relative speed of the simulation.
        """
        # this should be called *after* all objects have been added to the scene, hence it is called here, as opposed
        # to in __init__()
        self.initialize_processes()
        self.bind_scene()
        self.start_time = time.time()
        # Note that we can have a visualizer even when there is no display, since it's required to render a video, which
        # is independent of whether there is a display.
        if self.is_headless():
            print("Running in headless mode")
            if with_display:
                with_display = False
                print("Cannot open display")
            os.environ["DISPLAY"] = ":0"  # artificial display for mujoco renderer to work properly
        self.visualizer = Visualizer(self, fps, with_display=with_display)
        if with_display:
            self.add_process("visualize", self.visualizer.visualize, fps / speed)
            self.add_process("rate_limit", partial(self.rate_limit, speed=speed), fps/speed)
        self.tick()
        yield self
        self.visualizer.close()

    def rate_limit(self, speed: float = 1.0):
        """
        If we finished the last loop too quick and immediately running the next loop would cause the animation
        to be too fast, we need to wait by calling this method.

        Args:
            speed (float): The speed of the loop compared to wall clock time.
        """
        leftover_time = self.timestep * self.tick_count - self.wallclock_time * speed
        if leftover_time > 0:
            time.sleep(leftover_time)

    def bind_scene(self) -> None:
        """
        Similar to Scene.bind_to_model, except this binds the **data**, not the model. The distinction is that the data
        describes a particular run of the simulation for a given model, at a given time. Should be called once all the
        objects have been added to the simulation.
        """
        self.data = mujoco.MjData(self.model) # re-bind data in case the scene has changed since __init__ was called
        for obj in self.simulated_objects:
            obj.bind_to_data(self.data)

    def tick(self) -> None:
        """
        Tick the simulation: call every process that needs to be called. E.g.: the physics process always gets called,
        but if a process has 1/10th the frequency of the physics process, it only gets called every 10th round.
        """
        for process in self.processes.values():
            interval = max(1, math.ceil((1 / process.target_frequency) / self.timestep))
            if not process.paused and self.tick_count % interval == 0:
                process()

        # We may wish to do some optimization here: if each process step time (interval) is an integer multiple of
        # the process that's closest to it in frequency, then we can save some time by calling mj_step with an extra
        # argument nstep. This nstep may be the interval of the fastest process.
        # For example, if the physics is 1000Hz, the control is 100Hz, and the display is 50Hz, then we can call the
        # physics engine for 10 steps at every loop, call the control every loop and the display every other loop
        self.tick_count += 1

    def mj_step(self) -> None:
        """
        Process that steps the internal simulator physics.
        """
        mujoco.mj_step(self.model, self.data)

    def display_should_close(self) -> Optional[bool]:
        """
        If there is a display, returns whether the glfw window has been closed (X pressed).
        If there is no display, returns None.
        """
        if self.visualizer is not None and self.visualizer.with_display:
            return glfw.window_should_close(self.visualizer.display.window)
        else:
            return None












