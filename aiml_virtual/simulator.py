"""
Module that contains the class handling simulation.
"""
import math
import mujoco
import mujoco.viewer
from typing import Optional, Callable
from contextlib import contextmanager
import platform
import glfw

from aiml_virtual.scene import warning

if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time


from aiml_virtual import scene
from aiml_virtual.simulated_object import simulated_object

Scene = scene.Scene
SimulatedObject = simulated_object.SimulatedObject

class Simulator:
    """
    Class that uses the scene and the mujoco package to run the mujoco simulation and display the results.

    .. todo::
        Find a way to disable the default keybinds.
        Relevant reading material:

        - https://github.com/google-deepmind/mujoco/discussions/780
        - https://github.com/google-deepmind/mujoco/issues/766
        - https://github.com/google-deepmind/mujoco/issues/1151
        - https://github.com/google-deepmind/mujoco/issues/2038
    """

    class Process:
        """
        Class for containing all the data for a process that is tied to the Simulator. I.E. a process that may only
        run when the simulator ticks.
        """
        def __init__(self, func: Callable[[], None], frequency:float):
            self.func: Callable[[], None] = func  #: The function that gets called when the process gets its turn.
            self.frequency: float = frequency  #: The real-life frequency of the process if it's not stopped
            self.paused: bool = False  #: Whether the process should run when it gets the next chance.

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

        def toggle(self) -> None:
            """
            Starts the process if it was stopped, pauses it if it was running.
            """
            self.paused = not self.paused

        def __call__(self) -> None:
            """
            Calls the underyling function, or raises a warning if the process was called despite being paused.
            """
            if not self.paused:
                self.func()
            else:
                warning("Calling paused process.")

    def __init__(self, scene: Scene, update_freq: float = 100, target_fps: int = 50):
        self.scene: Scene = scene  #: The scene corresponding to the mujoco model.
        self.data: mujoco.MjData = mujoco.MjData(self.model)  #: The data corresponding to the model.
        self.viewer: Optional[mujoco.viewer.Handle] = None  #: The handler to be used for the passive viewer.
        self.tick_count: int = 0  #: The number of times self.tick was called. Still ticks when sim is stopped.
        self.processes: dict[str, Simulator.Process] = {}  #: The dictionary of simulator processes and their names
        self.start_time: float = time.time()  #: The time when the simulation starts.
        self.callback_dictionary: dict[tuple[int, bool], callable] = {
            (glfw.KEY_SPACE, False): self.pause_physics,
            (glfw.KEY_F, True): lambda: print("F key callback"),
            (glfw.KEY_R, True): lambda: self.processes["render"].toggle()
        }  #: A dictionary of what function to call when receiving a given keypress,and whether it requires a shift press.

        self.add_process("update", self.update_objects, update_freq)
        self.add_process("sync", self.sync, target_fps)
        self.add_process("physics", self.mj_step, 1/self.timestep)

        self.renderer = mujoco.Renderer(self.scene.model)
        self.add_process("render", self.render, 1)
        self.processes["render"].pause()
        self.frames = []


    @property
    def time(self) -> float:
        """
        Property to grab the time since the simulation started (time.time() with an offset).
        """
        return time.time() - self.start_time

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
    def simulated_objects(self):
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
        self.processes["update"].toggle()

    @contextmanager
    def launch_viewer(self) -> 'Simulator':
        """
        Wraps the mujoco.viewer.launch_passive function so that it handlers the simulator's initialization. As this
        is a context handler, it should be used like so:

        .. code-block:: python

            sim = simulator.Simulator(scene, control_freq=500, target_fps=100)
            with sim.launch_viewer():
                while sim.viewer.is_running():
                    sim.step()
        """
        self.bind_scene()
        try:
            # may need a 0th step here (mujoco.mj_forward)
            self.start_time = time.time()
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False,
                                                       key_callback=self.handle_keypress)
            yield self
        finally:
            if self.viewer:
                self.viewer.close()
                self.viewer = None

    def bind_scene(self) -> None:
        """
        Similar to Scene.bind_to_model, except this binds the **data**, not the model. The distinction is that the data
        describes a particular run of the simulation for a given model, at a given time.
        """
        for obj in self.simulated_objects:
            obj.bind_to_data(self.data)

    def add_process(self, name: str, func: Callable[[], None], frequency: float) -> None:
        """
        Register a process to be run at a specified frequency. The function will be ran after a given number of
        physics steps to match the frequency, rounded down.

        Args:
            name (str): The name of the process in the dictionary, to allow named access later.
            func (func: Callable[[], None]): The function that will run when the process gets its turn.
            frequency (float): The frequency (in Hz) at which to run the method.
        """
        # the method we're registering shall be called after interval number of physics loops, for example, if
        # the physics step is 1ms, and the control frequency is 100Hz, then the method calculating control inputs
        # must be called every 10th physics loop
        if name in self.processes.keys():
            warning(f"Process {name} already present.")
        else:
            self.processes[name] = Simulator.Process(func, frequency)

    def tick(self) -> None:
        """
        This is the function that wraps mj_step and does the housekeeping relating to it; namely:

        - Calling processes in self.processes if necessary.
        - Syncing to the wall clock.
        """
        for process in self.processes.values():
            interval = max(1, math.ceil((1 / process.frequency) / self.timestep))
            if not process.paused and self.tick_count % interval == 0:
                process()

        # We may wish to do some optimization here: if each process step time (interval) is an integer multiple of
        # the process that's closest to it in frequency, then we can save some time by calling mj_step with an extra
        # argument nstep. This nstep may be the interval of the fastest process.
        # For example, if the physics is 1000Hz, the control is 100Hz, and the display is 50Hz, then we can call the
        # physics engine for 10 steps at every loop, call the control every loop and the display every other loop
        dt = self.timestep * self.tick_count - self.time  # this puts a rate limit on the loop
        if dt > 0:
            time.sleep(dt)
        self.tick_count += 1

    def mj_step(self):
        """
        Process that steps the internal simulator physics.
        """
        mujoco.mj_step(self.model, self.data)

    def update_objects(self) -> None:
        """
        Each simulated object may have housekeeping to do: this process is their opportunity.
        """
        for obj in self.simulated_objects:
            obj.update(self.data.time)

    def sync(self) -> None:
        """
        Syncs the viewer to the underlying data, which also refreshes the image. This means that the frame rate of
        the resulting animation will be the rate of this process.
        """
        self.viewer.sync()

    def handle_keypress(self, keycode: int) -> None:
        """
        This method is passed to the viewer's initialization function to get called when a key is pressed. What should
        happen upon that specific keypress is then looked up in the callback dictionary.

        .. note::
            There are **tons** of default keybinds in the mujoco passive viewer that are (so far as I can tell) separate
            from this mechanism. As best I could determine, they each update something in the model/scene/data, most
            commonly flags, such as viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] or
            mujoco.mjtVisFlag.mjVIS_LIGHT, and then (and this caused me some headache) run the equivalent of sync();
            i.e. they also render the scene. This is an issue: almost all binds are taken by these features, and I
            cannot unbind them to redirect the keybind to our function, because their mechanism is separate. I tried
            to disable some of them by inverting the flag they change back to its original value in my callback, but
            this was unreliable: often my flag flips would get caught in some sort of buffer and only take effect upon
            the next option modification. There should be a way to disable or at least modify the keybinds, or even
            UI elements from python (like disabling the rendering section). By this, I mean completely disabling their
            functionality, not just hiding it (like hiding the left-right UI).
            Additionally, we should be able to get other info above the keypress: was it modified by alt-shit-etc? Was
            it a keypress or a key release?
            Currently, we have a hacky way to determine whether shift was pressed. The dictionary storing callbacks
            can decide whether shift is needed to call the function.
        """
        with self.viewer.lock(): # I'm not actually sure if this needs a lock
            window = glfw.get_current_context()
            if window is not None:
                shift_pressed = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) and keycode != glfw.KEY_LEFT_SHIFT
            else:
                shift_pressed = False
            key = (keycode, shift_pressed)
            if key in self.callback_dictionary:
                self.callback_dictionary[key]()

    def render(self):
        if self.viewer is not None and self.data is not None:
            self.renderer.update_scene(self.data, self.viewer.cam)
            self.frames.append(self.renderer.render())
            print(f"Rendered a frame (currently there are {len(self.frames)} frames).")









