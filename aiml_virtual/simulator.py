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
from functools import partial
if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time

from aiml_virtual.scene import warning
from aiml_virtual import scene
from aiml_virtual.simulated_object import simulated_object
from aiml_virtual import renderer

Scene = scene.Scene
SimulatedObject = simulated_object.SimulatedObject

class Simulator:
    """
    Class that uses the scene and the mujoco package to run the mujoco simulation and display the results.
    The way the user can drive the Simulator is by ticking it (calling tick()). When ticking, the simulator checks
    its processes, and decides which ones need to be called depending on their frequency and whether they are stoppod.
    The user can add any number of processes, but these ones are provided by default:

        - "physics": steps the physics loop
        - "sync": syncs changes to the internal MjvScene and displays the simulation to the monitor
        - a process for each of the simulated objects in the scene to update themselves (e.g. update control inputs)
        - "render": saves a frame for later video creation. The renderer process saves the frame relying on the
          simulator's internal cam and vOpt. When there is a display active, the camera and options used come from the
          viewer, but whenever you want to render video without display, make sure to verify cam and vOpt.

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
        def __init__(self, func: Callable, frequency: float, **kwargs):
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
        self.viewer: Optional[mujoco.viewer.Handle] = None  #: The handler to be used for the passive viewer.
        self.tick_count: int = 0  #: The number of times self.tick was called. Still ticks when sim is stopped.
        self.processes: dict[str, Simulator.Process] = {}  #: The dictionary of simulator processes and their names
        self.start_time: float = time.time()  #: The time when the simulation starts.
        self.callback_dictionary: dict[tuple[int, bool], callable] = {
            (glfw.KEY_SPACE, False): self.pause_physics,
            (glfw.KEY_F, True): lambda: print("shift+F key callback"),
            (glfw.KEY_R, True): self.toggle_render
        }  #: A dictionary of what function to call when receiving a given keypress,and whether it requires a shift press.
        self.cam: Optional[mujoco.MjvCamera] = None  #: The camera used for rendering. Comes from viewer then possible.
        self.vOpt: Optional[mujoco.MjvOption] = None  #: The visual options used for rendering. Comes from viewer then possible. Different from opt (model options).
        self.renderer: Optional[renderer.Renderer] = None  #: Renderer for video production

    def add_process(self, name: str, func: Callable, frequency: float, *args, **kwargs) -> None:
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

    def initialize_processes(self, renderer_fps: float) -> None:
        """
        Adds the necessary processes to the process dictionary. Should be called after all the simulated objects
        are set, since each object will have a process associated with it, where it can update.

        Args:
            renderer_fps (float): If the renderer is enabled, this will be the fps of the resulting video.
        """
        self.add_process("physics", self.mj_step, 1 / self.timestep)
        self.add_process("render", self.render_data, renderer_fps,
                         renderer_fps=renderer_fps)  # last argument is keyword argument for process
        self.processes["render"].pause()  # when the render process is on, frames get saved: start with it OFF
        for obj in self.simulated_objects:
            self.add_process(obj.name, obj.update, obj.update_frequency)

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
            self.processes[obj.name].toggle()

    @contextmanager
    def launch(self, with_display: bool = True, fps: float = 50, renderer_fps: float = 50) -> 'Simulator':
        """
        A context handler to wrap the initialization and cleanup of a simulation run. If the simulation has a display,
        it will wrap the mujoco.viewer.launch_passive function. When used with a display, it can be used like so:

        .. code-block:: python

            sim = simulator.Simulator(scene)
            with sim.launch():
                while sim.viewer.is_running():
                    sim.tick()

        When used without a display, sim.viewer will be None, and you should decide for yourself how many times you
        would like to step the simulation, like so:

        .. code-block:: python

            sim = simulator.Simulator(scene)
            with sim.launch():
                while sim.tick_count < 10000:
                    sim.tick()

        Args:
            with_display (bool): Whether to display the simulation in a window as it's being calculated.
            fps (float): The fps of the display animation (if present).
        """
        # this should be called *after* all objects have been added to the scene, hence it is called here, as opposed
        # to in __init__()
        self.initialize_processes(renderer_fps)
        self.bind_scene()
        self.start_time = time.time()
        try:
            if with_display:
                # sync is the process that displays the scene
                self.add_process("sync", self.sync, fps)
                # wrap the launch_passive context handler
                with mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False,
                                                  key_callback=self.handle_keypress) as viewer:
                    # if we have a viewer, rely on the viewer for camera and visual options
                    self.viewer = viewer
                    self.cam = viewer.cam
                    self.vOpt = viewer.opt
                    yield self
            else:  #if we don't have a viewer, make a fresh camera and visual options
                self.cam = mujoco.MjvCamera()
                self.vOpt = mujoco.MjvOption()
                yield self
        finally:
            if self.renderer is not None:
                self.renderer.writer.release()
                self.renderer = None
            self.viewer = None

    def bind_scene(self) -> None:
        """
        Similar to Scene.bind_to_model, except this binds the **data**, not the model. The distinction is that the data
        describes a particular run of the simulation for a given model, at a given time. Should be called once all the
        objects have been added to the simulation.
        """
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

    def sync(self) -> None:
        """
        Syncs the viewer to the underlying data, which also refreshes the image. This means that the frame rate of
        the resulting animation will be the rate of this process. Syncs to the wall clock in order to prevent the
        display from running ahead of the simulation.
        """
        self.viewer.sync()
        dt = self.timestep * self.tick_count - self.time  # this puts a rate limit on the display
        if dt > 0:
            time.sleep(dt)

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
        with self.viewer.lock(): # This needs a lock, because they callack may change the scene.
            window = glfw.get_current_context()
            if window is not None:
                shift_pressed = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) and keycode != glfw.KEY_LEFT_SHIFT
            else:
                shift_pressed = False
            key = (keycode, shift_pressed)
            if key in self.callback_dictionary:
                self.callback_dictionary[key]()

    def toggle_render(self) -> None:
        """
        Toggles whether the rendering process saves frames.
        """
        on = self.processes["render"].toggle()
        print(f"Rendering toggled {'ON' if on else 'OFF' }")

    def render_data(self, **kwargs) -> None:
        """
        Instructs the renderer to save the data in the passive viewer to a frame, if there is already a renderer
        present. If not, initializes the renderer first.
        """
        if self.renderer is None:
            self.renderer = renderer.Renderer(self.model, kwargs["renderer_fps"], 1080, 1920)
        if self.data is not None and self.vOpt is not None and self.cam is not None:
            self.renderer.render_frame(self.data, self.cam, self.vOpt)












