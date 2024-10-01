import os
import pathlib

import mujoco
import numpy as np

import aiml_virtual.scene as scene
import aiml_virtual.simulated_object.dynamic_object.controlled_object.bicycle as bicycle
import aiml_virtual.simulated_object.dynamic_object.controlled_object.drone.crazyflie as cf
import aiml_virtual.simulated_object.dynamic_object.controlled_object.drone.bumblebee as bb
from aiml_virtual.trajectory import dummy_drone_trajectory, skyc_trajectory

"""
- nagyon sok beépített keybind van
	* van, ami király
	* van, ami értelmetlen
	* amik értelmetlenek, elveszik tőlünk a helyet
	
alternatív verzió: használjunk renderert? talán nagyobb szabadság????
jegyzetelek hozzá kicsit:
structs: 
legtöbb helyen közvetlen férnek hozzá a memóriához
mjModel-nek nincs konstruktorja (helyette factory func), többinek igen (vagy mj_default vagy mj_make)

functions:
név, param mujoco.h szerint 
input: array->nparray/iterable(np convertible)
output: writeable numpy array
amikor egy C fgv meghívódik, amíg fut, el van engedve a GIL

named access:
minden objektumnak az xml-ben kell egy 'name' attribute, amelyhez szám rendelődik az mjModel-ben, amellyel indexelni lehet
ezekkel a nevekkel: m: mujoco.MjModel; m.geom('név) -> visszaadja a geom dolgait a modellben, hasonló data-ban

Rendering:
mjr_* renderel ez előtt kell egy OpenGL context: ctx = mujoco.GLContext(max_width, max_height), felszabadítás (törlés): ctx.free()
ennek kurrensnek kell lennie: ctx.make_current()
ey context egy adott pillanatban csak 1 szálnak lehet kurrens, és minden rendering callnak azon a szálon kell lennie

Ami érdekes, hogy a google colab-ban nincsenek ilyen opengl hívások:
with mujoco.Renderer(model) as renderer:
  mujoco.mj_forward(model, data)
  renderer.update_scene(data)
  media.show_image(renderer.render())
  
 Ennek a közepében láthatóan a Renderer van, ami egy olyan objektum amit (a passive viewerhez hasonlóan) nem 
 használt még Máté, helyette mjv_updateScene, mjr_render, mjr_overlay, mjr_text, etc.
 A Renderer úgy tűnik, megoldja a háttérben az opengl dolgokat?
 Van egy self._gl_context-je, tán ahhoz hozzá lehet férni a többi opengl dologért, egyébként pedig 2 fontos fgv:
 render(): pixeleket ad vissza numpy arrayben kijelzéshez (100000% fix, hogy ez Máténál is így van, csak meg kell keresni)
 udpate_scene(): frissíti a dolgokat amikből renderelünk (mivelhogy step miatt elváltoztak az előző render óta)
 Nekünk kelleni fognak keybindok ugyebár
 Végül nem jó, mert a renderer csak számol, nem displayel, és bonyolultabb displayelni mint amire én hajlandó vagyok
 """



if __name__ == "__main__":
    project_root = pathlib.Path(__file__).parents[1].resolve().as_posix()
    xml_directory = os.path.join(project_root, "xml_models")
    scene = scene.Scene(os.path.join(xml_directory, "bicycle.xml"))
    bike1 = bicycle.Bicycle()
    scene.add_object(bike1, "0 1 0", "1 0 0 0", "0.5 0.5 0.5 1")
    bike2 = bicycle.Bicycle()
    scene.add_object(bike2, "0 -1 0", "1 0 0 0", "0.5 0.5 0.5 1")
    cf0 = cf.Crazyflie()
    traj = skyc_trajectory.SkycTrajectory("skyc_example.skyc")
    cf0.trajectory = traj
    scene.add_object(cf0, "0 0 0", "1 0 0 0", "0.5 0.5 0.5 1")
    bb0 = bb.Bumblebee()
    bb0.trajectory = dummy_drone_trajectory.DummyDroneTrajectory(np.array([-1, 0, 1]))
    scene.add_object(bb0, "-1 0 0.5", "1 0 0 0", "0.5 0.5 0.5 1")

    model = scene.model
    data = mujoco.MjData(model)
    with mujoco.Renderer(scene.model) as renderer:
        for i in range(1000):
            mujoco.mj_step(scene.model, data)
            renderer.update_scene(data)
            pixels = renderer.render().flatten()


