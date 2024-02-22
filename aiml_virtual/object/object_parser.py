import mujoco
from aiml_virtual.util import mujoco_helper
from aiml_virtual.object.car import Fleet1Tenth, CarMocap
from aiml_virtual.object.drone import Crazyflie, Bumblebee, BumblebeeHooked, DroneMocap, DroneMocapHooked, HookMocap
from aiml_virtual.object.payload import BoxPayload, TeardropPayload, PayloadMocap
from aiml_virtual.object.bicycle import Bicycle
from aiml_virtual.object.airplane import Airplane


def parseMovingObjects(data: mujoco.MjData, model: mujoco.MjModel):

    moving_objects = []

    freejoint_names = mujoco_helper.get_freejoint_name_list(model)

    for i in range(len(freejoint_names)):
        name = freejoint_names[i]
        split_name = name.split("_")

        if split_name[0] in globals():
            class_to_be_created = globals()[split_name[0]]

            moving_obj = class_to_be_created(model, data, name)

            moving_objects += [moving_obj]

        else:
            print()
            print("[parseMovingObjects] Could not find class for ", split_name[0])

    
    return moving_objects


def parseMocapObjects(data: mujoco.MjData, model: mujoco.MjModel):

    mocap_objects = []

    obj_counter_dict = {
        "crazyflie" : 1,
        "bumblebee" : 1,
        "fleet1tenth" : 1,
        "hook" : 0,
        "payload" : 1
    }

    mocap_body_names = mujoco_helper.get_mocapbody_name_list(model)

    #print(mocap_body_names)

    for i in range(len(mocap_body_names)):

        name = mocap_body_names[i]

        split_name = name.split("_")

        try:
            class_to_be_created = globals()[split_name[0]]
        except Exception as e:
            print()
            print("[parseMocapObjects] Could not find class for ", e)
            continue

        mocapid = model.body(name).mocapid[0]


        if "crazyflie" in name:
            #name_in_motive = "cf" + str(obj_counter_dict["crazyflie"])
            #obj_counter_dict["crazyflie"] += 1
            name_in_motive = "cf" + split_name[-1]
            
        elif name.startswith("HookMocap"):
            name_in_motive = "hook12"

        elif "bumblebee" in name:
            #name_in_motive = "bb" + str(obj_counter_dict["bumblebee"])
            #obj_counter_dict["bumblebee"] += 1
            name_in_motive = "bb" + split_name[-1]
        
        elif "fleet1tenth" in name:
            if obj_counter_dict["fleet1tenth"] < 10:
                name_in_motive = "AI_car_0" + str(obj_counter_dict["fleet1tenth"])
            else:
                name_in_motive = "AI_car_" + str(obj_counter_dict["fleet1tenth"])

            obj_counter_dict["fleet1tenth"] += 1

        elif name.startswith("PayloadMocap"):
            #name_in_motive = "payload" + str(obj_counter_dict["payload"])
            #obj_counter_dict["payload"] += 1
            name_in_motive = "payload" + str(split_name[-1])
        
        else:
            name_in_motive = split_name[0]


        mocap_obj = class_to_be_created(model, data, mocapid, name, name_in_motive)

        #print("created a " + str(split_name[0]))

        mocap_objects += [mocap_obj]
    
    return mocap_objects

