import os
from aiml_virtual.xml_generator import SceneXmlGenerator
from aiml_virtual.simulator import ActiveSimulator
from aiml_virtual.gui import BuildingInputGui
from aiml_virtual.gui import VehicleInputGui
from aiml_virtual.gui import PayloadInputGui
from aiml_virtual.object.payload import PAYLOAD_TYPES
from aiml_virtual.object import parseMovingObjects, parseMocapObjects
from aiml_virtual.object.drone import DRONE_TYPES


# open the base on which we'll build
abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xmlBaseFileName = "scene_base.xml"
save_filename = "built_scene.xml"

build_based_on_optitrack = False
is_scene_cleared = True

scene = SceneXmlGenerator(xmlBaseFileName)

virt_parsers = [parseMovingObjects]
mocap_parsers = [parseMocapObjects]
simulator = ActiveSimulator(os.path.join(xml_path, xmlBaseFileName), None, 0.01, 0.02, virt_parsers, mocap_parsers, False)

simulator.cam.azimuth = 0
simulator.onBoard_elev_offset = 20

simulator.set_title("AIMotionLab-Virtual")


RED = "0.85 0.2 0.2 1.0"
BLUE = "0.2 0.2 0.85 1.0"

landing_zone_counter = 0
pole_counter = 0

def add_building():
    global scene, simulator, is_scene_cleared
    input_gui = BuildingInputGui()
    input_gui.show()

    # add airport
    if input_gui.building == "Airport":
        if input_gui.position != "" and input_gui.quaternion != "":
            splt_pos = input_gui.position.split()
            if len(splt_pos) == 3:
                # if the 3rd coordinate is 0
                # since it's a plane, need to move it up a little so that the carpet would not cover it
                if(splt_pos[2] == '0'):
                    splt_pos[2] = "0.001"
            input_gui.position = splt_pos[0] + " " + splt_pos[1] + " " + splt_pos[2]
            scene.add_airport(input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, simulator, os.path.join(xml_path, save_filename))
    
    # add parking lot
    elif input_gui.building == "Parking lot":
        if input_gui.position != "" and input_gui.quaternion != "":
            splt_pos = input_gui.position.split()
            if len(splt_pos) == 3:
                # if the 3rd coordinate is 0
                # since it's a plane, need to move it up a little so that the carpet would not cover it
                if(splt_pos[2] == '0'):
                    splt_pos[2] = "0.001"
            input_gui.position = splt_pos[0] + " " + splt_pos[1] + " " + splt_pos[2]
            scene.add_parking_lot(input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, simulator, os.path.join(xml_path, save_filename))

    # add hospital
    elif input_gui.building == "Hospital":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_hospital(input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, simulator, os.path.join(xml_path, save_filename))
    
    # add post-office
    elif input_gui.building == "Post office":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_post_office(input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))

    # add landing zone
    elif input_gui.building == "Landing zone":
        global landing_zone_counter
        lz_name = "landing_zone" + str(landing_zone_counter)
        landing_zone_counter += 1
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_landing_zone(lz_name, input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, simulator, save_filename)

    # add Sztaki
    elif input_gui.building == "Sztaki":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_sztaki(input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
    
    # add pole
    elif input_gui.building == "Pole":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_pole(input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
    
    else:
        print("Non-existent building")
        return
    
    is_scene_cleared = False
    

    
def add_vehicle():
    global scene, simulator, is_scene_cleared
    
    input_gui = VehicleInputGui()
    input_gui.show()
    if input_gui.needs_new_vehicle:
        if input_gui.vehicle_type == "Virtual crazyflie":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_drone(input_gui.position, input_gui.quaternion, RED, DRONE_TYPES.CRAZYFLIE)
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Virtual bumblebee":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_drone(input_gui.position, input_gui.quaternion, RED, DRONE_TYPES.BUMBLEBEE)
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Virtual bb with hook":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_drone(input_gui.position, input_gui.quaternion, RED, DRONE_TYPES.BUMBLEBEE_HOOKED)
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Mocap crazyflie":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_mocap_drone(input_gui.position, input_gui.quaternion, BLUE, DRONE_TYPES.CRAZYFLIE)
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Mocap bumblebee":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_mocap_drone(input_gui.position, input_gui.quaternion, BLUE, DRONE_TYPES.BUMBLEBEE)
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Mocap bb with hook":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_mocap_drone(input_gui.position, input_gui.quaternion, BLUE, DRONE_TYPES.BUMBLEBEE_HOOKED)
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Virtual Fleet1Tenth":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_car(input_gui.position, input_gui.quaternion, RED, True, False, "fleet1tenth")
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Virtual F1Tenth with rod":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_car(input_gui.position, input_gui.quaternion, RED, True, True, "fleet1tenth")
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Mocap Fleet1Tenth":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_car(input_gui.position, input_gui.quaternion, BLUE, False, False, "fleet1tenth")
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Mocap F1Tenth with rod":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_car(input_gui.position, input_gui.quaternion, BLUE, False, True, "fleet1tenth")
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        
        else:
            #print(input_gui.vehicle_type)
            print("Non-existent vehicle type: " + input_gui.vehicle_type)
            return
    
        is_scene_cleared = False

def add_payload():
    global scene, simulator
    input_gui = PayloadInputGui()
    input_gui.show()

    if input_gui.needs_new_payload:
        if input_gui.position != "" and input_gui.quaternion != "":
            if input_gui.is_mocap:
                if input_gui.type == PAYLOAD_TYPES.Box:
                    if input_gui.size == "":
                        print("Payload size was unspecified...")
                        return
                    scene.add_mocap_payload(input_gui.position, input_gui.size, None, input_gui.quaternion, input_gui.color, input_gui.type)
                elif input_gui.type == PAYLOAD_TYPES.Teardrop:
                    scene.add_mocap_payload(input_gui.position, input_gui.size, None, input_gui.quaternion, input_gui.color, input_gui.type)
                else:
                    print("Unknown payload type...")
            else:
                if input_gui.mass != "":
                    if input_gui.type == PAYLOAD_TYPES.Box:
                        if input_gui.size == "":
                            print("Payload size was unspecified...")
                            return
                        scene.add_payload(input_gui.position, input_gui.size, input_gui.mass, input_gui.quaternion, input_gui.color)
                    elif input_gui.type == PAYLOAD_TYPES.Teardrop:
                        scene.add_payload(input_gui.position, input_gui.size, input_gui.mass, input_gui.quaternion, input_gui.color, input_gui.type)
                    else:
                        print("Unknown payload type...")
                else:
                    print("Payload mass was unspecified...")

            save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))


def save_and_reload_model(scene, simulator, save_filename, vehicle_names_in_motive=None):
    scene.save_xml(save_filename)
    simulator.reload_model(save_filename, vehicle_names_in_motive)

def clear_scene():
    global scene, simulator, landing_zone_counter, is_scene_cleared
    if not is_scene_cleared:
        scene = SceneXmlGenerator(xmlBaseFileName)
        simulator.reload_model(os.path.join(xml_path, xmlBaseFileName))
        
        landing_zone_counter = 0
        is_scene_cleared = True


def build_from_optitrack():
    """ Try and build a complete scene based on information from Motive
    
    """
    global is_scene_cleared
    global scene, simulator

    if is_scene_cleared:

        vehicle_names_in_motive = []

        if not simulator.connect_to_optitrack:
            simulator.connect_to_Optitrack()

        simulator.mc.waitForNextFrame()
        for name, obj in simulator.mc.rigidBodies.items():
            print(name)
            # have to put rotation.w to the front because the order is different
            orientation = str(obj.rotation.w) + " " + str(obj.rotation.x) + " " + str(obj.rotation.y) + " " + str(obj.rotation.z)
            position = str(obj.position[0]) + " " + str(obj.position[1]) + " " + '0'


            # this part needs to be tested again because scene generator's been modified
            if name.startswith("cf"):
                #scene.add_landing_zone("lz_" + name, position, "1 0 0 0")
                position = str(obj.position[0]) + " " + str(obj.position[1]) + " " + str(obj.position[2])
                scene.add_mocap_drone(position, orientation, BLUE, DRONE_TYPES.CRAZYFLIE, int(name[2:]))
                vehicle_names_in_motive += [name]
            
            elif name.startswith("bb"):
                position = str(obj.position[0]) + " " + str(obj.position[1]) + " " + str(obj.position[2])
                scene.add_mocap_drone(position, orientation, BLUE, DRONE_TYPES.BUMBLEBEE, int(name[2:]))
                vehicle_names_in_motive += [name]

            elif name.startswith("hook12"):
                position = str(obj.position[0]) + " " + str(obj.position[1]) + " " + str(obj.position[2])
                scene.add_mocap_hook(position, "DroneMocapHooked_bumblebee_2")
                vehicle_names_in_motive += [name]

            elif name.startswith("payload"):
                position = str(obj.position[0]) + " " + str(obj.position[1]) + " " + str(obj.position[2])
                scene.add_mocap_payload(position, None, "1 0 0 0", ".1 .1 .1 1.0", PAYLOAD_TYPES.Teardrop, int(name[7:]))
                vehicle_names_in_motive += [name]

            elif name == "bu11":
                scene.add_hospital(position, "0.71 0 0 0.71")
            elif name == "bu12":
                scene.add_sztaki(position, "0.71 0 0 0.71")
            elif name == "bu13":
                scene.add_post_office(position, "0.71 0 0 0.71")
            elif name == "bu14":
                position = str(obj.position[0]) + " " + str(obj.position[1]) + " 0.001"
                #scene.add_airport(position, "0.71 0 0 0.71")
                scene.add_sztaki(position, "0.71 0 0 0.71")

            elif name.startswith("obs"):
                scene.add_pole(position, "0.3826834 0 0 0.9238795")
            
            elif ("RC_car" in name) or (name.startswith("Trailer")):
                position = str(obj.position[0]) + " " + str(obj.position[1]) + " " + '0.05'
                #scene.add_car(position, orientation, BLUE, False, True)
                scene.add_car(position, orientation, BLUE, False, False)
                vehicle_names_in_motive += [name]
                #car_added = True
            
            elif "AI_car" in name:
                position = str(obj.position[0]) + " " + str(obj.position[1]) + " " + '0.05'
                #scene.add_car(position, orientation, BLUE, False, True)
                scene.add_car(position, orientation, BLUE, False, True)
                vehicle_names_in_motive += [name]


        save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename), vehicle_names_in_motive)
        is_scene_cleared = False
        #if car_added:
        #    mocapid = simulator.model.body("car0").mocapid[0]
        #    car = CarMocap(simulator.model, simulator.data, mocapid, "car0", "AI_car_01")
        #    simulator.realdrones += [car]


def main():
    simulator.set_key_b_callback(add_building)
    simulator.set_key_o_callback(build_from_optitrack)
    simulator.set_key_t_callback(add_payload)
    simulator.set_key_v_callback(add_vehicle)
    simulator.set_key_delete_callback(clear_scene)
    
    #simulator.print_optitrack_data()


    if build_based_on_optitrack:
        build_from_optitrack()

    i = 0
    while not simulator.glfw_window_should_close():

        simulator.update()
        #simulator.cam.azimuth += 0.2
    
    simulator.close()


if __name__ == '__main__':
    main()
