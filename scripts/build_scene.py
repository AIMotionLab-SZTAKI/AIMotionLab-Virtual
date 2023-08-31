import os
from util import xml_generator
from classes.active_simulation import ActiveSimulator
from gui.building_input_gui import BuildingInputGui
from gui.vehicle_input_gui import VehicleInputGui
from gui.payload_input_gui import PayloadInputGui
from classes.payload import PAYLOAD_TYPES
from classes.object_parser import parseMovingObjects, parseMocapObjects


# open the base on which we'll build
abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "xml_models")
xmlBaseFileName = "scene.xml"
save_filename = "built_scene.xml"

build_based_on_optitrack = False

scene = xml_generator.SceneXmlGenerator(xmlBaseFileName)

virt_parsers = [parseMovingObjects]
mocap_parsers = [parseMocapObjects]
simulator = ActiveSimulator(os.path.join(xml_path, xmlBaseFileName), None, 0.01, 0.02, virt_parsers, mocap_parsers, False)
#simulator.set_drone_names()

drone_counter = 0

drone_positions = ["-1 -1 0.5", "1 -1 0.5", "-1 1 0.5", "1 1 0.5"]
drone_colors = ["0.1 0.9 0.1 1", "0.9 0.1 0.1 1", "0.1 0.1 0.9 1", "0.5 0.5 0.1 1"]

RED_COLOR = "0.85 0.2 0.2 1.0"
BLUE_COLOR = "0.2 0.2 0.85 1.0"

landing_zone_counter = 0
pole_counter = 0

def add_building():
    global scene, simulator
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
                    splt_pos[2] = "0.01"
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
                    splt_pos[2] = "0.01"
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
    

    
def add_vehicle():
    global scene, simulator
    
    input_gui = VehicleInputGui()
    input_gui.show()
    if input_gui.needs_new_vehicle:
        if input_gui.vehicle_type == "Virtual crazyflie":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_drone(input_gui.position, input_gui.quaternion, RED_COLOR, True, "crazyflie")
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Virtual bumblebee":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_drone(input_gui.position, input_gui.quaternion, RED_COLOR, True, "bumblebee", False)
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Virtual bb with hook":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_drone(input_gui.position, input_gui.quaternion, RED_COLOR, True, "bumblebee", True)
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Real crazyflie":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_drone(input_gui.position, input_gui.quaternion, BLUE_COLOR, False, "crazyflie")
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Real bumblebee":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_drone(input_gui.position, input_gui.quaternion, BLUE_COLOR, False, "bumblebee", False)
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Real bb with hook":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_drone(input_gui.position, input_gui.quaternion, BLUE_COLOR, False, "bumblebee", True)
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Virtual Fleet1Tenth":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_car(input_gui.position, input_gui.quaternion, RED_COLOR, True, False, "fleet1tenth")
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Virtual F1Tenth with rod":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_car(input_gui.position, input_gui.quaternion, RED_COLOR, True, True, "fleet1tenth")
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Real Fleet1Tenth":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_car(input_gui.position, input_gui.quaternion, BLUE_COLOR, False, False, "fleet1tenth")
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        elif input_gui.vehicle_type == "Real F1Tenth with rod":
            if input_gui.position != "" and input_gui.quaternion != "":
                scene.add_car(input_gui.position, input_gui.quaternion, BLUE_COLOR, False, True, "fleet1tenth")
                save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))
        
        else:
            #print(input_gui.vehicle_type)
            print("Non-existent vehicle type: " + input_gui.vehicle_type)

def add_load():
    global scene, simulator
    input_gui = PayloadInputGui()
    input_gui.show()

    if input_gui.needs_new_payload:
        if input_gui.position != "" and input_gui.quaternion != "":
            if input_gui.is_mocap:
                if input_gui.type == PAYLOAD_TYPES.Box.value:
                    if input_gui.size == "":
                        print("Payload size was unspecified...")
                        return
                    scene.add_load(input_gui.position, input_gui.size, None, input_gui.quaternion, input_gui.color, input_gui.type, input_gui.is_mocap)
                elif input_gui.type == PAYLOAD_TYPES.Teardrop.value:
                    scene.add_load(input_gui.position, input_gui.size, None, input_gui.quaternion, input_gui.color, input_gui.type, input_gui.is_mocap)
                else:
                    print("Unknown payload type...")
            else:
                if input_gui.mass != "":
                    if input_gui.type == PAYLOAD_TYPES.Box.value:
                        if input_gui.size == "":
                            print("Payload size was unspecified...")
                            return
                        scene.add_load(input_gui.position, input_gui.size, input_gui.mass, input_gui.quaternion, input_gui.color)
                    elif input_gui.type == PAYLOAD_TYPES.Teardrop.value:
                        scene.add_load(input_gui.position, input_gui.size, input_gui.mass, input_gui.quaternion, input_gui.color, input_gui.type)
                    else:
                        print("Unknown payload type...")
                else:
                    print("Payload mass was unspecified...")

            save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename))


def save_and_reload_model(scene, simulator, save_filename, vehicle_names_in_motive=None):
    scene.save_xml(save_filename)
    simulator.reload_model(save_filename, vehicle_names_in_motive)

def clear_scene():
    global scene, simulator, drone_counter, landing_zone_counter, pole_counter
    scene = xml_generator.SceneXmlGenerator(xmlBaseFileName)
    simulator.reload_model(os.path.join(xml_path, xmlBaseFileName))
    
    drone_counter = 0
    landing_zone_counter = 0
    pole_counter = 0


def build_from_optitrack():
    """ Try and build a complete scene based on information from Motive
    
    """
    global scene, simulator

    drone_names_in_motive = []
    car_names_in_motive = []

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
            scene.add_drone(position, orientation, BLUE_COLOR, False, "crazyflie", False)
            drone_names_in_motive += [name]
        
        elif name.startswith("bb"):
            position = str(obj.position[0]) + " " + str(obj.position[1]) + " " + str(obj.position[2])
            scene.add_drone(position, orientation, BLUE_COLOR, False, "bumblebee", False)
            drone_names_in_motive += [name]

        elif name.startswith("hook12"):
            position = str(obj.position[0]) + " " + str(obj.position[1]) + " " + str(obj.position[2])
            scene.add_mocap_hook(position, "bumblebee")
            drone_names_in_motive += [name]

        elif name.startswith("payload"):
            position = str(obj.position[0]) + " " + str(obj.position[1]) + " " + str(obj.position[2])
            scene.add_load(position, None, None, "1 0 0 0", ".1 .1 .1 1.0", PAYLOAD_TYPES.Teardrop.value, True)
            drone_names_in_motive += [name]




        elif name == "bu11":
            scene.add_hospital(position, "0.71 0 0 0.71")
        elif name == "bu12":
            scene.add_sztaki(position, "0.71 0 0 0.71")
        elif name == "bu13":
            scene.add_post_office(position, "0.71 0 0 0.71")
        elif name == "bu14":
            position = str(obj.position[0]) + " " + str(obj.position[1]) + " 0.01"
            #scene.add_airport(position, "0.71 0 0 0.71")
            scene.add_sztaki(position, "0.71 0 0 0.71")

        elif name.startswith("obs"):
            scene.add_pole(position, "0.3826834 0 0 0.9238795")
        
        elif ("RC_car" in name) or (name.startswith("Trailer")):
            position = str(obj.position[0]) + " " + str(obj.position[1]) + " " + '0.05'
            #scene.add_car(position, orientation, BLUE_COLOR, False, True)
            scene.add_car(position, orientation, BLUE_COLOR, False, False)
            car_names_in_motive += [name]
            #car_added = True
        
        elif "AI_car" in name:
            position = str(obj.position[0]) + " " + str(obj.position[1]) + " " + '0.05'
            #scene.add_car(position, orientation, BLUE_COLOR, False, True)
            scene.add_car(position, orientation, BLUE_COLOR, False, True)
            car_names_in_motive += [name]

    vehicle_names_in_motive = drone_names_in_motive + car_names_in_motive

    save_and_reload_model(scene, simulator, os.path.join(xml_path,save_filename), vehicle_names_in_motive)
    #if car_added:
    #    mocapid = simulator.model.body("car0").mocapid[0]
    #    car = CarMocap(simulator.model, simulator.data, mocapid, "car0", "AI_car_01")
    #    simulator.realdrones += [car]


def main():
    simulator.set_key_b_callback(add_building)
    simulator.set_key_d_callback(add_vehicle)
    simulator.set_key_o_callback(build_from_optitrack)
    simulator.set_key_t_callback(add_load)
    simulator.set_key_delete_callback(clear_scene)
    
    #simulator.print_optitrack_data()


    if build_based_on_optitrack:
        build_from_optitrack()

    i = 0
    while not simulator.glfw_window_should_close():

        simulator.update(i)
        #simulator.cam.azimuth += 0.2
        i += 1
    
    simulator.close()


if __name__ == '__main__':
    main()
