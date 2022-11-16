from dataclasses import dataclass
import numpy as np
import xml_generator
import drone_passive_simulation
from BuildingInputGui import BuildingDataGui


# open the base on which we'll build
xmlBaseFileName = "built_scene.xml"
save_filename = "built_scene.xml"
scene = xml_generator.SceneXmlGenerator(xmlBaseFileName)
display = drone_passive_simulation.PassiveDisplay(xmlBaseFileName, False)
#display.set_drone_names()

drone_counter = 0

drone_positions = ["-1 -1 0.5", "1 -1 0.5", "-1 1 0.5", "1 1 0.5"]
drone_colors = ["0.1 0.9 0.1 1", "0.9 0.1 0.1 1", "0.1 0.1 0.9 1", "0.5 0.5 0.1 1"]

landing_zone_counter = 0
pole_counter = 0

def add_building():
    global scene
    input_gui = BuildingDataGui()
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
            save_and_reload_model(scene, display, save_filename)
    
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
            save_and_reload_model(scene, display, save_filename)

    # add hospital
    elif input_gui.building == "Hospital":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_hospital(input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, display, save_filename)
    
    # add post-office
    elif input_gui.building == "Post office":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_post_office(input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, display, save_filename)

    # add landing zone
    elif input_gui.building == "Landing zone":
        global landing_zone_counter
        lz_name = "landing_zone" + str(landing_zone_counter)
        landing_zone_counter += 1
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_landing_zone(lz_name, input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, display, save_filename)

    # add tall Sztaki landing zone
    elif input_gui.building == "Sztaki landing zone":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_tall_landing_zone(input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, display, save_filename)
    
    # add pole
    elif input_gui.building == "Pole":
        global pole_counter
        p_name = "pole" + str(pole_counter)
        pole_counter += 1
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_pole(p_name, input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, display, save_filename)
    
    else:
        print("Non-existent building")
    

    
def add_drone():
    global scene, drone_counter
    # add drone at hard-coded positions
    # they'll be updated as soon as Optitrack data arrives
    if drone_counter < 4:
        drone_name = "drone" + str(drone_counter)
        scene.add_drone(drone_name, drone_positions[drone_counter], drone_colors[drone_counter])
        save_and_reload_model(scene, display, save_filename)
        drone_counter += 1


def save_and_reload_model(scene, display, save_filename):
        scene.save_xml(save_filename)
        display.reload_model(save_filename)



def main():
    display.set_key_b_callback(add_building)
    display.set_key_d_callback(add_drone)
    
    display.run()

if __name__ == '__main__':
    main()