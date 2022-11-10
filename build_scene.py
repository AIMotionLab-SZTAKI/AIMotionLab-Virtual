from dataclasses import dataclass
import dis
import mujoco
import glfw
import os
import numpy as np
import xml_generator
import drone_passive_simulation
import BuildingInputGui as bdg


# open the base on which we'll build
xmlBaseFileName = "scene.xml"
save_filename = "built_scene.xml"
scene = xml_generator.SceneXmlGenerator(xmlBaseFileName)
display = drone_passive_simulation.PassiveDisplay(xmlBaseFileName, False)

drone_counter = 0

drone_positions = ["-1 -1 0.5", "1 -1 0.5", "-1 1 0.5", "1 1 0.5"]
drone_colors = ["0.1 0.9 0.1 1", "0.9 0.1 0.1 1", "0.1 0.1 0.9 1", "0.5 0.5 0.1 1"]

landing_zone_counter = 0
pole_counter = 0

def add_building():
    global scene
    input_gui = bdg.BuildingDataGui()
    input_gui.show()

    # add airport
    if input_gui.building == "a":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_airport(input_gui.position, input_gui.quaternion)
            scene.save_xml(save_filename)
            display.reload_model(save_filename)
    
    # add parking lot
    if input_gui.building == "pl":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_parking_lot(input_gui.position, input_gui.quaternion)
            scene.save_xml(save_filename)
            display.reload_model(save_filename)

    # add hospital
    if input_gui.building == "h":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_hospital(input_gui.position, input_gui.quaternion)
            scene.save_xml(save_filename)
            display.reload_model(save_filename)
    
    # add post-office
    if input_gui.building == "p":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_post_office(input_gui.position, input_gui.quaternion)
            scene.save_xml(save_filename)
            display.reload_model(save_filename)

    # add landing zone
    if input_gui.building == "l":
        global landing_zone_counter
        lz_name = "landing_zone" + str(landing_zone_counter)
        landing_zone_counter += 1
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_landing_zone(lz_name, input_gui.position, input_gui.quaternion)
            scene.save_xml(save_filename)
            display.reload_model(save_filename)

    # add tall landing zone
    if input_gui.building == "lt":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_tall_landing_zone(input_gui.position, input_gui.quaternion)
            scene.save_xml(save_filename)
            display.reload_model(save_filename)
    
    # add pole (oszlop)
    if input_gui.building == "o":
        global pole_counter
        p_name = "pole" + str(pole_counter)
        pole_counter += 1
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_pole(p_name, input_gui.position, input_gui.quaternion)
            scene.save_xml(save_filename)
            display.reload_model(save_filename)
    

    
def add_drone():
    global scene, drone_counter
    if drone_counter < 4:
        drone_name = "drone" + str(drone_counter)
        scene.add_drone(drone_name, drone_positions[drone_counter], drone_colors[drone_counter])
        scene.save_xml(save_filename)
        display.reload_model(save_filename)
        drone_counter += 1


def main():
    display.set_key_b_callback(add_building)
    display.set_key_d_callback(add_drone)
    
    display.run()

if __name__ == '__main__':
    main()