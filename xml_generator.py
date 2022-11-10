import xml.etree.cElementTree as ET

class SceneXmlGenerator:

    def __init__(self, base_scene_filename):
        self.root = ET.Element("mujoco")
        ET.SubElement(self.root, "include", file=base_scene_filename)
        self.worldbody = ET.SubElement(self.root, "worldbody")
        self.actuator = ET.SubElement(self.root, "actuator")
        self.sensor = ET.SubElement(self.root, "sensor")
        self.parking_lot = None
        self.airport = None
        self.hospital = None
        self.post_office = None
        self.tall_landing_zone = None


    def add_airport(self, pos, quat=None):
        if self.airport is None:

            tag = "geom"
            name = "airport"
            size = "0.105 0.105 .05"
            type = "plane"
            material = "mat-airport"

            if quat is None:
                self.airport = ET.SubElement(self.worldbody, tag, name=name, pos=pos, size=size, type=type, material=material)
            else:
                self.airport = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat, size=size, type=type, material=material)
            return self.airport
        else:
            print("[SceneXmlGenerator] Airport already added")


    def add_parking_lot(self, pos, quat=None):
        if self.parking_lot is None:

            tag = "geom"
            name = "parking_lot"
            size = "0.105 0.115 .05"
            type = "plane"
            material = "mat-parking_lot"

            if quat is None:
                self.parking_lot = ET.SubElement(self.worldbody, tag, name=name, pos=pos, size=size, type=type, material=material)
            else:
                self.parking_lot = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat, size=size, type=type, material=material)
            return self.parking_lot
        else:
            print("[SceneXmlGenerator] Parking lot already added")
    

    def add_pole(self, name, pos, quat=None):
        tag = "body"
        if quat is None:
            pole = ET.SubElement(self.worldbody, tag, name=name, pos=pos)
        else:
            pole = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat)
        
        ET.SubElement(pole, "geom", {"class": "pole_top"})
        ET.SubElement(pole, "geom", {"class": "pole_bottom1"})
        ET.SubElement(pole, "geom", {"class": "pole_bottom2"})

        return pole


    def add_drone(self, name, pos, color):
        site_name = name + "_COG"

        drone = ET.SubElement(self.worldbody, "body", name=name, pos=pos)
        ET.SubElement(drone, "inertial", pos="0 0 0", diaginertia="1.4e-5 1.4e-5 2.17e-5", mass="0.028")
        ET.SubElement(drone, "joint", type="free")
        ET.SubElement(drone, "geom", name=name, type="mesh", pos="0 0 0", mesh="drone", rgba=color)
        ET.SubElement(drone, "site", name=site_name, pos="0 0 0")

        ET.SubElement(self.actuator, "general", site=site_name, gear=" 0 0 1 0 0 0", ctrllimited="true", ctrlrange="0 0.64")
        ET.SubElement(self.actuator, "general", site=site_name, gear=" 0 0 0 1 0 0", ctrllimited="true", ctrlrange="-0.01 0.01")
        ET.SubElement(self.actuator, "general", site=site_name, gear=" 0 0 0 0 1 0", ctrllimited="true", ctrlrange="-0.01 0.01")
        ET.SubElement(self.actuator, "general", site=site_name, gear=" 0 0 0 0 0 1", ctrllimited="true", ctrlrange="-0.01 0.01")

        ET.SubElement(self.sensor, "gyro", site=site_name)

        return drone


    def add_hospital(self, pos, quat=None):
        name = "hospital"
        if self.hospital is None:
            tag = "body"
            if quat is None:
                self.hospital = ET.SubElement(self.worldbody, tag, name=name, pos=pos)
            else:
                self.hospital = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat)

            ET.SubElement(self.hospital, "geom", name=name, type="box", pos="0 0 0.445", size="0.1275 0.13 0.445", material="mat-hospital")

            return self.hospital
        else:
            print("[SceneXmlGenerator] Hospital already added")


    def add_post_office(self, pos, quat=None):
        name = "post_office"
        if self.post_office is None:
            tag = "body"
            if quat is None:
                self.post_office = ET.SubElement(self.worldbody, tag, name=name, pos=pos)
            else:
                self.post_office = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat)

            ET.SubElement(self.post_office, "geom", name=name, type="box", pos="0 0 0.205", size="0.1275 0.1275 0.205", material="mat-post_office")

            return self.post_office
        else:
            print("[SceneXmlGenerator] Post office already added")


    def add_landing_zone(self, name, pos, quat=None):
        tag = "body"
        if quat is None:
            landing_zone = ET.SubElement(self.worldbody, tag, name=name, pos=pos)
        else:
            landing_zone = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat)
        
        ET.SubElement(landing_zone, "geom", {"class" : "landing_zone"})

        return landing_zone


    def add_tall_landing_zone(self, pos, quat):
        if self.tall_landing_zone is None:
            name = "tall_landing_zone"
            
            self.tall_landing_zone = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat)

            ET.SubElement(self.tall_landing_zone, "geom", name=name, type="box", pos="0 0 0.0925", size="0.105 0.105 0.0925", rgba="0.8 0.8 0.8 1.0", material="mat-sztaki")

            return self.tall_landing_zone

        else:
            print("[SceneXmlGenerator] Tall Sztaki landing zone already added")


    def save_xml(self, file_name):
        
        tree = ET.ElementTree(self.root)
        ET.indent(tree, space="\t", level=0)
        tree.write(file_name)


"""
scene = SceneXmlGenerator("scene.xml")

scene.add_airport(pos="0.5 -1.2 0.0025")
scene.add_parking_lot(pos="-0.5 1.2 0.0025")

scene.add_drone(name="drone0", pos="0 -1 0.2", color="0.1 0.9 0.1 1")
scene.add_drone(name="drone1", pos="0 1 0.2", color="0.1 0.9 0.1 1")

scene.add_pole(name="pole1", pos="0.25 0.25 0")
scene.add_pole(name="pole2", pos="-0.25 0.25 0")
scene.add_pole(name="pole3", pos="0.3 -0.3 0", quat="0.924 0 0 0.383")
scene.add_pole(name="pole4", pos="-0.3 -0.3 0", quat="0.924 0 0 0.383")

scene.add_hospital(pos="-1 1 0")
scene.add_post_office(pos="1 1.255 0")

scene.add_landing_zone(name="landing_zone1", pos="-1.2 -0.7 0", quat="0.924 0 0 0.383")
scene.add_landing_zone(name="landing_zone2", pos="-1 -0.9 0", quat="0.924 0 0 0.383")
scene.add_landing_zone(name="landing_zone3", pos="-0.8 -1.1 0", quat="0.924 0 0 0.383")
scene.add_landing_zone(name="landing_zone4", pos="-0.6 -1.3 0", quat="0.924 0 0 0.383")

scene.add_tall_landing_zone(pos="0 0 0", quat="1 0 0 0")

scene.save_xml("first_xml.xml")
"""