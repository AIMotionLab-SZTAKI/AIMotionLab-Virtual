import os

pkg_dir = os.path.dirname(os.path.abspath(__file__))
resource_directory = os.path.join(pkg_dir, "resources")
windflow_data_directory = os.path.join(resource_directory, "windflow_data")
xml_directory = os.path.join(resource_directory, "xml_models")
grouping_element_tags = ["asset", "contact", "deformable", "equality", "tendon", "actuator", "sensor", "keyframe",
                             "custom", "extension", "worldbody"]
__all__ =["resource_directory", "xml_directory"]