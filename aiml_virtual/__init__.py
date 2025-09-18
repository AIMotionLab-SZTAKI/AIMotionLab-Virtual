import os

pkg_dir = os.path.dirname(os.path.abspath(__file__))
resource_directory = os.path.join(pkg_dir, "resources")
xml_directory = os.path.join(resource_directory, "xml_models")
airflow_data = os.path.join(resource_directory, "airflow_data")
skyc_folder = os.path.join(resource_directory, "skyc")
airflow_luts_pressure = os.path.join(airflow_data, "airflow_luts_pressure")
airflow_luts_velocity = os.path.join(airflow_data, "airflow_luts_velocity")
grouping_element_tags = ["asset", "contact", "deformable", "equality", "tendon", "actuator", "sensor", "keyframe",
                             "custom", "extension", "worldbody"]
__all__ =["resource_directory", "xml_directory", "airflow_luts_pressure", "airflow_luts_velocity"]