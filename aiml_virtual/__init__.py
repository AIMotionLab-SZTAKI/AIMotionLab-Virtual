import os

pkg_dir = os.path.dirname(os.path.abspath(__file__))
resource_directory = os.path.join(pkg_dir, "resources")
xml_directory = os.path.join(resource_directory, "xml_models")
__all__ =["resource_directory", "xml_directory"]