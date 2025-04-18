import numpy as np
import trimesh


class Utility:

    @staticmethod
    def make_trimesh_object(primitive):
        primitive_type = primitive.get_description().type
        mesh = None

        match primitive_type:
            case 'box':
                extents = primitive.get_extents()
                mesh = trimesh.creation.box(extents=[extents[0] * 2.0, extents[1] * 2.0, extents[2] * 2.0])

            case 'sphere':
                radius = primitive.get_radius()
                mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)

            case 'cylinder':
                radius, height = primitive.get_attributes()
                mesh = trimesh.creation.cylinder(radius=radius, height=height * 2.0)

            case 'capsule':
                radius, height = primitive.get_attributes()
                mesh = trimesh.creation.capsule(height=height * 2.0, radius=radius)

            case 'ellipsoid':
                radii = primitive.get_radii()
                ellipsoid_mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
                ellipsoid_mesh.apply_scale(radii)
                mesh = ellipsoid_mesh

            case 'mesh':
                path = primitive.get_stl_path()
                mesh = trimesh.load(path)

        mesh.apply_transform(primitive.get_transform())
        return mesh

    @staticmethod
    def extract_rgba_from_xml(xml_geom):
        rgba_attribute = xml_geom.get('rgba')
        if rgba_attribute != None:
            rgba_attribute = Utility.convert_to_float(rgba_attribute)
        return rgba_attribute

    @staticmethod
    def extract_position_from_xml(xml_geom):
        position_attribute = xml_geom.get('pos')
        if position_attribute != None:
            return Utility.convert_to_float(position_attribute)
        return np.array([0.0, 0.0, 0.0])

    @staticmethod
    def extract_rotation_from_xml(xml_geom):
        euler_angles_attribute = xml_geom.get('euler')
        if euler_angles_attribute != None:
            euler_angles = Utility.convert_to_float(euler_angles_attribute)
            return trimesh.transformations.quaternion_from_euler(euler_angles[0], euler_angles[1], euler_angles[2],
                                                                 'rxyz')

        quaternion_attribute = xml_geom.get('quat')
        if quaternion_attribute != None:
            return Utility.convert_to_float(quaternion_attribute)

        return np.array([1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def convert_to_float(attribute_str):
        return list(map(float, attribute_str.split()))

    @staticmethod
    def get_translation_transform_matrix(translation):
        return trimesh.transformations.translation_matrix(translation)

    @staticmethod
    def get_orientation_transform_matrix(quaternion):
        return trimesh.transformations.quaternion_matrix([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])

    @staticmethod
    def get_parent_transform_from_body(body):
        position = Utility.extract_position_from_xml(body)
        rotation = Utility.extract_rotation_from_xml(body)

        return Utility.get_translation_transform_matrix(position) @ \
            Utility.get_orientation_transform_matrix(rotation)
