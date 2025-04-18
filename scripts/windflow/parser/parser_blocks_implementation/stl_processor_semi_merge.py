import trimesh

from parser_blocks_interface.istl_processor import ISTLProcessor
from primitives.utility import Utility


class STLProcessorSemiMerge(ISTLProcessor):
    def __init__(self):
        self._meshes = []
        self._is_mesh_already_visited = {}
        self._mesh_groups = []
        self._stl_names = []

    def generate_stl(self, primitives, stl_filename):
        for primitive in primitives:
            mesh = Utility.make_trimesh_object(primitive)
            self._meshes.append(mesh)

        for i in range(len(self._meshes)):
            self._is_mesh_already_visited[i] = False

        for i in range(len(self._meshes)):
            if not self._is_mesh_already_visited[i]:
                group = self._get_intersection_group(i)
                self._mesh_groups.append(group)

        for i, index_group in enumerate(self._mesh_groups):
            mesh_group = [self._meshes[i] for i in index_group]
            merged_mesh = trimesh.boolean.union(mesh_group)

            stl_name = stl_filename + '_group' + str(i) + '.stl'
            self._stl_names.append(stl_name)
            merged_mesh.export(stl_name)

        return self._stl_names

    def _get_intersection_group(self, target_mesh_index):
        self._is_mesh_already_visited[target_mesh_index] = True

        group = [target_mesh_index]
        group_index = 0
        while group_index != len(group):
            current_mesh_index = group[group_index]

            for i in range(len(self._meshes)):
                if self._is_mesh_already_visited[i] or (i == current_mesh_index):
                    continue

                do_meshes_intersect_by_bounding_box = self._do_meshes_intersect_by_bounding_box(self._meshes[i],
                                                                                                self._meshes[
                                                                                                    current_mesh_index])
                if not do_meshes_intersect_by_bounding_box:
                    continue

                do_meshes_intersect_precisely = self._do_meshes_intersect_precisely(self._meshes[i],
                                                                                    self._meshes[current_mesh_index])
                if not do_meshes_intersect_precisely:
                    continue

                self._is_mesh_already_visited[i] = True
                group.append(i)

            group_index += 1

        return group

    def _do_meshes_intersect_by_bounding_box(self, mesh1, mesh2):
        mesh1_box_min = mesh1.bounds[0]
        mesh1_box_max = mesh1.bounds[1]
        mesh2_box_min = mesh2.bounds[0]
        mesh2_box_max = mesh2.bounds[1]

        x_overlap = (mesh1_box_max[0] >= mesh2_box_min[0]) and (mesh2_box_max[0] >= mesh1_box_min[0])
        y_overlap = (mesh1_box_max[1] >= mesh2_box_min[1]) and (mesh2_box_max[1] >= mesh1_box_min[1])
        z_overlap = (mesh1_box_max[2] >= mesh2_box_min[2]) and (mesh2_box_max[2] >= mesh1_box_min[2])

        return x_overlap and y_overlap and z_overlap

    def _do_meshes_intersect_precisely(self, mesh1, mesh2):
        intersection = trimesh.boolean.intersection([mesh1, mesh2])
        return not intersection.is_empty
