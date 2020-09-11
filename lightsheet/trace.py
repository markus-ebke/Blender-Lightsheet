# ##### BEGIN GPL LICENSE BLOCK #####
#
#  Lightsheet is a Blender addon for creating fake caustics that can be
#  rendered with Cycles and EEVEE.
#  Copyright (C) 2020  Markus Ebke
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ##### END GPL LICENSE BLOCK #####
"""Functions that are used for tracing rays and creating the caustic bmeshes.

Tracing of rays is done via
- trace_lightsheet
- trace_scene_recursive
- scene_raycast

Helper functions:
- calc_normal_and_uv
- get_eval_mesh (uses cache to speed up access)
- new_caustic_bmesh

Note that after tracing you should cleanup the generated meshes via
for obj in trace.meshes_cache:
    obj.to_mesh_clear()
trace.meshes_cache.clear()
"""

from collections import defaultdict, namedtuple
from math import copysign, exp

import bmesh
import bpy
from mathutils import Color, Vector
from mathutils.geometry import (barycentric_transform, intersect_point_tri,
                                tessellate_polygon)

from lightsheet import material

# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
# organize ray information
Ray = namedtuple("Ray", ["origin", "direction", "tint", "path"])
Ray.__doc__ = """Record ray information for tracing.

    We want to cast a ray from the ray origin (mathutils.Vector) in the given
    direction (mathutils.Vector). Further we need the tint of the ray
    (mathutils.Color) and a record of the chain of interaction along the
    already taken raypath (a tuple of Link instances).
    """

# organize links in interaction chains
Link = namedtuple("Link", ["object", "kind", "volume_params"])
Link.__doc__ = """One link in the chain of interactions of a ray.

    Includes the object that was hit, the kind of interaction as a string (one
    of 'DIFFUSE', 'REFLECT', 'REFRACT', 'TRANSPARENT') and any volume
    parameters as a tuple.
    """

# cache evaluated meshes for faster access, meshes_cache is a dict of form
# {object: mesh datablock of evaluated object}
meshes_cache = dict()


# -----------------------------------------------------------------------------
# Trace functions
# -----------------------------------------------------------------------------
def trace_lightsheet(lightsheet, depsgraph, max_bounces):
    """Trace rays from lighsheet and return all caustics coordinates in dict"""
    # get lightsheet mesh
    ls_bmesh = bmesh.new(use_operators=False)
    ls_bmesh.from_object(lightsheet, depsgraph)

    # get lightsheet coordinate system
    lightsheet_eval = lightsheet.evaluated_get(depsgraph)
    matrix = lightsheet_eval.matrix_world.copy()
    origin = matrix.to_translation()

    # setup first ray of given vertex coordinate depending on light type
    assert lightsheet.parent is not None and lightsheet.parent.type == 'LIGHT'
    if lightsheet.parent.data.type == 'SUN':
        # parallel projection along -z axis (local coordinates)
        # note that origin = matrix @ Vector((0, 0, 0))
        target = matrix @ Vector((0, 0, -1))
        minus_z_axis = (target - origin).normalized()

        def first_ray_coords(vert_co):
            return matrix @ vert_co, minus_z_axis
    else:
        # project from origin of lightsheet coordinate system
        def first_ray_coords(vert_co):
            direction = matrix @ vert_co - origin
            return origin, direction.normalized()

    ls_id = ls_bmesh.verts.layers.int["id"]
    # path_bm = {path: (caustic bmesh, normal dict, color dict, uv dict)}
    path_bm = defaultdict(new_caustic_bmesh)
    for ls_vert in ls_bmesh.verts:
        # setup first ray
        ray_origin, ray_direction = first_ray_coords(ls_vert.co)
        ray = Ray(ray_origin, ray_direction, Color((1.0, 1.0, 1.0)), tuple())

        # trace ray
        vert_id = ls_vert[ls_id]
        trace_scene_recursive(ray, vert_id, depsgraph, max_bounces, path_bm)

    return path_bm


def trace_scene_recursive(ray, vert_id, depsgraph, max_bounces, path_bm):
    """Recursively trace a ray, add interaction to path_bm dict"""
    ray_origin, ray_direction, color, old_path = ray
    result = scene_raycast(ray_origin, ray_direction, depsgraph)
    obj, location, normal, uv, face_index = result

    # if we hit the background we reached the end of the path
    if obj is None:
        return

    # not the end of the path, right???
    assert location is not None and normal is not None, (location, normal)

    # get hit face
    obj_mesh = get_eval_mesh(obj, depsgraph)
    face = obj_mesh.polygons[face_index]

    # get material on hit face with given index
    if obj.material_slots:
        mat_idx = face.material_index
        mat = obj.material_slots[mat_idx].material
    else:
        mat = None

    # determine surface interactions from material (i.e. does the ray
    # continue and if yes, how?)
    surface_shader, volume_params = material.get_material_shader(mat)
    for kind, new_direction, tint in surface_shader(ray_direction, normal):
        # diffuse interactions will show reflective or refractive caustics,
        # non-diffuse interactions will add new rays if the new path is not
        # longer than the bounce limit
        if kind == 'DIFFUSE':
            # add vertex to caustic only for reflective and refractive caustics
            if any(step[1] in {'REFLECT', 'REFRACT'} for step in old_path):
                # get the caustic bmesh for this chain of interactions
                caustic_key = tuple(old_path + (Link(obj, kind, None),))
                bm, normal_dict, color_dict, uv_dict = path_bm[caustic_key]

                # create caustic vertex in front of object
                offset = copysign(1e-4, -ray_direction.dot(face.normal))
                position = location + offset * face.normal
                vert = bm.verts.new(position)

                # set vertex id so that we can find its source vertex later
                id_layer = bm.verts.layers.int["id"]
                vert[id_layer] = vert_id

                # record normal from hit face, vertex color from path and
                # uv-coordinate from target object, we will set them later when
                # creating faces
                normal_dict[vert_id] = face.normal
                color_dict[vert_id] = color
                if uv is not None:
                    uv_dict[vert_id] = uv
        elif len(old_path) < max_bounces:
            assert new_direction is not None
            # move the starting point a safe distance away from the object
            offset = copysign(1e-5, new_direction.dot(face.normal))
            new_origin = location + offset * face.normal

            # compute volume absorption from previous interaction
            volume = (1.0, 1.0, 1.0)
            if old_path:
                previous_link = old_path[-1]
                if previous_link.volume_params is not None:
                    assert previous_link.kind in {'REFRACT', 'TRANSPARENT'}
                    # compute transmittance via Beer-Lambert law
                    volume_color, volume_density = previous_link.volume_params
                    ray_length = (location - ray_origin).length
                    volume = (exp(-(1 - val) * volume_density * ray_length)
                              for val in volume_color)

            # tint the color of the new ray
            mult = (c * t * v for (c, t, v) in zip(color, tint, volume))
            new_color = Color(mult)

            # extend path of ray
            headed_inside = new_direction.dot(normal) < 0
            if kind in {'REFRACT', 'TRANSPARENT'} and headed_inside:
                # consider volume absorption
                new_path = old_path + (Link(obj, kind, volume_params),)
            else:
                # no volume absorption necessary
                new_path = old_path + (Link(obj, kind, None),)

            # trace new ray
            new_ray = Ray(new_origin, new_direction, new_color, new_path)
            trace_scene_recursive(new_ray, vert_id, depsgraph, max_bounces,
                                  path_bm)


def scene_raycast(ray_origin, ray_direction, depsgraph):
    """Raycast all visible objects in the set scene and view layer"""
    # cast the ray
    scene = depsgraph.scene
    if bpy.app.version < (2, 91, 0):
        hit, position, normal, face_index, obj, matrix = scene.ray_cast(
            view_layer=depsgraph.view_layer,
            origin=ray_origin,
            direction=ray_direction)
    else:
        # API change: https://developer.blender.org/rBA82ed41ec6324
        hit, position, normal, face_index, obj, matrix = scene.ray_cast(
            depsgraph=depsgraph,
            origin=ray_origin,
            direction=ray_direction)

    if not hit:
        # no hit
        return None, None, None, None, None

    # calculate (smooth) normal
    pos_obj = matrix.inverted() @ position
    normal_obj, uv = calc_normal_and_uv(obj, depsgraph, face_index, pos_obj)
    normal = (matrix @ (pos_obj + normal_obj) - position).normalized()

    return obj, position, normal, uv, face_index


def calc_normal_and_uv(obj, depsgraph, face_index, point):
    """Calculate (smooth) normal vector and uv coordinates for given point"""
    # get the face that we hit
    mesh = get_eval_mesh(obj, depsgraph)
    face = mesh.polygons[face_index]

    # do we have to smooth the normals? do we have uv coordinates?
    use_smooth = face.use_smooth
    uv_layer = mesh.uv_layers.active

    # result for flat shading and no uvs is obvious
    if not use_smooth and uv_layer is None:
        return (face.normal, None)

    # coordinates, normals and uv for the vertices of the face
    vertex_coords, vertex_normals, vertex_uvs = [], [], []
    for idx in face.loop_indices:
        loop = mesh.loops[idx]
        loop_vert = mesh.vertices[loop.vertex_index]
        vertex_coords.append(loop_vert.co)

        if use_smooth:
            vertex_normals.append(loop.normal)

        if uv_layer is not None:
            vertex_uvs.append(uv_layer.data[loop.index].uv)

    # tessellate face and find the triangle that contains the point
    # if no triangle is found (is that even possible???) use the last one
    triangles = tessellate_polygon((vertex_coords,))
    assert len(triangles) > 0  # will loop => will define tri and v1, v2, v3
    for tri in triangles:
        v1, v2, v3 = (vertex_coords[idx] for idx in tri)
        if intersect_point_tri(point, v1, v2, v3):
            # found triangle
            # note that tri and v1, v2, v3 are defined now
            break

    # calculate normal at given point
    if use_smooth:
        # smooth shading => interpolate normal via barycentric transformation
        n1, n2, n3 = (vertex_normals[idx] for idx in tri)
        normal = barycentric_transform(point, v1, v2, v3, n1, n2, n3)
        normal = normal.normalized()  # length not invariant, recalculate
    else:
        # flat shading => face normal
        assert not vertex_normals, vertex_normals
        normal = face.normal

    # calculate uv coordinate
    if uv_layer is not None:
        # interpolate uv-coordinates via barycentric transformation
        # note that barycentric transform only works with 3d vectors
        uv1, uv2, uv3 = (vertex_uvs[idx].to_3d() for idx in tri)
        uv = barycentric_transform(point, v1, v2, v3, uv1, uv2, uv3)
        uv = uv.to_2d()
    else:
        # no uv-layer => don't calculate uvs
        assert not vertex_uvs, vertex_uvs
        uv = None

    return (normal, uv)


def get_eval_mesh(obj, depsgraph):
    """Return mesh of object with modifiers, etc. applied"""
    # check cache
    mesh = meshes_cache.get(obj)
    if mesh is not None:
        return mesh

    # get mesh of object
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    mesh.calc_normals_split()  # for loop normals

    meshes_cache[obj] = mesh
    return mesh


def new_caustic_bmesh():
    """Create empty bmesh with data layers used for caustics"""
    bm = bmesh.new()

    # create id layer = vertex index from source mesh
    bm.verts.layers.int.new("id")

    # create uv layer for caustic squeeze = ratio of source face area to
    # projected face area
    bm.loops.layers.uv.new("Caustic Squeeze")

    # create vertex color layer for caustic tint
    bm.loops.layers.color.new("Caustic Tint")

    # create uv layer for transplanted coordinates
    bm.loops.layers.uv.new("UVMap")

    # after we have created faces we need to ensure that the face normal
    # points in the right direction, in trace we will therefore record the
    # normal at each vertex and later compare the face normal with the average
    # vertex normal
    normal_dict = dict()

    # since we can apply vertex colors and uv coordinates only to loops and
    # not to vertices directly, we will record the colors and coordinates in
    # these dicts and set them later when we iterate over the faces of the
    # bmesh
    color_dict = dict()
    uv_dict = dict()

    return bm, normal_dict, color_dict, uv_dict
