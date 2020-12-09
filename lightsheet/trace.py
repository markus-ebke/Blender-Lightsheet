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
- trace_scene_recursive
- scene_raycast
- trace_along_chain
- object_raycast

Helper functions:
- calc_normal
- calc_uv
- get_eval_mesh (uses cache to speed up access)

Note that after tracing you should cleanup the generated meshes via
for obj in trace.meshes_cache:
    obj.to_mesh_clear()
trace.meshes_cache.clear()
"""

from collections import namedtuple
from math import copysign, exp

import bpy
from mathutils import Color
from mathutils.geometry import (barycentric_transform, intersect_point_tri,
                                tessellate_polygon)

from lightsheet import material

# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
# organize ray information
Ray = namedtuple("Ray", ["origin", "direction", "tint", "chain"])
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

# organize caustic vertex info
CausticData = namedtuple("CausticData", ["location", "color", "uv", "normal",
                                         "face_index"])
CausticData.__doc__ = """Data for a caustic vertex at the specified location.

    location (mathutils.Vector): location in scene space of final point
    color (mathutils.Color): color of caustic at the final point
    uv (mathutils.Vector): uv-coordinates on the hit object (may be None)
    normal (mathutils.Vector): points away from the hit face (away from front
        or backside of face depending on the side that was hit)
    face_index: index of hit face from the mesh of the hit object
    """

# cache evaluated meshes for faster access, meshes_cache is a dict of form
# {object: mesh datablock of evaluated object}
meshes_cache = dict()


# -----------------------------------------------------------------------------
# Trace functions
# -----------------------------------------------------------------------------
def trace_scene_recursive(ray, sheet_pos, depsgraph, max_bounces, traced):
    """Recursively trace a ray through the scene, add data to traced dict."""
    ray_origin, ray_direction, color, old_chain = ray
    obj, location, normal, face_index, matrix = scene_raycast(
        ray_origin, ray_direction, depsgraph)

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
            if any(step[1] in {'REFLECT', 'REFRACT'} for step in old_chain):
                # get the caustic data mapping for this chain of interactions
                caustic_key = old_chain + (Link(obj, kind, None),)
                sheet_to_data = traced[caustic_key]

                # calculate uv-coordinates if any
                if obj.data.uv_layers.active:
                    location_obj = matrix.inverted() @ location
                    uv = calc_uv(obj, depsgraph, face_index, location_obj)
                else:
                    uv = None

                # setup vector perpendicular to face (will be used to offset
                # caustic), if we hit the backside use the reversed normal
                normal = face.normal
                if ray_direction.dot(normal) > 0:
                    normal = -normal

                # set data
                cdata = CausticData(location, color, uv, normal, face_index)
                sheet_to_data[sheet_pos] = cdata
        elif len(old_chain) < max_bounces:
            assert new_direction is not None
            # move the starting point a safe distance away from the object
            offset = copysign(1e-5, new_direction.dot(face.normal))
            new_origin = location + offset * face.normal

            # compute volume absorption from previous interaction
            volume = (1.0, 1.0, 1.0)
            if old_chain:
                previous_link = old_chain[-1]
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
                new_chain = old_chain + (Link(obj, kind, volume_params),)
            else:
                # no volume absorption necessary
                new_chain = old_chain + (Link(obj, kind, None),)

            # trace new ray
            new_ray = Ray(new_origin, new_direction, new_color, new_chain)
            trace_scene_recursive(new_ray, sheet_pos, depsgraph, max_bounces,
                                  traced)


def scene_raycast(ray_origin, ray_direction, depsgraph):
    """Raycast all visible objects in the set scene, return hit object info."""
    # cast the ray
    scene = depsgraph.scene
    if bpy.app.version < (2, 91, 0):
        success, location, normal, face_index, obj, matrix = scene.ray_cast(
            view_layer=depsgraph.view_layer,
            origin=ray_origin,
            direction=ray_direction)
    else:
        # API change: https://developer.blender.org/rBA82ed41ec6324
        success, location, normal, face_index, obj, matrix = scene.ray_cast(
            depsgraph=depsgraph,
            origin=ray_origin,
            direction=ray_direction)

    if not success:
        # no hit
        return None, None, None, None, None

    # calculate (smooth) normal
    location_obj = matrix.inverted() @ location
    normal_obj = calc_normal(obj, depsgraph, face_index, location_obj)
    normal = (matrix @ (location_obj + normal_obj) - location).normalized()

    return obj, location, normal, face_index, matrix


def trace_along_chain(ray, depsgraph, chain_to_follow):
    """(Re)trace a ray, but only follow the given chain of interactions."""
    assert len(chain_to_follow) > 0
    assert all(link.kind != 'DIFFUSE' for link in chain_to_follow[:-1])
    assert chain_to_follow[-1].kind == 'DIFFUSE'

    # record intermediate locations for visualizing the raypath
    trail = [ray.origin]

    # trace along the given chain of interactions
    for obj, kind_to_follow, _ in chain_to_follow:
        ray_origin, ray_direction, color, old_chain = ray
        location, normal, face_index, matrix = object_raycast(
            obj, ray_origin, ray_direction, depsgraph)
        trail.append(location)

        # if we missed the object, the caustic will be empty at this position
        if location is None:
            return (None, trail)

        # get hit face
        obj_mesh = get_eval_mesh(obj, depsgraph)
        face = obj_mesh.polygons[face_index]

        # get material on hit face with given index
        if obj.material_slots:
            mat_idx = face.material_index
            mat = obj.material_slots[mat_idx].material
        else:
            mat = None

        # determine surface interactions from material, but focus only on the
        # interaction that follows the path of the caustic
        surface_shader, volume_params = material.get_material_shader(mat)
        found_intersection = None
        for interaction in surface_shader(ray_direction, normal):
            if interaction.kind == kind_to_follow:
                found_intersection = interaction
                break

        # if we cannot find this interaction, the caustic will be empty here
        if found_intersection is None:
            return (None, trail)

        kind, new_direction, tint = found_intersection
        if kind == 'DIFFUSE':
            caustic_key = old_chain + (Link(obj, kind, None),)

            # calculate uv-coordinates if any
            if obj.data.uv_layers.active:
                location_obj = matrix.inverted() @ location
                uv = calc_uv(obj, depsgraph, face_index, location_obj)
            else:
                uv = None

            # setup vector perpendicular to face (will be used to offset
            # caustic), if we hit the backside use the reversed normal
            normal = face.normal
            if ray_direction.dot(normal) > 0:
                normal = -normal
        else:
            assert new_direction is not None
            # move the starting point a safe distance away from the object
            offset = copysign(1e-5, new_direction.dot(face.normal))
            new_origin = location + offset * face.normal

            # compute volume absorption from previous interaction
            volume = (1.0, 1.0, 1.0)
            if old_chain:
                previous_link = old_chain[-1]
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
                new_chain = old_chain + (Link(obj, kind, volume_params),)
            else:
                # no volume absorption necessary
                new_chain = old_chain + (Link(obj, kind, None),)

            # prepare new ray
            ray = Ray(new_origin, new_direction, new_color, new_chain)

    # check chain
    assert len(caustic_key) == len(chain_to_follow)
    for link_followed, link_to_follow in zip(caustic_key, chain_to_follow):
        assert link_followed.object == link_to_follow.object
        assert link_followed.kind == link_to_follow.kind

    cdata = CausticData(location, color, uv, normal, face_index)
    return (cdata, trail)


def object_raycast(obj, ray_origin, ray_direction, depsgraph):
    """Raycast only one object, return hit info."""
    # transform to object space
    matrix = obj.evaluated_get(depsgraph).matrix_world.copy()
    matrix_inv = matrix.inverted()
    ray_origin_obj = matrix_inv @ ray_origin
    ray_target_obj = matrix_inv @ (ray_origin + ray_direction)
    ray_direction_obj = ray_target_obj - ray_origin_obj

    # raycast
    success, location_obj, normal_obj, face_index, = obj.ray_cast(
        ray_origin_obj, ray_direction_obj, depsgraph=depsgraph)

    if not success:
        return None, None, None, None

    # calculate (smooth) normal
    normal_obj = calc_normal(obj, depsgraph, face_index, location_obj)
    location = matrix @ location_obj
    normal = (matrix @ (location_obj + normal_obj) - location).normalized()

    return location, normal, face_index, matrix


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def calc_normal(obj, depsgraph, face_index, point):
    """Calculate (smooth) normal vector and uv coordinates for given point"""
    # get the face that we hit
    mesh = get_eval_mesh(obj, depsgraph)
    face = mesh.polygons[face_index]

    # # do we have to smooth the normals? if no then flat shading
    use_smooth = face.use_smooth
    if not use_smooth:
        return face.normal

    # coordinates, normals and uv for the vertices of the face
    vertex_coords, vertex_normals = [], []
    for idx in face.loop_indices:
        loop = mesh.loops[idx]
        loop_vert = mesh.vertices[loop.vertex_index]
        vertex_coords.append(loop_vert.co)
        vertex_normals.append(loop.normal)

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

    # calculate smooth normal at given point, interpolate via barycentric
    # transformation, note that interpolated vector might not be normalized
    n1, n2, n3 = (vertex_normals[idx] for idx in tri)
    normal = barycentric_transform(point, v1, v2, v3, n1, n2, n3)

    return normal.normalized()


def calc_uv(obj, depsgraph, face_index, point):
    """Calculate (smooth) normal vector and uv coordinates for given point"""
    # get the face that we hit
    mesh = get_eval_mesh(obj, depsgraph)

    # do we have-uv coordinates? if no then we are done
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        return None

    # coordinates, normals and uv for the vertices of the face
    face = mesh.polygons[face_index]
    vertex_coords, vertex_uvs = [], []
    for idx in face.loop_indices:
        loop = mesh.loops[idx]
        loop_vert = mesh.vertices[loop.vertex_index]
        vertex_coords.append(loop_vert.co)
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

    # interpolate uv-coordinates via barycentric transformation, note that
    # barycentric transform only works with 3d vectors
    uv1, uv2, uv3 = (vertex_uvs[idx].to_3d() for idx in tri)
    uv = barycentric_transform(point, v1, v2, v3, uv1, uv2, uv3)

    return uv.to_2d()


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
