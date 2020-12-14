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

Setup rays via
- setup_lightsheet_first_ray

Tracing of rays is done via
- trace_scene_recursive
- scene_raycast
- trace_along_chain
- object_raycast

Helper functions:
- calc_normal
- calc_uv
- get_eval_mesh (uses cache to speed up access)
- cache_clear (use after tracing to cleanup any generated meshes)
"""

from collections import namedtuple
from functools import lru_cache
from math import copysign, exp

import bpy
from mathutils import Color, Vector
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

# during tracing the evaluated meshes from hit objects will be saved in a
# functools.lru_cache, when we clear the cache we also have to clear the
# generated meshes via obj.to_mesh_clear(), therefore we will record in this
# set for which objects we have generated meshes
meshed_objects = set()


# -----------------------------------------------------------------------------
# Ray setup
# -----------------------------------------------------------------------------
def setup_lightsheet_first_ray(lightsheet):
    """Generate a function that returns the ray for a given sheet position."""
    sheet_to_world = lightsheet.matrix_world
    origin = sheet_to_world.to_translation()

    # setup first ray of given vertex coordinate depending on light type
    assert lightsheet.parent is not None and lightsheet.parent.type == 'LIGHT'
    white = Color((1.0, 1.0, 1.0))
    white.freeze()  # will use as default value, therefore should be immutable
    if lightsheet.parent.data.type == 'SUN':
        # parallel projection along -z axis (local coordinates)
        # note that origin = matrix @ Vector((0, 0, 0))
        target = sheet_to_world @ Vector((0, 0, -1))
        minus_z_axis = (target - origin).normalized()
        minus_z_axis.freeze()  # will use as default value, should be immutable

        # parallel projection in sun direction
        def first_ray(sheet_pos):
            ray_origin = sheet_to_world @ Vector(sheet_pos)
            return Ray(ray_origin, minus_z_axis, white, tuple())
    else:
        origin.freeze()  # will use as default value, should be immutable

        # project from origin of lightsheet coordinate system
        def first_ray(sheet_pos):
            ray_direction = sheet_to_world @ Vector(sheet_pos) - origin
            ray_direction.normalize()
            return Ray(origin, ray_direction, white, tuple())

    return first_ray


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
    face = get_eval_mesh(obj, depsgraph).polygons[face_index]

    # get material on hit face with given index
    if obj.material_slots:
        mat = obj.material_slots[face.material_index].material
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

                # calculate uv-coordinates (None if obj has no active uv-layer)
                uv = calc_uv(obj, depsgraph, face_index,
                             point=matrix.inverted() @ location)

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
                    # compute transmittance via Beer-Lambert law
                    volume_color, volume_density = previous_link.volume_params
                    ray_length = (location - ray_origin).length
                    volume = [exp(-(1 - val) * volume_density * ray_length)
                              for val in volume_color]

            # tint the color of the new ray
            mult = [c * t * v for (c, t, v) in zip(color, tint, volume)]
            new_color = Color(mult)

            # extend path of ray
            if new_direction.dot(normal) < 0:
                # consider volume absorption if headed inside
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
    if bpy.app.version < (2, 91, 0):
        result = depsgraph.scene.ray_cast(
            view_layer=depsgraph.view_layer,
            origin=ray_origin,
            direction=ray_direction)
    else:
        # API change: https://developer.blender.org/rBA82ed41ec6324
        result = depsgraph.scene.ray_cast(
            depsgraph=depsgraph,
            origin=ray_origin,
            direction=ray_direction)
    success, location, normal, face_index, obj, matrix = result
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
        face = get_eval_mesh(obj, depsgraph).polygons[face_index]

        # get material on hit face with given index
        if obj.material_slots:
            mat = obj.material_slots[face.material_index].material
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

            # calculate uv-coordinates (None if obj has no active uv-layer)
            uv = calc_uv(obj, depsgraph, face_index,
                         point=matrix.inverted() @ location)

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
                    # compute transmittance via Beer-Lambert law
                    volume_color, volume_density = previous_link.volume_params
                    ray_length = (location - ray_origin).length
                    volume = [exp(-(1 - val) * volume_density * ray_length)
                              for val in volume_color]

            # tint the color of the new ray
            mult = [c * t * v for (c, t, v) in zip(color, tint, volume)]
            new_color = Color(mult)

            # extend path of ray
            if new_direction.dot(normal) < 0:
                # consider volume absorption if headed inside
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

    # raycast
    result = obj.ray_cast(ray_origin_obj, ray_target_obj - ray_origin_obj,
                          depsgraph=depsgraph)
    success, location_obj, normal_obj, face_index = result
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
    if not face.use_smooth:
        return face.normal

    # coordinates, normals and uv for the vertices of the face
    vert_co, vert_normal = [], []
    for idx in face.loop_indices:
        loop = mesh.loops[idx]
        loop_vert = mesh.vertices[loop.vertex_index]
        vert_co.append(loop_vert.co)
        vert_normal.append(loop.normal)

    # tessellate face and find the triangle that contains the point
    # if no triangle is found (is that even possible???) use the last one
    triangles = tessellate_polygon((vert_co,))
    assert len(triangles) > 0  # will loop => will define tri and v1, v2, v3
    for tri in triangles:
        v1, v2, v3 = vert_co[tri[0]], vert_co[tri[1]], vert_co[tri[2]]
        if intersect_point_tri(point, v1, v2, v3):
            # found triangle, note that tri and v1, v2, v3 are defined now
            break

    # calculate smooth normal at given point, interpolate via barycentric
    # transformation, note that interpolated vector might not be normalized
    n1, n2, n3 = vert_normal[tri[0]], vert_normal[tri[1]], vert_normal[tri[2]]
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
    vert_co, vert_uv = [], []
    for idx in face.loop_indices:
        loop = mesh.loops[idx]
        loop_vert = mesh.vertices[loop.vertex_index]
        vert_co.append(loop_vert.co)
        vert_uv.append(uv_layer.data[loop.index].uv)

    # tessellate face and find the triangle that contains the point
    # if no triangle is found (is that even possible???) use the last one
    triangles = tessellate_polygon((vert_co,))
    assert len(triangles) > 0  # will loop => will define tri and v1, v2, v3
    for tri in triangles:
        v1, v2, v3 = vert_co[tri[0]], vert_co[tri[1]], vert_co[tri[2]]
        if intersect_point_tri(point, v1, v2, v3):
            # found triangle, note that tri and v1, v2, v3 are defined now
            break

    # interpolate uv-coordinates via barycentric transformation, note that
    # barycentric transform only works with 3d vectors
    uv1 = vert_uv[tri[0]].to_3d()
    uv2 = vert_uv[tri[1]].to_3d()
    uv3 = vert_uv[tri[2]].to_3d()
    uv = barycentric_transform(point, v1, v2, v3, uv1, uv2, uv3)

    return uv.to_2d()


@lru_cache(maxsize=None)
def get_eval_mesh(obj, depsgraph):
    """Return mesh of object with modifiers, etc. applied"""
    # get mesh of object
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    mesh.calc_normals_split()  # for loop normals

    # record object for clearing later
    meshed_objects.add(obj)
    return mesh


def cache_clear():
    """Clear the cache used by get_eval_mesh and cleanup generated meshes."""
    get_eval_mesh.cache_clear()
    while meshed_objects:
        obj = meshed_objects.pop()
        obj.to_mesh_clear()
