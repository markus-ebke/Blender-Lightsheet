# ##### BEGIN GPL LICENSE BLOCK #####
#
#  Lightsheet is a Blender addon for creating fake caustics that can be
#  rendered with Cycles and EEVEE.
#  Copyright (C) 2021  Markus Ebke
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

Setup rays and depsgraph via
- setup_lightsheet_first_ray
- configure_for_trace (is context manager)

Tracing of rays is done via
- trace_scene
- trace_along_chain
- scene_raycast

Helper functions:
- calc_normal
- calc_uv
- get_eval_mesh (uses cache to speed up access)
- cache_clear (use after tracing to cleanup any generated meshes)
"""

from collections import namedtuple
from contextlib import contextmanager
from functools import lru_cache
from math import copysign, exp

import bpy
from mathutils import Vector
from mathutils.geometry import (barycentric_transform, intersect_point_tri,
                                tessellate_polygon)

from lightsheet import material, utils

# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
# organize ray information
Ray = namedtuple("Ray", ["origin", "direction", "color", "chain"])
Ray.__doc__ = """Record ray information for tracing.

    We want to cast a ray from the ray origin (mathutils.Vector) in the given
    direction (mathutils.Vector). Additionally we need the color of the ray
    (3-tuple of floats) and a record of the chain of interaction along the
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
CausticData = namedtuple("CausticData", ["location", "color", "uv", "perp",
                                         "face_index"])
CausticData.__doc__ = """Data for a caustic vertex at the specified location.

    location (mathutils.Vector): location in scene space of final point
    color (3-tuple): color of caustic at the final point
    uv (mathutils.Vector): uv-coordinates on the hit object (may be None)
    perp (mathutils.Vector): direction in scene space pointing perpendicularly
        away from the hit face (away from the front- or the backside depending
        on from which side the face was hit)
    face_index: index of hit face from the mesh of the hit object
    """

# during tracing the evaluated meshes from hit objects will be saved in a
# functools.lru_cache, when we clear the cache we also have to clear the
# generated meshes via obj.to_mesh_clear(), therefore we will record in this
# set for which objects we have generated meshes
meshed_objects = set()


# -----------------------------------------------------------------------------
# Prepare for trace
# -----------------------------------------------------------------------------
def setup_lightsheet_first_ray(lightsheet, color=(1.0, 1.0, 1.0)):
    """Generate a function that returns the ray for a given sheet position."""
    sheet_to_world = lightsheet.matrix_world
    origin = sheet_to_world.to_translation()

    # setup first ray of given vertex coordinate depending on light type
    if lightsheet.parent.data.type == 'SUN':
        # parallel projection along -z axis (local coordinates)
        # note that origin = matrix @ Vector((0, 0, 0))
        target = sheet_to_world @ Vector((0, 0, -1))
        minus_z_axis = (target - origin).normalized()
        minus_z_axis.freeze()  # will use as default value, should be immutable

        # parallel projection in sun direction
        def first_ray(sheet_pos):
            ray_origin = sheet_to_world @ sheet_pos
            return Ray(ray_origin, minus_z_axis, color, tuple())
    else:
        origin.freeze()  # will use as default value, should be immutable

        # project from origin of lightsheet coordinate system
        def first_ray(sheet_pos):
            ray_direction = sheet_to_world @ sheet_pos - origin
            ray_direction.normalize()
            return Ray(origin, ray_direction, color, tuple())

    return first_ray


@contextmanager
def configure_for_trace(context):
    """Contextmanager that configures the depsgraph for caustic tracing."""
    # gather lightsheets and caustics
    ls_coll = utils.get_collection_for_scene(context.scene, "lightsheets",
                                             force=False)
    cau_coll = utils.get_collection_for_scene(context.scene, "caustics",
                                              force=False)

    # hide lightsheets and caustics from raycast, note that hiding selected
    # objects will unselect them
    hidden = []
    for coll in [ls_coll, cau_coll]:
        if coll is None:
            # we have nothing to hide here
            continue

        # hide every object in this collection, but save the previous state
        for obj in coll.objects:
            hidden.append((obj, obj.hide_viewport, obj.select_get()))
            obj.hide_viewport = True

    # make sure that caches are clean
    cache_clear()
    material.cache_clear()

    try:
        yield context.view_layer.depsgraph
    finally:
        # cleanup caches
        cache_clear()
        material.cache_clear()

        # restore original state for hidden lightsheets and caustics
        for obj, view_state, select_state in hidden:
            obj.hide_viewport = view_state
            obj.select_set(select_state)


# -----------------------------------------------------------------------------
# Trace functions
# -----------------------------------------------------------------------------
def trace_scene(ray, depsgraph, max_bounces):
    """Recursively trace a ray through the scene, add data to traced dict."""
    caustic_data = []
    stack = [ray]  # implement recursion via stack
    while stack:
        ray_origin, ray_direction, color, old_chain = stack.pop()
        res = scene_raycast(ray_origin, ray_direction, depsgraph)
        obj, location, smooth_normal, face_normal, face_index, matrix = res

        # if we hit the background we reached the end of the path
        if obj is None:
            continue

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
        for interaction in surface_shader(ray_direction, smooth_normal):
            kind, new_direction, tint = interaction
            # diffuse interactions will show reflective or refractive
            # caustics, non-diffuse interactions will add new rays if the new
            # path is not longer than the bounce limit
            if kind == 'DIFFUSE':
                # add vertex only for reflective and refractive caustics
                if any(s[1] in {'REFLECT', 'REFRACT'} for s in old_chain):
                    # calc uv-coordinates (None if obj has no active uv-layer)
                    uv = calc_uv(obj, depsgraph, face_index,
                                 point=matrix.inverted() @ location)

                    # setup vector perpendicular to face (will be used to
                    # offset caustic), if we hit the backside use reversed
                    # normal
                    if ray_direction.dot(face_normal) < 0:
                        perp = face_normal
                    else:
                        perp = -face_normal

                    # setup data and add to result
                    chain = old_chain + (Link(obj, kind, None),)
                    cdata = CausticData(location, color, uv, perp, face_index)
                    caustic_data.append((chain, cdata))
            elif len(old_chain) < max_bounces:
                # assert new_direction is not None

                # move the starting point a safe distance away from the object
                offset = copysign(1e-5, new_direction.dot(face_normal))
                new_origin = location + offset * face_normal

                # compute volume absorption from previous interaction
                volume = (1.0, 1.0, 1.0)
                if old_chain:
                    previous_link = old_chain[-1]
                    if previous_link.volume_params is not None:
                        # compute transmittance via Beer-Lambert law
                        vol_color, vol_density = previous_link.volume_params
                        ray_length = (location - ray_origin).length
                        volume = [exp(-(1 - val) * vol_density * ray_length)
                                  for val in vol_color]

                # tint the color of the new ray
                # new_color = [c*t*v for (c, t, v) in zip(color, tint, volume)]
                new_color = (color[0] * tint[0] * volume[0],
                             color[1] * tint[1] * volume[1],
                             color[2] * tint[2] * volume[2])

                # extend path of ray
                if new_direction.dot(face_normal) < 0:
                    # consider volume absorption if headed inside
                    new_chain = old_chain + (Link(obj, kind, volume_params),)
                else:
                    # no volume absorption necessary
                    new_chain = old_chain + (Link(obj, kind, None),)

                # trace new ray
                new_ray = Ray(new_origin, new_direction, new_color, new_chain)
                stack.append(new_ray)

    return caustic_data


def trace_along_chain(ray, depsgraph, chain_to_follow):
    """(Re)trace a ray, but only follow the given chain of interactions."""
    # assert len(chain_to_follow) > 0
    # assert all(link.kind != 'DIFFUSE' for link in chain_to_follow[:-1])
    # assert chain_to_follow[-1].kind == 'DIFFUSE'

    # record intermediate locations for visualizing the raypath
    trail = [ray.origin]

    # trace along the given chain of interactions
    for obj, kind_to_follow, _ in chain_to_follow:
        ray_origin, ray_direction, color, old_chain = ray
        res = scene_raycast(ray_origin, ray_direction, depsgraph, obj)
        hit_obj, location, smooth_normal, face_normal, face_index, matrix = res
        trail.append(location)

        # if we missed the object, the caustic will be empty at this position
        if hit_obj is None:
            return (None, trail)
        # assert hit_obj is obj, (hit_obj, obj)

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
        for interaction in surface_shader(ray_direction, smooth_normal):
            if interaction.kind == kind_to_follow:
                found_intersection = interaction
                break

        # if we cannot find this interaction, the caustic will be empty here
        if found_intersection is None:
            return (None, trail)

        kind, new_direction, tint = found_intersection
        if kind == 'DIFFUSE':
            # calculate uv-coordinates (None if obj has no active uv-layer)
            uv = calc_uv(obj, depsgraph, face_index,
                         point=matrix.inverted() @ location)

            # setup vector perpendicular to face (will be used to offset
            # caustic), if we hit the backside use reversed normal
            if ray_direction.dot(face_normal) < 0:
                perp = face_normal
            else:
                perp = -face_normal

            # setup data
            # chain = old_chain + (Link(obj, kind, None),)
            cdata = CausticData(location, color, uv, perp, face_index)
        else:
            # assert new_direction is not None

            # move the starting point a safe distance away from the object
            offset = copysign(1e-5, new_direction.dot(face_normal))
            new_origin = location + offset * face_normal

            # compute volume absorption from previous interaction
            volume = (1.0, 1.0, 1.0)
            if old_chain:
                previous_link = old_chain[-1]
                if previous_link.volume_params is not None:
                    # compute transmittance via Beer-Lambert law
                    vol_color, vol_density = previous_link.volume_params
                    ray_length = (location - ray_origin).length
                    volume = [exp(-(1 - val) * vol_density * ray_length)
                              for val in vol_color]

            # tint the color of the new ray
            # new_color = [c*t*v for (c, t, v) in zip(color, tint, volume)]
            new_color = (color[0] * tint[0] * volume[0],
                         color[1] * tint[1] * volume[1],
                         color[2] * tint[2] * volume[2])

            # extend path of ray
            if new_direction.dot(face_normal) < 0:
                # consider volume absorption if headed inside
                new_chain = old_chain + (Link(obj, kind, volume_params),)
            else:
                # no volume absorption necessary
                new_chain = old_chain + (Link(obj, kind, None),)

            # prepare new ray
            ray = Ray(new_origin, new_direction, new_color, new_chain)

    # check chain
    # assert len(chain) == len(chain_to_follow)
    # for link_followed, link_to_follow in zip(chain, chain_to_follow):
    #     assert link_followed.object == link_to_follow.object
    #     assert link_followed.kind == link_to_follow.kind

    return (cdata, trail)


def scene_raycast(ray_origin, ray_direction, depsgraph, obj=None):
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
    success, location, face_normal, face_index, hit_obj, matrix = result

    # check if hit is valid
    if not success or (obj is not None and hit_obj is not obj):
        return None, None, None, None, None, None

    # calculate smoothed normal (!= true normal if material uses bump mapping)
    # note that normals transform differently than position vectors because
    # scaling can mess up the perpendicularness of the normal, explanation:
    # https://computergraphics.stackexchange.com/q/1502
    normal_local = calc_normal(hit_obj, depsgraph, face_index,
                               point=matrix.inverted() @ location)
    smooth_normal = matrix.inverted().transposed().to_3x3() @ normal_local
    smooth_normal.normalize()

    return hit_obj, location, smooth_normal, face_normal, face_index, matrix


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def calc_normal(obj, depsgraph, face_index, point):
    """Calculate (smooth) normal vector and uv coordinates for given point"""
    # get the face that we hit
    mesh = get_eval_mesh(obj, depsgraph)
    face = mesh.polygons[face_index]

    # do we have to smooth the normals? if no then flat shading
    if not face.use_smooth:
        return face.normal

    # coordinates, normals and uv for the vertices of the face
    vert_co, vert_normal = [], []
    for idx in face.loop_indices:
        loop = mesh.loops[idx]
        vert_co.append(mesh.vertices[loop.vertex_index].co)
        vert_normal.append(loop.normal)

    # tessellate face and find the triangle that contains the point
    # if no triangle is found (is that even possible???) use the last one
    triangles = tessellate_polygon((vert_co,))
    # assert len(triangles) > 0  # will loop => will define tri and v1, v2, v3
    for tri in triangles:
        v1, v2, v3 = vert_co[tri[0]], vert_co[tri[1]], vert_co[tri[2]]
        if intersect_point_tri(point, v1, v2, v3):
            # found triangle, note that tri and v1, v2, v3 are defined now
            break

    # calculate smooth normal at given point via barycentric transformation,
    # note that the interpolated vector is (in general) not normalized, but
    # probably close to length 1 (if normals at vertices don't differ too much)
    n1, n2, n3 = vert_normal[tri[0]], vert_normal[tri[1]], vert_normal[tri[2]]
    normal = barycentric_transform(point, v1, v2, v3, n1, n2, n3)

    # return normal.normalized()
    return normal  # normalized not needed if we transform and normalize later


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
        vert_co.append(mesh.vertices[loop.vertex_index].co)
        vert_uv.append(uv_layer.data[loop.index].uv)

    # tessellate face and find the triangle that contains the point
    # if no triangle is found (is that even possible???) use the last one
    triangles = tessellate_polygon((vert_co,))
    # assert len(triangles) > 0  # will loop => will define tri and v1, v2, v3
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
