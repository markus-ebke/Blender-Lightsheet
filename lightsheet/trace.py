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

Several of the functions use scene, view_layer and depsgraph, so to shorten
function arguments they are global variables. Set/unset them via
- setup(scene, view_layer, depsgraph)
- cleanup()

Tracing of rays works via
- trace_lightsheet
- trace_scene_recursive

Helper functions:
- calc_normal_and_uv
- get_eval_mesh
- new_caustic_bmesh (uses cache to speed up access)
"""

from collections import defaultdict, namedtuple
from math import copysign

import bmesh
from mathutils import Color, Vector
from mathutils.geometry import (barycentric_transform, intersect_point_tri,
                                tessellate_polygon)

from lightsheet import material

print("lightsheet trace.py")

# use namedtuple to organize ray information
Ray = namedtuple("Ray", ["origin", "direction", "tint", "path"])
Ray.__doc__ = """Record ray information for tracing.

    We want to cast a ray from the ray origin (mathutils.Vector) in the given
    direction (mathutils.Vector). Further we need the tint of the ray
    (mathutils.Color) and a record of the chain of interaction along the
    already taken raypath (a tuple of Link instances).
    """

Link = namedtuple("Link", ["object", "kind"])
Link.__doc__ = """One link in the chain of interactions of a ray.

    Includes the object that was hit, the kind of interaction as a string (one
    of "diffuse", "reflect", "refract", "transparent").
    """


# -----------------------------------------------------------------------------
# Handle global variables
# -----------------------------------------------------------------------------
# simplify function calls by using scene, view_layer and depsgraph as globals
_SCENE = None
_VIEW_LAYER = None
_DEPSGRAPH = None

# cache evaluated meshes for faster access, meshes_cache is a dict of form
# {object: mesh datablock of evaluated object}
meshes_cache = None


def setup(scene, view_layer, depsgraph):
    """Setup empty caches and set global variables for trace"""
    global _SCENE, _VIEW_LAYER, _DEPSGRAPH, meshes_cache

    # setup empty caches
    meshes_cache = dict()
    material.materials_cache = dict()

    # setup globals
    _SCENE = scene
    _VIEW_LAYER = view_layer
    _DEPSGRAPH = depsgraph


def cleanup():
    """Cleanup caches and reset global variables for trace"""
    global meshes_cache, _SCENE, _VIEW_LAYER, _DEPSGRAPH

    # cleanup generated meshes
    for obj in meshes_cache:
        obj.to_mesh_clear()

    # cleanup caches
    meshes_cache.clear()
    material.materials_cache.clear()

    # reset globals
    _SCENE = None
    _VIEW_LAYER = None
    _DEPSGRAPH = None


# -----------------------------------------------------------------------------
# Trace functions
# -----------------------------------------------------------------------------
def trace_lightsheet(ls_mesh, matrix, max_bounces, projection_mode="point"):
    """Trace rays from lighsheet and return all caustics coordinates in dict"""
    origin = matrix.to_translation()

    # seutp first ray of given vertex coordinate depending on projection mode
    assert projection_mode in {"point", "parallel"}
    if projection_mode == "point":
        # project from origin of lightsheet coordinate system
        def first_ray_coords(vert_co):
            direction = matrix @ vert_co - origin
            return origin, direction.normalized()
    else:
        # parallel projection along -z axis (local coordinates)
        # note that origin = matrix @ Vector((0, 0, 0))
        target = matrix @ Vector((0, 0, -1))
        minus_z_axis = (target - origin).normalized()

        def first_ray_coords(vert_co):
            return matrix @ vert_co, minus_z_axis

    ls_id_data = ls_mesh.vertex_layers_int["id"].data
    white = Color((1.0, 1.0, 1.0))

    # path_bm = {path: (caustic bmesh, vertex uv dict, vertex color dict)}
    path_bm = defaultdict(new_caustic_bmesh)
    for ls_vert in ls_mesh.vertices:
        # setup first ray
        ray_origin, ray_direction = first_ray_coords(ls_vert.co)
        ray = Ray(ray_origin, ray_direction, white, tuple())

        # trace ray
        vert_id = ls_id_data[ls_vert.index].value
        assert vert_id != -1, "lightsheet vertex id uninitialized"
        trace_scene_recursive(ray, vert_id, max_bounces, path_bm)

    return path_bm


def trace_scene_recursive(ray, vert_id, max_bounces, path_bm):
    """Recursively trace a ray, add interaction to path_bm dict"""
    ray_origin, ray_direction, color, old_path = ray
    result = scene_raycast(ray_origin, ray_direction)
    obj, location, normal, uv, face_index = result
    # print(ray_origin, ray_direction, obj, location)

    # if we hit the background we reached the end of the path
    if obj is None:
        return

    # not the end of the path, right???
    assert location is not None
    assert normal is not None

    # get hit face
    face = get_eval_mesh(obj).polygons[face_index]

    # get material on hit face with given index
    if obj.material_slots:
        mat_idx = face.material_index
        mat = obj.material_slots[mat_idx].material
    else:
        mat = None

    # determine surface interactions from material (i.e. does the ray
    # continue and if yes, how?)
    surface_shader = material.get_material_shader(mat)
    for kind, new_direction, tint in surface_shader(ray_direction, normal):
        # diffuse interactions will show reflective or refractive caustics,
        # non-diffuse interactions will add new rays if the new path is not
        # longer than the bounce limit
        if kind == "diffuse":
            # add caustic vert if it is a reflective or refractive caustic
            if any(step[1] in {"reflect", "refract"} for step in old_path):
                # get the caustic bmesh for this chain of interactions
                caustic_path = old_path + (Link(obj, "diffuse"),)
                bm, uv_dict, color_dict = path_bm[tuple(caustic_path)]

                # create caustic vertex slightly above object
                position = location + 2e-4 * face.normal  # is 2e-5 enough?
                vert = bm.verts.new(position)

                # set vertex id so that we can find its source vertex later
                id_layer = bm.verts.layers.int["id"]
                vert[id_layer] = vert_id

                # record uv coordinates from target object and record color as
                # vertex color, will set them later when creating faces
                if uv is not None:
                    uv_dict[vert_id] = uv
                color_dict[vert_id] = color
        elif len(old_path) < max_bounces:
            assert new_direction is not None
            # move the starting point a safe distance away from the object
            offset = copysign(1e-5, new_direction.dot(face.normal))
            new_origin = location + offset * face.normal

            # tint the color of the new ray
            mult = (c * t for c, t in zip(color, tint))
            new_color = Color(mult)

            # extend path of ray
            new_path = old_path + (Link(obj, kind),)

            # trace new ray
            new_ray = Ray(new_origin, new_direction, new_color, new_path)
            trace_scene_recursive(new_ray, vert_id, max_bounces, path_bm)


def scene_raycast(ray_origin, ray_direction):
    """Raycast all visible objects in the set scene and view layer"""
    # cast the ray
    result = _SCENE.ray_cast(_VIEW_LAYER, ray_origin, ray_direction)
    success, location, normal, face_index, obj, matrix = result

    if not success:
        # no hit
        return None, None, None, None, None

    # calculate (smooth) normal
    location_obj = matrix.inverted() @ location
    normal_obj, uv = calc_normal_and_uv(obj, face_index, location_obj)
    normal = (matrix @ (location_obj + normal_obj) - location).normalized()

    return obj, location, normal, uv, face_index


def calc_normal_and_uv(obj, face_index, point):
    """Calculate (smooth) normal vector and uv coordinates for given point"""
    # get the face that we hit
    mesh = get_eval_mesh(obj)
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


def get_eval_mesh(obj):
    """Return mesh of object with modifiers, etc. applied"""
    # check cache
    mesh = meshes_cache.get(obj)
    if mesh is not None:
        return mesh

    # get mesh of evaluated object
    obj_eval = obj.evaluated_get(_DEPSGRAPH)
    mesh = obj_eval.to_mesh()
    mesh.calc_normals_split()  # for loop normals

    meshes_cache[obj] = mesh
    return mesh


def new_caustic_bmesh(obj=None):
    """Create bmesh (empty or imported) with data layers used for caustics"""
    bm = bmesh.new()

    # if object is given, import mesh of evaluated object
    if obj is not None:
        bm.from_object(obj, _DEPSGRAPH)

    # get or create id layer = vertex index from source mesh
    id_layer = bm.verts.layers.int.get("id")
    if id_layer is None:
        id_layer = bm.verts.layers.int.new("id")

    # get or create uv layer for transplanted coordinates
    uv_layer = bm.loops.layers.uv.get("Transplanted UVMap")
    if uv_layer is None:
        uv_layer = bm.loops.layers.uv.new("Transplanted UVMap")

    # get or create uv layer for caustic squeeze = ratio of source face area
    # to projected face area
    squeeze_layer = bm.loops.layers.uv.get("Caustic Squeeze")
    if squeeze_layer is None:
        squeeze_layer = bm.loops.layers.uv.new("Caustic Squeeze")

    # get or create vertex color layer
    color_layer = bm.loops.layers.color.get("Caustic Tint")
    if color_layer is None:
        color_layer = bm.loops.layers.color.new("Caustic Tint")

    # since we can apply uv coordinates and vertex colors only via loops and
    # not to vertices directly, we will record the coordinates and colors in
    # these dicts and set them later when we iterate over the faces of the
    # bmesh
    uv_dict = dict()
    color_dict = dict()

    return bm, uv_dict, color_dict
