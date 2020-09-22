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
from mathutils.geometry import (area_tri, barycentric_transform,
                                intersect_point_tri, tessellate_polygon)

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
    ls_bmesh.from_mesh(lightsheet.data)

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

        def first_ray_coords(sheet_pos):
            return matrix @ Vector(sheet_pos), minus_z_axis
    else:
        # project from origin of lightsheet coordinate system
        def first_ray_coords(sheet_pos):
            direction = matrix @ Vector(sheet_pos) - origin
            return origin, direction.normalized()

    # sheet coordinate access
    sheet_x = ls_bmesh.verts.layers.float["Lightsheet X"]
    sheet_y = ls_bmesh.verts.layers.float["Lightsheet Y"]
    sheet_z = ls_bmesh.verts.layers.float["Lightsheet Z"]

    # path_bm = {path: (caustic bmesh, normal dict, color dict, uv dict)}
    path_bm = defaultdict(new_caustic_bmesh)
    for ls_vert in ls_bmesh.verts:
        # get sheet position of this vertex
        sheet_pos = (ls_vert[sheet_x], ls_vert[sheet_y], ls_vert[sheet_z])

        # setup first ray
        ray_origin, ray_direction = first_ray_coords(sheet_pos)
        ray = Ray(ray_origin, ray_direction, Color((1.0, 1.0, 1.0)), tuple())

        # trace ray
        trace_scene_recursive(ray, sheet_pos, depsgraph, max_bounces, path_bm)

    ls_bmesh.free()
    return path_bm


def trace_scene_recursive(ray, sheet_pos, depsgraph, max_bounces, path_bm):
    """Recursively trace a ray, add interaction to path_bm dict"""
    ray_origin, ray_direction, color, old_path = ray
    obj, location, normal, uv, face_index = scene_raycast(
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
            if any(step[1] in {'REFLECT', 'REFRACT'} for step in old_path):
                # get the caustic bmesh for this chain of interactions
                caustic_key = tuple(old_path + (Link(obj, kind, None),))
                bm, normal_dict, color_dict, uv_dict = path_bm[caustic_key]

                # create caustic vertex in front of object
                offset = copysign(1e-4, -ray_direction.dot(face.normal))
                position = location + offset * face.normal
                vert = bm.verts.new(position)

                # set sheet coordinates
                sx, sy, sz = sheet_pos
                vert[bm.verts.layers.float["Lightsheet X"]] = sx
                vert[bm.verts.layers.float["Lightsheet Y"]] = sy
                vert[bm.verts.layers.float["Lightsheet Z"]] = sz

                # record normal from hit face, vertex color from path and
                # uv-coordinate from target object, we will set them later when
                # creating faces
                normal_dict[sheet_pos] = face.normal
                color_dict[sheet_pos] = color
                if uv is not None:
                    uv_dict[sheet_pos] = uv
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
            trace_scene_recursive(new_ray, sheet_pos, depsgraph, max_bounces,
                                  path_bm)


def scene_raycast(ray_origin, ray_direction, depsgraph):
    """Raycast all visible objects in the set scene and view layer"""
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
    pos_obj = matrix.inverted() @ location
    normal_obj, uv = calc_normal_and_uv(obj, depsgraph, face_index, pos_obj)
    normal = (matrix @ (pos_obj + normal_obj) - location).normalized()

    return obj, location, normal, uv, face_index


def refine_caustic(caustic_obj, depsgraph, relative_error):
    """Do one adaptive subdivision of caustic bmesh."""
    # convert caustic to bmesh
    caustic_bm = bmesh.new()
    caustic_bm.from_mesh(caustic_obj.data)

    # world to caustic object coordinate transformation
    obj_to_world = caustic_obj.parent.matrix_world
    world_to_obj = obj_to_world.inverted()

    # caustic info
    caustic_info = caustic_obj.caustic_info
    lightsheet = caustic_info.lightsheet
    assert lightsheet is not None
    path = [Link(link.object, link.kind, None) for link in caustic_info.path]

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

        def first_ray_coords(sheet_pos):
            return matrix @ Vector(sheet_pos), minus_z_axis
    else:
        # project from origin of lightsheet coordinate system
        def first_ray_coords(sheet_pos):
            direction = matrix @ Vector(sheet_pos) - origin
            return origin, direction.normalized()

    # coordinates of source position on lighsheet
    sheet_x = caustic_bm.verts.layers.float["Lightsheet X"]
    sheet_y = caustic_bm.verts.layers.float["Lightsheet Y"]
    sheet_z = caustic_bm.verts.layers.float["Lightsheet Z"]

    # create vert -> sheet lookup table for faster access
    vert_to_sheet = dict()
    for vert in caustic_bm.verts:
        sx = vert[sheet_x]
        sy = vert[sheet_y]
        sz = vert[sheet_z]
        vert_to_sheet[vert] = (sx, sy, sz)

    def bla(edge):
        # get sheet positions of edge endpoints
        vert1, vert2 = edge.verts
        sheet_pos1 = Vector(vert_to_sheet[vert1])
        sheet_pos2 = Vector(vert_to_sheet[vert2])

        # calculate coordinates of edge midpoint and setup ray
        sheet_mid = (sheet_pos1 + sheet_pos2) / 2
        ray_origin, ray_direction = first_ray_coords(sheet_mid)

        # trace ray
        position, normal, color, uv = trace_along_path(
            ray_origin, ray_direction, depsgraph, path)

        return tuple(sheet_mid), position, normal, color, uv

    # gather all edges that we have to split
    refine_edges = set()  # edges to subdivide
    # {edge to subdivide: sheet and target coords of midpoint}
    refine_targets = dict()
    for edge in caustic_bm.edges:
        # add every edge of a boundary face
        if edge.is_boundary:
            assert len(edge.link_faces) == 1
            face = edge.link_faces[0]
            for other_edges in face.edges:
                refine_edges.add(other_edges)
            continue

        # always refine wires
        if edge.is_wire:
            refine_edges.add(edge)

        # skip edges that were not marked from last time
        if not edge.seam:
            continue

        # calc midpoint and target coordinates
        sheet_mid, target_pos, normal, color, uv = bla(edge)
        refine_targets[edge] = (sheet_mid, target_pos, normal, color, uv)

        # calc error and wether we should keep the edge
        if target_pos is None:
            refine_edges.add(edge)
        else:
            vert1, vert2 = edge.verts
            edge_mid = obj_to_world @ (vert1.co + vert2.co) / 2
            edge_length = (obj_to_world @ (vert1.co - vert2.co)).length
            rel_err = (edge_mid - target_pos).length / edge_length
            if rel_err > relative_error:
                refine_edges.add(edge)

    # if two edges of a triangle will be refined, also subdivide the other
    # edge, this will make sure that we always end up with triangles
    last_added = list(refine_edges)
    newly_added = list()
    while last_added:  # crawl through the mesh and select edges that we need
        for edge in last_added:  # only last edges can change futher selection
            for face in edge.link_faces:
                assert len(face.edges) == 3  # must be a triangle
                not_refined = [
                    ed for ed in face.edges if ed not in refine_edges]
                if len(not_refined) == 1:
                    ed = not_refined[0]
                    refine_edges.add(ed)
                    newly_added.append(ed)

        last_added = newly_added
        newly_added = list()

    # calculate targets for the missing edges
    for edge in refine_edges:
        if edge not in refine_targets:
            refine_targets[edge] = bla(edge)

    # split edges
    edgelist = list(refine_edges)
    splits = bmesh.ops.subdivide_edges(caustic_bm, edges=edgelist, cuts=1,
                                       use_grid_fill=True,
                                       use_single_edge=True)

    # ensure triangles
    triang_less = [face for face in caustic_bm.faces if len(face.edges) > 3]
    if triang_less:
        print(f"We have to triangulate {len(triang_less)} faces")
        bmesh.ops.triangulate(caustic_bm, faces=triang_less)

    # gather the newly added vertices
    dirty_verts = set()
    for item in splits['geom_inner']:
        if isinstance(item, bmesh.types.BMVert):
            dirty_verts.add(item)

    # gather the edges that we may have to split next
    dirty_edges = set()
    for item in splits['geom']:
        if isinstance(item, bmesh.types.BMEdge):
            dirty_edges.add(item)

    # set coordinates
    delete_verts = []
    normal_dict, color_dict, uv_dict = dict(), dict(), dict()
    for edge in refine_edges:
        # find the newly added vertex, it is one of the endpoints of a
        # refined edge
        v1, v2 = edge.verts
        if v1 in dirty_verts:
            vert = v1
        else:
            assert v2 in dirty_verts, (v1.co, v2.co)
            vert = v2

        # set coordinates
        sheet_mid, target_pos, normal, color, uv = refine_targets[edge]
        if target_pos is None:
            delete_verts.append(vert)
        else:
            vert.co = world_to_obj @ target_pos

            # set sheet coords
            sx, sy, sz = sheet_mid
            vert[sheet_x] = sx
            vert[sheet_y] = sy
            vert[sheet_z] = sz

            # save orther settings in mappings and set them later via loops
            normal_dict[sheet_mid] = normal
            color_dict[sheet_mid] = color
            if uv is not None:
                uv_dict[sheet_mid] = uv

    # remove verts that have no target
    dirty_verts.difference_update(delete_verts)
    bmesh.ops.delete(caustic_bm, geom=delete_verts, context='VERTS')

    # mark edges for next refinement step
    for edge in caustic_bm.edges:
        edge.seam = edge in dirty_edges

    # select only the dirty verts
    for face in caustic_bm.faces:
        face.select_set(False)
    for vert in dirty_verts:
        vert.select_set(True)

    # gather the faces where we have to recalulate squeeze
    dirty_faces = set()
    for vert in dirty_verts:
        for face in vert.link_faces:
            dirty_faces.add(face)

    # squeeze layer
    squeeze_layer = caustic_bm.loops.layers.uv["Caustic Squeeze"]

    # sheet coordinates uv-layers
    uv_sheet_xy = caustic_bm.loops.layers.uv["Lightsheet XY"]
    uv_sheet_xz = caustic_bm.loops.layers.uv["Lightsheet XZ"]

    # uv-layers and vertex color access
    if uv_dict:
        uv_layer = caustic_bm.loops.layers.uv["UVMap"]
    color_layer = caustic_bm.loops.layers.color["Caustic Tint"]

    # set face info
    for face in dirty_faces:
        assert len(face.verts) == 3, len(face.verts)
        sheet_triangle = []
        target_triangle = []
        for vert in face.verts:
            sx = vert[sheet_x]
            sy = vert[sheet_y]
            sz = vert[sheet_z]
            sheet_pos = (sx, sy, sz)
            sheet_triangle.append(sheet_pos)
            target_triangle.append(obj_to_world @ vert.co)

        # set squeeze factor = ratio of source area to projected area
        source_area = area_tri(*sheet_triangle)
        target_area = area_tri(*target_triangle)
        squeeze = source_area / target_area
        for loop in face.loops:
            loop[squeeze_layer].uv = (0, squeeze)

        # set face info: uv-coordinates (sheet and transplanted map), normal
        # and vertex colors
        vert_normal_sum = Vector((0, 0, 0))
        for loop in face.loops:
            # get sheet position
            vert = loop.vert
            sx = vert[sheet_x]
            sy = vert[sheet_y]
            sz = vert[sheet_z]
            sheet_pos = (sx, sy, sz)

            if sheet_pos not in normal_dict:
                # we do not need to set this vertex
                continue

            # sheet position to uv-coordinates
            loop[uv_sheet_xy].uv = (sx, sy)
            loop[uv_sheet_xz].uv = (sx, sz)

            vert_normal_sum += normal_dict[sheet_pos]
            if uv_dict:  # only set uv-coords if we have uv-coords
                loop[uv_layer].uv = uv_dict[sheet_pos]
            loop[color_layer] = tuple(color_dict[sheet_pos]) + (1,)

        # if face normal does not point in the same general direction as
        # the averaged vertex normal, then flip the face normal
        face.normal_update()
        if face.normal.dot(vert_normal_sum) < 0:
            face.normal_flip()

    caustic_bm.normal_update()

    # convert bmesh back to object
    caustic_bm.to_mesh(caustic_obj.data)
    caustic_bm.free()


def trace_along_path(ray_origin, ray_direction, depsgraph, path_to_follow):
    color = Color((1.0, 1.0, 1.0))
    old_path = tuple()

    # trace along the given chain of interactions
    for obj, focus_on_interaction, _ in path_to_follow:
        location, normal, uv, face_index = obj_raycast(
            obj, ray_origin, ray_direction, depsgraph)

        if location is None:
            return None, None, None, None

        # get hit face
        obj_mesh = get_eval_mesh(obj, depsgraph)
        face = obj_mesh.polygons[face_index]

        # get material on hit face with given index
        if obj.material_slots:
            mat_idx = face.material_index
            mat = obj.material_slots[mat_idx].material
        else:
            mat = None

        # determine surface interactions from material
        surface_shader, volume_params = material.get_material_shader(mat)

        # focus on the interaction that that follows the path of the caustic
        found = False
        for kind, new_direction, tint in surface_shader(ray_direction, normal):
            if kind == focus_on_interaction:
                found = True
                break

        # if we cannot find this interaction, then the caustic will be empty
        # at this position
        if not found:
            return None, None, None, None

        if kind == 'DIFFUSE':
            # see after the loop
            break

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

        # prepare new ray
        ray_origin = new_origin
        ray_direction = new_direction
        color = new_color
        old_path = new_path

    # create caustic vertex in front of object
    offset = copysign(1e-4, -ray_direction.dot(face.normal))
    position = location + offset * face.normal

    return position, face.normal, color, uv


def obj_raycast(obj, ray_origin, ray_direction, depsgraph):
    # transform to object space
    matrix = obj.evaluated_get(depsgraph).matrix_world
    matrix_inv = matrix.inverted()
    ray_origin_obj = matrix_inv @ ray_origin
    ray_target_obj = matrix_inv @ (ray_origin + ray_direction)
    ray_direction_obj = ray_target_obj - ray_origin_obj

    # raycast
    success, pos_obj, normal_obj, face_index, = obj.ray_cast(
        ray_origin_obj, ray_direction_obj, depsgraph=depsgraph)

    if not success:
        return None, None, None, None

    # calculate (smooth) normal
    normal_obj, uv = calc_normal_and_uv(obj, depsgraph, face_index, pos_obj)
    location = matrix @ pos_obj
    normal = (matrix @ (pos_obj + normal_obj) - location).normalized()

    return location, normal, uv, face_index


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

    # create vertex layers for sheet coordinates
    bm.verts.layers.float.new("Lightsheet X")
    bm.verts.layers.float.new("Lightsheet Y")
    bm.verts.layers.float.new("Lightsheet Z")

    # create uv-layers for sheet coordinates
    bm.loops.layers.uv.new("Lightsheet XY")
    bm.loops.layers.uv.new("Lightsheet XZ")

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
