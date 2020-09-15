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
"""Utility functions for operators.

LIGHTSHEET_OT_create_lightsheet:
- create_bmesh_square
- create_bmesh_disk
- create_bmesh_sphere
- convert_to_lightsheet

LIGHTSHEET_OT_trace_lighsheet:
- verify_lighsheet
- convert_caustics_to_objects
- fill_caustic_faces
- set_caustic_face_info

LIGHTSHEET_OT_finalize_caustics:
- smooth_caustic_squeeze
- cleanup_caustic
"""

from math import sqrt

import bmesh
import bpy
from mathutils import Vector
from mathutils.geometry import barycentric_transform


# -----------------------------------------------------------------------------
# Functions for create lightsheet operator
# -----------------------------------------------------------------------------
def create_bmesh_square(sidelength, resolution):
    """Create bmesh for a square filled with triangles."""
    # horizontal and vertical strides, note that odd horizontal strips are
    # shifted by dx/2 and the height of an equilateral triangle with base a
    # is h = sqrt(3)/2 a
    dx = sidelength / (resolution - 1 / 2)
    dy = sqrt(3) / 2 * dx

    # horizontal and vertical resolution, note that we need to correct the
    # vertical resolution because triangle height is less than base length
    xres = resolution
    yres = int(resolution * 2 / sqrt(3))
    ydiff = sidelength - (yres - 1) * dy  # height error we make with triangles

    bm = bmesh.new()

    # place vertices
    strips = []  # each entry is a horizontal strip of vertices
    for j in range(yres):
        py = -sidelength / 2 + j * dy
        py += ydiff / 2  # center in y-direction

        strip = []
        for i in range(xres):
            px = -sidelength / 2 + i * dx
            px += (j % 2) * dx / 2  # shift the odd strips to the right
            vert = bm.verts.new((px, py, 0))
            strip.append(vert)
        strips.append(strip)

    # fill in faces
    for j in range(yres - 1):
        # lower and upper horizontal strips
        lower = strips[j]
        upper = strips[j + 1]

        if j % 2 == 0:
            # fill triangles in up,down,up,down,... configuration
            for i in range(xres - 1):
                bm.faces.new((lower[i], upper[i], lower[i + 1]))
                bm.faces.new((lower[i + 1], upper[i], upper[i + 1]))
        else:
            # fill triangles in down,up,down,up,... configuration
            for i in range(xres - 1):
                bm.faces.new((lower[i], upper[i], upper[i + 1]))
                bm.faces.new((lower[i + 1], lower[i], upper[i + 1]))

    return bm


def create_bmesh_disk(radius, resolution):
    """Create bmesh for a circle filled with triangles."""
    # the easiest way to create a circle is to create a square and then cut out
    # the circle
    bm = create_bmesh_square(2 * radius, resolution)

    # gather vertices that lie outside the circle and delete them, note that
    # this will also delete some faces on the edge of the circle which may
    # look weird for low resolutions
    outside_verts = [vert for vert in bm.verts
                     if vert.co[0] ** 2 + vert.co[1] ** 2 > radius ** 2]
    bmesh.ops.delete(bm, geom=outside_verts, context="VERTS")

    return bm


def create_bmesh_sphere(resolution, radius=1):
    """Create a spherical bmesh based on a subdivided icosahedron."""
    # template icosahedron
    bm_template = bmesh.new()
    bmesh.ops.create_icosphere(bm_template, subdivisions=0, diameter=1.0)

    # size of source triangle and smaller filler triangles
    source_triangle = [
        Vector((0, 0, 0)),
        Vector((resolution - 1, 0, 0)),
        Vector((0, resolution - 1, 0))
    ]

    # replace every triangular face with a triangle mesh of given resolution
    bm = bmesh.new()
    for face in bm_template.faces:
        assert len(face.verts) == 3, len(face.verts)
        target_triangle = [vert.co for vert in face.verts]

        # place vertices
        strips = []  # each entry is a horizontal strip of vertices
        for j in range(resolution):
            strip = []
            for i in range(resolution - j):  # less vertices as we go higher
                coords = barycentric_transform(Vector((i, j, 0)),
                                               *source_triangle,
                                               *target_triangle)
                coords *= radius / coords.length  # place on sphere
                vert = bm.verts.new(coords)
                strip.append(vert)
            strips.append(strip)

        # fill in faces
        for j in range(resolution - 1):
            # lower and upper horizontal strips
            lower = strips[j]
            upper = strips[j + 1]

            # fill triangles in up,down,up,down,...,up configuration
            end = resolution - j - 2
            for i in range(end):
                bm.faces.new((lower[i], lower[i + 1], upper[i]))
                bm.faces.new((lower[i + 1], upper[i + 1], upper[i]))
            bm.faces.new((lower[end], lower[end + 1], upper[end]))

    # we created overlapping vertices at the edges and corners, merge them
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.1)

    bm_template.free()
    return bm


def convert_to_lightsheet(bm, light):
    """Convert given bmesh to lightsheet object with given light as parent."""
    # create id layer and assign values
    id_layer = bm.verts.layers.int.new("id")
    for idx, vert in enumerate(bm.verts):
        vert[id_layer] = idx

    # think of a good name
    name = f"Lightsheet for {light.name}"

    # convert bmesh to mesh data block
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()

    # create new object
    lightsheet = bpy.data.objects.new(name, me)
    lightsheet.parent = light

    # adjust drawing and visibility
    lightsheet.display_type = 'WIRE'
    lightsheet.hide_render = True
    lightsheet.cycles_visibility.camera = False
    lightsheet.cycles_visibility.shadow = False
    lightsheet.cycles_visibility.diffuse = False
    lightsheet.cycles_visibility.transmission = False
    lightsheet.cycles_visibility.scatter = False

    return lightsheet


# -----------------------------------------------------------------------------
# Functions for trace lightsheet operator
# -----------------------------------------------------------------------------
def verify_lightsheet(obj):
    """Confirm that the given object can be used as a lightsheet."""
    assert obj is not None and isinstance(obj, bpy.types.Object)

    # verify name
    if "lightsheet" not in obj.name.lower():
        obj.name = f"Lightsheet {obj.name}"

    # verify that lightsheet mesh has id layer = persistent vertex index
    mesh = obj.to_mesh()
    ls_id = mesh.vertex_layers_int.get("id")
    if ls_id is None:
        ls_id = mesh.vertex_layers_int.new(name="id")
        for vert in mesh.vertices:
            ls_id.data[vert.index].value = vert.index

    # check id data before we continue
    assert all(ls_id.data[v.index].value != -1 for v in mesh.vertices)

    return obj


def convert_caustics_to_objects(lightsheet, path, traced):
    """Convert caustic bmesh to blender object with filled in faces."""
    bm, normal_dict, color_dict, uv_dict = traced
    assert len(bm.edges) == 0 and len(bm.faces) == 0  # bmesh with only verts

    # fill faces
    fill_caustic_faces(lightsheet, bm)
    set_caustic_face_info(bm, normal_dict, color_dict, uv_dict)

    # parent of caustic = last object
    last_link = path[-1]
    parent_obj = last_link.object
    assert last_link.kind == 'DIFFUSE', path  # check consistency of path

    # consistency check for copied uv-coordinates from the parent object
    if not uv_dict:
        assert not parent_obj.data.uv_layers, parent_obj.data.uv_layers[:]
    else:
        assert parent_obj.data.uv_layers

    # think of a good name
    name = f"Caustic of {lightsheet.name} on {parent_obj.name}"

    # new mesh data block
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()

    # rename transplanted uvmap to avoid confusion
    if uv_dict:
        parent_uv_layer = parent_obj.data.uv_layers.active
        assert parent_obj.data.uv_layers.active is not None
        uv_layer = me.uv_layers["UVMap"]
        uv_layer.name = parent_uv_layer.name

    # before setting parent, undo parent transform
    me.transform(parent_obj.matrix_world.inverted())

    # new object with given mesh
    caustic = bpy.data.objects.new(name, me)
    caustic.parent = parent_obj

    # fill out caustic_info property
    caustic.caustic_info.lightsheet = lightsheet
    caustic_path = caustic.caustic_info.path
    for obj, kind, _ in path:
        item = caustic_path.add()
        item.object = obj
        item.kind = kind

    return caustic


def fill_caustic_faces(lightsheet, bm):
    """Fill in faces of caustic bmesh using faces from lightsheet."""
    id_layer = bm.verts.layers.int["id"]
    id_cache = {vert[id_layer]: vert for vert in bm.verts}
    squeeze_layer = bm.loops.layers.uv["Caustic Squeeze"]

    # create corresponding faces
    ls_mesh = lightsheet.to_mesh()
    ls_id = ls_mesh.vertex_layers_int["id"]
    for ls_face in ls_mesh.polygons:
        caustic_verts = []
        for ls_vert_index in ls_face.vertices:
            ls_vert_id = ls_id.data[ls_vert_index].value
            vert = id_cache.get(ls_vert_id)
            if vert is not None:
                caustic_verts.append(vert)

        # create edge or face from vertices
        if len(caustic_verts) == 2:
            # can only create an edge here, but this edge may already exist
            # if we worked through a neighboring face before
            if bm.edges.get(caustic_verts) is None:
                bm.edges.new(caustic_verts)
        elif len(caustic_verts) > 2:
            caustic_face = bm.faces.new(caustic_verts)

            # set squeeze factor = ratio of source area to projected area
            squeeze = ls_face.area / caustic_face.calc_area()
            for loop in caustic_face.loops:
                # TODO can we use the u-coordinate for something useful?
                loop[squeeze_layer].uv = (0, squeeze)


def set_caustic_face_info(bm, normal_dict, color_dict, uv_dict):
    """Set caustic face normals, vertex colors and uv-coordinates (if any)."""
    # set transplanted uv-coordinates and vertex colors
    id_layer = bm.verts.layers.int["id"]
    if uv_dict:
        uv_layer = bm.loops.layers.uv["UVMap"]
    color_layer = bm.loops.layers.color["Caustic Tint"]
    for face in bm.faces:
        vert_normal_sum = Vector((0, 0, 0))
        for loop in face.loops:
            vert_id = loop.vert[id_layer]

            vert_normal_sum += normal_dict[vert_id]
            if uv_dict:  # only set uv-coords if we have uv-coords
                loop[uv_layer].uv = uv_dict[vert_id]
            loop[color_layer] = tuple(color_dict[vert_id]) + (1,)

        # if face normal does not point in the same general direction as
        # the averaged vertex normal, then flip the face normal
        face.normal_update()
        if face.normal.dot(vert_normal_sum) < 0:
            face.normal_flip()

    bm.normal_update()


# -----------------------------------------------------------------------------
# Functions for finalize caustics operator
# -----------------------------------------------------------------------------
def smooth_caustic_squeeze(bm):
    """Poke faces and smooth out caustic squeeze."""
    squeeze_layer = bm.loops.layers.uv["Caustic Squeeze"]

    # poke every face (i.e. place a vertex in the middle)
    result = bmesh.ops.poke(bm, faces=bm.faces)

    # interpolate squeeze for the original vertices
    poked_verts = set(result["verts"])
    for vert in bm.verts:
        # skip the new vertices
        if vert in poked_verts:
            continue

        # find neighbouring poked vertices (= original faces)
        poked_squeeze = dict()
        for loop in vert.link_loops:
            for other_vert in loop.face.verts:
                if other_vert in poked_verts:
                    squeeze = loop[squeeze_layer].uv[1]
                    poked_squeeze[other_vert] = squeeze

        # cannot process vertices that are not connected to a face, these
        # vertices will be removed anyway
        if len(poked_squeeze) == 0:
            assert len(vert.link_faces) == 0
            break

        # interpolate squeeze of original vertex from squeeze of neighbouring
        # original faces, weight each face according to its distance (note that
        # there is a correct solution that we could find via tracing rays that
        # start very close to the source lightsheet vertex)
        weightsum = 0.0
        squeeze = 0.0
        for other_vert in poked_squeeze:
            dist = (other_vert.co - vert.co).length
            assert dist > 0
            weight = 1 / dist**2  # an arbitrary but sensible choice
            weightsum += weight
            squeeze += weight * poked_squeeze[other_vert]
        squeeze /= weightsum

        # set squeeze for this vertex
        for loop in vert.link_loops:
            loop[squeeze_layer].uv[1] = squeeze


def cleanup_caustic(bm, intensity_threshold):
    """Remove invisible faces and cleanup resulting mesh."""
    squeeze_layer = bm.loops.layers.uv["Caustic Squeeze"]
    color_layer = bm.loops.layers.color["Caustic Tint"]

    # remove faces with intensity less than intensity_threshold
    invisible_faces = []
    for face in bm.faces:
        visible = False
        for loop in face.loops:
            squeeze = loop[squeeze_layer].uv[1]
            tint_v = max(loop[color_layer][:3])  # = Color(...).v
            if squeeze * tint_v > intensity_threshold:
                # vertex is intense enough, face is visible
                visible = True
                break

        if not visible:
            invisible_faces.append(face)

    bmesh.ops.delete(bm, geom=invisible_faces, context='FACES_ONLY')

    # clean up mesh, remove vertices and edges not connected to faces
    lonely_verts = [vert for vert in bm.verts if vert.is_wire]
    bmesh.ops.delete(bm, geom=lonely_verts, context='VERTS')
    lonely_edges = [edge for edge in bm.edges if edge.is_wire]
    bmesh.ops.delete(bm, geom=lonely_edges, context='EDGES')

    # check if cleanup was successful
    assert not any(vert.is_wire for vert in bm.verts)
    assert not any(edge.is_wire for edge in bm.edges)
