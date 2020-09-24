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
- setup_as_lighsheet
- trace_lightsheet
- setup_lightsheet_first_ray
- convert_caustics_to_objects
- setup_caustic_bmesh
- fill_caustic_faces
- set_caustic_squeeze
- set_caustic_face_data

LIGHTSHEET_OT_refine_caustics:
- refine_caustic

LIGHTSHEET_OT_finalize_caustics:
- smooth_caustic_squeeze
- cleanup_caustic
"""
from collections import defaultdict
from math import sqrt

import bmesh
import bpy
from mathutils import Color, Matrix, Vector
from mathutils.geometry import area_tri, barycentric_transform

from lightsheet import material, trace


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


def create_bmesh_sphere(resolution, radius=1.0):
    """Create a spherical bmesh based on a subdivided icosahedron."""
    # use icosahedron as template
    bm_template = bmesh.new()
    bmesh.ops.create_icosphere(bm_template, subdivisions=0, diameter=1.0)

    # we will generate points with coordinates (i, j, 0), where i, j are
    # integers with i >= 0, j >= 0, i + j <= resolution - 1
    source_triangle = [
        Vector((0, 0, 0)),
        Vector((resolution - 1, 0, 0)),
        Vector((0, resolution - 1, 0))
    ]

    # replace every triangular face in the template icosahedron with a grid of
    # triangles with the given resolution
    bm = bmesh.new()
    shared_verts = []
    for face in bm_template.faces:
        assert len(face.verts) == 3, len(face.verts)
        target_triangle = [vert.co for vert in face.verts]

        # place vertices
        strips = []  # each entry is a horizontal line of vertices
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

        # record which vertices are at the edge because these will have to be
        # merged with other edge vertices later
        shared_verts.extend(strips[0])  # bottom edge
        for j in range(1, resolution - 1):
            strip = strips[j]
            shared_verts.append(strip[0])  # left edge
            shared_verts.append(strip[-1])  # diagonal edge
        shared_verts.append(strips[resolution - 1][0])  # tip of triangle

    # merge the overlapping vertices at the edges and corners
    bmesh.ops.remove_doubles(bm, verts=shared_verts,
                             dist=0.1*radius/resolution)

    bm_template.free()
    return bm


def convert_to_lightsheet(bm, light):
    """Convert given bmesh to lightsheet object with given light as parent."""
    # create sheet coordinate layer and set values
    sheet_x = bm.verts.layers.float.new("Lightsheet X")
    sheet_y = bm.verts.layers.float.new("Lightsheet Y")
    sheet_z = bm.verts.layers.float.new("Lightsheet Z")
    for vert in bm.verts:
        vx, vy, vz = vert.co
        vert[sheet_x] = vx
        vert[sheet_y] = vy
        vert[sheet_z] = vz

    # create sheet coordinates as uv-layers for easier visualization
    uv_sheet_xy = bm.loops.layers.uv.new("Lightsheet XY")
    uv_sheet_xz = bm.loops.layers.uv.new("Lightsheet XZ")
    for face in bm.faces:
        for loop in face.loops:
            vert = loop.vert
            sx, sy, sz = (vert[sheet_x], vert[sheet_y], vert[sheet_z])
            loop[uv_sheet_xy].uv = (sx, sy)
            loop[uv_sheet_xz].uv = (sx, sz)

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
def setup_as_lightsheet(obj):
    """Setup the given object so that it can be used as a lightsheet."""
    assert obj is not None and isinstance(obj, bpy.types.Object)

    # verify name
    if "lightsheet" not in obj.name.lower():
        obj.name = f"Lightsheet {obj.name}"

    # convert to bmesh to simplify the following steps
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # verify that lightsheet mesh has sheet coordinate layer
    sheet_x = bm.verts.layers.float.get("Lightsheet X")
    if sheet_x is None:
        sheet_x = bm.verts.layers.float.new("Lightsheet X")
    sheet_y = bm.verts.layers.float.get("Lightsheet Y")
    if sheet_y is None:
        sheet_y = bm.verts.layers.float.new("Lightsheet Y")
    sheet_z = bm.verts.layers.float.get("Lightsheet Z")
    if sheet_z is None:
        sheet_z = bm.verts.layers.float.new("Lightsheet Z")

    # always recalculate the sheet coordinates because we can't know if the
    # geometry was changed after creating the lightsheet
    for vert in bm.verts:
        vx, vy, vz = vert.co
        vert[sheet_x] = vx
        vert[sheet_y] = vy
        vert[sheet_z] = vz

    # verify that mesh has uv-layers for sheet coordinates
    uv_sheet_xy = bm.loops.layers.uv.get("Lightsheet XY")
    if uv_sheet_xy is None:
        uv_sheet_xy = bm.loops.layers.uv.new("Lightsheet XY")
    uv_sheet_xz = bm.loops.layers.uv.get("Lightsheet XZ")
    if uv_sheet_xz is None:
        uv_sheet_xz = bm.loops.layers.uv.new("Lightsheet XZ")

    # recalculate sheet coordinates
    for face in bm.faces:
        for loop in face.loops:
            vert = loop.vert
            sx = vert[sheet_x]
            sy = vert[sheet_y]
            sz = vert[sheet_z]
            loop[uv_sheet_xy].uv = (sx, sy)
            loop[uv_sheet_xz].uv = (sx, sz)

    # convert back to object
    bm.to_mesh(obj.data)
    bm.free()


def trace_lightsheet(lightsheet, depsgraph, max_bounces):
    """Trace rays from lighsheet and return all caustics coordinates in dict"""
    first_ray = setup_lightsheet_first_ray(lightsheet)

    # convert lightsheet to bmesh
    ls_bmesh = bmesh.new(use_operators=False)
    ls_bmesh.from_mesh(lightsheet.data)

    # coordinates of source position on lighsheet
    sheet_x = ls_bmesh.verts.layers.float["Lightsheet X"]
    sheet_y = ls_bmesh.verts.layers.float["Lightsheet Y"]
    sheet_z = ls_bmesh.verts.layers.float["Lightsheet Z"]

    # make sure caches are clean
    trace.meshes_cache.clear()
    material.materials_cache.clear()

    # traced = {chain: {sheet_pos: CausticVert(location, color, uv, normal)}}
    traced = defaultdict(dict)
    for vert in ls_bmesh.verts:
        sx = vert[sheet_x]
        sy = vert[sheet_y]
        sz = vert[sheet_z]
        sheet_pos = (sx, sy, sz)
        trace.trace_scene_recursive(first_ray(sheet_pos), sheet_pos, depsgraph,
                                    max_bounces, traced)

    # cleanup generated meshes and caches
    ls_bmesh.free()
    for obj in trace.meshes_cache:
        obj.to_mesh_clear()
    trace.meshes_cache.clear()
    material.materials_cache.clear()

    return traced


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
            return trace.Ray(ray_origin, minus_z_axis, white, tuple())
    else:
        origin.freeze()  # will use as default value, should be immutable

        # project from origin of lightsheet coordinate system
        def first_ray(sheet_pos):
            ray_direction = sheet_to_world @ Vector(sheet_pos) - origin
            ray_direction.normalize()
            return trace.Ray(origin, ray_direction, white, tuple())

    return first_ray


def convert_caustic_to_objects(lightsheet, chain, caustic_data):
    """Convert caustic bmesh to blender object with filled in faces."""
    # setup and fill caustic bmesh
    caustic_bm = setup_caustic_bmesh(caustic_data)
    fill_caustic_faces(caustic_bm, lightsheet)
    set_caustic_squeeze(caustic_bm, sheet_to_world=lightsheet.matrix_world)
    set_caustic_face_data(caustic_bm, caustic_data)

    # mark all edges of the caustic for refinement
    for edge in caustic_bm.edges:
        edge.seam = True

    # parent of caustic = object in last link
    parent_obj = chain[-1].object
    assert chain[-1].kind == 'DIFFUSE', chain  # check consistency of path

    # consistency check for copied uv-coordinates from the parent object
    if caustic_bm.loops.layers.uv.get("UVMap") is not None:
        assert parent_obj.data.uv_layers
    else:
        assert not parent_obj.data.uv_layers, parent_obj.data.uv_layers[:]

    # think of a good name
    name = f"Caustic of {lightsheet.name} on {parent_obj.name}"

    # new mesh data block
    me = bpy.data.meshes.new(name)
    caustic_bm.to_mesh(me)
    caustic_bm.free()

    # rename transplanted uvmap to avoid confusion
    if parent_obj.data.uv_layers:
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
    for obj, kind, _ in chain:
        item = caustic_path.add()
        item.object = obj
        item.kind = kind

    return caustic


def setup_caustic_bmesh(caustic_data):
    """Create empty bmesh with data layers used for caustics"""
    bm = bmesh.new()

    # create vertex layers for sheet coordinates
    sheet_x = bm.verts.layers.float.new("Lightsheet X")
    sheet_y = bm.verts.layers.float.new("Lightsheet Y")
    sheet_z = bm.verts.layers.float.new("Lightsheet Z")

    # create uv-layers for sheet coordinates
    bm.loops.layers.uv.new("Lightsheet XY")
    bm.loops.layers.uv.new("Lightsheet XZ")

    # create uv layer for caustic squeeze = ratio of source face area to
    # projected face area
    bm.loops.layers.uv.new("Caustic Squeeze")

    # create vertex color layer for caustic tint
    bm.loops.layers.color.new("Caustic Tint")

    # create uv layer for transplanted coordinates (if any)
    if all(data.uv is not None for data in caustic_data.values()):
        bm.loops.layers.uv.new("UVMap")

    # create vertices and given positions and set sheet coordinates
    for sheet_pos in caustic_data:
        # create vertex
        position = caustic_data[sheet_pos].position
        vert = bm.verts.new(position)

        # set sheet coordinates
        sx, sy, sz = sheet_pos
        vert[sheet_x] = sx
        vert[sheet_y] = sy
        vert[sheet_z] = sz

    return bm


def fill_caustic_faces(caustic_bm, lightsheet):
    """Fill in faces of caustic bmesh using faces from lightsheet."""
    assert len(caustic_bm.edges) == 0, len(caustic_bm.edges)
    assert len(caustic_bm.faces) == 0, len(caustic_bm.faces)

    # convert lightsheet to bmesh
    ls_bm = bmesh.new()
    ls_bm.from_mesh(lightsheet.data)

    # get sheet coordinates from lightsheet
    ls_sheet_x = ls_bm.verts.layers.float["Lightsheet X"]
    ls_sheet_y = ls_bm.verts.layers.float["Lightsheet Y"]
    ls_sheet_z = ls_bm.verts.layers.float["Lightsheet Z"]

    # create vert -> sheet lookup table for faster access
    ls_vert_to_sheet = dict()
    for vert in ls_bm.verts:
        sx = vert[ls_sheet_x]
        sy = vert[ls_sheet_y]
        sz = vert[ls_sheet_z]
        ls_vert_to_sheet[vert] = (sx, sy, sz)

    # get sheet coordinates from caustic
    caustic_sheet_x = caustic_bm.verts.layers.float["Lightsheet X"]
    caustic_sheet_y = caustic_bm.verts.layers.float["Lightsheet Y"]
    caustic_sheet_z = caustic_bm.verts.layers.float["Lightsheet Z"]

    # create sheet -> vert lookup table for faster access
    sheet_to_caustic_vert = dict()
    for vert in caustic_bm.verts:
        sx = vert[caustic_sheet_x]
        sy = vert[caustic_sheet_y]
        sz = vert[caustic_sheet_z]
        assert (sx, sy, sz) not in sheet_to_caustic_vert
        sheet_to_caustic_vert[(sx, sy, sz)] = vert

    # iterate over lighsheet faces and create corresponding caustic faces
    for ls_face in ls_bm.faces:
        # gather verts and sheet positions of caustic face
        caustic_verts = []
        for ls_vert in ls_face.verts:
            sheet_pos = ls_vert_to_sheet[ls_vert]
            vert = sheet_to_caustic_vert.get(sheet_pos)
            if vert is not None:
                caustic_verts.append(vert)

        # create edge or face from vertices
        if len(caustic_verts) == 2:
            # can only create an edge here, but this edge may already exist
            # if we created a neighboring face before
            if caustic_bm.edges.get(caustic_verts) is None:
                caustic_bm.edges.new(caustic_verts)
        elif len(caustic_verts) > 2:
            # create new caustic face
            assert len(caustic_verts) == 3, len(caustic_verts)
            caustic_bm.faces.new(caustic_verts)
    ls_bm.free()


def set_caustic_squeeze(caustic_bm, sheet_to_world=None, caustic_to_world=None,
                        faces=None):
    """Set squeeze (= source area / projected area) for the given faces."""
    # if no matrices given assume identity
    if sheet_to_world is None:
        sheet_to_world = Matrix()
    if caustic_to_world is None:
        caustic_to_world = Matrix()

    # if no faces given, iterate over every face
    if faces is None:
        faces = caustic_bm.faces

    # sheet coordinate access
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

    # iterate over caustic faces and calculate squeeze
    squeeze_layer = caustic_bm.loops.layers.uv["Caustic Squeeze"]
    for face in faces:
        assert len(face.verts) == 3, len(face.verts)

        # gather coordinates and sheet positions of caustic face
        source_triangle = []
        target_triangle = []
        for vert in face.verts:
            sheet_pos = vert_to_sheet[vert]
            source_triangle.append(sheet_to_world @ Vector(sheet_pos))
            target_triangle.append(caustic_to_world @ vert.co)

        # set squeeze factor = ratio of source area to projected area
        squeeze = area_tri(*source_triangle) / area_tri(*target_triangle)
        for loop in face.loops:
            loop[squeeze_layer].uv = (0, squeeze)


def set_caustic_face_data(caustic_bm, sheet_to_data, faces=None):
    """Set caustic vertex colors, uv-coordinates (if any) and face normals."""
    if faces is None:
        faces = caustic_bm.faces

    # sheet coordinate access
    sheet_x = caustic_bm.verts.layers.float["Lightsheet X"]
    sheet_y = caustic_bm.verts.layers.float["Lightsheet Y"]
    sheet_z = caustic_bm.verts.layers.float["Lightsheet Z"]

    # create vert -> sheet lookup table for faster access
    vert_to_sheet = dict()
    for vert in caustic_bm.verts:
        sx = vert[sheet_x]
        sy = vert[sheet_y]
        sz = vert[sheet_z]
        sheet_pos = (sx, sy, sz)
        vert_to_sheet[vert] = sheet_pos

    # sheet coordinates uv-layers
    uv_sheet_xy = caustic_bm.loops.layers.uv["Lightsheet XY"]
    uv_sheet_xz = caustic_bm.loops.layers.uv["Lightsheet XZ"]

    # vertex color and uv-layer access (take the first one that we can find)
    color_layer = caustic_bm.loops.layers.color["Caustic Tint"]
    uv_layer = None
    for layer in caustic_bm.loops.layers.uv.values():
        if layer not in (uv_sheet_xy, uv_sheet_xz):
            uv_layer = layer

    # set transplanted uv-coordinates and vertex colors for faces
    for face in faces:
        vert_normal_sum = Vector((0.0, 0.0, 0.0))
        for loop in face.loops:
            # get data for this vertex (if any)
            sheet_pos = vert_to_sheet[loop.vert]
            data = sheet_to_data.get(sheet_pos)
            if data is not None:
                # sheet position to uv-coordinates
                sx, sy, sz = sheet_pos
                loop[uv_sheet_xy].uv = (sx, sy)
                loop[uv_sheet_xz].uv = (sx, sz)

                # set face data
                loop[color_layer] = tuple(data.color) + (1,)
                if uv_layer is not None:
                    assert data.uv is not None
                    loop[uv_layer].uv = data.uv
                vert_normal_sum += data.normal

        # if face normal does not point in the same general direction as
        # the averaged vertex normal, then flip the face normal
        face.normal_update()
        if face.normal.dot(vert_normal_sum) < 0:
            face.normal_flip()

    caustic_bm.normal_update()


# -----------------------------------------------------------------------------
# Functions for refine caustics operator
# -----------------------------------------------------------------------------
def refine_caustic(caustic, depsgraph, relative_tolerance):
    """Do one adaptive subdivision of caustic bmesh."""
    # world to caustic object coordinate transformation
    world_to_caustic = caustic.matrix_world.inverted()

    # caustic info
    caustic_info = caustic.caustic_info
    lightsheet = caustic_info.lightsheet
    assert lightsheet is not None
    first_ray = setup_lightsheet_first_ray(lightsheet)

    # convert caustic_info.path into chain for trace
    chain = []
    for item in caustic_info.path:
        chain.append(trace.Link(item.object, item.kind, None))

    # convert caustic to bmesh
    caustic_bm = bmesh.new()
    caustic_bm.from_mesh(caustic.data)

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

    # make sure caches are clean
    trace.meshes_cache.clear()
    material.materials_cache.clear()

    def trace_midpoint(edge):
        # get sheet positions of edge endpoints
        vert1, vert2 = edge.verts
        sheet_pos1 = Vector(vert_to_sheet[vert1])
        sheet_pos2 = Vector(vert_to_sheet[vert2])

        # calculate coordinates of edge midpoint and trace ray
        sheet_mid = (sheet_pos1 + sheet_pos2) / 2
        ray = first_ray(sheet_mid)
        data = trace.trace_along_chain(ray, depsgraph, chain)

        return tuple(sheet_mid), data

    # gather all edges that we have to split
    refine_edges = set()  # edges to subdivide
    edge_to_sheet = dict()  # {edge: sheet pos of midpoint}
    sheet_to_data = dict()  # {sheet pos: target data}

    # gather edges that exceed the tolerance
    for edge in (ed for ed in caustic_bm.edges if ed.seam):
        # calc midpoint and target coordinates
        sheet_mid, target_data = trace_midpoint(edge)
        edge_to_sheet[edge] = sheet_mid
        sheet_to_data[sheet_mid] = target_data

        # calc error and wether we should keep the edge
        if target_data is None:
            refine_edges.add(edge)
        else:
            vert1, vert2 = edge.verts
            edge_mid = (vert1.co + vert2.co) / 2
            co = world_to_caustic @ target_data.position
            rel_err = (edge_mid - co).length / edge.calc_length()
            if rel_err >= relative_tolerance:
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
        if edge not in edge_to_sheet:
            sheet_mid, target_data = trace_midpoint(edge)
            edge_to_sheet[edge] = sheet_mid
            sheet_to_data[sheet_mid] = target_data

    # cleanup generated meshes and caches
    for obj in trace.meshes_cache:
        obj.to_mesh_clear()
    trace.meshes_cache.clear()
    material.materials_cache.clear()

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

    # gather newly added vertices
    dirty_verts = set()
    for item in splits['geom_inner']:
        if isinstance(item, bmesh.types.BMVert):
            dirty_verts.add(item)

    # gather edges that we may have to split next
    dirty_edges = set()
    for item in splits['geom']:
        if isinstance(item, bmesh.types.BMEdge):
            dirty_edges.add(item)

    # verify newly added vertices
    delete_verts = []
    for edge in refine_edges:
        # find the newly added vertex, it is one of the endpoints of a
        # refined edge
        v1, v2 = edge.verts
        if v1 in dirty_verts:
            vert = v1
        else:
            assert v2 in dirty_verts, (v1.co, v2.co)
            vert = v2
        assert vert not in vert_to_sheet

        # get data for this vertex
        sheet_pos = edge_to_sheet[edge]
        data = sheet_to_data[sheet_pos]
        if data is None:
            delete_verts.append(vert)
            del edge_to_sheet[edge]
            del sheet_to_data[sheet_pos]
        else:
            # set correct vertex coordinates
            vert.co = world_to_caustic @ data.position

            # set sheet coords
            sx, sy, sz = sheet_pos
            vert[sheet_x] = sx
            vert[sheet_y] = sy
            vert[sheet_z] = sz

    # remove verts that have no target
    dirty_verts.difference_update(delete_verts)
    bmesh.ops.delete(caustic_bm, geom=delete_verts, context='VERTS')

    # gather the faces where we changed at least one vertex
    dirty_faces = set()
    for vert in dirty_verts:
        for face in vert.link_faces:
            dirty_faces.add(face)

    # recalculate squeeze for dirty faces
    sheet_to_world = lightsheet.matrix_world
    caustic_to_world = caustic.matrix_world
    set_caustic_squeeze(caustic_bm, sheet_to_world, caustic_to_world,
                        faces=dirty_faces)

    # set face data from color_dict, uv_dict and normal_dict
    set_caustic_face_data(caustic_bm, sheet_to_data, faces=dirty_faces)

    # mark edges for next refinement step
    for edge in caustic_bm.edges:
        edge.seam = edge in dirty_edges

    # select only the dirty verts
    for face in caustic_bm.faces:
        face.select_set(False)
    for vert in dirty_verts:
        vert.select_set(True)

    # convert bmesh back to object
    caustic_bm.to_mesh(caustic.data)
    caustic_bm.free()


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
