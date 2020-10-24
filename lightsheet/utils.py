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

- bmesh_delete_loose (reimplementation of bpy.ops.mesh.delete_loose)

LIGHTSHEET_OT_create_lightsheet:
- create_bmesh_square
- create_bmesh_disk
- create_bmesh_sphere
- convert_bmesh_to_lightsheet
- verify_lightsheet_coordinate_layers

LIGHTSHEET_OT_trace_lighsheet:
- verify_object_is_lighsheet
- setup_sheet_property
- trace_lightsheet
- setup_lightsheet_first_ray
- chain_complexity
- convert_caustics_to_objects
- setup_caustic_bmesh
- fill_caustic_faces
- set_caustic_squeeze
- set_caustic_face_data

LIGHTSHEET_OT_refine_caustics:
- refine_caustic
- split_caustic_edges
- grow_caustic_boundary

LIGHTSHEET_OT_finalize_caustics:
- finalize_caustic
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


def bmesh_delete_loose(bm, use_verts=True, use_edges=True):
    """Delete loose vertices or edges."""
    # cleanup loose edges
    if use_edges:
        loose_edges = [edge for edge in bm.edges if edge.is_wire]
        bmesh.ops.delete(bm, geom=loose_edges, context='EDGES')

    # cleanup loose verts
    if use_verts:
        loose_verts = [vert for vert in bm.verts if not vert.link_edges]
        bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')


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
    # to create a circle we create a square and then cut out the circle
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


def convert_bmesh_to_lightsheet(bm, light):
    """Convert a bmesh or object to a lightsheet for the given light."""
    # verify coordinates of vertices
    if light.data.type == 'SUN':
        # lighsheet should be in xy-plane
        for vert in bm.verts:
            vert.co.z = 0.0
    else:
        assert light.data.type in ('POINT', 'SPOT'), light.data.type
        # lightsheet should be spherical
        for vert in bm.verts:
            vert.co.normalize()
    verify_lightsheet_coordinate_layers(bm)

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


def verify_lightsheet_coordinate_layers(bm):
    """Verify that bmesh has sheet coordinate layers and update coordinates."""
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

    # calculate the sheet coordinates
    vert_to_sheet = dict()  # cache for later
    for vert in bm.verts:
        sx, sy, sz = vert.co  # sheet coordinates = vert coordinates
        vert[sheet_x] = sx
        vert[sheet_y] = sy
        vert[sheet_z] = sz
        vert_to_sheet[vert] = (sx, sy, sz)

    # verify that mesh has uv-layers for sheet coordinates
    uv_sheet_xy = bm.loops.layers.uv.get("Lightsheet XY")
    if uv_sheet_xy is None:
        uv_sheet_xy = bm.loops.layers.uv.new("Lightsheet XY")
    uv_sheet_xz = bm.loops.layers.uv.get("Lightsheet XZ")
    if uv_sheet_xz is None:
        uv_sheet_xz = bm.loops.layers.uv.new("Lightsheet XZ")

    # set sheet uv-coordinates
    for face in bm.faces:
        for loop in face.loops:
            sx, sy, sz = vert_to_sheet[loop.vert]  # retrieve from cache
            loop[uv_sheet_xy].uv = (sx, sy)
            loop[uv_sheet_xz].uv = (sx, sz)


# -----------------------------------------------------------------------------
# Functions for trace lightsheet operator
# -----------------------------------------------------------------------------
def verify_object_is_lightsheet(obj):
    """Setup the given object so that it can be used as a lightsheet."""
    assert obj is not None and isinstance(obj, bpy.types.Object)

    # verify name
    if "lightsheet" not in obj.name.lower():
        obj.name = f"Lightsheet {obj.name}"

    # verify coordinate layers
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    verify_lightsheet_coordinate_layers(bm)
    bmesh.ops.triangulate(bm, faces=bm.faces)  # ensure triangles
    bm.to_mesh(obj.data)
    bm.free()

    return obj


def setup_sheet_property(bm):
    """Return getter and setter functions for sheet coordinate access."""
    # vertex layers for coordinate access
    sheet_x = bm.verts.layers.float["Lightsheet X"]
    sheet_y = bm.verts.layers.float["Lightsheet Y"]
    sheet_z = bm.verts.layers.float["Lightsheet Z"]

    # getter
    def get_sheet(vert):
        return Vector((vert[sheet_x], vert[sheet_y], vert[sheet_z]))

    # setter
    def set_sheet(vert, sheet_pos):
        vert[sheet_x], vert[sheet_y], vert[sheet_z] = sheet_pos

    return get_sheet, set_sheet


def trace_lightsheet(lightsheet, depsgraph, max_bounces):
    """Trace rays from lighsheet and return caustic coordinates in dict."""
    # hide lightsheets and caustics from raycast
    hidden = []
    for obj in depsgraph.view_layer.objects:
        if "lightsheet" in obj.name.lower() or obj.caustic_info.path:
            hidden.append((obj, obj.hide_viewport))
            obj.hide_viewport = True

    # convert lightsheet to bmesh
    lightsheet_bm = bmesh.new(use_operators=False)
    lightsheet_bm.from_mesh(lightsheet.data)

    # coordinates of source position on lighsheet
    get_sheet, _ = setup_sheet_property(lightsheet_bm)

    # make sure caches are clean
    trace.meshes_cache.clear()
    material.materials_cache.clear()

    # traced = {chain: {sheet_pos: CausticVert(location, color, uv, normal)}}
    traced = defaultdict(dict)
    try:
        first_ray = setup_lightsheet_first_ray(lightsheet)
        for vert in lightsheet_bm.verts:
            sheet_pos = get_sheet(vert)
            trace.trace_scene_recursive(first_ray(sheet_pos), tuple(sheet_pos),
                                        depsgraph, max_bounces, traced)
    finally:
        # cleanup generated meshes and caches
        lightsheet_bm.free()
        for obj in trace.meshes_cache:
            obj.to_mesh_clear()
        trace.meshes_cache.clear()
        material.materials_cache.clear()

        # restore original state for hidden lightsheets and caustics
        for obj, state in hidden:
            obj.hide_viewport = state

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


def chain_complexity(chain):
    """Calculate a number representing the complexity of the given ray path."""
    weights = {'DIFFUSE': 1, 'TRANSPARENT': 2, 'REFLECT': 3, 'REFRACT': 4}
    cplx = 0
    for link in chain:
        # each link gets its own power of ten
        cplx = 10 * cplx + weights[link.kind]
    return cplx


def convert_caustic_to_objects(lightsheet, chain, sheet_to_data):
    """Convert caustic bmesh to blender object with filled in faces."""
    # setup and fill caustic bmesh
    caustic_bm = setup_caustic_bmesh(sheet_to_data)
    fill_caustic_faces(caustic_bm, lightsheet)
    bmesh_delete_loose(caustic_bm)
    set_caustic_squeeze(caustic_bm, matrix_sheet=lightsheet.matrix_world)
    set_caustic_face_data(caustic_bm, sheet_to_data)

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


def setup_caustic_bmesh(sheet_to_data):
    """Create empty bmesh with data layers used for caustics"""
    caustic_bm = bmesh.new()

    # create vertex color layer for caustic tint
    caustic_bm.loops.layers.color.new("Caustic Tint")

    # create uv-layer for transplanted coordinates (if any)
    if all(data.uv is not None for data in sheet_to_data.values()):
        caustic_bm.loops.layers.uv.new("UVMap")

    # create uv-layer for squeeze = ratio of source area to projected area
    caustic_bm.loops.layers.uv.new("Caustic Squeeze")

    # create vertex layers for sheet coordinates
    sheet_x = caustic_bm.verts.layers.float.new("Lightsheet X")
    sheet_y = caustic_bm.verts.layers.float.new("Lightsheet Y")
    sheet_z = caustic_bm.verts.layers.float.new("Lightsheet Z")

    # create uv-layers for sheet coordinates
    caustic_bm.loops.layers.uv.new("Lightsheet XY")
    caustic_bm.loops.layers.uv.new("Lightsheet XZ")

    # create vertices and given positions and set sheet coordinates
    for sheet_pos, data in sheet_to_data.items():
        assert data is not None, sheet_pos
        vert = caustic_bm.verts.new(data.position)
        vert[sheet_x], vert[sheet_y], vert[sheet_z] = sheet_pos

    return caustic_bm


def fill_caustic_faces(caustic_bm, lightsheet):
    """Fill in faces of caustic bmesh using faces from lightsheet."""
    assert len(caustic_bm.edges) == 0, len(caustic_bm.edges)
    assert len(caustic_bm.faces) == 0, len(caustic_bm.faces)

    # convert lightsheet to bmesh
    ls_bm = bmesh.new()
    ls_bm.from_mesh(lightsheet.data)

    # get sheet coordinates from lightsheet
    ls_get_sheet, _ = setup_sheet_property(ls_bm)

    # get sheet coordinates from caustic
    caustic_get_sheet, _ = setup_sheet_property(caustic_bm)

    # create sheet -> vert lookup table for faster access
    # note that mathutils.Vector is mutable and must be converted to tuple
    # before using it as a key for dict
    sheet_to_caustic_vert = dict()
    for vert in caustic_bm.verts:
        sheet_key = tuple(caustic_get_sheet(vert))
        assert sheet_key not in sheet_to_caustic_vert
        sheet_to_caustic_vert[sheet_key] = vert

    # iterate over lighsheet faces and create corresponding caustic faces
    for ls_face in ls_bm.faces:
        # gather verts and sheet positions of caustic face
        caustic_verts = []
        for ls_vert in ls_face.verts:
            sheet_key = tuple(ls_get_sheet(ls_vert))
            vert = sheet_to_caustic_vert.get(sheet_key)
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


def set_caustic_squeeze(caustic_bm, matrix_sheet=None, matrix_caustic=None,
                        faces=None):
    """Set squeeze (= source area / projected area) for the given faces."""
    # if no matrices given assume identity
    if matrix_sheet is None:
        matrix_sheet = Matrix()
    if matrix_caustic is None:
        matrix_caustic = Matrix()

    # if no faces given, iterate over every face
    if faces is None:
        faces = caustic_bm.faces

    # sheet coordinate access
    get_sheet, _ = setup_sheet_property(caustic_bm)

    # iterate over caustic faces and calculate squeeze
    squeeze_layer = caustic_bm.loops.layers.uv["Caustic Squeeze"]
    for face in faces:
        assert len(face.verts) == 3, len(face.verts)

        # gather coordinates and sheet positions of caustic face
        source_triangle = []
        target_triangle = []
        for vert in face.verts:
            source_triangle.append(matrix_sheet @ get_sheet(vert))
            target_triangle.append(matrix_caustic @ vert.co)

        # set squeeze factor = ratio of source area to projected area
        source_area = area_tri(*source_triangle)
        target_area = area_tri(*target_triangle)
        squeeze = source_area / max(target_area, 1e-15)
        for loop in face.loops:
            loop[squeeze_layer].uv = (0, squeeze)


def set_caustic_face_data(caustic_bm, sheet_to_data, faces=None):
    """Set caustic vertex colors, uv-coordinates (if any) and face normals."""
    if faces is None:
        faces = caustic_bm.faces

    # sheet coordinate access
    get_sheet, _ = setup_sheet_property(caustic_bm)

    # sheet coordinates uv-layers
    uv_sheet_xy = caustic_bm.loops.layers.uv["Lightsheet XY"]
    uv_sheet_xz = caustic_bm.loops.layers.uv["Lightsheet XZ"]

    # vertex color and uv-layer access (take the first one that we can find)
    color_layer = caustic_bm.loops.layers.color["Caustic Tint"]
    reserved_layers = {"Lightsheet XY", "Lightsheet XZ", "Caustic Squeeze"}
    uv_layer = None
    for layer in caustic_bm.loops.layers.uv.values():
        if layer.name not in reserved_layers:
            uv_layer = layer

    # set transplanted uv-coordinates and vertex colors for faces
    for face in faces:
        vert_normal_sum = Vector((0.0, 0.0, 0.0))
        for loop in face.loops:
            # get data for this vertex (if any)
            sheet_key = tuple(get_sheet(loop.vert))
            data = sheet_to_data.get(sheet_key)
            if data is not None:
                # sheet position to uv-coordinates
                sx, sy, sz = sheet_key
                loop[uv_sheet_xy].uv = (sx, sy)
                loop[uv_sheet_xz].uv = (sx, sz)

                # set face data
                loop[color_layer] = tuple(data.color) + (1,)
                if uv_layer is not None:
                    assert data.uv is not None, uv_layer.name
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
def refine_caustic(caustic, depsgraph, relative_tolerance, grow_boundary=True):
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
    get_sheet, _ = setup_sheet_property(caustic_bm)

    # make sure caches are clean
    trace.meshes_cache.clear()
    material.materials_cache.clear()

    def calc_sheet_midpoint(edge):
        vert1, vert2 = edge.verts
        sheet_pos1 = get_sheet(vert1)
        sheet_pos2 = get_sheet(vert2)
        sheet_mid = (sheet_pos1 + sheet_pos2) / 2
        return sheet_mid

    # gather all edges that we have to split
    refine_edges = dict()  # {edge: sheet pos of midpoint}
    sheet_to_data = dict()  # {sheet pos: target data (trace.CausticData)}

    # gather edges that exceed the tolerance
    deleted_edges = set()
    for edge in (ed for ed in caustic_bm.edges if ed.seam):
        # calc midpoint and target data
        sheet_mid = calc_sheet_midpoint(edge)
        ray = first_ray(sheet_mid)
        target_data = trace.trace_along_chain(ray, depsgraph, chain)
        sheet_to_data[tuple(sheet_mid)] = target_data

        if target_data is None:
            # edge will be deleted, split it and see later what happens
            deleted_edges.add(edge)
            refine_edges[edge] = sheet_mid
        else:
            # calc error and wether we should keep the edge
            vert1, vert2 = edge.verts
            edge_mid = (vert1.co + vert2.co) / 2
            mid_target = world_to_caustic @ target_data.position

            rel_err = (edge_mid - mid_target).length / edge.calc_length()
            if rel_err >= relative_tolerance:
                refine_edges[edge] = sheet_mid

    if grow_boundary:
        # edges that belong to a face where at least one edge will be deleted
        future_boundary_edges = set()
        for face in caustic_bm.faces:
            if any(edge in deleted_edges for edge in face.edges):
                # face will disappear => new boundary edges
                for edge in face.edges:
                    future_boundary_edges.add(edge)

        # include all edges of (present or future) boundary faces
        for edge in caustic_bm.edges:
            if edge.is_boundary or edge in future_boundary_edges:
                for face in edge.link_faces:
                    for other_edge in face.edges:
                        sheet_mid = calc_sheet_midpoint(other_edge)
                        refine_edges[other_edge] = sheet_mid

    # modify bmesh
    split_verts, split_edges = split_caustic_edges(caustic_bm, refine_edges)
    if grow_boundary:
        # find any offending vertices?
        is_growable = True
        for vert in caustic_bm.verts:
            if vert.is_boundary and len(vert.link_edges) > 6:
                sheet_vec = get_sheet(vert)
                print(f"Sheet: {sheet_vec}, links: {len(vert.link_edges)}")
                is_growable = False

        if is_growable:
            boundary_verts = grow_caustic_boundary(caustic_bm)
        else:
            # raise exception
            msg = "cannot grow boundary, found verts with too many neighbours"
            info = "printed sheet coordinates of offending points to terminal"
            raise RuntimeError(f"{msg}; {info}")
    else:
        boundary_verts = []

    # ensure triangles
    triang_less = [face for face in caustic_bm.faces if len(face.edges) > 3]
    if triang_less:
        print(f"We have to triangulate {len(triang_less)} faces")
        bmesh.ops.triangulate(caustic_bm, faces=triang_less)

    # verify newly added vertices
    new_verts = split_verts + boundary_verts
    dead_verts = []
    for vert in new_verts:
        # get sheet coords and convert to key for dict (cannot use Vectors as
        # keys because they are mutable)
        sheet_pos = get_sheet(vert)
        sheet_key = tuple(sheet_pos)

        # trace ray if necessary
        if sheet_key in sheet_to_data:
            target_data = sheet_to_data[sheet_key]
        else:
            ray = first_ray(sheet_pos)
            target_data = trace.trace_along_chain(ray, depsgraph, chain)
            sheet_to_data[sheet_key] = target_data

        # set coordinates or mark vertex for deletion
        if target_data is None:
            dead_verts.append(vert)
            del sheet_to_data[sheet_key]
        else:
            # set correct vertex coordinates
            vert.co = world_to_caustic @ target_data.position

    # remove verts that have no target
    bmesh.ops.delete(caustic_bm, geom=dead_verts, context='VERTS')
    bmesh_delete_loose(caustic_bm)
    new_verts = [vert for vert in new_verts if vert.is_valid]
    assert all(data is not None for data in sheet_to_data.values())

    # cleanup generated meshes and caches
    for obj in trace.meshes_cache:
        obj.to_mesh_clear()
    trace.meshes_cache.clear()
    material.materials_cache.clear()

    # gather the edges that we may split next and the faces where we changed at
    # least one vertex (and now have to recalculate face data)
    dirty_edges = set(split_edges)
    for vert in boundary_verts:
        if vert.is_valid:
            for ed in vert.link_edges:
                dirty_edges.add(ed)
    dirty_faces = {face for vert in new_verts for face in vert.link_faces}

    # recalculate squeeze and set face data for dirty faces
    set_caustic_squeeze(caustic_bm, matrix_sheet=lightsheet.matrix_world,
                        matrix_caustic=caustic.matrix_world, faces=dirty_faces)
    set_caustic_face_data(caustic_bm, sheet_to_data, faces=dirty_faces)

    # mark edges for next refinement step
    for edge in caustic_bm.edges:
        edge.seam = edge in dirty_edges

    # select only the newly added verts
    for face in caustic_bm.faces:
        face.select_set(False)
    for vert in new_verts:
        vert.select_set(True)

    # convert bmesh back to object
    caustic_bm.to_mesh(caustic.data)
    caustic_bm.free()


def split_caustic_edges(caustic_bm, refine_edges):
    """Subdivide the given edges and return the new vertices."""
    # sheet coordinate access
    get_sheet, set_sheet = setup_sheet_property(caustic_bm)

    def calc_sheet_midpoint(edge):
        vert1, vert2 = edge.verts
        sheet_pos1 = get_sheet(vert1)
        sheet_pos2 = get_sheet(vert2)
        sheet_mid = (sheet_pos1 + sheet_pos2) / 2
        return sheet_mid

    # balance refinement: if two edges of a triangle will be refined, also
    # subdivide the other edge => after splitting we always get triangles
    last_added = list(refine_edges.keys())
    newly_added = list()
    while last_added:  # crawl through the mesh and select edges that we need
        for edge in last_added:  # only last edges can change futher selection
            for face in edge.link_faces:
                assert len(face.edges) == 3  # must be a triangle
                not_refined = [
                    ed for ed in face.edges if ed not in refine_edges]
                if len(not_refined) == 1:
                    ed = not_refined[0]
                    refine_edges[ed] = calc_sheet_midpoint(ed)
                    newly_added.append(ed)

        last_added = newly_added
        newly_added = list()

    # split edges
    edgelist = list(refine_edges.keys())
    splits = bmesh.ops.subdivide_edges(caustic_bm, edges=edgelist, cuts=1,
                                       use_grid_fill=True,
                                       use_single_edge=True)

    # gather newly added vertices
    dirty_verts = set()
    for item in splits['geom_inner']:
        if isinstance(item, bmesh.types.BMVert):
            dirty_verts.add(item)

    # get all newly added verts and set their sheet coordinates
    split_verts = []
    for edge, sheet_pos in refine_edges.items():
        # one of the endpoints of a refined edge is a newly added vertex
        v1, v2 = edge.verts
        vert = v1 if v1 in dirty_verts else v2
        assert vert in dirty_verts, sheet_pos

        set_sheet(vert, sheet_pos)
        split_verts.append(vert)

    # gather edges that were split
    split_edges = []
    for item in splits['geom']:
        if isinstance(item, bmesh.types.BMEdge):
            split_edges.append(item)

    return split_verts, split_edges


def grow_caustic_boundary(caustic_bm):
    """Exand the boundary of the given caustic outwards in the lightsheet."""
    # names of places:
    # boundary: at the boundary of the original mesh
    # outside:  extended outwards, at the boundary of the new mesh
    # fan:      vertices around a central point (corner vertex at boundary)

    # sheet coordinate access
    get_sheet, set_sheet = setup_sheet_property(caustic_bm)

    # categorize connections of vertices at the boundary, note that adding new
    # faces will change .link_edges of boundary vertices, therefore we have to
    # save the original connections here before creating new faces
    original_boundary_connections = dict()
    for vert in (v for v in caustic_bm.verts if v.is_boundary):
        # categorize linked edges based on connected faces
        wire_edges, boundary_edges, inside_edges = [], [], []
        for edge in vert.link_edges:
            if edge.is_wire:
                assert len(edge.link_faces) == 0
                wire_edges.append(edge)
            elif edge.is_boundary:
                assert len(edge.link_faces) == 1
                boundary_edges.append(edge)
            else:
                assert len(edge.link_faces) == 2
                inside_edges.append(edge)

        assert len(boundary_edges) in (0, 2, 4), len(boundary_edges)
        conn = (wire_edges, boundary_edges, inside_edges)
        original_boundary_connections[vert] = conn

    # record new vertices as they are created
    outside_verts = []

    def create_vert(sheet_pos):
        new_vert = caustic_bm.verts.new()
        outside_verts.append(new_vert)
        set_sheet(new_vert, sheet_pos)
        return new_vert

    # record new faces as they are created
    new_faces = []  # list of new faces

    def create_triangle(v1, v2, v3):
        face = caustic_bm.faces.new((v1, v2, v3))
        new_faces.append(face)
        return face

    # create outside pointing triangle with a boundary edge at the base and an
    # outside vertex at the tip
    boundary_edge_to_outside_vert = dict()  # {boundary edge: new outside vert}
    original_boundary_edges = [ed for ed in caustic_bm.edges if ed.is_boundary]
    for edge in original_boundary_edges:
        # get verts that are connected by the edge
        vert_first, vert_second = edge.verts

        # get the vertex opposite of the edge
        assert len(edge.link_faces) == 1
        face = edge.link_faces[0]
        assert len(face.verts) == 3
        other_verts = [ve for ve in face.verts if ve not in edge.verts]
        assert len(other_verts) == 1
        vert_opposite = other_verts[0]

        # sheet coordinates of the three vertices
        sheet_first = get_sheet(vert_first)
        sheet_second = get_sheet(vert_second)
        sheet_opposite = get_sheet(vert_opposite)

        # mirror opposite vertex across the edge to get the outside vertex
        # want: sheet_midpoint == (sheet_opposite + sheet_outside) / 2
        sheet_midpoint = (sheet_first + sheet_second) / 2
        sheet_outside = 2 * sheet_midpoint - sheet_opposite
        vert_outside = create_vert(sheet_outside)

        # add outside vertex to mappings
        boundary_edge_to_outside_vert[edge] = vert_outside

        # create face
        create_triangle(vert_first, vert_second, vert_outside)

    # create inside pointing trinagle with a boundary vertex at the tip and an
    # outside edge at the base
    targetmap = dict()  # {vert to replace: replacement vert}
    for vert, conn in original_boundary_connections.items():
        sheet_vert = get_sheet(vert)
        wire_edges, boundary_edges, inside_edges = conn

        # degree = number of neighbours
        degree = len(boundary_edges) + len(inside_edges)
        assert 2 <= degree <= 6, (sheet_vert, boundary_edges, inside_edges)

        if degree == 2:
            # 300° outside angle, needs two more vertices to create three faces
            assert len(boundary_edges) == 2, (sheet_vert, boundary_edges)
            # assert len(inside_edges) == 0, (sheet_vert, inside_edges)

            # get sheet coordinates of neighbouring vertices
            verts_fan = []
            for edge in boundary_edges:
                # get other vert in this edge and its sheet coordinates
                vert_opposite = edge.other_vert(vert)
                sheet_opposite = get_sheet(vert_opposite)

                # mirror across vertex
                # want: sheet_vert == (sheet_opposite + sheet_outside) / 2
                sheet_fan = 2 * sheet_vert - sheet_opposite
                vert_fan = create_vert(sheet_fan)

                # add to list
                verts_fan.append((vert_fan, edge))
            vert_fan_a, from_edge_a = verts_fan[0]
            vert_fan_b, from_edge_b = verts_fan[1]

            # get the outside neighbours of vert_fan_a/b, note that because
            # of mirroring we change the order in which we connect the vertices
            vert_outside_a = boundary_edge_to_outside_vert[from_edge_b]
            vert_outside_b = boundary_edge_to_outside_vert[from_edge_a]

            # create three new faces
            create_triangle(vert, vert_outside_a, vert_fan_a)
            create_triangle(vert, vert_fan_a, vert_fan_b)
            create_triangle(vert, vert_fan_b, vert_outside_b)
        elif degree == 3:
            # 240° outside angle, needs one more vertex to create two faces
            assert len(boundary_edges) == 2, (sheet_vert, boundary_edges)
            assert len(inside_edges) == 1, (sheet_vert, inside_edges)

            # get vertex of edge on the inside and its sheet coordinates
            inside_edge = inside_edges[0]
            vert_opposite = inside_edge.other_vert(vert)
            sheet_opposite = get_sheet(vert_opposite)

            # mirror across vertex
            # want: sheet_vert == (sheet_opposite + sheet_fan) / 2
            sheet_fan = 2 * sheet_vert - sheet_opposite
            vert_fan = create_vert(sheet_fan)

            # get neighbours of outside vert
            edge_a, edge_b = boundary_edges
            vert_outside_a = boundary_edge_to_outside_vert[edge_a]
            vert_outside_b = boundary_edge_to_outside_vert[edge_b]

            # create two new faces
            create_triangle(vert, vert_outside_a, vert_fan)
            create_triangle(vert, vert_fan, vert_outside_b)
        elif degree == 4:
            # the vert is at an 180° angle or at an X-shaped intersection
            assert len(boundary_edges) in (2, 4), (sheet_vert, boundary_edges)
            # assert len(inside_edges) in (0, 2), (sheet_vert, inside_edges)

            if len(boundary_edges) == 2:  # 180° angle
                edge_a, edge_b = boundary_edges
                vert_outside_a = boundary_edge_to_outside_vert[edge_a]
                vert_outside_b = boundary_edge_to_outside_vert[edge_b]
                create_triangle(vert, vert_outside_a, vert_outside_b)
            else:  # X-shaped intersection of two 120° angles
                # pairs of boundary edges have the same outside vertex, merge
                # the corresponding pairs of outside vertices
                assert len(boundary_edges) == 4, (sheet_vert, boundary_edges)

                # get outside verts
                outside_stuff = []
                for edge in boundary_edges:
                    vert_outside = boundary_edge_to_outside_vert[edge]
                    sheet_outside = get_sheet(vert_outside)
                    outside_stuff.append((edge, vert_outside, sheet_outside))

                # pair up the verts via distance in sheet
                sheet_outside_a = outside_stuff[0][2]
                outside_stuff.sort(
                    key=lambda item: (item[2] - sheet_outside_a).length
                )
                pair_1 = outside_stuff[0], outside_stuff[1]
                pair_2 = outside_stuff[2], outside_stuff[3]

                # merge pairs
                for pair in (pair_1, pair_2):
                    outside_a, outside_b = pair
                    edge_a, vert_outside_a, sheet_outside_a = outside_a
                    edge_b, vert_outside_b, sheet_outside_b = outside_b

                    # merge vertices at median point
                    sheet_merge = (sheet_outside_a + sheet_outside_b) / 2
                    set_sheet(vert_outside_a, sheet_merge)
                    targetmap[vert_outside_b] = vert_outside_a
        elif degree == 5:
            # 120° outside angle => the boundary edges connected to this vertex
            # have the same outside vertex, merge these vertices
            edge_a, edge_b = boundary_edges

            # get verts to merge and their sheet coordinates
            vert_outside_a = boundary_edge_to_outside_vert[edge_a]
            vert_outside_b = boundary_edge_to_outside_vert[edge_b]
            sheet_outside_a = get_sheet(vert_outside_a)
            sheet_outside_b = get_sheet(vert_outside_b)

            # merge vertices at median point
            sheet_merge = (sheet_outside_a + sheet_outside_b) / 2
            set_sheet(vert_outside_a, sheet_merge)
            targetmap[vert_outside_b] = vert_outside_a
        elif degree == 6:
            # 60° outside angle, one face of a complete hexagon is missing,
            # except triangles that grew from the boundary edges already lie on
            # top of this face
            edge_a, edge_b = boundary_edges
            vert_a = edge_a.other_vert(vert)
            vert_b = edge_b.other_vert(vert)
            vert_outside_a = boundary_edge_to_outside_vert[edge_a]
            vert_outside_b = boundary_edge_to_outside_vert[edge_b]

            # we can merge vert_a and vert_outside_b because they overlap and
            # do the same for vert_b and vert_outside_a, note that when we
            # merge the vertices later one of the faces over edge_a and edge_b
            # will be deleted
            set_sheet(vert_outside_a, get_sheet(vert_b))
            targetmap[vert_outside_a] = vert_b
            set_sheet(vert_outside_b, get_sheet(vert_a))
            targetmap[vert_outside_b] = vert_a
        else:
            # degree < 2 or degree > 6 should not be possible, but it might
            # happen if suddenly a hole starts to appear in an area with uneven
            # subdivision (usually inside the caustic and not at the boundary)
            pass

    # remove the doubled verts (will delete faces if case degree == 6 happend)
    bmesh.ops.weld_verts(caustic_bm, targetmap=targetmap)
    outside_verts = [vert for vert in outside_verts if vert.is_valid]

    # update uv coordinates and vertex colors for new faces
    new_faces = [face for face in new_faces if face.is_valid]
    ret = bmesh.ops.face_attribute_fill(caustic_bm, faces=new_faces,
                                        use_normals=False,  # calc normal later
                                        use_data=True)
    assert len(ret["faces_fail"]) == 0, ret["faces_fail"]

    return outside_verts


# -----------------------------------------------------------------------------
# Functions for finalize caustics operator
# -----------------------------------------------------------------------------
def finalize_caustic(caustic, intensity_threshold):
    """Finalize caustic mesh."""
    # convert from object
    caustic_bm = bmesh.new()
    caustic_bm.from_mesh(caustic.data)

    # smooth out and cleanup
    smooth_caustic_squeeze(caustic_bm)
    cleanup_caustic(caustic_bm, intensity_threshold)
    # TODO for cycles overlapping faces should be stacked in layers

    # convert bmesh back to object
    caustic_bm.to_mesh(caustic.data)
    caustic_bm.free()

    # mark as finalized
    caustic.caustic_info.finalized = True


def smooth_caustic_squeeze(caustic_bm):
    """Poke faces and smooth out caustic squeeze."""
    squeeze_layer = caustic_bm.loops.layers.uv["Caustic Squeeze"]

    # poke every face (i.e. place a vertex in the middle)
    result = bmesh.ops.poke(caustic_bm, faces=caustic_bm.faces,
                            center_mode='MEAN')
    poked_verts = set(result["verts"])

    # interpolate squeeze for the original vertices
    for vert in (ve for ve in caustic_bm.verts if ve not in poked_verts):
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
            print("Found vertex belonging to unpoked face", vert.co)
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


def cleanup_caustic(caustic_bm, intensity_threshold):
    """Remove invisible faces and cleanup resulting mesh."""
    squeeze_layer = caustic_bm.loops.layers.uv["Caustic Squeeze"]
    color_layer = caustic_bm.loops.layers.color["Caustic Tint"]

    # mark faces with intensity less than intensity_threshold
    invisible_faces = []
    for face in caustic_bm.faces:
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

    # delete invisible faces and cleanup mesh
    bmesh.ops.delete(caustic_bm, geom=invisible_faces, context='FACES_ONLY')
    bmesh_delete_loose(caustic_bm)
