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
"""Helper functions used by more than one operator.

- bmesh_delete_loose (bmesh reimplementation of bpy.ops.mesh.delete_loose)
- verify_lightsheet_layers (used by create_lightsheet and trace_lightsheet)
- setup_sheet_property (used by trace_lightsheet and refine_caustics)
- setup_lightsheet_first_ray (used by trace_lightsheet and refine_caustics)
- set_caustic_squeeze (used by trace_lightsheet and refine_caustics)
- set_caustic_face_data (used by trace_lightsheet and refine_caustics)
"""

import bmesh
from mathutils import Color, Matrix, Vector
from mathutils.geometry import area_tri

from lightsheet import trace


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


def verify_lightsheet_layers(bm):
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
    vert_to_sheet = dict()  # lookup table for later
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
            sx, sy, sz = vert_to_sheet[loop.vert]
            loop[uv_sheet_xy].uv = (sx, sy)
            loop[uv_sheet_xz].uv = (sx, sz)


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


def set_caustic_squeeze(caustic_bm, matrix_sheet=None, matrix_caustic=None,
                        verts=None):
    """Calc squeeze (= source area / projected area) for the given verts."""
    # if no matrices given assume identity
    if matrix_sheet is None:
        matrix_sheet = Matrix()
    if matrix_caustic is None:
        matrix_caustic = Matrix()

    # if no faces given, iterate over every face
    if verts is None:
        verts = caustic_bm.verts

    # sheet coordinate access
    get_sheet, _ = setup_sheet_property(caustic_bm)

    # face source and target area cache
    face_to_area = dict()

    # the squeeze factor models how strongly a bundle of rays gets focused at
    # the target, we estimate the true squeeze value at a given caustic vertex
    # from the source and target area of a small region around the vertex
    squeeze_layer = caustic_bm.loops.layers.uv["Caustic Squeeze"]
    for vert in verts:
        # imagine we merge the faces connected to vert into one bigger polygon
        # with the vert somewhere in the middle, then the source and target
        # area of this polygon is the sum of the source and target areas of
        # the individual faces
        source_area_sum = 0.0
        target_area_sum = 0.0
        for face in vert.link_faces:
            if face in face_to_area:
                # found face in cache
                source_area, target_area = face_to_area[face]
            else:
                assert len(face.verts) == 3, face.verts[:]

                # gather coordinates and sheet positions of caustic face
                source_triangle = []
                target_triangle = []
                for face_vert in face.verts:
                    source_triangle.append(matrix_sheet @ get_sheet(face_vert))
                    target_triangle.append(matrix_caustic @ face_vert.co)

                # compute area
                source_area = area_tri(*source_triangle)
                target_area = area_tri(*target_triangle)

                # add to cache
                face_to_area[face] = (source_area, target_area)

            # accumulate areas
            source_area_sum += source_area
            target_area_sum += target_area

        # squeeze factor = ratio of source area to projected area
        squeeze = source_area_sum / max(target_area_sum, 1e-15)
        for loop in vert.link_loops:
            loop[squeeze_layer].uv[1] = squeeze


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
