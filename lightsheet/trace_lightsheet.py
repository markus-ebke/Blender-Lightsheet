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
"""Trace the selected lightsheet and create caustic objects.

LIGHTSHEET_OT_trace_lighsheet: Operator for tracing lightsheet

Helper functions:
- verify_object_is_lighsheet
- trace_lightsheet
- convert_caustics_to_objects
- setup_caustic_bmesh
- fill_caustic_faces
"""

from collections import defaultdict
from time import perf_counter

import bmesh
import bpy
from bpy.types import Operator

from lightsheet import material, trace, utils


class LIGHTSHEET_OT_trace_lightsheet(Operator):
    """Trace rays from active lightsheet and create caustics"""
    bl_idname = "lightsheet.trace"
    bl_label = "Trace Lightsheet"
    bl_options = {'REGISTER', 'UNDO'}

    max_bounces: bpy.props.IntProperty(
        name="Max Bounces", description="Maximum number of light bounces",
        default=4, min=0
    )
    dismiss_empty_caustics: bpy.props.BoolProperty(
        name="Dismiss empty caustics",
        description="Don't create caustics for raypaths that produce no faces",
        default=True
    )

    @classmethod
    def poll(cls, context):
        # operator makes sense only for lightsheets (must have light as parent)
        obj = context.object
        if obj is not None and obj.type == 'MESH':
            parent = obj.parent
            return parent is not None and parent.type == 'LIGHT'

        return False

    def invoke(self, context, event):
        # set properties via dialog window
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        # parameters for tracing
        lightsheet = verify_object_is_lightsheet(context.object)
        depsgraph = context.view_layer.depsgraph
        max_bounces = self.max_bounces

        # raytrace lightsheet and convert resulting caustics to objects
        tic = perf_counter()
        traced = trace_lightsheet(lightsheet, depsgraph, max_bounces)
        traced_sorted = sorted(
            traced.items(), key=lambda item: utils.chain_complexity(item[0]))
        caustics = []
        for chain, sheet_to_data in traced_sorted:
            obj = convert_caustic_to_objects(lightsheet, chain, sheet_to_data)

            # if wanted delete empty caustics
            if self.dismiss_empty_caustics and len(obj.data.polygons) == 0:
                bpy.data.objects.remove(obj)
            else:
                caustics.append(obj)
        toc = perf_counter()

        # get or setup collection for caustics
        coll_name = f"Caustics in {context.scene.name}"
        coll = context.scene.collection.children.get(coll_name)
        if coll is None:
            coll = bpy.data.collections.get(coll_name)
            if coll is None:
                coll = bpy.data.collections.new(coll_name)
            context.scene.collection.children.link(coll)

        # get material and add caustic object to caustic collection
        for obj in caustics:
            mat = material.get_caustic_material(lightsheet.parent, obj.parent)
            obj.data.materials.append(mat)
            coll.objects.link(obj)

        # report statistics
        c_stats = "{} caustics".format(len(caustics))
        t_stats = "{:.1f}s".format((toc - tic))
        self.report({"INFO"}, f"Created {c_stats} in {t_stats}")

        return {"FINISHED"}


# -----------------------------------------------------------------------------
# Functions used by trace lightsheet operator
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
    utils.verify_lightsheet_layers(bm)
    bmesh.ops.triangulate(bm, faces=bm.faces)  # ensure triangles
    bm.to_mesh(obj.data)
    bm.free()

    return obj


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
    get_sheet, _ = utils.setup_sheet_property(lightsheet_bm)

    # make sure caches are clean
    trace.cache_clear()
    material.cache_clear()

    # traced = {chain: {sheet_pos: CausticData(location, color, uv, normal)}}
    traced = defaultdict(dict)
    try:
        first_ray = trace.setup_lightsheet_first_ray(lightsheet)
        for vert in lightsheet_bm.verts:
            sheet_pos = get_sheet(vert)
            trace.trace_scene_recursive(first_ray(sheet_pos), tuple(sheet_pos),
                                        depsgraph, max_bounces, traced)
    finally:
        # cleanup generated meshes and caches
        lightsheet_bm.free()
        trace.cache_clear()
        material.cache_clear()

        # restore original state for hidden lightsheets and caustics
        for obj, state in hidden:
            obj.hide_viewport = state

    return traced


def convert_caustic_to_objects(lightsheet, chain, sheet_to_data):
    """Convert caustic bmesh to blender object with filled in faces."""
    # setup and fill caustic bmesh
    caustic_bm = setup_caustic_bmesh(sheet_to_data)
    fill_caustic_faces(caustic_bm, lightsheet)
    utils.bmesh_delete_loose(caustic_bm)
    utils.set_caustic_squeeze(caustic_bm, matrix_sheet=lightsheet.matrix_world)
    utils.set_caustic_face_data(caustic_bm, sheet_to_data)

    # mark all edges of the caustic for refinement
    for edge in caustic_bm.edges:
        edge.seam = True

    # parent of caustic = object in last link
    parent_obj = chain[-1].object
    assert chain[-1].kind == 'DIFFUSE', chain  # check consistency of path

    # consistency check for copied uv-coordinates from the parent object
    if "UVMap" in caustic_bm.loops.layers.uv is not None:
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
    """Create caustic bmesh with vertices and data layers."""
    caustic_bm = bmesh.new()

    # create vertex color layer for caustic tint
    caustic_bm.loops.layers.color.new("Caustic Tint")

    # create uv-layer for transplanted coordinates (if any)
    if any(data.uv is not None for data in sheet_to_data.values()):
        # assume that all data objects have uv coordinates
        caustic_bm.loops.layers.uv.new("UVMap")

    # create vertex layer for face index
    face_index = caustic_bm.verts.layers.int.new("Face Index")

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

        # create caustic vertex, offset so that caustic wraps around object
        position = data.location + 1e-4 * data.normal
        vert = caustic_bm.verts.new(position)

        # setup vertex data
        vert[face_index] = data.face_index
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
    ls_get_sheet, _ = utils.setup_sheet_property(ls_bm)

    # get sheet coordinates from caustic
    caustic_get_sheet, _ = utils.setup_sheet_property(caustic_bm)

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
            if sheet_key in sheet_to_caustic_vert:
                vert = sheet_to_caustic_vert[sheet_key]
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
