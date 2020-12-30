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
- convert_caustics_to_object
- setup_caustic_bmesh
- fill_caustic_faces
- chain_complexity
"""

from collections import defaultdict
from time import process_time as stopwatch

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
        name="Max Bounces",
        description="Maximum number of glossy/transmission/transparent "
        "bounces until hitting a diffuse surface",
        default=4, min=0
    )

    @classmethod
    def poll(cls, context):
        # operator makes sense only for lightsheets (must have light as parent)
        def is_light(obj):
            return obj is not None and obj.type == 'LIGHT'

        if context.selected_objects:
            return all(obj.type == 'MESH' and is_light(obj.parent)
                       for obj in context.selected_objects)
        return False

    def invoke(self, context, event):
        # cancel with error message
        def cancel(obj, reasons):
            msg = f"Cannot trace '{obj.name}' because {reasons}!"
            self.report({"ERROR"}, msg)
            return {'CANCELLED'}

        # cancel operator for area lights
        for obj in context.selected_objects:
            assert obj.parent is not None, obj  # poll failed us!
            light_type = obj.parent.data.type
            if light_type not in ('SUN', 'SPOT', 'POINT'):
                reasons = f"{light_type.lower()} lights are not supported"
                return cancel(obj, reasons)

        # set properties via dialog window
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        # verify lightsheets
        lightsheets = [verify_lightsheet(obj, context.scene)
                       for obj in context.selected_objects]

        # setup progress indicator
        prog = utils.ProgressIndicator(total_jobs=len(lightsheets))

        # raytrace lightsheet and gather all caustics, note that caustics are
        # not yet linked to scene collection, so they will not interfere with
        # lightsheet tracings that may follow if multiple lightsheets were
        # selected
        tic = stopwatch()
        all_caustics = []
        with trace.configure_for_trace(context) as depsgraph:
            for lightsheet in lightsheets:
                prog.start_job(lightsheet.name)
                caustics = trace_lightsheet(lightsheet, depsgraph,
                                            self.max_bounces, prog)
                prog.stop_job()
                all_caustics.extend(caustics)
            prog.end()
        toc = stopwatch()

        # add caustics to the right scene collection
        coll = utils.verify_collection_for_scene(context.scene, "caustics")
        for obj in all_caustics:
            coll.objects.link(obj)

        # report statistics
        # prog.print_stats()
        c_stats = f"{len(all_caustics)} caustic(s)"
        t_stats = f"{toc-tic:.1f}s"
        self.report({"INFO"}, f"Created {c_stats} in {t_stats}")

        return {"FINISHED"}


# -----------------------------------------------------------------------------
# Functions used by trace lightsheet operator
# -----------------------------------------------------------------------------
def verify_lightsheet(obj, scene):
    """Ensure that the given object can be used as a lightsheet."""
    assert obj.type == 'MESH'
    assert obj.parent is not None and obj.parent.type == 'LIGHT'

    # verify coordinate layers
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    utils.verify_lightsheet_layers(bm)
    bmesh.ops.triangulate(bm, faces=bm.faces)  # must be triangle mesh
    bm.to_mesh(obj.data)
    bm.free()

    # make sure that lightsheet belongs to the right collection, otherwise we
    # can't hide it for tracing
    ls_coll = utils.verify_collection_for_scene(scene, "lightsheets")
    if (obj.name not in ls_coll.objects
            or obj is not ls_coll.objects.get(obj.name)):
        ls_coll.objects.link(obj)

    return obj


def trace_lightsheet(lightsheet, depsgraph, max_bounces, prog):
    """Trace rays from lighsheet and return list of caustics."""
    assert lightsheet.parent is not None and lightsheet.parent.type == 'LIGHT'

    # convert lightsheet to bmesh
    lightsheet_bm = bmesh.new(use_operators=False)
    lightsheet_bm.from_mesh(lightsheet.data)

    # coordinates of source position on lighsheet
    get_sheet, _ = utils.setup_sheet_property(lightsheet_bm)

    # trace rays and record results
    prog.start_task("tracing rays", total_steps=len(lightsheet_bm.verts))
    traced = defaultdict(dict)  # traced = {chain: {sheet_pos: CausticData}}
    first_ray = trace.setup_lightsheet_first_ray(lightsheet)
    for vert in lightsheet_bm.verts:
        sheet_pos = get_sheet(vert)

        # trace ray
        result = trace.trace_scene(first_ray(sheet_pos), depsgraph,
                                   max_bounces)

        # add data to traced
        sheet_key = sheet_pos.to_tuple()
        for chain, cdata in result:
            traced[chain][sheet_key] = cdata

        prog.update_progress()
    lightsheet_bm.free()

    # convert to Blender objects
    prog.start_task("converting to objects", total_steps=len(traced))
    traced_sorted = sorted(traced.items(),
                           key=lambda item: chain_complexity(item[0]))
    caustics = []
    for item in traced_sorted:
        chain, sheet_to_data = item
        caustic = convert_caustic_to_object(lightsheet, chain, sheet_to_data)
        if caustic is not None:
            caustics.append(caustic)
        prog.update_progress()

    return caustics


def convert_caustic_to_object(lightsheet, chain, sheet_to_data):
    """Convert caustic bmesh to Blender object with filled in faces."""
    # setup and fill caustic bmesh
    caustic_bm = setup_caustic_bmesh(sheet_to_data)
    fill_caustic_faces(caustic_bm, lightsheet)
    utils.bmesh_delete_loose(caustic_bm)

    # dismiss caustics without faces
    if len(caustic_bm.faces) == 0:
        return None

    # paint caustic
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
    name = f"{lightsheet.parent.name} on {parent_obj.name} Caustic"

    # new mesh data block
    me = bpy.data.meshes.new(name)
    caustic_bm.to_mesh(me)
    caustic_bm.free()

    # rename transplanted uvmap to avoid confusion
    if parent_obj.data.uv_layers:
        parent_uv_layer = parent_obj.data.uv_layers.active
        assert parent_obj.data.uv_layers.active is not None
        me.uv_layers["UVMap"].name = parent_uv_layer.name

    # before setting parent, undo parent transform
    me.transform(parent_obj.matrix_world.inverted())

    # new object with given mesh
    caustic = bpy.data.objects.new(name, me)
    caustic.parent = parent_obj

    # add material
    mat = material.get_caustic_material(lightsheet.parent, parent_obj)
    caustic.data.materials.append(mat)

    # for Cycles we need to make the caustic visible to diffuse rays only
    caustic.cycles_visibility.camera = False
    caustic.cycles_visibility.diffuse = True
    caustic.cycles_visibility.glossy = False
    caustic.cycles_visibility.transmission = False
    caustic.cycles_visibility.scatter = False
    caustic.cycles_visibility.shadow = False

    # add shrinkwrap modifier
    mod = caustic.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
    mod.wrap_mode = 'ABOVE_SURFACE'
    mod.target = caustic.parent
    mod.offset = 1e-4  # * chain_complexity(chain)

    # fill out caustic_info property
    caustic.caustic_info.lightsheet = lightsheet
    for obj, kind, _ in chain:
        item = caustic.caustic_info.path.add()
        item.object = obj
        item.kind = kind

    return caustic


def setup_caustic_bmesh(sheet_to_data):
    """Create caustic bmesh with vertices and data layers."""
    caustic_bm = bmesh.new()

    # create vertex color layer for caustic tint
    caustic_bm.loops.layers.color.new("Caustic Tint")

    # create uv-layer for transplanted coordinates (if any)
    if any(cdata.uv is not None for cdata in sheet_to_data.values()):
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
    for sheet_pos, cdata in sheet_to_data.items():
        assert cdata is not None, sheet_pos

        # create caustic vertex
        vert = caustic_bm.verts.new(cdata.location)

        # setup vertex data
        vert[face_index] = cdata.face_index
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
        sheet_key = caustic_get_sheet(vert).to_tuple()
        assert sheet_key not in sheet_to_caustic_vert
        sheet_to_caustic_vert[sheet_key] = vert

    # iterate over lighsheet faces and create corresponding caustic faces
    for ls_face in ls_bm.faces:
        # gather verts and sheet positions of caustic face
        caustic_verts = []
        for ls_vert in ls_face.verts:
            sheet_key = ls_get_sheet(ls_vert).to_tuple()
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


def chain_complexity(chain):
    """Calculate a number representing the complexity of the given ray path."""
    weights = {'TRANSPARENT': 0, 'REFLECT': 1, 'REFRACT': 2}

    # check that only the last link is diffuse
    assert chain[-1].kind == 'DIFFUSE', chain[-1]
    assert all(link.kind in weights for link in chain[:-1])

    # complexity = kind of interactions in base 3
    cplx = 0
    for link in chain[:-1]:
        cplx = 3 * cplx + weights[link.kind]

    return cplx
