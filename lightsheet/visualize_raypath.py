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
"""Visualize the caustic raypath that lead to the selected vertices.

LIGHTSHEET_OT_visualize_raypath: Operator to retrace raypath for selected verts
"""
from time import perf_counter

import bmesh
import bpy
from bpy.types import Operator

from lightsheet import trace, utils


class LIGHTSHEET_OT_visualize_raypath(Operator):
    """Visualize the raypath for the selected vertices of the active caustic"""
    bl_idname = "lightsheet.visualize"
    bl_label = "Visualize Raypath"
    bl_options = {'REGISTER', 'UNDO'}

    num_verts: bpy.props.IntProperty(name="Number of Selected Vertices")

    @classmethod
    def poll(cls, context):
        # operator makes sense only for caustics
        obj = context.object
        return obj is not None and obj.caustic_info.path

    def invoke(self, context, event):
        obj = context.object
        assert obj is not None

        # cancel with error message
        def cancel(obj, reasons):
            msg = f"Cannot visualize raypath for {obj.name} because {reasons}!"
            self.report({"ERROR"}, msg)
            return {'CANCELLED'}

        # check that caustic has a lightsheet
        lightsheet = obj.caustic_info.lightsheet
        if lightsheet is None:
            return cancel(obj, reasons="it has no lightsheet")

        # check that light (parent of lightsheet) is valid
        light = lightsheet.parent
        if light is None or light.type != 'LIGHT':
            return cancel(obj, reasons="lightsheet parent is not a light")

        # check that light type is supported
        light_type = light.data.type
        if light_type not in {'SUN', 'SPOT', 'POINT'}:
            reasons = f"{light_type.capitalize()} lights are not supported"
            return cancel(obj, reasons)

        # count selected vertices and show confirmation dialog
        self.num_verts = sum(1 for vert in obj.data.vertices if vert.select)
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        # draw info about selected vertices for user to confirm
        txt = f"Visualize raypath for {self.num_verts:,} selected vertices?"
        self.layout.label(text=txt)

    def execute(self, context):
        obj = context.object

        # visualize
        with trace.configure_for_trace(context) as depsgraph:
            tic = perf_counter()
            trails = gather_trails(obj, depsgraph)
            path_obj = convert_trails_to_objects(trails, obj)
            toc = perf_counter()

        # add path to caustic collection
        coll = utils.verify_collection_for_scene(context.scene, "caustics")
        coll.objects.link(path_obj)

        # report statistics
        v_stats = f"Retraced {len(trails):,} verts"
        t_stats = f"{toc-tic:.1f}s"
        self.report({"INFO"}, f"{v_stats} in {t_stats}")

        return {"FINISHED"}


# -----------------------------------------------------------------------------
# Functions used by visualize raypath operator
# -----------------------------------------------------------------------------
def gather_trails(caustic, depsgraph):
    """Visualize the raypath of the selected vertices in caustic."""
    # setup rays from lightsheet
    lightsheet = caustic.caustic_info.lightsheet
    assert lightsheet is not None
    first_ray = trace.setup_lightsheet_first_ray(lightsheet)

    # chain for retracing
    chain = []
    for item in caustic.caustic_info.path:
        chain.append(trace.Link(item.object, item.kind, None))

    # convert caustic to bmesh
    caustic_bm = bmesh.new()
    caustic_bm.from_mesh(caustic.data)

    # coordinates of source position on lighsheet
    get_sheet, _ = utils.setup_sheet_property(caustic_bm)

    # (re)trace rays for selected vertices and save trails
    trails = []
    for vert in (v for v in caustic_bm.verts if v.select):
        # trace ray
        ray = first_ray(get_sheet(vert))
        cdata, trail = trace.trace_along_chain(ray, depsgraph, chain)

        # if raypath is invalid something is wrong and we should not continue
        if cdata is None:
            msg = "Existing caustic vertex cannot be projected?!"
            raise RuntimeError(msg)

        # for last vertex add offset so that trail is flush with the caustic
        position = cdata.location + 1e-4 * cdata.perp
        trail[-1] = position
        trails.append(trail)

    # free caustic bmesh
    caustic_bm.free()

    return trails


def convert_trails_to_objects(trails, caustic):
    """Convert trails to Blender object and parent to caustic."""
    path_bm = bmesh.new()
    path_name = f"Raypath of {caustic.name}"

    # create vertices and connect with edges
    for trail in trails:
        # create verts
        trail_verts = [path_bm.verts.new(position) for position in trail]

        # create edges along trail
        for start_vert, end_vert in zip(trail_verts[:-1], trail_verts[1:]):
            path_bm.edges.new((start_vert, end_vert))

    # new mesh data block
    me = bpy.data.meshes.new(path_name)
    path_bm.to_mesh(me)
    path_bm.free()

    # before setting caustic as parent, undo parent transform
    me.transform(caustic.matrix_world.inverted())

    # new object with given mesh
    path_obj = bpy.data.objects.new(path_name, me)
    path_obj.parent = caustic

    return path_obj
