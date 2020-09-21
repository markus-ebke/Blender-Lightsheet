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
"""Operators that do stuff with lightsheets.

LIGHTSHEET_OT_create_lightsheet: Create a lightsheet, the kind of sheet depends
on the light type.
LIGHTSHEET_OT_trace_lighsheet: Trace the selected lightsheet and create caustic
objects.
LIGHTSHEET_OT_finalize_caustics: Smooth out and cleanup selected caustics.
"""

from functools import partial
from math import tan
from time import perf_counter

import bmesh
import bpy
from bpy.types import Operator

from lightsheet import material, trace, utils


# -----------------------------------------------------------------------------
# Create lightsheet
# -----------------------------------------------------------------------------
class LIGHTSHEET_OT_create_lightsheet(Operator):
    """Create a lightsheet"""
    bl_idname = "lightsheet.create"
    bl_label = "Create Lightsheet"
    bl_options = {'REGISTER', 'UNDO'}

    resolution: bpy.props.IntProperty(
        name="Resolution", description="Resolution of lightsheet mesh",
        default=10, min=2
    )

    @classmethod
    def poll(cls, context):
        # operator makes sense only for light objects
        return context.object is not None and context.object.type == 'LIGHT'

    def invoke(self, context, event):
        obj = context.object

        # cancel operator for area lights
        light_type = obj.data.type
        if light_type not in {'SUN', 'SPOT', 'POINT'}:
            message = f"{light_type.capitalize()} lights are not supported"
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        # set resolution via dialog window
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        # build lightsheet around active object (should be a light source)
        light = context.object

        # setup lightsheet bmesh, type of lightsheet depends on type of light
        light_type = light.data.type
        assert light_type in {'SUN', 'SPOT', 'POINT'}, light.data.type

        tic = perf_counter()
        if light_type == 'SUN':
            # sun gets a square grid, because we don't know anything better
            sidelength = 2  # scale by hand if not the right size
            bm = utils.create_bmesh_square(sidelength, self.resolution)
        elif light_type == 'SPOT':
            # intersect cone of spot with shifted plane => circle with
            # radius = tan(halfangle) (because size of circle = sin(...) and
            # shift = cos(...), but we want shift = 1, so divide by cos(...)
            # and radius becomes sin / cos = tan)
            angle = light.data.spot_size  # between 0° and 180°, but in radians
            radius = min(tan(angle / 2), 10)  # restrict for angles near 180°
            bm = utils.create_bmesh_disk(radius, self.resolution)

            # shift circle to inside of cone
            for vert in bm.verts:
                vert.co.z = -1
        else:
            # lightsheet that surrounds the point light
            bm = utils.create_bmesh_sphere(self.resolution)
        toc = perf_counter()

        # convert to lightsheet object and add to scene
        lightsheet = utils.convert_to_lightsheet(bm, light)
        coll = context.scene.collection
        coll.objects.link(lightsheet)

        # report statistics
        v_stats = "{:,} vertices".format(len(lightsheet.data.vertices))
        f_stats = "{:,} faces".format(len(lightsheet.data.polygons))
        t_stats = "{:.3f}s".format((toc - tic))
        self.report({"INFO"}, f"Created {v_stats} and {f_stats} in {t_stats}")

        return {"FINISHED"}


# -----------------------------------------------------------------------------
# Trace lightsheet
# -----------------------------------------------------------------------------
class LIGHTSHEET_OT_trace_lightsheet(Operator):
    """Trace rays from active lightsheet and create caustics"""
    bl_idname = "lightsheet.trace"
    bl_label = "Trace Lightsheet"
    bl_options = {'REGISTER', 'UNDO'}

    max_bounces: bpy.props.IntProperty(
        name="Max Bounces", description="Maximum number of light bounces",
        default=4, min=0
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
        lightsheet = context.object
        utils.verify_lightsheet(lightsheet)
        depsgraph = context.view_layer.depsgraph
        max_bounces = self.max_bounces

        # hide lightsheets and caustics from raycast
        hidden = []
        for obj in depsgraph.view_layer.objects:
            if "lightsheet" in obj.name.lower() or obj.caustic_info.path:
                hidden.append((obj, obj.hide_viewport))
                obj.hide_viewport = True

        # raytrace lightsheet and convert resulting caustics to objects
        tic = perf_counter()
        path_bm = trace.trace_lightsheet(lightsheet, depsgraph, max_bounces)
        cau_to_obj = partial(utils.convert_caustic_to_objects, lightsheet)
        caustics = [cau_to_obj(pth, trc) for pth, trc in path_bm.items()]
        toc = perf_counter()

        # cleanup generated meshes and caches
        for obj in trace.meshes_cache:
            obj.to_mesh_clear()
        trace.meshes_cache.clear()
        material.materials_cache.clear()

        # get or setup collection for caustics
        coll = bpy.data.collections.get("Caustics")
        if coll is None:
            coll = bpy.data.collections.new("Caustics")
            context.scene.collection.children.link(coll)

        # get material and add caustic object to caustic collection
        light = lightsheet.parent
        for obj in caustics:
            mat = material.get_caustic_material(light, obj.parent)
            obj.data.materials.append(mat)
            coll.objects.link(obj)

        # restore original state for hidden lightsheets and caustics
        for obj, state in hidden:
            obj.hide_viewport = state

        # report statistics
        c_stats = "{} caustics".format(len(caustics))
        t_stats = "{:.1f}s".format((toc - tic))
        self.report({"INFO"}, f"Created {c_stats} in {t_stats}")

        return {"FINISHED"}


# -----------------------------------------------------------------------------
# Finalize caustics
# -----------------------------------------------------------------------------
class LIGHTSHEET_OT_finalize_caustics(Operator):
    """Smooth and cleanup selected caustics"""
    bl_idname = "lightsheet.finalize"
    bl_label = "Finalize Caustic"
    bl_options = {'REGISTER', 'UNDO'}

    intensity_threshold: bpy.props.FloatProperty(
        name="Intensity Treshold",
        description="Remove faces less intense than this cutoff",
        default=0.00001, min=0.0, precision=6
    )
    delete_empty_caustics: bpy.props.BoolProperty(
        name="Delete empty caustics",
        description="If after cleanup no faces remain, delete the caustic",
        default=True
    )

    @classmethod
    def poll(cls, context):
        # operator makes sense only if some caustics are selected
        objects = context.selected_objects
        if objects:
            return any(len(obj.caustic_info.path) > 0 for obj in objects)
        return False

    def invoke(self, context, event):
        # set properties via dialog window
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        tic = perf_counter()
        finalized, skipped, deleted = 0, 0, 0
        for obj in context.selected_objects:
            # skip objects that are not caustics or are already finalized
            if not obj.caustic_info.path or obj.caustic_info.finalized:
                skipped += 1
                continue

            # convert from object
            bm = bmesh.new()
            bm.from_mesh(obj.data)

            # smooth out and cleanup
            utils.smooth_caustic_squeeze(bm)
            utils.cleanup_caustic(bm, self.intensity_threshold)
            # TODO for cycles overlapping faces should be stacked in layers

            # if no faces remain, delete the caustic (if wanted by the user)
            if len(bm.faces) == 0 and self.delete_empty_caustics:
                bm.free()
                bpy.data.objects.remove(obj)
                deleted += 1
                continue

            # convert bmesh back to object
            bm.to_mesh(obj.data)
            bm.free()

            # mark as finalized
            obj.caustic_info.finalized = True
            finalized += 1

        toc = perf_counter()

        # report statistics
        f_stats = f"Finalized {finalized}"
        s_stats = f"skipped {skipped}"
        d_stats = f"deleted {deleted}"
        t_stats = "{:.3f}s".format(toc-tic)
        if self.delete_empty_caustics:
            message = f"{f_stats}, {s_stats}, {d_stats} in {t_stats}"
            self.report({"INFO"}, message)
        else:
            self.report({"INFO"}, f"{f_stats}, {s_stats} in {t_stats}")

        return {"FINISHED"}
