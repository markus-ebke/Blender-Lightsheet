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
"""Trace, refine and finalize the selected caustics over several frames.

LIGHTSHEET_OT_animated_trace: Operator for automatic tracing
"""

from collections import defaultdict
from time import process_time as stopwatch

import bpy
from bpy.types import Operator


class LIGHTSHEET_OT_animated_trace(Operator):
    """Trace, refine and finalize the selected caustics over several frames"""
    bl_idname = "lightsheet.animate"
    bl_label = "Animated Trace"
    bl_options = {'REGISTER', 'UNDO'}

    frame_start: bpy.props.IntProperty(name="Start Frame")
    frame_end: bpy.props.IntProperty(name="End Frame")

    @classmethod
    def poll(cls, context):
        # operator makes sense only for caustics
        if context.selected_objects:
            return all(obj.caustic_info.path
                       for obj in context.selected_objects)
        return False

    def invoke(self, context, event):
        # cancel with error message
        def cancel(obj, reasons):
            msg = f"Can't refine '{obj.name}' because {reasons}!"
            self.report({"ERROR"}, msg)
            return {'CANCELLED'}

        # check all caustics
        for obj in context.selected_objects:
            assert obj.caustic_info.path, obj  # poll failed us!

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
            if light_type not in ('SUN', 'SPOT', 'POINT'):
                reasons = f"{light_type.lower()} lights are not supported"
                return cancel(obj, reasons)

        # use same start and end frames as in scene
        scene = context.scene
        self.frame_start = scene.frame_start
        self.frame_end = scene.frame_end

        # set properties via dialog window
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def draw(self, context):
        caustics = context.selected_objects

        # info about selected caustics
        self.layout.label(text=f"Found {len(caustics)} selected caustics:")
        box = self.layout.box().column(align=True)
        for obj in caustics:
            num_refine = len(obj.caustic_info.refinements)
            box.label(text=f"{obj.name} ({num_refine} refinements)")

        # start and end frame properties
        row = self.layout.row(align=True)
        row.scale_x = 0.8
        row.prop(self, "frame_start", text="Start")
        row.prop(self, "frame_end", text="End")

        # show how to cancel compuations
        msg = "Press Ctrl-C in the console window to cancel"
        self.layout.label(text=msg, icon='INFO')

    def execute(self, context):
        wm = context.window_manager
        frame_current = context.scene.frame_current
        reference_caustics = context.selected_objects[:]

        # show progress via window manager progress counter
        wm.progress_begin(self.frame_start, self.frame_end + 1)

        # animated trace
        tic = stopwatch()
        for frame in range(self.frame_start, self.frame_end + 1):
            # update window manager progress counter
            wm.progress_update(frame)
            print(f"Lightsheet: Animate trace for frame {frame}")

            context.scene.frame_set(frame)
            if frame != frame_current:
                new_caustics = auto_trace(context, reference_caustics, frame)
            else:
                new_caustics = reference_caustics

            # check for errors
            if new_caustics is None:
                return {'CANCELLED'}

            print(f"Frame {frame}: {len(new_caustics)} caustics")
            # unhide the caustic only for this one frame
            for obj in new_caustics:
                # show in this frame
                obj.hide_viewport = obj.hide_render = False
                obj.keyframe_insert(data_path="hide_viewport", frame=frame)
                obj.keyframe_insert(data_path="hide_render", frame=frame)

                # hide in other frames before and after
                obj.hide_viewport = obj.hide_render = True
                obj.keyframe_insert(data_path="hide_viewport", frame=frame - 1)
                obj.keyframe_insert(data_path="hide_render", frame=frame - 1)
                obj.keyframe_insert(data_path="hide_viewport", frame=frame + 1)
                obj.keyframe_insert(data_path="hide_render", frame=frame + 1)
        wm.progress_update(self.frame_end + 1)

        # reset scene and give reference caustics the correct name
        context.scene.frame_set(frame_current)
        for obj in reference_caustics:
            obj.name = f"{obj.name} f{frame_current:0>4}"

        toc = stopwatch()

        # stop window indicator
        wm.progress_end()

        # report statistics
        c_stats = f"{len(reference_caustics)} caustics(s)"
        f_stats = f"{self.frame_end-self.frame_start+1} frames"
        t_stats = f"{(toc-tic)/60:.1f}min"
        self.report({"INFO"}, f"Animated {c_stats} for {f_stats} in {t_stats}")

        return {"FINISHED"}


def auto_trace(context, reference_caustics, frame):
    """Trace, refine and finalize caustics for the given lightsheets."""
    # match reference to new caustics by their path
    def caustic_key(caustic_info):
        path_key = [(caustic_info.lightsheet, "SOURCE")]
        for link in caustic_info.path:
            path_key.append((link.object, link.kind))
        return tuple(path_key)

    # map lightsheets to reference caustics
    lightsheet_to_paths = defaultdict(list)
    path_to_reference = dict()
    for obj in reference_caustics:
        path_key = caustic_key(obj.caustic_info)
        lightsheet_to_paths[obj.caustic_info.lightsheet].append(path_key)
        path_to_reference[path_key] = obj

    # remember all old caustics
    scene = context.scene
    caustic_coll = scene.collection.children.get(f"Caustics in {scene.name}")
    assert caustic_coll is not None  # how else can we have reference caustics?
    old_caustics = set(caustic_coll.objects)
    # print(f"Found {len(old_caustics)} old caustics")

    # trace lightsheets
    for lightsheet, path_keys in lightsheet_to_paths.items():
        max_bounces = max(len(path_key) - 2 for path_key in path_keys)

        override = context.copy()
        override["selected_objects"] = [lightsheet]
        ret = bpy.ops.lightsheet.trace(override, max_bounces=max_bounces)
        if 'CANCELLED' in ret:
            return None

    # collect and categorize new caustics
    new_caustics = [obj for obj in caustic_coll.objects
                    if obj not in old_caustics]
    # print(f"Found {len(new_caustics)} new caustics")
    caustic_and_reference = []
    for obj in new_caustics:
        path_key = caustic_key(obj.caustic_info)

        # find matching reference caustic, if none remove caustic
        if path_key in path_to_reference:
            ref_obj = path_to_reference[path_key]
            caustic_and_reference.append((obj, ref_obj))

            # give the new caustic a better name
            obj.name = f"{ref_obj.name} f{frame:0>4}"
        else:
            bpy.data.objects.remove(obj)

    # refine all new caustics
    for obj, ref_obj in caustic_and_reference:
        # print(f"Refining {obj.name} (ref: {ref_obj.name})")
        for step in ref_obj.caustic_info.refinements:
            settings = {
                "adaptive_subdivision": step.adaptive_subdivision,
                "error_threshold": step.error_threshold,
                "span_faces": step.span_faces,
                "grow_boundary": step.grow_boundary
            }

            override = context.copy()
            override["selected_objects"] = [obj]
            ret = bpy.ops.lightsheet.refine(override, **settings)
            if 'CANCELLED' in ret:
                return None

    # finalize all new caustics
    for obj, ref_obj in caustic_and_reference:
        # print(f"Finalizing {obj.name} (ref: {ref_obj.name})")
        caustic_info = ref_obj.caustic_info
        settings = {
            "fade_boundary": caustic_info.fade_boundary,
            "remove_dim_faces": caustic_info.remove_dim_faces,
            "emission_cutoff": caustic_info.emission_cutoff,
            "delete_empty_caustics": False,  # for predictable caustic number
            "fix_overlap": caustic_info.fix_overlap
        }

        override = context.copy()
        override["selected_objects"] = [obj]
        ret = bpy.ops.lightsheet.finalize(override, **settings)
        if 'CANCELLED' in ret:
            return None

    return [obj for obj, ref_obj in caustic_and_reference]
