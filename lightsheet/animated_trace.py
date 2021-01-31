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

Helper functions:
- auto_trace
- categorize_new_caustics
- caustics_key
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

    frame_reference: bpy.props.IntProperty(
        name="Reference Frame",
        description="Frame in which the selected caustics were traced"
    )
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
            self.report({'ERROR'}, msg)
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
            if light_type not in {'SUN', 'SPOT', 'POINT'}:
                reasons = f"{light_type.lower()} lights are not supported"
                return cancel(obj, reasons)

        # use same start and end frames as in scene
        scene = context.scene
        self.frame_reference = scene.frame_current
        self.frame_start = scene.frame_start
        self.frame_end = scene.frame_end

        # set properties via dialog window
        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=400)

    def draw(self, context):
        layout = self.layout
        caustics = context.selected_objects

        # info about selected caustics
        layout.label(text=f"Found {len(caustics)} selected caustics:")
        show_lines = 8
        if len(caustics) > show_lines:
            show_lines -= 1  # make room for last line
        box = layout.box().column(align=True)
        for obj in caustics[:show_lines]:
            state = f"{len(obj.caustic_info.refinements)} refinements"
            if obj.caustic_info.finalized:
                state += ", finalized"
            box.label(text=f"{obj.name} ({state})")
        if len(caustics) > show_lines:
            box.label(text=f"...{len(caustics)-show_lines} more objects...")

        # start and end frame properties
        row = layout.row(align=True)
        row.prop(self, "frame_reference")
        row.prop(self, "frame_start")
        row.prop(self, "frame_end")

        # show time and memory warning
        msg = "This will take a long time and may use a lot of memory"
        layout.label(text=msg, icon='ERROR')

        # show how to cancel compuations
        msg = "To cancel press Ctrl-C in the console window"
        layout.label(text=msg, icon='INFO')

    def execute(self, context):
        wm = context.window_manager
        frame_current = context.scene.frame_current
        reference_caustics = context.selected_objects[:]

        # animated trace, show progress via window manager progress counter
        tic = stopwatch()
        wm.progress_begin(self.frame_start, self.frame_end + 1)
        for frame in range(self.frame_start, self.frame_end + 1):
            # update window manager progress counter
            wm.progress_update(frame)
            # print(f"Lightsheet: Animating trace for frame {frame}")

            context.scene.frame_set(frame)
            if frame == self.frame_reference:
                # caustics for this frame are the reference caustics
                new_caustics = reference_caustics

                # add frame number to the name of the reference caustic
                for obj in reference_caustics:
                    ref_name = obj.name
                    if ref_name[-3:].isnumeric() and ref_name[-5:-3] == " f":
                        ref_name = ref_name[:-5]
                    obj.name = f"{ref_name} f{frame:0>3}"
                    obj.data.name = obj.name
            else:
                # trace caustics for this frame
                new_caustics = auto_trace(context, reference_caustics, frame)

            # check for errors
            if new_caustics is None:
                return {'CANCELLED'}

            print(f"Lightsheet: {len(new_caustics)} caustics in frame {frame}")
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

        # reset scene
        wm.progress_update(self.frame_end + 1)
        context.scene.frame_set(frame_current)

        # stop window indicator
        wm.progress_end()
        toc = stopwatch()

        # report statistics
        c_stats = f"{len(reference_caustics)} caustics(s)"
        f_stats = f"{self.frame_end-self.frame_start+1} frames"
        t_stats = f"{(toc-tic)/60:.1f}min"
        self.report({'INFO'}, f"Animated {c_stats} for {f_stats} in {t_stats}")

        return {'FINISHED'}


def auto_trace(context, reference_caustics, frame):
    """Trace, refine and finalize caustics for the given lightsheets."""
    # map lightsheets to reference caustics
    lightsheet_to_paths = defaultdict(list)
    path_to_reference = dict()
    for obj in reference_caustics:
        path_key = caustic_key(obj.caustic_info)
        lightsheet_to_paths[obj.caustic_info.lightsheet].append(path_key)
        path_to_reference[path_key] = obj

    # remember old caustics
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

    # collect new caustics, categorize by refinement and finalization settings
    new_caustics = [obj for obj in caustic_coll.objects
                    if obj not in old_caustics]
    # print(f"Found {len(new_caustics)} new caustics")
    result = categorize_new_caustics(new_caustics, path_to_reference, frame)
    refinement_to_caustics, finalization_to_caustics, delete_caustics = result

    # delete unused caustics
    # print(f"Deleting {len(delete_caustics)} of {len(new_caustics)} caustics")
    for obj in delete_caustics:
        bpy.data.objects.remove(obj)

    # refine new caustics
    for settings_tuple, caustics in refinement_to_caustics.items():
        for step in settings_tuple:
            settings = {
                "adaptive_subdivision": step[0],
                "error_threshold": step[1],
                "grow_boundary": step[2]
            }

            override = context.copy()
            override["selected_objects"] = caustics
            ret = bpy.ops.lightsheet.refine(override, **settings)
            if 'CANCELLED' in ret:
                return None

    # finalize new caustics
    for settings_tuple, caustics in finalization_to_caustics.items():
        settings = {
            "fade_boundary": settings_tuple[0],
            "remove_dim_faces": settings_tuple[1],
            "emission_cutoff": settings_tuple[2],
            "delete_empty_caustics": True,
            "fix_overlap": settings_tuple[3],
            "delete_coordinates": settings_tuple[4],
        }

        override = context.copy()
        override["selected_objects"] = caustics
        ret = bpy.ops.lightsheet.finalize(override, **settings)
        if 'CANCELLED' in ret:
            return None

    # return remaining new caustics
    return [obj for obj in caustic_coll.objects if obj not in old_caustics]


def categorize_new_caustics(new_caustics, path_to_reference, frame):
    """Categorize new caustics by settings of their reference caustics."""
    refinement_to_caustics = defaultdict(list)
    finalization_to_caustics = defaultdict(list)
    delete_caustics = []

    for obj in new_caustics:
        path_key = caustic_key(obj.caustic_info)

        # find matching reference caustic, if none remove caustic
        if path_key in path_to_reference:
            ref_obj = path_to_reference[path_key]
            assert obj.parent == ref_obj.parent, (obj.parent, ref_obj.parent)

            # add frame number to the name of the new caustic
            ref_name = ref_obj.name
            if ref_name[-3:].isnumeric() and ref_name[-5:-3] == " f":
                ref_name = ref_name[:-5]
            obj.name = f"{ref_name} f{frame:0>3}"
            obj.data.name = obj.name

            # copy settings for shrinkwrap modifier (if any)
            ref_mod = ref_obj.modifiers.get("Shrinkwrap")
            if ref_mod is not None:
                # check if reference modifier is valid
                if ref_mod.type != 'SHRINKWRAP':
                    msg = f"{ref_obj.name} has no valid shrinkwrap modifier"
                    raise ValueError(msg)
                if ref_mod.target != obj.parent:
                    msg = f"Target of {ref_obj.name} shrinkwrap is not parent"
                    raise ValueError(msg)

                # get modifier of new caustic and copy offset
                mod = obj.modifiers.get("Shrinkwrap")
                assert mod is not None and mod.type == 'SHRINKWRAP', mod
                assert mod.target == obj.parent, mod.target
                mod.offset = ref_mod.offset

            # get refinement settings from reference
            caustic_info = ref_obj.caustic_info
            settings_list = []
            for step in caustic_info.refinements:
                step_settings = (
                    step.adaptive_subdivision,
                    step.error_threshold,
                    step.grow_boundary
                )
                settings_list.append(step_settings)
            refinement_to_caustics[tuple(settings_list)].append(obj)

            # get finalization settings from reference
            if caustic_info.finalized:
                settings_tuple = (
                    caustic_info.fade_boundary,
                    caustic_info.remove_dim_faces,
                    caustic_info.emission_cutoff,
                    caustic_info.fix_overlap,
                    caustic_info.delete_coordinates
                )
                finalization_to_caustics[settings_tuple].append(obj)
        else:
            # we don't need caustics that have no reference
            delete_caustics.append(obj)

    return refinement_to_caustics, finalization_to_caustics, delete_caustics


def caustic_key(caustic_info):
    """Identify caustics by their raypath."""
    path_key = [(caustic_info.lightsheet, "SOURCE")]
    for link in caustic_info.path:
        path_key.append((link.object, link.kind))
    return tuple(path_key)
