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
"""Smooth out and cleanup selected caustics.

LIGHTSHEET_OT_finalize_caustics: Operator for finalizing caustics

Helper functions:
- finalize_caustic
- smooth_caustic_squeeze
- cleanup_caustic
"""

from time import perf_counter

import bmesh
import bpy
from bpy.types import Operator

from lightsheet import utils


class LIGHTSHEET_OT_finalize_caustic(Operator):
    """Smooth and cleanup selected caustics"""
    bl_idname = "lightsheet.finalize"
    bl_label = "Finalize Caustic"
    bl_options = {'REGISTER', 'UNDO'}

    intensity_threshold: bpy.props.FloatProperty(
        name="Intensity Treshold",
        description="Remove faces that are less intense than this cutoff "
        "(caustic emission strength < intensity threshold * light strength)",
        default=0.00001, min=0.0, precision=6, subtype='FACTOR'
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
            for obj in objects:
                caustic_info = obj.caustic_info
                if caustic_info.path and not caustic_info.finalized:
                    return True
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

            finalize_caustic(obj, self.intensity_threshold)
            if self.delete_empty_caustics and len(obj.data.polygons) == 0:
                # delete caustic object
                bpy.data.objects.remove(obj)
                deleted += 1
            else:
                # count as finalized
                finalized += 1
        toc = perf_counter()

        if not self.delete_empty_caustics:
            assert deleted == 0, (finalized, deleted, skipped)

        # report statistics
        f_stats = f"Finalized {finalized}"
        d_stats = f"deleted {deleted}"
        s_stats = f"skipped {skipped}"
        t_stats = "{:.3f}s".format(toc-tic)
        message = f"{f_stats}, {d_stats} and {s_stats} in {t_stats}"
        self.report({"INFO"}, message)

        return {"FINISHED"}


# -----------------------------------------------------------------------------
# Functions used by finalize caustics operator
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
    utils.bmesh_delete_loose(caustic_bm)
