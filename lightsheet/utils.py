# ##### BEGIN GPL LICENSE BLOCK #####
#
#  Lightsheet is a Blender addon for creating fake caustics that can be
#  rendered with Cycles and EEVEE.
#  Copyright (C) 2021  Markus Ebke
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
"""Helper functions used in other files.

- bmesh_delete_loose (bmesh reimplementation of bpy.ops.mesh.delete_loose, used
    by trace_lightsheet, refine_caustic and finalize_caustic)
- srgb_to_linear (used in finalize_caustic and material)
- linear_to_srgb (used here in set_caustic_face_data and in finalize_caustic)
- get_collection_for_scene (used by create_lightsheet, trace_lightsheet,
    animated_trace, visualize_raypath, ui and trace)
- verify_lightsheet_layers (used by create_lightsheet and trace_lightsheet)
- setup_sheet_property (used by trace_lightsheet and refine_caustics)
- get_uv_map (used here in set_caustic_face_data and in finalize_caustics)
- set_caustic_squeeze (used by trace_lightsheet and refine_caustics)
- set_caustic_face_data (used by trace_lightsheet and refine_caustics)
- ProgressIndicator (used by trace_lightsheet, refine_caustic and
    finalize_caustic)
"""

import sys
import time

import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils.geometry import area_tri


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


def srgb_to_linear(col):
    """Convert color in sRGB to linear RGB."""
    # return [u / 12.82 if u <= 0.04045 else ((u + 0.055) / 1.055)**2.4
    #         for u in col]
    # list comprehension unrolled for speed:
    return (
        col[0]/12.82 if col[0] <= 0.04045 else ((col[0]+0.055)/1.055)**2.4,
        col[1]/12.82 if col[1] <= 0.04045 else ((col[1]+0.055)/1.055)**2.4,
        col[2]/12.82 if col[2] <= 0.04045 else ((col[2]+0.055)/1.055)**2.4,
    )


def linear_to_srgb(col):
    """Convert color in linear RGB to sRGB."""
    # return [12.92 * u if u <= 0.0031308 else 1.055 * u**(1/2.4) - 0.055
    #         for u in col]
    # list comprehension unrolled for speed:
    return (
        12.92*col[0] if col[0] <= 0.0031308 else 1.055*col[0]**(1/2.4)-0.055,
        12.92*col[1] if col[1] <= 0.0031308 else 1.055*col[1]**(1/2.4)-0.055,
        12.92*col[2] if col[2] <= 0.0031308 else 1.055*col[2]**(1/2.4)-0.055,
    )


def get_collection_for_scene(scene, objects="lightsheets", force=True):
    """Get or create the lightsheets/caustics collection for the scene."""
    coll_name = f"{objects.capitalize()} in {scene.name}"

    # see if we can find the collection in the scene
    coll = scene.collection.children.get(coll_name)
    if coll is None:
        # collection not in scene, is it anywhere in the file?
        coll = bpy.data.collections.get(coll_name)
        if coll is None:
            # if we don't want to create a new collection, return nothing
            if not force:
                return None

            # create a new collection
            coll = bpy.data.collections.new(coll_name)

        # don't forget to link to scene (if that is wanted)
        if force:
            scene.collection.children.link(coll)

    return coll


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


def get_uv_map(bm):
    """Get uv-layer for non-caustic data (if any)."""
    reserved_layers = {"Lightsheet XY", "Lightsheet XZ", "Caustic Squeeze"}
    for layer in bm.loops.layers.uv.values():
        if layer.name not in reserved_layers:
            return layer
    return None


def set_caustic_squeeze(caustic_bm, matrix_sheet=None, matrix_caustic=None,
                        verts=None):
    """Calculate squeeze (= source area / projected area) for given verts."""
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
        source_area_sum, target_area_sum = 0.0, 0.0
        for face in vert.link_faces:
            if face in face_to_area:
                # found face in cache
                source_area, target_area = face_to_area[face]
            else:
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
    uv_layer = get_uv_map(caustic_bm)

    # set transplanted uv-coordinates and vertex colors for faces
    for face in faces:
        vert_normal_sum = Vector((0.0, 0.0, 0.0))
        for loop in face.loops:
            # get data for this vertex (if any)
            sheet_key = get_sheet(loop.vert).to_tuple()
            cdata = sheet_to_data.get(sheet_key)
            if cdata is not None:
                # sheet position to uv-coordinates
                sx, sy, sz = sheet_key
                loop[uv_sheet_xy].uv = (sx, sy)
                loop[uv_sheet_xz].uv = (sx, sz)

                # set face data
                loop[color_layer] = linear_to_srgb(cdata.color) + (1,)
                if uv_layer is not None:
                    loop[uv_layer].uv = cdata.uv
                vert_normal_sum += cdata.perp

        # if face normal does not point in the same general direction as
        # the averaged vertex normal, then flip the face normal
        face.normal_update()
        if face.normal.dot(vert_normal_sum) < 0:
            face.normal_flip()

    caustic_bm.normal_update()


class ProgressIndicator:
    """Show progress of tasks in terminal."""

    def __init__(self, total_jobs, update_delay=0.1):
        # jobs
        self.total_jobs = total_jobs
        self.current_job = 0
        self.job_name = ""

        # tasks and steps
        self.task_name = ""
        self.total_steps = None
        self.current_step = None

        # description of job and task shown in front of progress counter
        self._description = ""

        # timing for updates
        self.update_delay = update_delay
        self.clock = time.monotonic
        self.last_update = self.clock()

        # stats
        self.stopwatch = time.process_time
        self.tic_job = None
        self.tic_task = None
        self.job_stats = []
        self.task_stats = []

    def _clear_output(self):
        """Clear last output and reset cursor to start of line."""
        clear_length = len(self._description) + 7  # include percentage
        sys.stdout.write("\r" + " " * clear_length + "\r")
        sys.stdout.flush()

    def start_job(self, job_name):
        """Switch to next job."""
        # stop last job if not done already
        self.stop_job()

        # setup new job
        self.current_job += 1
        self.job_name = job_name

        # start job timer
        self.tic_job = self.stopwatch()

    def stop_job(self):
        """Stop job and task timers and record time."""
        # stop last task
        self.stop_task()

        # update time of last job, if timer is active
        if self.tic_job is not None:
            toc = self.stopwatch()
            job_timing = (self.job_name, toc - self.tic_job, self.task_stats)
            self.task_stats = []
            self.job_stats.append(job_timing)

        # deactivate timer
        self.tic_job = None

    def start_task(self, task_name, total_steps=1):
        """Switch to next task."""
        # stop last task if not done already
        self.stop_task()

        # setup new task
        self.task_name = task_name

        # set description
        self._clear_output()
        if len(self.job_name) > 40:  # show at most 40 character of job name
            # truncate name such that we see the tail
            job_name = f"[...]{self.job_name[-35:]}"
        else:
            job_name = self.job_name
        self._description = (f"{self.current_job}/{self.total_jobs} "
                             f"{job_name}: {self.task_name}")

        # reset progress
        self.total_steps = total_steps
        self.update_progress(step=0, force=True)

        # start task timer
        self.tic_task = self.stopwatch()

    def stop_task(self):
        """Stop task timer and record time."""
        # update time of last task, if timer is active
        if self.tic_task is not None:
            toc = self.stopwatch()
            self.task_stats.append((self.task_name, toc - self.tic_task))

        # deactivate timer
        self.tic_task = None

    def update_progress(self, step=None, force=False):
        """Update progress of current task."""
        if step is None:
            self.current_step += 1
        else:
            self.current_step = step

        # what time is it?
        if self.clock() - self.last_update > self.update_delay or force:
            # IT'S UPDATE TIME!!!
            # update terminal output, note that \r will go back to the start of
            # the line so that we can overwrite the previous output
            # idea taken from https://blender.stackexchange.com/a/30739
            if self.total_steps > 1:
                task_progress = self.current_step / self.total_steps
                msg = f"\r{self._description} ({task_progress:.0%})"
            else:
                # for less than two steps a percentage counter is useless
                msg = f"\r{self._description}"
            sys.stdout.write(msg)
            sys.stdout.flush()

            # reset timer
            self.last_update = self.clock()

    def end(self):
        """Stop last job and end indicator."""
        # stop last job if not done already
        self.stop_job()

        # clear last output
        self._clear_output()

    def print_stats(self):
        """Print time stats for each job and task."""
        job_time_sum = 0.0
        for job_name, job_time, task_stats in self.job_stats:
            # print info about job
            print(f"{job_name}: {job_time:.3f}s")
            job_time_sum += job_time

            # print info about tasks
            task_time_sum = 0.0
            for task_name, task_time in task_stats:
                print(f"    {task_name}: {task_time:.3f}s")
                task_time_sum += task_time
            print(f"    <other>: {job_time-task_time_sum:.3f}s")

        print(f"Total time ({len(self.job_stats)} jobs): {job_time_sum:.3f}s")
