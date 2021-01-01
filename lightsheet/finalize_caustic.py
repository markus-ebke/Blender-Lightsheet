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
"""Cleanup selected caustics.

LIGHTSHEET_OT_finalize_caustics: Operator for finalizing caustics

Helper functions:
- finalize_caustic
- fadeout_caustic_boundary
- remove_dim_faces
- stack_overlapping_in_levels
- collect_by_plane
- crawl_faces
- group_by_collision_table
- build_collision_table
- compute_pointcloud_bounds
- intersect_triangle_triangle
- det_2d
"""

from collections import defaultdict, deque
from math import pi
from time import process_time as stopwatch

import bmesh
import bpy
from bpy.types import Operator
from mathutils import Matrix

from lightsheet import utils


class LIGHTSHEET_OT_finalize_caustic(Operator):
    """Smooth and cleanup selected caustics"""
    bl_idname = "lightsheet.finalize"
    bl_label = "Finalize Caustic"
    bl_options = {'REGISTER', 'UNDO'}

    fade_boundary: bpy.props.BoolProperty(
        name="Fade Out Boundary",
        description="Disguise the boundary by fading out",
        default=True
    )
    remove_dim_faces: bpy.props.BoolProperty(
        name="Remove Dim Faces",
        description="Remove faces that emit less power than the cutoff below",
        default=True
    )
    emission_cutoff: bpy.props.FloatProperty(
        name="Emission Cutoff",
        description="Remove face if its emission strength (in W/m^2) is lower "
        "than this value (includes the current light strength)",
        default=0.001, min=0.0, precision=4
    )
    delete_empty_caustics: bpy.props.BoolProperty(
        name="Delete Empty Caustics",
        description="If no faces remain after cleanup, delete the caustic",
        default=True
    )
    fix_overlap: bpy.props.BoolProperty(
        name="Cycles: Fix Overlap Artifacts",
        description="WARNING: SLOW! Prevent render artifacts in Cycles caused "
        "by overlapping faces, will find intersecting faces and stack them on "
        "top of each other. First use the offset of the shrinkwrap modifiers "
        "to separate different caustics from each other",
        default=False
    )

    @classmethod
    def poll(cls, context):
        # operator makes sense only for caustics
        if context.selected_objects:
            return all(obj.caustic_info.path and not obj.caustic_info.finalized
                       for obj in context.selected_objects)
        return False

    def invoke(self, context, event):
        # cancel with error message
        def cancel(obj, reasons):
            msg = f"Cannot finalize '{obj.name}' because {reasons}!"
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        # check all caustics
        for obj in context.selected_objects:
            assert obj.caustic_info.path, obj  # poll failed us!
            assert not obj.caustic_info.finalized, obj

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

        # set properties via dialog window
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        layout.prop(self, "fade_boundary")

        # remove dim faces and emission cutoff in one row
        heading = layout.column(align=False, heading="Remove Dim Faces")
        row = heading.row(align=True)
        row.prop(self, "remove_dim_faces", text="")
        sub = row.row()
        sub.active = self.remove_dim_faces
        sub.prop(self, "emission_cutoff")

        # if we don't remove dim faces, deleting empty caustics is useless
        sub = layout.column()
        sub.active = self.remove_dim_faces
        sub.prop(self, "delete_empty_caustics")

        layout.prop(self, "fix_overlap")

    def execute(self, context):
        caustics = context.selected_objects

        # set emission cutoff
        if self.remove_dim_faces:
            emission_cutoff = self.emission_cutoff
        else:
            emission_cutoff = None

        # setup progress indicator
        prog = utils.ProgressIndicator(total_jobs=len(caustics))

        # finalize selected caustics
        tic = stopwatch()
        finalized, deleted = 0, 0
        for caustic in caustics:
            prog.start_job(caustic.name)
            result = finalize_caustic(caustic, self.fade_boundary,
                                      emission_cutoff,
                                      self.delete_empty_caustics,
                                      self.fix_overlap, prog)
            if result is None:
                deleted += 1
            else:
                finalized += 1
            prog.stop_job()
        prog.end()
        toc = stopwatch()

        # report statistics
        # prog.print_stats()
        f_stats = f"Finalized {finalized}"
        d_stats = f"deleted {deleted} caustics"
        t_stats = f"{toc-tic:.1f}s"
        self.report({'INFO'}, f"{f_stats} and {d_stats} in {t_stats}")

        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Functions used by finalize caustics operator
# -----------------------------------------------------------------------------
def finalize_caustic(caustic, fade_boundary, emission_cutoff, delete_empty,
                     fix_overlap, prog):
    """Finalize caustic mesh."""
    # convert from object
    caustic_bm = bmesh.new()
    caustic_bm.from_mesh(caustic.data)

    # fade out boundary
    if fade_boundary:
        prog.start_task("fading out boundary")
        fadeout_caustic_boundary(caustic_bm)

    # remove dim faces
    if emission_cutoff is not None:
        prog.start_task("removing dim faces")
        light = caustic.caustic_info.lightsheet.parent
        remove_dim_faces(caustic_bm, light, emission_cutoff)

        # if wanted by user delete caustic objects without faces
        if delete_empty and len(caustic_bm.faces) == 0:
            bpy.data.objects.remove(caustic)
            return None
    prog.stop_task()

    # stack overlapping faces in layers
    if fix_overlap:
        prog.start_task("fixing overlap", total_steps=3)

        # remove shrinkwrap modifier because we need to apply it by hand
        mod = caustic.modifiers.get("Shrinkwrap")
        if mod is not None:
            offset = mod.offset
            caustic.modifiers.remove(mod)
        else:
            # modifier must have been removed by hand
            assert False, list(caustic.modifiers)
            offset = 0.0

        # stack
        stack_overlapping_in_levels(caustic_bm, caustic.matrix_world, offset,
                                    prog)
        prog.stop_task()
    else:
        offset = 0.0

    # convert bmesh back to object
    caustic_bm.to_mesh(caustic.data)
    caustic_bm.free()

    # fill out caustic_info property
    caustic_info = caustic.caustic_info
    caustic_info.finalized = True
    caustic_info.fade_boundary = fade_boundary
    caustic_info.remove_dim_faces = emission_cutoff is not None
    if emission_cutoff is not None:
        caustic_info.emission_cutoff = emission_cutoff
    caustic_info.fix_overlap = fix_overlap
    caustic_info.shrinkwrap_offset = offset

    return caustic


def fadeout_caustic_boundary(caustic_bm):
    """Set vertex color to black for vertices at the boundary."""
    color_layer = caustic_bm.loops.layers.color["Caustic Tint"]
    for vert in (v for v in caustic_bm.verts if v.is_boundary):
        for loop in vert.link_loops:
            loop[color_layer] = (0.0, 0.0, 0.0, 0.0)


def remove_dim_faces(caustic_bm, light, emission_cutoff):
    """Remove invisible faces and cleanup resulting mesh."""
    assert light is not None and light.type == 'LIGHT', light

    # emission strength and color from caustic
    squeeze_layer = caustic_bm.loops.layers.uv["Caustic Squeeze"]
    color_layer = caustic_bm.loops.layers.color["Caustic Tint"]

    # light color and strength, see also material.py: add_drivers_from_light
    if light.data.type == 'SUN':
        light_strength = light.data.energy / pi
    else:
        assert light.data.type in {'SPOT', 'POINT'}, light
        light_strength = light.data.energy / (4 * pi**2)
    light_strength *= light.data.color.v

    # gather faces with low emission strength
    invisible_faces = []
    for face in caustic_bm.faces:
        visible = False
        for loop in face.loops:
            squeeze = loop[squeeze_layer].uv[1]
            tint_v = max(loop[color_layer][:3])  # = Color(...).v
            if light_strength * tint_v * squeeze > emission_cutoff:
                # vertex is intense enough, face is visible
                visible = True
                break

        if not visible:
            invisible_faces.append(face)

    # delete invisible faces and cleanup mesh
    bmesh.ops.delete(caustic_bm, geom=invisible_faces, context='FACES_ONLY')
    utils.bmesh_delete_loose(caustic_bm)


def stack_overlapping_in_levels(bm, matrix_world, offset, prog,
                                separation=2e-4):
    """Stack intersecting faces such that they don't overlap anymore."""
    # find affine planes and the faces they contain
    affine_planes = collect_by_plane(bm, safe_distance=100*separation)
    prog.total_steps = 2 + len(affine_planes)
    prog.update_progress()

    # assign to each face a level
    level_to_faces = defaultdict(list)
    for _, faces, shapes in affine_planes:
        # group faces into non-intersecting sets, sort groups by area
        groups = group_by_collision_table(faces, shapes)
        groups.sort(key=lambda ics: sum(faces[idx].calc_area() for idx in ics),
                    reverse=True)

        for level, indices in enumerate(groups):
            level_to_faces[level].extend(faces[idx] for idx in indices)

        prog.update_progress()

    # split off and elevate faces
    world_to_local = matrix_world.inverted()
    for level, faces in level_to_faces.items():
        geom = bmesh.ops.split(bm, geom=faces)["geom"]
        verts = (g for g in geom if isinstance(g, bmesh.types.BMVert))
        for vert in verts:
            # get vert normal from face, because normal at boundary might not
            # point in the right direction
            vert_normal = vert.link_faces[0].normal

            # get vertex normal in world coordinates
            location = matrix_world @ vert.co
            world_normal = matrix_world @ (vert.co + vert_normal) - location
            world_normal.normalize()

            # displace in world coordinates and transform to local coordinates
            displacement = (offset + level * separation) * world_normal
            vert.co = world_to_local @ (location + displacement)

    prog.update_progress()


def collect_by_plane(bm, safe_distance=1e-4):
    """Group faces of bmesh if they live in the same plane."""
    # affine plane = (projection matrix, list of faces, list of 2D-shapes),
    # the projection matrix projects points in global space to (u, v, w) where
    # (u, v) are local coordinates in the plane and z is the distance to the
    # plane, each shape is a list of three 2D-vectors which are the (u, v)
    # coordinates of face.verts
    face_to_plane = dict()  # cache plane index for processed faces
    affine_planes = []
    for face in crawl_faces(bm.faces):
        # get face vertices in CCW-direction from loop cycle
        face_verts = []
        loop = face.loops[0]
        for _ in range(len(face.loops)):
            face_verts.append(loop.vert)
            loop = loop.link_loop_next
        assert len(face_verts) == 3, face_verts

        # check only planes of the immediate neighbours, we may miss a valid
        # plane but this is a lot faster if we have many planes (like when the
        # caustic wraps around a curved object)
        check_planes_indices = set()
        for vert in face.verts:
            for other_face in vert.link_faces:
                if other_face in face_to_plane:
                    guess_plane_idx = face_to_plane[other_face]
                    if guess_plane_idx not in check_planes_indices:
                        check_planes_indices.add(guess_plane_idx)

        # if we could not guess any planes, check all of them
        if not check_planes_indices:
            check_planes_indices = range(len(affine_planes))

        # check if face is part of plane and if not then create one for it
        for plane_idx in check_planes_indices:
            projector, faces, shapes = affine_planes[plane_idx]

            # project face onto plane
            shape = []
            for vert in face_verts:
                point = projector @ vert.co
                if abs(point.z) > safe_distance:
                    # point is too far away from affine plane, break loop
                    # over face.verts (will skip else clause)
                    break
                shape.append(point.xy)
            else:
                # found the right plane to add triangle
                assert len(faces) == len(shapes)
                faces.append(face)
                shapes.append(shape)
                face_to_plane[face] = plane_idx

                # break loop over affine planes, skip else clause below
                break
        else:
            # did not find a plane that contains this face, create a new plane
            v1, v2, v3 = (vert.co for vert in face_verts)
            s1, s2 = v2 - v1, v3 - v1  # vectors along the sides

            # get orthonormal basis from Gram-Schmidt orthogonalization
            u1 = s1.normalized()
            u2 = (s2 - u1.dot(s2) * u1).normalized()  # perpendicular to u1
            u3 = s1.cross(s2).normalized()  # parallel to face normal

            # want local plane coordinates from global scene coordinates
            # global coords = Matrix(columns=(u1, u2, u3)) @ local coords + v1
            mat = Matrix((u1, u2, u3))
            mat.transpose()  # want u1, u2 and u3 as columns
            mat = mat.to_4x4()  # to setup affine transformation
            mat.translation = v1
            projector = mat.inverted()  # local coords = proj @ global coords

            # project to plane to get 2D-shape
            shape = [(projector @ co).xy for co in (v1, v2, v3)]

            # check that v1 maps to origin and the other points map to xy-plane
            # and have correct side lengths
            p1, p2, p3 = (projector @ co for co in (v1, v2, v3))
            assert p1.length < 1e-5, (p1, p1.length)
            assert abs(p2.z) < 1e-5, (p2, p2.z)
            assert abs(p3.z) < 1e-5, (p3, p3.z)
            assert abs(p2.length - s1.length) < 1e-5, (p2.length, s1.length)
            assert abs(p3.length - s2.length) < 1e-5, (p3.length. s2.length)

            # add new affine plane
            affine_planes.append((projector, [face], [shape]))
            face_to_plane[face] = len(affine_planes) - 1

    return affine_planes


def crawl_faces(faces):
    """Generator for iterating over faces in a connected manner."""
    unvisited_faces = set(faces)
    while unvisited_faces:
        root = unvisited_faces.pop()

        # record connected faces in FIFO queue
        queue = deque([root])
        while queue:
            # get the oldest face that we have not yet iterated over
            face = queue.popleft()
            yield face

            # find connected faces and put unprocessed ones into queue
            for vert in face.verts:
                for other_face in vert.link_faces:
                    if other_face in unvisited_faces:
                        # transfer from set of faces to queue
                        unvisited_faces.remove(other_face)
                        queue.append(other_face)


def group_by_collision_table(faces, shapes):
    """Sort faces into non-intersecting groups (group is set of indices)."""
    collision_table = build_collision_table(shapes)

    groups = []
    indices_to_process = set(range(len(faces)))
    while indices_to_process:
        if not groups:
            # on the first iteration try to start with a collision-free face
            for idx in indices_to_process:
                if idx not in collision_table:
                    # found one!
                    indices_to_process.remove(idx)
                    group_indices = {idx}
                    break
            else:
                # could not find one, start with a random unprocessed face
                group_indices = {indices_to_process.pop()}
        else:
            # start with a random unprocessed face
            group_indices = {indices_to_process.pop()}

        # add more faces if they don't intersect with the already selected
        # faces for this group, start from the faces immediately connected to
        # selected faces and grow outwards
        newly_added = group_indices
        while newly_added:
            neighbours = set()
            for idx in newly_added:
                for vert in faces[idx].verts:
                    for other_face in vert.link_faces:
                        if other_face.index in indices_to_process:
                            neighbours.add(other_face.index)

            newly_added = []
            for idx in neighbours:
                # if neighbour does not intersect with any face that we have
                # already selected, then add it to the group
                if not (idx in collision_table and
                        group_indices.intersection(collision_table[idx])):
                    indices_to_process.remove(idx)
                    group_indices.add(idx)
                    newly_added.append(idx)

        # try to add other faces
        for idx in indices_to_process.copy():
            if not group_indices.intersection(collision_table[idx]):
                indices_to_process.remove(idx)
                group_indices.add(idx)

        # tested all faces that were available, group is complete
        groups.append(group_indices)

    return groups


def build_collision_table(shapes):
    """Build collision table for given shapes via axis sweeping."""
    # make sure that all triangles have counter-clockwise winding
    shapes_ccw = []
    for triangle in shapes:
        v1, v2, v3 = triangle
        if det_2d(v1, v2, v3) < 0:
            shapes_ccw.append((v1, v3, v2))
        shapes_ccw.append(triangle)

    # axis-aligned bounding box for each triangle
    bounds = [compute_pointcloud_bounds(triangle) for triangle in shapes_ccw]

    # collision table: list index -> {indices of colliding triangles}
    collision_table = defaultdict(set)

    # sort triangles left to right, then sweep along x-axis
    number = len(shapes_ccw)
    indices_xaxis = sorted(range(number), key=lambda i: bounds[i][0])
    for sweep_index in range(number):
        # get x-bounds of current triangle
        idx = indices_xaxis[sweep_index]
        triangle = shapes_ccw[idx]
        max_x, max_y = bounds[idx][2:]

        # sweep further along x-axis, record triangles in current interval
        overlap_indices = []
        for sweep_ahead in range(sweep_index+1, number):
            jdx = indices_xaxis[sweep_ahead]
            if bounds[jdx][0] >= max_x:
                # arrived at end of x-interval, skip all following triangles
                break
            overlap_indices.append(jdx)

        # sort found triangles bottom to top, then sweep along y-axis
        overlap_indices.sort(key=lambda i: bounds[i][1])
        for jdx in overlap_indices:
            if bounds[jdx][1] >= max_y:
                # arrived at end of y-interval, skip all following triangles
                break

            # narrow-phase collision detection
            if intersect_triangle_triangle(triangle, shapes_ccw[jdx]):
                collision_table[idx].add(jdx)
                collision_table[jdx].add(idx)

    return collision_table


def compute_pointcloud_bounds(points):
    """Compute corners of axis-aligned bounding box containing the points."""
    p1 = points[0]
    lo_x = hi_x = p1.x
    lo_y = hi_y = p1.y

    for p in points[1:]:
        if p.x < lo_x:
            lo_x = p.x
        elif p.x > hi_x:
            hi_x = p.x

        if p.y < lo_y:
            lo_y = p.y
        elif p.y > hi_y:
            hi_y = p.y

    return (lo_x, lo_y, hi_x, hi_y)


# triangle-triangle intersection ----------------------------------------------
# adapted from https://rosettacode.org/wiki/Determine_if_two_triangles_overlap

def intersect_triangle_triangle(triangle1, triangle2):
    """Test if two triangles intersect using the separating axis theorem."""
    v1, v2, v3 = triangle1
    w1, w2, w3 = triangle2

    # check if triangles share an edge, then we know due to winding on which
    # side of the edge the other points lie and if the triangles intersect
    if v1 == w1:
        if v2 == w2:
            return True
        if v2 == w3:
            return False
    elif v1 == w2:
        if v2 == w3:
            return True
        if v2 == w1:
            return False
    elif v1 == w3:
        if v2 == w1:
            return True
        if v2 == w2:
            return False

    # see if one of the edges of triangle1 is a separating line
    for start, end in [(v1, v2), (v2, v3), (v3, v1)]:
        # if all vertices of triangle2 lie to the right side of the edge, then
        # this edge separates triangle1 (on the left) from triangle2 (on the
        # right)
        if (det_2d(start, end, w1) <= 0 and det_2d(start, end, w2) <= 0 and
                det_2d(start, end, w3) <= 0):
            return False

    # see if one of the edges of triangle2 is a separating line
    for start, end in [(w1, w2), (w2, w3), (w3, w1)]:
        if (det_2d(start, end, v1) <= 0 and det_2d(start, end, v2) <= 0 and
                det_2d(start, end, v3) <= 0):
            return False

    # if we arrived here, then none of the edges is a separating line and the
    # two triangles (which are convex) must intersect by the separating axis
    # theorem
    return True


def det_2d(p1, p2, p3):
    """Compute signed area of parallelogram spanned by p2 - p1, p3 - p1."""
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
