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
from time import perf_counter

import bmesh
import bpy
from bpy.types import Operator
from mathutils import Matrix

from lightsheet import trace, utils


class LIGHTSHEET_OT_finalize_caustic(Operator):
    """Smooth and cleanup selected caustics"""
    bl_idname = "lightsheet.finalize"
    bl_label = "Finalize Caustic"
    bl_options = {'REGISTER', 'UNDO'}

    fade_boundary: bpy.props.BoolProperty(
        name="Fade out boundary",
        description="Hide boundary by making it transparent",
        default=True
    )
    remove_dim_faces: bpy.props.BoolProperty(
        name="Remove dim faces",
        description="Remove faces that are less intense than the cutoff below",
        default=True
    )
    intensity_threshold: bpy.props.FloatProperty(
        name="Intensity Treshold",
        description="Remove face if for every vertex: caustic squeeze * tint "
        "<= threshold (note: light strength is not included)",
        default=0.0001, min=0.0, precision=5, subtype='FACTOR'
    )
    delete_empty_caustics: bpy.props.BoolProperty(
        name="Delete empty caustics",
        description="If after cleanup no faces remain, delete the caustic",
        default=True
    )
    fix_overlap: bpy.props.BoolProperty(
        name="Cycles: Fix overlap artifacts",
        description="WARNING: SLOW! Prevent render artifacts in Cycles caused "
        "by overlapping faces, will find intersecting faces (slow!) and stack "
        "them on top of each other",
        default=False
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

            min_dim = self.intensity_threshold if self.remove_dim_faces else 0
            finalize_caustic(obj, self.fade_boundary, min_dim,
                             self.fix_overlap)
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
def finalize_caustic(caustic, fade_boundary, intensity_threshold, fix_overlap):
    """Finalize caustic mesh."""
    # convert from object
    caustic_bm = bmesh.new()
    caustic_bm.from_mesh(caustic.data)

    # fade out boundary
    if fade_boundary:
        fadeout_caustic_boundary(caustic_bm)

    # smooth out and cleanup
    if intensity_threshold > 0:
        remove_dim_faces(caustic_bm, intensity_threshold)

    # stack overlapping faces in layers
    if fix_overlap:
        # derive offset from complexity of raypath chain so that different
        # caustics are also separated
        chain = []
        for item in caustic.caustic_info.path:
            chain.append(trace.Link(item.object, item.kind, None))
        offset = 1e-4 * utils.chain_complexity(chain)

        stack_overlapping_in_levels(caustic_bm, caustic.matrix_world, offset)

    # convert bmesh back to object
    caustic_bm.to_mesh(caustic.data)
    caustic_bm.free()

    # mark as finalized
    caustic.caustic_info.finalized = True


def fadeout_caustic_boundary(caustic_bm):
    """Set vertex color to black for vertices at the boundary."""
    color_layer = caustic_bm.loops.layers.color["Caustic Tint"]
    for vert in (v for v in caustic_bm.verts if v.is_boundary):
        for loop in vert.link_loops:
            loop[color_layer] = (0.0, 0.0, 0.0, 0.0)


def remove_dim_faces(caustic_bm, intensity_threshold):
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


def stack_overlapping_in_levels(bm, matrix_world, offset, separation=1e-4):
    """Stack intersecting faces such that they don't overlap anymore."""
    # find affine planes and the faces they contain
    affine_planes = collect_by_plane(bm, safe_distance=100*separation)

    # assign to each face a level
    level_to_faces = defaultdict(list)
    for _, faces, shapes in affine_planes:
        # group faces into non-intersecting sets, sort groups by area
        groups = group_by_collision_table(faces, shapes)
        groups.sort(key=lambda ics: sum(faces[idx].calc_area() for idx in ics),
                    reverse=True)

        for level, indices in enumerate(groups):
            level_to_faces[level].extend(faces[idx] for idx in indices)

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
        check_planes = []
        for vert in face.verts:
            for other_face in vert.link_faces:
                guess_plane = face_to_plane.get(other_face, None)
                if guess_plane is not None and guess_plane not in check_planes:
                    check_planes.append(guess_plane)

        # if we could not guess any planes, check all of them
        if not check_planes:
            check_planes = affine_planes

        # check if face is part of plane and if not then create one for it
        for projector, faces, shapes in check_planes:
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
                face_to_plane[face] = projector, faces, shapes

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
            face_to_plane[face] = affine_planes[-1]

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
