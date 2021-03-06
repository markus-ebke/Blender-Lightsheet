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
- remove_layers
- remove_dim_faces
- collect_by_plane
- create_affine_plane
- bmesh_duplicate_faces
- merge_in_plane
- det_2d
"""

from bisect import bisect_left, bisect_right
from math import pi
from time import process_time as stopwatch

import bmesh
import bpy
from bpy.types import Operator
from mathutils import Matrix, Vector
from mathutils.geometry import barycentric_transform, delaunay_2d_cdt

from lightsheet import utils


class LIGHTSHEET_OT_finalize_caustics(Operator):
    """Cleanup the selected caustics"""
    bl_idname = "lightsheet.finalize"
    bl_label = "Finalize Caustics"
    bl_options = {'REGISTER', 'UNDO'}

    fade_boundary: bpy.props.BoolProperty(
        name="Fade Out Boundary",
        description="Disguise the caustic boundary with a fade out",
        default=True
    )
    remove_dim_faces: bpy.props.BoolProperty(
        name="Remove Dim Faces",
        description="Remove faces that emit less power than the given cutoff",
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
        description="WARNING: SLOW AND USES A LOT OF MEMORY! Fix render "
        "artifacts in Cycles caused by overlapping faces, will merge faces "
        "into a single non-overlapping mesh. NOTE: use the offset of the "
        "shrinkwrap modifier to separate different caustics from each other",
        default=False
    )
    delete_coordinates: bpy.props.BoolProperty(
        name="Delete Lightsheet Coordinates",
        description="Delete the two UV-layers that hold lightsheet "
        "coordinates, will save memory but could be used for other effects. "
        "If overlapping faces are fixed, then these coordinates will always "
        "be removed",
        default=True
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
        if bpy.app.version < (2, 90, 0):
            layout.prop(self, "remove_dim_faces")
            row = layout
        else:
            heading = layout.column(heading="Remove Dim Faces")
            row = heading.row(align=True)
            row.prop(self, "remove_dim_faces", text="")
        sub = row.row()
        sub.active = self.remove_dim_faces
        sub.prop(self, "emission_cutoff")

        # if we don't remove dim faces, deleting empty caustics is not needed
        sub = layout.column()
        sub.active = self.remove_dim_faces
        sub.prop(self, "delete_empty_caustics")

        layout.prop(self, "fix_overlap")

        # always remove coordinates if fix_overlap is enabled
        sub = layout.column()
        sub.active = not self.fix_overlap
        sub.prop(self, "delete_coordinates")

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
            result = finalize_caustic(
                caustic, self.fade_boundary, emission_cutoff,
                self.delete_empty_caustics, self.fix_overlap,
                self.delete_coordinates, prog)
            if result is None:
                deleted += 1
            else:
                finalized += 1
            prog.stop_job()
        prog.end()
        toc = stopwatch()

        # report statistics
        # prog.print_stats()  # uncomment for profiling
        f_stats = f"Finalized {finalized}"
        d_stats = f"deleted {deleted} caustics"
        t_stats = f"{toc-tic:.1f}s"
        self.report({'INFO'}, f"{f_stats} and {d_stats} in {t_stats}")

        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Functions used by finalize caustics operator
# -----------------------------------------------------------------------------
def finalize_caustic(caustic, fade_boundary, emission_cutoff, delete_empty,
                     fix_overlap, delete_coordinates, prog):
    """Finalize caustic mesh."""
    # convert from object
    caustic_bm = bmesh.new()
    caustic_bm.from_mesh(caustic.data)

    # remove internal layers used for refining
    prog.start_task("deleting coordinates")
    remove_layers(caustic_bm, delete_coordinates or fix_overlap)

    # fade out boundary
    if fade_boundary:
        prog.start_task("fading out boundary")
        color_layer = caustic_bm.loops.layers.color["Caustic Tint"]
        for vert in caustic_bm.verts:
            if vert.is_boundary:
                for loop in vert.link_loops:
                    loop[color_layer] = (0.0, 0.0, 0.0, 0.0)

    # remove dim faces
    if emission_cutoff is not None:
        prog.start_task("removing dim faces")
        light = caustic.caustic_info.lightsheet.parent
        remove_dim_faces(caustic_bm, light, emission_cutoff)

        # if wanted by user delete caustic objects without faces
        if delete_empty and len(caustic_bm.faces) == 0:
            bpy.data.objects.remove(caustic)
            return None

    # fix overlapping faces
    if fix_overlap:
        num_faces = len(caustic_bm.faces)

        # sort faces into affine planes
        prog.start_task("finding affine planes", total_steps=num_faces)
        proper_planes, individual_faces = collect_by_plane(caustic_bm, prog)

        # add individual faces from trivial planes
        prog.start_task("merging faces", total_steps=num_faces)
        new_bm, vert_map = bmesh_duplicate_faces(caustic_bm, individual_faces)
        prog.update_progress(step=len(individual_faces))

        # merge faces in affine planes and add to new bmesh
        for projector, faces in proper_planes:
            vert_map = merge_within_plane(caustic_bm, projector, faces,
                                          new_bm, vert_map, prog)

        # replace old bmesh by new merged bmesh
        caustic_bm.free()
        caustic_bm = new_bm

    # stop any tasks that are still running
    prog.stop_task()

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
    caustic_info.delete_coordinates = delete_coordinates

    return caustic


def remove_layers(caustic_bm, delete_coordinates):
    """Delete internal caustic vertex layers and uv-layers if wanted."""
    # face index vertex layer
    layer = caustic_bm.verts.layers.int.get("Face Index")
    if layer is not None:
        caustic_bm.verts.layers.int.remove(layer)

    # lightsheet coordinate vertex layer
    for co in ["X", "Y", "Z"]:
        layer = caustic_bm.verts.layers.float.get(f"Lightsheet {co}")
        if layer is not None:
            caustic_bm.verts.layers.float.remove(layer)

    # lightsheet coordinate uv-layer
    if delete_coordinates:
        for co in ["XY", "XZ"]:
            layer = caustic_bm.loops.layers.uv.get(f"Lightsheet {co}")
            if layer is not None:
                caustic_bm.loops.layers.uv.remove(layer)


def remove_dim_faces(caustic_bm, light, emission_cutoff):
    """Remove invisible faces and cleanup resulting mesh."""
    # assert light is not None and light.type == 'LIGHT', light

    # emission strength and color from caustic
    squeeze_layer = caustic_bm.loops.layers.uv["Caustic Squeeze"]
    color_layer = caustic_bm.loops.layers.color["Caustic Tint"]

    # light color and strength, see also material.py: add_drivers_from_light
    if light.data.type == 'SUN':
        light_strength = light.data.energy / pi
    else:
        # assert light.data.type in {'SPOT', 'POINT'}, light
        light_strength = light.data.energy / (4 * pi**2)
    light_strength *= light.data.color.v

    # gather faces with low emission strength
    invisible_faces = []
    for face in caustic_bm.faces:
        visible = False
        for loop in face.loops:
            squeeze = loop[squeeze_layer].uv[1]
            color = utils.srgb_to_linear(loop[color_layer].to_3d())
            if light_strength * squeeze * max(color) > emission_cutoff:
                # vertex is intense enough, face is visible
                visible = True
                break

        if not visible:
            invisible_faces.append(face)

    # delete invisible faces and cleanup mesh
    bmesh.ops.delete(caustic_bm, geom=invisible_faces, context='FACES_ONLY')
    utils.bmesh_delete_loose(caustic_bm)


# fix overlap: collection ----------------------------------------------------
def collect_by_plane(bm, prog, safe_distance=1e-4):
    """Group faces of bmesh if they live in the same plane."""
    # sort faces by obliqueness, because if the height is very small compared
    # to the base then a small error in position of the tip vertex can have a
    # big effect on the normal of the face and therefore on the orientation of
    # the affine plane (we don't want to use these oblique faces to setup the
    # planes)
    def height_to_base_ratio(face):
        base_length = max(edge.calc_length() for edge in face.edges)
        return face.calc_area() / base_length**2  # * 2 unimportant for sort
    sorted_faces = sorted(bm.faces, key=height_to_base_ratio, reverse=True)

    # affine plane = (projection matrix, list of faces), the projection matrix
    # maps points in global space to (u, v, w) where (u, v) are local
    # coordinates in the plane and z is the distance to the plane
    affine_planes = []

    # record which faces were collected into which planes and try these planes
    # first for neighboring faces
    bm.faces.ensure_lookup_table()
    face_to_plane = dict()  # {face index: plane index}

    # save indices of planes in the order such that the planes are sorted by
    # their normal coordinates, save normals in sorted order for faster access
    order = ([], [], [])
    normals = ([], [], [])

    # generator for finding good plane indices
    def plane_indices(face):
        nonlocal face_to_plane, order, normals

        # check planes of the immediate neighbours first (if any)
        neighbor_indices = set()
        for vert in face.verts:
            for other_face in vert.link_faces:
                if other_face.index in face_to_plane:
                    plane_index = face_to_plane[other_face.index]
                    if plane_index not in neighbor_indices:
                        yield plane_index
                        neighbor_indices.add(plane_index)

        # get indices for planes with normal similar to face
        axis_idx = max([0, 1, 2], key=lambda idx: abs(face.normal[idx]))
        order_axis, normals_axis = order[axis_idx], normals[axis_idx]
        start = bisect_left(normals_axis, face.normal[axis_idx] - 0.1)
        stop = bisect_right(normals_axis, face.normal[axis_idx] + 0.1)
        for idx in range(start, stop):
            yield order_axis[idx]

    # process faces, starting with the most equilateral ones
    for face in sorted_faces:
        # check if face can be added to any existing affine plane and if not
        # then create one for it
        for plane_index in plane_indices(face):
            projector, faces = affine_planes[plane_index]
            for vert in face.verts:
                # project vert and check if lies within the plane
                point = projector @ vert.co
                if abs(point.z) > safe_distance:
                    # point is too far away from affine plane, break innermost
                    # loop over face.verts (will skip its else clause)
                    break
            else:
                # found an acceptable plane, add face and remember plane
                faces.append(face)
                face_to_plane[face.index] = plane_index

                # break loop over affine planes, skip else clause below
                break
        else:
            # create and add new affine plane
            affine_planes.append(create_affine_plane(face))
            face_to_plane[face.index] = len(affine_planes) - 1

            # sort plane indices by normal coordinates
            for axis_idx in [0, 1, 2]:
                order_axis, normals_axis = order[axis_idx], normals[axis_idx]
                sort_idx = bisect_right(normals_axis, face.normal[axis_idx])
                order_axis.insert(sort_idx, len(affine_planes) - 1)
                normals_axis.insert(sort_idx, face.normal[axis_idx])
        prog.update_progress()

    # filter out trivial planes that contain only one face
    proper_planes, individual_faces = [], []
    for plane in affine_planes:
        faces = plane[1]
        if len(faces) > 1:
            proper_planes.append(plane)
        else:
            individual_faces.extend(faces)

    return proper_planes, individual_faces


def create_affine_plane(face):
    """Create an affine plane for the given face with vertices in CCW order."""
    # did not find a plane that contains this face, create a new plane
    v1, v2, v3 = [vert.co for vert in face.verts]
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

    # check that v1 maps to origin and the other points map to xy-plane
    # and have correct side lengths
    # p1, p2, p3 = [projector @ co for co in (v1, v2, v3)]
    # assert p1.length < 1e-5, (p1, p1.length)
    # assert abs(p2.z) < 1e-5, (p2, p2.z)
    # assert abs(p3.z) < 1e-5, (p3, p3.z)
    # assert abs(p2.length - s1.length) < 1e-5, (p2.length, s1.length)
    # assert abs(p3.length - s2.length) < 1e-5, (p3.length. s2.length)

    return (projector, [face])


# fix overlap: merging -------------------------------------------------------
# We need the following function because bmesh.ops.duplicate does not work for
# two bmeshes: NotImplementedError when dest is another bmesh. Aaargh!!! I hope
# that programmer gets mauled by the rabbit of Caerbannog!
def bmesh_duplicate_faces(source_bm, faces):
    """Create new bmesh and copy faces (with caustic data) from old bmesh."""
    # get caustic layers of source bmesh
    squeeze_layer = source_bm.loops.layers.uv["Caustic Squeeze"]
    color_layer = source_bm.loops.layers.color["Caustic Tint"]
    uv_layer = utils.get_uv_map(source_bm)

    # create fresh bmesh and setup caustic data layers
    new_bm = bmesh.new()
    new_squeeze_layer = new_bm.loops.layers.uv.new("Caustic Squeeze")
    new_color_layer = new_bm.loops.layers.color.new("Caustic Tint")

    # create new verts
    verts = {vert for face in faces for vert in face.verts}
    vert_map = dict()
    for vert in verts:
        vert_map[vert] = new_bm.verts.new(vert.co)

    # create new faces
    face_map = dict()
    for face in faces:
        # get verts corresponding to old face and create new face
        new_face_verts = [vert_map[vert] for vert in face.verts]
        new_face = new_bm.faces.new(new_face_verts)
        face_map[face] = new_face

        # copy face data from old loops to new loops
        for loop, new_loop in zip(face.loops, new_face.loops):
            # assert new_loop.vert is vert_map[loop.vert]
            new_loop[new_squeeze_layer].uv = loop[squeeze_layer].uv
            new_loop[new_color_layer] = loop[color_layer]

    # copy uv-layer data if it exists
    if uv_layer is not None:
        new_uv_layer = new_bm.loops.layers.uv.new(uv_layer.name)
        for face, new_face in zip(faces, new_bm.faces):
            # assert new_face is face_map[face], (new_face, face_map[face])
            for loop, new_loop in zip(face.loops, new_face.loops):
                # assert new_loop.vert is vert_map[loop.vert]
                new_loop[new_uv_layer] = loop[uv_layer]

    return new_bm, vert_map


def merge_within_plane(bm, projector, faces, new_bm, vert_map, prog):
    """Merge faces that lie in the projector plane and add to new bmesh."""
    # in this function progress from 0% to 100% means going from
    # prog.current_step to prog.current_step + num_faces
    current_step = prog.current_step
    substeps = len(faces)

    # get caustic layers
    squeeze_layer = bm.loops.layers.uv["Caustic Squeeze"]
    color_layer = bm.loops.layers.color["Caustic Tint"]

    # read out vert coordinates and caustic color for each face
    srgb_to_linear = utils.srgb_to_linear
    face_points, face_color = [], []
    for face in faces:
        point_list, color_list = [], []
        for loop in face.loops:
            point_list.append(loop.vert.co.copy())
            squeeze = loop[squeeze_layer].uv.y
            color = srgb_to_linear(loop[color_layer].to_3d())
            color_list.append(squeeze * Vector(color))
        face_points.append(point_list)
        face_color.append(color_list)

    # read out uv-coordinates, if any
    uv_layer = utils.get_uv_map(bm)
    if uv_layer is not None:
        face_uv = []
        for face in faces:
            # note that barycentric_transform expects 3d-vectors
            face_uv.append([loop[uv_layer].uv.to_3d() for loop in face.loops])

    # convert vertices to 2d-points and remember the vert-index relation
    verts = list({vert for face in faces for vert in face.verts})
    points = [(projector @ vert.co).xy for vert in verts]
    vert_to_index = {vert: i for (i, vert) in enumerate(verts)}

    # convert edges to list of two vertex indices
    edge_indices = list({tuple(vert_to_index[vert] for vert in edge.verts)
                         for face in faces for edge in face.edges})

    # convert faces to list of three vertex indices in CCW-order
    face_indices = []
    for face in faces:
        i1, i2, i3 = [vert_to_index[vert] for vert in face.verts]
        if det_2d(points[i1], points[i2], points[i3]) >= 0:
            face_indices.append((i1, i2, i3))
        else:
            face_indices.append((i1, i3, i2))
    prog.update_progress(step=current_step+substeps*0.05)  # 5% done

    # constrained delaunay triangulation, note that this step uses a lot of
    # memory because the returned lists will be very big
    cdt_result = delaunay_2d_cdt(points, edge_indices, face_indices, 1, 1e-12)
    new_points, _, new_faces, orig_verts, _, orig_faces = cdt_result
    del _, cdt_result  # mark for garbage collection to clear memory
    prog.update_progress(step=current_step+substeps*0.25)  # 25% done

    # rebuild vertices in new bmesh
    mat = projector.inverted()
    cdt_verts = []
    for point, orig_idx in zip(new_points, orig_verts):
        # if an appropriate vert has already been created, use that one
        if orig_idx:
            vert = verts[orig_idx[0]]
            if vert in vert_map:
                cdt_verts.append(vert_map[vert])
                continue

        # add new vertex at the correct un-projected position
        new_vert = new_bm.verts.new(mat @ point.to_3d())
        cdt_verts.append(new_vert)

        # if there is an original vertex, add new vertex to vert_map
        if orig_idx:
            vert = verts[orig_idx[0]]
            vert_map[vert] = new_vert

    del new_points, orig_verts  # mark for garbage collection to clear memory
    prog.update_progress(step=current_step+substeps*0.3)  # 30% done

    # get new caustic layers
    new_squeeze_layer = new_bm.loops.layers.uv["Caustic Squeeze"]
    new_color_layer = new_bm.loops.layers.color["Caustic Tint"]
    if uv_layer is not None:
        new_uv_layer = utils.get_uv_map(new_bm)
        # assert new_uv_layer is not None

    # the loop over new faces takes a long time, use a local progress counter
    subcounter = prog.current_step  # local counter
    subincr = substeps / len(new_faces) * 0.7  # loop uses ~70% of the time

    # rebuild faces in new bmesh and interpolate their caustic color
    linear_to_srgb = utils.linear_to_srgb
    while new_faces:
        # get indices for new face and indices of old face, but don't use
        # zip(new_faces, orig_faces) because .pop() will help clear memory
        vert_indices = new_faces.pop()
        orig_indices = orig_faces.pop()

        # skip faces that fill holes, because they would be invisible anyway
        if not orig_indices:
            subcounter += subincr  # always update counter
            continue

        # create face
        face_verts = [cdt_verts[i] for i in vert_indices]
        try:
            face = new_bm.faces.new(face_verts)
        except ValueError:
            # ValueError: faces.new(verts): face already exists
            # This happens if collect_by_plane messed up: it did not sort the
            # face with the given verts into the current plane because another
            # plane was sufficient first (=> safe_distance too large). However
            # I don't want to set safe_distance to a lower value because we
            # might get a worse tradeoff between false positives vs. false
            # negatives. Instead just get the offending face and continue.
            face = new_bm.faces.get(face_verts)

        # set face data (squeeze and color)
        for loop in face.loops:
            point = loop.vert.co

            # add up colors from old faces
            color = Vector((0, 0, 0))
            for i in orig_indices:
                color += barycentric_transform(point, *face_points[i],
                                               *face_color[i])

            # normalize color and use its maximum as squeeze (= intensity)
            squeeze = max(*color, 1e-15)
            loop[new_squeeze_layer].uv[1] = squeeze
            loop[new_color_layer] = linear_to_srgb(color / squeeze) + (1,)

        # set face data (uv-coordinates if any)
        if uv_layer is not None:
            i = orig_indices[0]  # one index is enough, no average needed
            for loop in face.loops:
                uv = barycentric_transform(loop.vert.co, *face_points[i],
                                           *face_uv[i])
                loop[new_uv_layer].uv = uv.xy

        # update progress counter
        subcounter += subincr
        prog.update_progress(step=subcounter)

    # final state of progress counter
    prog.update_progress(step=current_step+substeps)

    # return updated vert map, this is not strictly neccessary because dicts
    # are mutable, but "explicit is better than implicit"
    return vert_map


def det_2d(p1, p2, p3):
    """Compute signed area of parallelogram spanned by p2 - p1, p3 - p1."""
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
