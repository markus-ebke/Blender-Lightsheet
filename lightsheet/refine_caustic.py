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
"""Adaptively subdivide selected caustics.

LIGHTSHEET_OT_refine_caustic: Operator for refining caustics

Helper functions:
- refine_caustic
- split_caustic_edges
- grow_caustic_boundary
"""

from time import perf_counter

import bmesh
import bpy
from bpy.types import Operator

from lightsheet import material, trace, utils


class LIGHTSHEET_OT_refine_caustic(Operator):
    """Refine selected caustics by adaptively subdividing edges"""
    bl_idname = "lightsheet.refine"
    bl_label = "Refine Caustic"
    bl_options = {'REGISTER', 'UNDO'}

    adaptive_subdivision: bpy.props.BoolProperty(
        name="Adaptive Subdivision",
        description="Subdivide edges based on projection error",
        default=True
    )
    error_threshold: bpy.props.FloatProperty(
        name="Error Threshold",
        description="Subdivide edge if distance between projected midpoint and"
        " midpoint of edge if >= threshold * 1/2 length of edge",
        default=0.1, min=0.0, soft_max=1.0, precision=2,
    )
    span_faces: bpy.props.BoolProperty(
        name="Subdivide edges spanning faces",
        description="Always subdivide edges whose vertices lie on different "
        "faces of the underlying object (better for wrapping around curved "
        "objects)",
        default=True
    )
    grow_boundary: bpy.props.BoolProperty(
        name="Grow Boundary",
        description="Always subdivide edges on the boundary, then add a strip "
        "of triangles on the outside",
        default=True
    )

    @classmethod
    def poll(cls, context):
        # operator makes sense only for caustics
        objects = context.selected_objects
        if objects:
            for obj in objects:
                caustic_info = obj.caustic_info
                if caustic_info.path and not caustic_info.finalized:
                    return True
        return False

    def invoke(self, context, event):
        assert context.selected_objects
        for obj in context.selected_objects:
            # skip non-refinable objects
            if not obj.caustic_info.path or obj.caustic_info.finalized:
                continue

            # check that we have a lightsheet
            lightsheet = obj.caustic_info.lightsheet
            if lightsheet is None:
                reasons = "it has no lightsheet"
                msg = f"Cannot refine {obj.name} because {reasons}!"
                self.report({"ERROR"}, msg)
                return {'CANCELLED'}

            # check that light (parent of lightsheet) is valid
            light = lightsheet.parent
            if light is None or light.type != 'LIGHT':
                reasons = "lightsheet parent is not a light"
                msg = f"Cannot refine {obj.name} because {reasons}!"
                self.report({"ERROR"}, msg)
                return {'CANCELLED'}

            # check that light type is supported
            light_type = light.data.type
            if light_type not in {'SUN', 'SPOT', 'POINT'}:
                reasons = f"{light_type.capitalize()} lights are not supported"
                msg = f"Cannot refine {obj.name} because {reasons}!"
                self.report({"ERROR"}, msg)
                return {'CANCELLED'}

        # set properties via dialog window
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        # parameters for tracing
        depsgraph = context.view_layer.depsgraph

        # set relative tolerance
        if self.adaptive_subdivision:
            relative_tolerance = self.error_threshold
        else:
            relative_tolerance = None

        num_verts_old, num_verts_now = 0, 0
        tic = perf_counter()
        for obj in context.selected_objects:
            # skip non-refinable objects
            if not obj.caustic_info.path or obj.caustic_info.finalized:
                continue

            # refine
            num_verts_old += len(obj.data.vertices)
            refine_caustic(obj, depsgraph, relative_tolerance, self.span_faces,
                           self.grow_boundary)
            num_verts_now += len(obj.data.vertices)
        toc = perf_counter()

        # report statistics
        v_stats = f"Added {num_verts_now-num_verts_old:,} verts"
        t_stats = f"{toc-tic:.3f}s"
        self.report({"INFO"}, f"{v_stats} in {t_stats}")

        return {"FINISHED"}


# -----------------------------------------------------------------------------
# Functions used by refine caustics operator
# -----------------------------------------------------------------------------
def refine_caustic(caustic, depsgraph, relative_tolerance=None,
                   span_faces=True, grow_boundary=True):
    """Do one adaptive subdivision of caustic bmesh."""
    # world to caustic object coordinate transformation
    world_to_caustic = caustic.matrix_world.inverted()

    # caustic info
    caustic_info = caustic.caustic_info
    lightsheet = caustic_info.lightsheet
    assert lightsheet is not None
    first_ray = utils.setup_lightsheet_first_ray(lightsheet)

    # chain for tracing
    chain = []
    for item in caustic_info.path:
        chain.append(trace.Link(item.object, item.kind, None))

    # convert caustic to bmesh
    caustic_bm = bmesh.new()
    caustic_bm.from_mesh(caustic.data)

    # setup face index from hit object
    face_index = caustic_bm.verts.layers.int["Face Index"]

    # coordinates of source position on lighsheet
    get_sheet, _ = utils.setup_sheet_property(caustic_bm)

    # make sure caches are clean
    trace.cache_clear()
    material.cache_clear()

    def calc_sheet_midpoint(edge):
        vert1, vert2 = edge.verts
        sheet_pos1 = get_sheet(vert1)
        sheet_pos2 = get_sheet(vert2)
        sheet_mid = (sheet_pos1 + sheet_pos2) / 2
        return sheet_mid

    # gather all edges that we have to split
    refine_edges = dict()  # {edge: sheet pos of midpoint}
    sheet_to_data = dict()  # {sheet pos: target data (trace.CausticData)}

    # gather edges that need to be split but skip non-seams
    deleted_edges = set()
    for edge in (ed for ed in caustic_bm.edges if ed.seam):
        # calc sheet midpoint, will be put into refine_edges
        sheet_mid = calc_sheet_midpoint(edge)

        if span_faces:
            # split edge if it spans different faces
            vert1, vert2 = edge.verts
            if vert1[face_index] != vert2[face_index]:
                refine_edges[edge] = sheet_mid

        if relative_tolerance is not None:
            # split edge if projection is not straight enough
            ray = first_ray(sheet_mid)
            cdata, _ = trace.trace_along_chain(ray, depsgraph, chain)
            sheet_to_data[tuple(sheet_mid)] = cdata

            if cdata is None:
                # edge will be deleted, split it and see later what happens
                deleted_edges.add(edge)
                refine_edges[edge] = sheet_mid
            else:
                # midpoint of edge
                vert1, vert2 = edge.verts
                edge_mid = (vert1.co + vert2.co) / 2

                # projected midpoint with offset
                position = cdata.location + 1e-4 * cdata.normal
                mid_target = world_to_caustic @ position

                # calc error and whether we should keep the edge
                rel_err = (edge_mid - mid_target).length / edge.calc_length()
                if rel_err >= relative_tolerance:
                    refine_edges[edge] = sheet_mid

    if grow_boundary:
        # edges that belong to a face where at least one edge will be deleted
        future_boundary_edges = set()
        for face in caustic_bm.faces:
            if any(edge in deleted_edges for edge in face.edges):
                # face will disappear => new boundary edges
                for edge in face.edges:
                    future_boundary_edges.add(edge)

        # include all edges of (present or future) boundary faces
        for edge in caustic_bm.edges:
            if edge.is_boundary or edge in future_boundary_edges:
                for face in edge.link_faces:
                    for other_edge in face.edges:
                        sheet_mid = calc_sheet_midpoint(other_edge)
                        refine_edges[other_edge] = sheet_mid

    # modify bmesh
    split_verts, split_edges = split_caustic_edges(caustic_bm, refine_edges)
    if grow_boundary:
        # find any offending vertices?
        is_growable = True
        for vert in caustic_bm.verts:
            if vert.is_boundary and len(vert.link_edges) > 6:
                sheet_vec = get_sheet(vert)
                print(f"Sheet: {sheet_vec}, links: {len(vert.link_edges)}")
                is_growable = False

        if is_growable:
            boundary_verts = grow_caustic_boundary(caustic_bm)
        else:
            # raise exception
            msg = "cannot grow boundary, found verts with too many neighbours"
            info = "printed sheet coordinates of offending points to terminal"
            raise RuntimeError(f"{msg}; {info}")
    else:
        boundary_verts = []

    # ensure triangles
    triang_less = [face for face in caustic_bm.faces if len(face.edges) > 3]
    if triang_less:
        print(f"We have to triangulate {len(triang_less)} faces")
        bmesh.ops.triangulate(caustic_bm, faces=triang_less)

    # verify newly added vertices
    new_verts = split_verts + boundary_verts
    dead_verts = []
    for vert in new_verts:
        # get sheet coords and convert to key for dict (cannot use Vectors as
        # keys because they are mutable)
        sheet_pos = get_sheet(vert)
        sheet_key = tuple(sheet_pos)

        # trace ray if necessary
        if sheet_key in sheet_to_data:
            cdata = sheet_to_data[sheet_key]
        else:
            ray = first_ray(sheet_pos)
            cdata, _ = trace.trace_along_chain(ray, depsgraph, chain)
            sheet_to_data[sheet_key] = cdata

        # set coordinates or mark vertex for deletion
        if cdata is None:
            dead_verts.append(vert)
            del sheet_to_data[sheet_key]
        else:
            # set correct vertex coordinates and face index
            position = cdata.location + 1e-4 * cdata.normal
            vert.co = world_to_caustic @ position
            vert[face_index] = cdata.face_index

    # remove verts that have no target
    bmesh.ops.delete(caustic_bm, geom=dead_verts, context='VERTS')
    utils.bmesh_delete_loose(caustic_bm)
    new_verts = [vert for vert in new_verts if vert.is_valid]
    assert all(data is not None for data in sheet_to_data.values())

    # cleanup generated meshes and caches
    trace.cache_clear()
    material.cache_clear()

    # gather the vertices whose neighbours have changed (to recalculate
    # squeeze), the edges that we may split next and the faces where we
    # changed at least one vertex (to recalculate face data)
    dirty_verts = {neighbour for vert in new_verts
                   for edge in vert.link_edges
                   for neighbour in edge.verts}
    dirty_edges = set(split_edges)
    for vert in new_verts:
        if vert.is_valid:
            for ed in vert.link_edges:
                dirty_edges.add(ed)
    dirty_faces = {face for vert in new_verts for face in vert.link_faces}

    # recalculate squeeze and set face data for dirty faces
    utils.set_caustic_squeeze(caustic_bm, matrix_sheet=lightsheet.matrix_world,
                              matrix_caustic=caustic.matrix_world,
                              verts=dirty_verts)
    utils.set_caustic_face_data(caustic_bm, sheet_to_data, faces=dirty_faces)

    # mark edges for next refinement step
    for edge in caustic_bm.edges:
        edge.seam = edge in dirty_edges

    # select only the newly added verts
    for face in caustic_bm.faces:
        face.select_set(False)
    for vert in new_verts:
        vert.select_set(True)

    # convert bmesh back to object
    caustic_bm.to_mesh(caustic.data)
    caustic_bm.free()


def split_caustic_edges(caustic_bm, refine_edges):
    """Subdivide the given edges and return the new vertices."""
    # sheet coordinate access
    get_sheet, set_sheet = utils.setup_sheet_property(caustic_bm)

    def calc_sheet_midpoint(edge):
        vert1, vert2 = edge.verts
        sheet_pos1 = get_sheet(vert1)
        sheet_pos2 = get_sheet(vert2)
        sheet_mid = (sheet_pos1 + sheet_pos2) / 2
        return sheet_mid

    # balance refinement: if two edges of a triangle will be refined, also
    # subdivide the other edge => after splitting we always get triangles
    last_added = list(refine_edges.keys())
    newly_added = list()
    while last_added:  # crawl through the mesh and select edges that we need
        for edge in last_added:  # only last edges can change futher selection
            for face in edge.link_faces:
                assert len(face.edges) == 3  # must be a triangle
                not_refined = [
                    ed for ed in face.edges if ed not in refine_edges]
                if len(not_refined) == 1:
                    ed = not_refined[0]
                    refine_edges[ed] = calc_sheet_midpoint(ed)
                    newly_added.append(ed)

        last_added = newly_added
        newly_added = list()

    # split edges
    edgelist = list(refine_edges.keys())
    splits = bmesh.ops.subdivide_edges(caustic_bm, edges=edgelist, cuts=1,
                                       use_grid_fill=True,
                                       use_single_edge=True)

    # gather newly added vertices
    dirty_verts = set()
    for item in splits['geom_inner']:
        if isinstance(item, bmesh.types.BMVert):
            dirty_verts.add(item)

    # get all newly added verts and set their sheet coordinates
    split_verts = []
    for edge, sheet_pos in refine_edges.items():
        # one of the endpoints of a refined edge is a newly added vertex
        v1, v2 = edge.verts
        vert = v1 if v1 in dirty_verts else v2
        assert vert in dirty_verts, sheet_pos

        set_sheet(vert, sheet_pos)
        split_verts.append(vert)

    # gather edges that were split
    split_edges = []
    for item in splits['geom']:
        if isinstance(item, bmesh.types.BMEdge):
            split_edges.append(item)

    return split_verts, split_edges


def grow_caustic_boundary(caustic_bm):
    """Exand the boundary of the given caustic outwards in the lightsheet."""
    # names of places:
    # boundary: at the boundary of the original mesh
    # outside:  extended outwards, at the boundary of the new mesh
    # fan:      vertices around a central point (corner vertex at boundary)

    # sheet coordinate access
    get_sheet, set_sheet = utils.setup_sheet_property(caustic_bm)

    # categorize connections of vertices at the boundary, note that adding new
    # faces will change .link_edges of boundary vertices, therefore we have to
    # save the original connections here before creating new faces
    original_boundary_connections = dict()
    for vert in (v for v in caustic_bm.verts if v.is_boundary):
        # categorize linked edges based on connected faces
        wire_edges, boundary_edges, inside_edges = [], [], []
        for edge in vert.link_edges:
            if edge.is_wire:
                assert len(edge.link_faces) == 0
                wire_edges.append(edge)
            elif edge.is_boundary:
                assert len(edge.link_faces) == 1
                boundary_edges.append(edge)
            else:
                assert len(edge.link_faces) == 2
                inside_edges.append(edge)

        assert len(boundary_edges) in (0, 2, 4), len(boundary_edges)
        conn = (wire_edges, boundary_edges, inside_edges)
        original_boundary_connections[vert] = conn

    # record new vertices as they are created
    outside_verts = []

    def create_vert(sheet_pos):
        new_vert = caustic_bm.verts.new()
        outside_verts.append(new_vert)
        set_sheet(new_vert, sheet_pos)
        return new_vert

    # record new faces as they are created
    new_faces = []  # list of new faces

    def create_triangle(v1, v2, v3):
        face = caustic_bm.faces.new((v1, v2, v3))
        new_faces.append(face)
        return face

    # create outside pointing triangle with a boundary edge at the base and an
    # outside vertex at the tip
    boundary_edge_to_outside_vert = dict()  # {boundary edge: new outside vert}
    original_boundary_edges = [ed for ed in caustic_bm.edges if ed.is_boundary]
    for edge in original_boundary_edges:
        # get verts that are connected by the edge
        vert_first, vert_second = edge.verts

        # get the vertex opposite of the edge
        assert len(edge.link_faces) == 1
        face = edge.link_faces[0]
        assert len(face.verts) == 3
        other_verts = [ve for ve in face.verts if ve not in edge.verts]
        assert len(other_verts) == 1
        vert_opposite = other_verts[0]

        # sheet coordinates of the three vertices
        sheet_first = get_sheet(vert_first)
        sheet_second = get_sheet(vert_second)
        sheet_opposite = get_sheet(vert_opposite)

        # mirror opposite vertex across the edge to get the outside vertex
        # want: sheet_midpoint == (sheet_opposite + sheet_outside) / 2
        sheet_midpoint = (sheet_first + sheet_second) / 2
        sheet_outside = 2 * sheet_midpoint - sheet_opposite
        vert_outside = create_vert(sheet_outside)

        # add outside vertex to mappings
        boundary_edge_to_outside_vert[edge] = vert_outside

        # create face
        create_triangle(vert_first, vert_second, vert_outside)

    # create inside pointing trinagle with a boundary vertex at the tip and an
    # outside edge at the base
    targetmap = dict()  # {vert to replace: replacement vert}
    for vert, conn in original_boundary_connections.items():
        sheet_vert = get_sheet(vert)
        wire_edges, boundary_edges, inside_edges = conn

        # degree = number of neighbours
        degree = len(boundary_edges) + len(inside_edges)
        assert 2 <= degree <= 6, (sheet_vert, boundary_edges, inside_edges)

        if degree == 2:
            # 300° outside angle, needs two more vertices to create three faces
            assert len(boundary_edges) == 2, (sheet_vert, boundary_edges)
            # assert len(inside_edges) == 0, (sheet_vert, inside_edges)

            # get sheet coordinates of neighbouring vertices
            verts_fan = []
            for edge in boundary_edges:
                # get other vert in this edge and its sheet coordinates
                vert_opposite = edge.other_vert(vert)
                sheet_opposite = get_sheet(vert_opposite)

                # mirror across vertex
                # want: sheet_vert == (sheet_opposite + sheet_outside) / 2
                sheet_fan = 2 * sheet_vert - sheet_opposite
                vert_fan = create_vert(sheet_fan)

                # add to list
                verts_fan.append((vert_fan, edge))
            vert_fan_a, from_edge_a = verts_fan[0]
            vert_fan_b, from_edge_b = verts_fan[1]

            # get the outside neighbours of vert_fan_a/b, note that because
            # of mirroring we change the order in which we connect the vertices
            vert_outside_a = boundary_edge_to_outside_vert[from_edge_b]
            vert_outside_b = boundary_edge_to_outside_vert[from_edge_a]

            # create three new faces
            create_triangle(vert, vert_outside_a, vert_fan_a)
            create_triangle(vert, vert_fan_a, vert_fan_b)
            create_triangle(vert, vert_fan_b, vert_outside_b)
        elif degree == 3:
            # 240° outside angle, needs one more vertex to create two faces
            assert len(boundary_edges) == 2, (sheet_vert, boundary_edges)
            assert len(inside_edges) == 1, (sheet_vert, inside_edges)

            # get vertex of edge on the inside and its sheet coordinates
            inside_edge = inside_edges[0]
            vert_opposite = inside_edge.other_vert(vert)
            sheet_opposite = get_sheet(vert_opposite)

            # mirror across vertex
            # want: sheet_vert == (sheet_opposite + sheet_fan) / 2
            sheet_fan = 2 * sheet_vert - sheet_opposite
            vert_fan = create_vert(sheet_fan)

            # get neighbours of outside vert
            edge_a, edge_b = boundary_edges
            vert_outside_a = boundary_edge_to_outside_vert[edge_a]
            vert_outside_b = boundary_edge_to_outside_vert[edge_b]

            # create two new faces
            create_triangle(vert, vert_outside_a, vert_fan)
            create_triangle(vert, vert_fan, vert_outside_b)
        elif degree == 4:
            # the vert is at an 180° angle or at an X-shaped intersection
            assert len(boundary_edges) in (2, 4), (sheet_vert, boundary_edges)
            # assert len(inside_edges) in (0, 2), (sheet_vert, inside_edges)

            if len(boundary_edges) == 2:  # 180° angle
                edge_a, edge_b = boundary_edges
                vert_outside_a = boundary_edge_to_outside_vert[edge_a]
                vert_outside_b = boundary_edge_to_outside_vert[edge_b]
                create_triangle(vert, vert_outside_a, vert_outside_b)
            else:  # X-shaped intersection of two 120° angles
                # pairs of boundary edges have the same outside vertex, merge
                # the corresponding pairs of outside vertices
                assert len(boundary_edges) == 4, (sheet_vert, boundary_edges)

                # get outside verts
                outside_stuff = []
                for edge in boundary_edges:
                    vert_outside = boundary_edge_to_outside_vert[edge]
                    sheet_outside = get_sheet(vert_outside)
                    outside_stuff.append((edge, vert_outside, sheet_outside))

                # pair up the verts via distance in sheet
                sheet_outside_a = outside_stuff[0][2]
                outside_stuff.sort(
                    key=lambda item: (item[2] - sheet_outside_a).length
                )
                pair_1 = outside_stuff[0], outside_stuff[1]
                pair_2 = outside_stuff[2], outside_stuff[3]

                # merge pairs
                for pair in (pair_1, pair_2):
                    outside_a, outside_b = pair
                    edge_a, vert_outside_a, sheet_outside_a = outside_a
                    edge_b, vert_outside_b, sheet_outside_b = outside_b

                    # merge vertices at median point
                    sheet_merge = (sheet_outside_a + sheet_outside_b) / 2
                    set_sheet(vert_outside_a, sheet_merge)
                    targetmap[vert_outside_b] = vert_outside_a
        elif degree == 5:
            # 120° outside angle => the boundary edges connected to this vertex
            # have the same outside vertex, merge these vertices
            edge_a, edge_b = boundary_edges

            # get verts to merge and their sheet coordinates
            vert_outside_a = boundary_edge_to_outside_vert[edge_a]
            vert_outside_b = boundary_edge_to_outside_vert[edge_b]
            sheet_outside_a = get_sheet(vert_outside_a)
            sheet_outside_b = get_sheet(vert_outside_b)

            # merge vertices at median point
            sheet_merge = (sheet_outside_a + sheet_outside_b) / 2
            set_sheet(vert_outside_a, sheet_merge)
            targetmap[vert_outside_b] = vert_outside_a
        elif degree == 6:
            # 60° outside angle, one face of a complete hexagon is missing,
            # except triangles that grew from the boundary edges already lie on
            # top of this face
            edge_a, edge_b = boundary_edges
            vert_a = edge_a.other_vert(vert)
            vert_b = edge_b.other_vert(vert)
            vert_outside_a = boundary_edge_to_outside_vert[edge_a]
            vert_outside_b = boundary_edge_to_outside_vert[edge_b]

            # we can merge vert_a and vert_outside_b because they overlap and
            # do the same for vert_b and vert_outside_a, note that when we
            # merge the vertices later one of the faces over edge_a and edge_b
            # will be deleted
            set_sheet(vert_outside_a, get_sheet(vert_b))
            targetmap[vert_outside_a] = vert_b
            set_sheet(vert_outside_b, get_sheet(vert_a))
            targetmap[vert_outside_b] = vert_a
        else:
            # degree < 2 or degree > 6 should not be possible, but it might
            # happen if suddenly a hole starts to appear in an area with uneven
            # subdivision (usually inside the caustic and not at the boundary)
            pass

    # remove the doubled verts (will delete faces if case degree == 6 happend)
    bmesh.ops.weld_verts(caustic_bm, targetmap=targetmap)
    outside_verts = [vert for vert in outside_verts if vert.is_valid]

    # update uv coordinates and vertex colors for new faces
    new_faces = [face for face in new_faces if face.is_valid]
    ret = bmesh.ops.face_attribute_fill(caustic_bm, faces=new_faces,
                                        use_normals=False,  # calc normal later
                                        use_data=True)
    assert len(ret["faces_fail"]) == 0, ret["faces_fail"]

    return outside_verts
