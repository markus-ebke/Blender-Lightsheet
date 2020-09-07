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

LIGHTSHEET_OT_create_lightsheet: Create a lightsheet, uses the functions
create_bmesh_square, create_bmesh_circle and create_bmesh_icosahedron to create
lightsheets depending on light type.

LIGHTSHEET_OT_trace_lighsheet: Trace the selected lightsheet and create caustic
objects, tracing is done with functions from trace.py
"""

from math import sqrt, tan
from time import time

import bmesh
import bpy
from bpy.types import Operator
from mathutils import Vector
from mathutils.geometry import barycentric_transform

from lightsheet import material, trace

print("lightsheet operators.py")


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
        assert obj is not None and obj.type == 'LIGHT', obj

        # cancel operator for area lights
        light_type = obj.data.type
        if light_type not in {'SUN', 'SPOT', 'POINT'}:
            message = f"{light_type} lights are not supported"
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        # set resolution via dialog window
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        start = time()

        # build lightsheet around active object (should be a light source)
        obj = context.object
        assert obj is not None and obj.type == 'LIGHT', obj

        # setup lightsheet bmesh, type of lightsheet depends on type of light
        light_type = obj.data.type
        assert light_type in {'SUN', 'SPOT', 'POINT'}, obj.data.type
        if light_type == 'SUN':
            # sun gets a square grid, because we don't know anything better
            sidelength = 2  # scale by hand if not the right size
            bm = create_bmesh_square(sidelength, self.resolution)
        elif light_type == 'SPOT':
            # intersect cone of spot with shifted plane => circle with
            # radius = tan(halfangle) (because size of circle = sin(...) and
            # shift = cos(...), but we want shift = 1, so divide by cos(...)
            # and radius becomes sin / cos = tan)
            angle = obj.data.spot_size  # between 0° and 180°, but in radians
            radius = min(tan(angle / 2), 10)  # restrict for angles near 180°
            bm = create_bmesh_circle(radius, self.resolution)

            # shift circle to inside of cone
            for vert in bm.verts:
                vert.co.z = -1
        else:
            # icosahedron because lightsheet should surround the point light
            bm = create_bmesh_icosahedron(self.resolution)

        # create id layer and assign values
        bm_id = bm.verts.layers.int.new("id")
        for idx, vert in enumerate(bm.verts):
            vert[bm_id] = idx

        # think of a good name
        name = f"Lightsheet for {obj.name}"

        # convert bmesh to mesh data block and create new object
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        lightsheet = bpy.data.objects.new(name, me)

        # adjust drawing and visibility
        lightsheet.display_type = 'WIRE'
        lightsheet.hide_render = True
        lightsheet.cycles_visibility.camera = False
        lightsheet.cycles_visibility.shadow = False
        lightsheet.cycles_visibility.diffuse = False
        lightsheet.cycles_visibility.transmission = False
        lightsheet.cycles_visibility.scatter = False

        # add to scene
        coll = context.scene.collection
        coll.objects.link(lightsheet)
        lightsheet.parent = obj

        # report statistics
        v_stats = "{} vertices".format(len(me.vertices))
        f_stats = "{} faces".format(len(me.polygons))
        t_stats = "{:.1f}ms".format((time() - start) * 1000)
        message = f"Created {v_stats} and {f_stats} in {t_stats}"
        self.report({"INFO"}, message)

        return {"FINISHED"}


def create_bmesh_square(sidelength, resolution):
    """Create bmesh for a square filled with triangles."""
    # horizontal and vertical strides, note that odd horizontal strips are
    # shifted by dx/2 and the height of an equilateral triangle with base a
    # is h = sqrt(3)/2 a
    dx = sidelength / (resolution - 1 / 2)
    dy = sqrt(3) / 2 * dx

    # horizontal and vertical resolution, note that we need to correct the
    # vertical resolution because triangle height is less than base length
    xres = resolution
    yres = int(resolution * 2 / sqrt(3))
    ydiff = sidelength - (yres - 1) * dy  # height error we make with triangles

    bm = bmesh.new()

    # place vertices
    strips = []  # each entry is a horizontal strip of vertices
    for j in range(yres):
        py = -sidelength / 2 + j * dy
        py += ydiff / 2  # center in y-direction

        strip = []
        for i in range(xres):
            px = -sidelength / 2 + i * dx
            px += (j % 2) * dx / 2  # shift the odd strips to the right
            vert = bm.verts.new((px, py, 0))
            strip.append(vert)
        strips.append(strip)

    # fill in faces
    for j in range(yres - 1):
        # lower and upper horizontal strips
        lower = strips[j]
        upper = strips[j + 1]

        if j % 2 == 0:
            # fill triangles in up,down,up,down,... configuration
            for i in range(xres - 1):
                bm.faces.new((lower[i], upper[i], lower[i + 1]))
                bm.faces.new((lower[i + 1], upper[i], upper[i + 1]))
        else:
            # fill triangles in down,up,down,up,... configuration
            for i in range(xres - 1):
                bm.faces.new((lower[i], upper[i], upper[i + 1]))
                bm.faces.new((lower[i + 1], lower[i], upper[i + 1]))

    return bm


def create_bmesh_circle(radius, resolution):
    """Create bmesh for a circle filled with triangles."""
    # the easiest way to create a circle is to create a square and then cut out
    # the circle
    bm = create_bmesh_square(2 * radius, resolution)

    # gather vertices that lie outside the circle and delete them, note that
    # this will also delete some faces on the edge of the circle which may
    # look weird for low resolutions
    outside_verts = [vert for vert in bm.verts
                     if vert.co[0] ** 2 + vert.co[1] ** 2 > radius ** 2]
    bmesh.ops.delete(bm, geom=outside_verts, context="VERTS")

    return bm


def create_bmesh_icosahedron(resolution, radius=1):
    """Create an icosahedral bmesh with faces filled by triangles."""
    # template icosahedron
    bm_template = bmesh.new()
    bmesh.ops.create_icosphere(bm_template, subdivisions=0, diameter=radius)

    # size of source triangle and smaller filler triangles
    source_triangle = [
        Vector((0, 0, 0)),
        Vector((resolution - 1, 0, 0)),
        Vector((0, resolution - 1, 0))
    ]

    # replace every triangular face with a triangle mesh of given resolution
    bm = bmesh.new()
    for face in bm_template.faces:
        assert len(face.verts) == 3, len(face.verts)
        target_triangle = [vert.co for vert in face.verts]

        # place vertices
        strips = []  # each entry is a horizontal strip of vertices
        for j in range(resolution):
            strip = []
            for i in range(resolution - j):  # less vertices as we go higher
                coords = barycentric_transform(Vector((i, j, 0)),
                                               *source_triangle,
                                               *target_triangle)
                vert = bm.verts.new(coords)
                strip.append(vert)
            strips.append(strip)

        # fill in faces
        for j in range(resolution - 1):
            # lower and upper horizontal strips
            lower = strips[j]
            upper = strips[j + 1]

            # fill triangles in up,down,up,down,...,up configuration
            end = resolution - j - 2
            for i in range(end):
                bm.faces.new((lower[i], lower[i + 1], upper[i]))
                bm.faces.new((lower[i + 1], upper[i + 1], upper[i]))
            bm.faces.new((lower[end], lower[end + 1], upper[end]))

    # we created overlapping vertices at the edges and corners, merge them
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.1)

    bm_template.free()
    return bm


# -----------------------------------------------------------------------------
# Trace lightsheet
# -----------------------------------------------------------------------------
class LIGHTSHEET_OT_trace_lightsheet(Operator):
    """Trace rays from selected lightsheet to create caustics"""
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
        if obj is not None:
            parent = obj.parent
            return parent is not None and parent.type == 'LIGHT'

        return False

    def invoke(self, context, event):
        obj = context.object
        assert obj is not None and obj.parent.type == 'LIGHT', obj

        # set max_bounces via dialog window
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        start = time()

        # context variables
        scene = context.scene
        view_layer = context.view_layer
        depsgraph = context.evaluated_depsgraph_get()

        # set globals for trace
        trace.setup(scene, view_layer, depsgraph)

        # lightsheet (source of rays) = active object
        lightsheet = context.object
        ls_mesh = lightsheet.to_mesh()
        lightsheet_eval = lightsheet.evaluated_get(depsgraph)
        matrix = lightsheet_eval.matrix_world.copy()

        # verify that lightsheet has id layer = persistent vertex index, if not
        # then create one and assign values
        ls_id = ls_mesh.vertex_layers_int.get("id")
        if ls_id is None:
            ls_id = ls_mesh.vertex_layers_int.new(name="id")
            for vert in ls_mesh.vertices:
                ls_id.data[vert.index].value = vert.index

        # check id data before we continue
        assert all(ls_id.data[v.index].value != -1 for v in ls_mesh.vertices)

        # hide lightsheets and caustics from raycast
        hidden = []
        for obj in view_layer.objects:
            if "Lightsheet" in obj.name or "Caustic" in obj.name:
                hidden.append((obj, obj.hide_viewport))
                obj.hide_viewport = True

        # raytrace lightsheet and convert resulting caustics to objects
        light_type = lightsheet.parent.data.type
        mode = "parallel" if light_type == 'SUN' else "point"
        path_bm = trace.trace_lightsheet(
            ls_mesh, matrix, self.max_bounces, mode)
        caustics = convert_to_objects(lightsheet, path_bm)

        # get or setup collection for caustics
        coll = bpy.data.collections.get("Caustics")
        if coll is None:
            coll = bpy.data.collections.new("Caustics")
            scene.collection.children.link(coll)

        # add caustic objects to caustic collection
        for obj in caustics:
            coll.objects.link(obj)

        # restore original state for hidden lightsheets and caustics
        for obj, state in hidden:
            obj.hide_viewport = state

        # cleanup generated meshes and reset trace globals
        lightsheet.to_mesh_clear()
        trace.cleanup()

        # report statistics
        c_stats = "{} caustics".format(len(caustics))
        t_stats = "{:.3f}s".format((time() - start))
        message = f"Created {c_stats} in {t_stats}"
        self.report({"INFO"}, message)

        return {"FINISHED"}


def convert_to_objects(lightsheet, path_bm):
    """Convert caustic bmeshes to blender objects with filled in faces."""
    ls_mesh = lightsheet.to_mesh()
    ls_name = lightsheet.name
    ls_id = ls_mesh.vertex_layers_int["id"]

    # check consistency
    assert (ls_id.data[vert.index].value != -1 for vert in ls_mesh.vertices)

    # create faces and turn bmeshes into objects
    caustic_objects = []
    for path, (bm, uv_dict, color_dict) in path_bm.items():
        # parent of caustic = last object
        parent_obj = path[-1][0]
        assert path[-1][1] == "diffuse", path[-1]  # check consistency of path

        id_layer = bm.verts.layers.int["id"]
        id_cache = {vert[id_layer]: vert for vert in bm.verts}
        squeeze_layer = bm.loops.layers.uv["Caustic Squeeze"]

        # create corresponding faces
        for ls_face in ls_mesh.polygons:
            caustic_verts = []
            for ls_vert_index in ls_face.vertices:
                ls_vert_id = ls_id.data[ls_vert_index].value
                vert = id_cache.get(ls_vert_id)
                if vert is not None:
                    caustic_verts.append(vert)

            # create edge or face from vertices
            if len(caustic_verts) == 2:
                # can only create an edge here, but this edge may already exist
                # if we worked through a neighboring face before
                if bm.edges.get(caustic_verts) is None:
                    bm.edges.new(caustic_verts)
            elif len(caustic_verts) > 2:
                caustic_face = bm.faces.new(caustic_verts)

                # set squeeze factor = ratio of source area to projected area
                squeeze = ls_face.area / caustic_face.calc_area()
                for loop in caustic_face.loops:
                    # TODO can we use the u-coordinate for something useful?
                    loop[squeeze_layer].uv = (0, squeeze)

        # set transplanted uv-coordinates and vertex colors
        uv_layer = bm.loops.layers.uv["Transplanted UVMap"]
        color_layer = bm.loops.layers.color["Caustic Tint"]
        for face in bm.faces:
            for loop in face.loops:
                vert_id = loop.vert[id_layer]
                if uv_dict:  # only set uv-coords if we have uv-coords
                    loop[uv_layer].uv = uv_dict[vert_id]
                loop[color_layer] = tuple(color_dict[vert_id]) + (1,)

        # think of a good name
        name = f"Caustic of {ls_name}"
        for obj, interaction, _ in path:
            name += f" -> {obj.name} ({interaction})"

        # if we didn't copy any uv-coordinates from the parent object, then we
        # don't need the transplanted uvmap layer
        if not uv_dict:
            assert not parent_obj.data.uv_layers, parent_obj.data.uv_layers[:]
            bm.loops.layers.uv.remove(uv_layer)

        # new mesh data block
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()

        # rename transplanted uvmap to avoid confusion
        if uv_dict:
            parent_uv_layer = parent_obj.data.uv_layers.active
            assert parent_obj.data.uv_layers.active is not None
            uv_layer = me.uv_layers["Transplanted UVMap"]
            uv_layer.name = parent_uv_layer.name

        # before setting parent, undo parent transform
        me.transform(parent_obj.matrix_world.inverted())

        # new object with given mesh
        caustic = bpy.data.objects.new(name, me)
        caustic.parent = parent_obj

        # get or setup caustic material
        mat = material.get_caustic_material(lightsheet.parent, parent_obj)
        caustic.data.materials.append(mat)

        caustic_objects.append(caustic)

    return caustic_objects
