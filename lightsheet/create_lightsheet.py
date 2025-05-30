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
"""Create a lightsheet, the kind of sheet depends on the light type.

LIGHTSHEET_OT_create_lightsheet: Blender operator to create lightsheet

Helper functions:
- setup_lightsheet
- create_bmesh_square
- create_bmesh_disk
- create_bmesh_sphere
- convert_bmesh_to_lightsheet
"""

from math import sqrt, tan
from time import process_time as stopwatch

import bmesh
import bpy
from bpy.types import Operator
from mathutils import Vector
from mathutils.geometry import barycentric_transform

from . import utils


class LIGHTSHEET_OT_create_lightsheets(Operator):
    """Create lightsheets for the selected lights (Sun, Spot or Point lamps)"""
    bl_idname = "lightsheet.create"
    bl_label = "Create Lightsheets"
    bl_options = {'REGISTER', 'UNDO'}

    resolution: bpy.props.IntProperty(
        name="Resolution", description="Resolution of lightsheet mesh",
        default=50, min=2
    )

    @classmethod
    def poll(cls, context):
        # operator makes sense only for light objects
        if context.selected_objects:
            return all(obj.type == 'LIGHT' for obj in context.selected_objects)
        return False

    def invoke(self, context, event):
        # cancel with error message
        def cancel(obj, reasons):
            verb = "create lightsheet for"
            msg = f"Cannot {verb} '{obj.name}' because {reasons}!"
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        # cancel operator for area lights
        for obj in context.selected_objects:
            light_type = obj.data.type
            if light_type not in {'SUN', 'SPOT', 'POINT'}:
                reasons = f"{light_type.lower()} lights are not supported"
                return cancel(obj, reasons)

        # set resolution via dialog window
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        # create lightsheets for every selected object (will all be lamps)
        tic = stopwatch()
        lightsheets = [setup_lightsheet(obj, self.resolution)
                       for obj in context.selected_objects]
        toc = stopwatch()

        # add lightsheets to the right scene collection
        coll = utils.get_collection_for_scene(context.scene, "lightsheets")
        for obj in lightsheets:
            coll.objects.link(obj)

        # report statistics
        c_stats = f"{len(lightsheets)} lightsheet(s)"
        t_stats = f"{toc - tic:.1f}s"
        self.report({'INFO'}, f"Created {c_stats} in {t_stats}")

        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Functions used by create lightsheet operator
# -----------------------------------------------------------------------------
def setup_lightsheet(light, resolution):
    """Setup lightsheet object, type of lightsheet depends on type of light."""
    light_type = light.data.type
    if light_type == 'SUN':
        # sun gets a square grid, because we don't know anything better
        sidelength = 2  # scale by hand if not the right size
        bm = create_bmesh_square(sidelength, resolution)
    elif light_type == 'SPOT':
        # intersect cone of spot with shifted plane => circle with
        # radius = tan(halfangle) (because size of circle = sin(...) and
        # shift = cos(...), but we want shift = 1, so divide by cos(...)
        # and radius becomes sin / cos = tan)
        angle = light.data.spot_size  # between 0° and 180°, but in radians
        radius = min(tan(angle / 2), 10)  # restrict for angles near 180°
        bm = create_bmesh_disk(radius, resolution)

        # shift circle to inside of cone
        for vert in bm.verts:
            vert.co.z = -1
    else:
        # assert light_type == 'POINT', (light, light_type)
        # lightsheet that surrounds the point light
        bm = create_bmesh_sphere(resolution)
    lightsheet = convert_bmesh_to_lightsheet(bm, light)

    return lightsheet


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


def create_bmesh_disk(radius, resolution):
    """Create bmesh for a circle filled with triangles."""
    # to create a circle we create a square and then cut out the circle
    bm = create_bmesh_square(2 * radius, resolution)

    # gather vertices that lie outside the circle and delete them, note that
    # this will also delete some faces on the edge of the circle which may
    # look weird for low resolutions
    outside_verts = [vert for vert in bm.verts
                     if vert.co[0] ** 2 + vert.co[1] ** 2 > radius ** 2]
    bmesh.ops.delete(bm, geom=outside_verts, context="VERTS")

    return bm


def create_bmesh_sphere(resolution, radius=1.0):
    """Create a spherical bmesh based on a subdivided icosahedron."""
    # use icosahedron as template
    bm_template = bmesh.new()
    if bpy.app.version < (3, 0, 0):
        bmesh.ops.create_icosphere(bm_template, subdivisions=0, diameter=1.0)
    else:
        # API change: https://developer.blender.org/rB9b2b32a3338d
        bmesh.ops.create_icosphere(bm_template, subdivisions=0, radius=1.0)

    # we will generate points with coordinates (i, j, 0), where i, j are
    # integers with i >= 0, j >= 0, i + j <= resolution - 1
    source_triangle = [
        Vector((0, 0, 0)),
        Vector((resolution - 1, 0, 0)),
        Vector((0, resolution - 1, 0))
    ]

    # replace every triangular face in the template icosahedron with a grid of
    # triangles with the given resolution
    bm = bmesh.new()
    shared_verts = []
    for face in bm_template.faces:
        # assert len(face.verts) == 3, len(face.verts)
        target_triangle = [vert.co for vert in face.verts]

        # place vertices
        strips = []  # each entry is a horizontal line of vertices
        for j in range(resolution):
            strip = []
            for i in range(resolution - j):  # less vertices as we go higher
                coords = barycentric_transform(Vector((i, j, 0)),
                                               *source_triangle,
                                               *target_triangle)
                coords *= radius / coords.length  # place on sphere
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

        # record which vertices are at the edge because these will have to be
        # merged with other edge vertices later
        shared_verts.extend(strips[0])  # bottom edge
        for j in range(1, resolution - 1):
            strip = strips[j]
            shared_verts.append(strip[0])  # left edge
            shared_verts.append(strip[-1])  # diagonal edge
        shared_verts.append(strips[resolution - 1][0])  # tip of triangle

    # merge the overlapping vertices at the edges and corners
    bmesh.ops.remove_doubles(bm, verts=shared_verts,
                             dist=0.1*radius/resolution)

    bm_template.free()
    return bm


def convert_bmesh_to_lightsheet(bm, light):
    """Convert a bmesh or object to a lightsheet for the given light."""
    # verify coordinates of vertices
    if light.data.type == 'SUN':
        # lighsheet should be in xy-plane
        for vert in bm.verts:
            vert.co.z = 0.0
    else:
        # assert light.data.type in {'SPOT', 'POINT'}, light.data.type
        # lightsheet should be spherical
        for vert in bm.verts:
            vert.co.normalize()
    utils.verify_lightsheet_layers(bm)

    # think of a good name
    name = f"{light.name} Lightsheet"

    # convert bmesh to mesh data block
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()

    # create new object
    lightsheet = bpy.data.objects.new(name, me)
    lightsheet.parent = light

    # adjust drawing and visibility
    lightsheet.display_type = 'WIRE'
    lightsheet.hide_render = True
    if bpy.app.version < (3, 0, 0):
        lightsheet.cycles_visibility.camera = False
        lightsheet.cycles_visibility.diffuse = False
        lightsheet.cycles_visibility.glossy = False
        lightsheet.cycles_visibility.transmission = False
        lightsheet.cycles_visibility.scatter = False
        lightsheet.cycles_visibility.shadow = False
    else:
        # API change: https://developer.blender.org/rBca64bd0aacdaa
        lightsheet.visible_camera = False
        lightsheet.visible_diffuse = False
        lightsheet.visible_glossy = False
        lightsheet.visible_transmission = False
        lightsheet.visible_volume_scatter = False
        lightsheet.visible_shadow = False

    return lightsheet
