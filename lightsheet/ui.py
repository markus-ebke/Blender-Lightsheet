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
"""User interface for lightsheet addon.

LIGHTSHEET_PT_tools: Panel in sidebar to access the operators.
LIGHTSHEET_PT_object: Information about the active object.
LIGHTSHEET_PT_caustic: Summary of caustic raypath and settings.
LIGHTSHEET_PT_raypath: Summary of caustic raypath for raypath visualization.

Helper functions:
- display_object_info
- display_mesh_info
- display_materials_info
- display_caustic_raypath
- display_caustic_refinement
- display_caustic_finalization
"""

import textwrap

from bpy.types import Panel
from mathutils import Vector

from lightsheet import material


class LIGHTSHEET_PT_tools(Panel):
    """Lightsheet tool panel in the 3d-view sidebar."""
    bl_label = "Lightsheet Tools"
    bl_context = 'objectmode'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Lightsheet"

    def draw(self, context):
        layout = self.layout

        # prepare layout for important operators
        col = layout.column(align=True)
        col.scale_y = 1.5

        # here come the important operators
        col.operator("lightsheet.create", icon='LIGHTPROBE_PLANAR')
        col.operator("lightsheet.trace", icon='HIDE_OFF')
        col.operator("lightsheet.refine", icon='MOD_MULTIRES')
        col.operator("lightsheet.finalize", icon='OUTPUT')

        # place animated trace operator visually separate from the others
        col = layout.column()
        col.scale_y = 1.5
        col.operator("lightsheet.animate", icon='RENDER_ANIMATION')


class LIGHTSHEET_PT_object(Panel):
    """Active object info panel in the 3d-view sidebar."""
    bl_label = "Active Object Info"
    bl_context = 'objectmode'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Lightsheet"

    def draw(self, context):
        layout = self.layout

        # info about the active object and selected objects
        obj = context.object
        if obj is not None:
            layout.label(text=f"Name: {obj.name}")

            # type of object and relation to lightsheet add-on
            ls_coll_name = f"Lightsheets in {context.scene.name}"
            ls_coll = context.scene.collection.children.get(ls_coll_name)
            box = layout.box().column(align=True)
            display_object_info(obj, ls_coll, box)

            if obj.type == 'MESH':
                # display mesh stats
                box = layout.box().column(align=True)
                display_mesh_info(obj.data, box)

                # display material info (except for caustics)
                if not obj.caustic_info.path:
                    materials = [slot.material for slot in obj.material_slots]
                    box = layout.box().column(align=True)
                    display_materials_info(materials, box)


class LIGHTSHEET_PT_caustic(Panel):
    """Create a caustic info panel in the 3d-view sidebar."""
    bl_label = "Caustic Info"
    bl_context = 'objectmode'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Lightsheet"

    @classmethod
    def poll(cls, context):
        # show panel only for caustics
        return context.object is not None and context.object.caustic_info.path

    def draw(self, context):
        layout = self.layout
        caustic = context.object
        caustic_info = caustic.caustic_info

        # display caustic info and raypath operator
        box = layout.box().column(align=True)
        display_caustic_raypath(caustic_info, box)
        box.operator("lightsheet.visualize", icon='CURVE_PATH')

        # display refinement settings
        box = layout.box().column(align=True)
        display_caustic_refinement(caustic_info, box)

        # display finalization
        box = layout.box().column(align=True)
        display_caustic_finalization(caustic_info, box)


class LIGHTSHEET_PT_raypath(Panel):
    """Create a raypath info panel in the 3d-view sidebar."""
    bl_label = "Raypath Visualization"
    bl_context = 'objectmode'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Lightsheet"

    @classmethod
    def poll(cls, context):
        # show panel only for caustics
        obj = context.object
        if obj is not None and obj.parent is not None:
            return bool(obj.parent.caustic_info.path)
        return False

    def draw(self, context):
        layout = self.layout
        obj = context.object
        caustic = obj.parent
        assert caustic is not None

        # display caustic info
        display_caustic_raypath(caustic.caustic_info, layout.column())

        box = layout.box()

        # display mesh stats
        verts_per_ray = len(caustic.caustic_info.path) + 1
        num_rays = len(obj.data.vertices) // verts_per_ray
        box.label(text="Info about mesh:")
        box.label(text=f"Rays: {num_rays:,}", icon='CURVE_PATH')
        box.label(text=f"Verts per ray: {verts_per_ray}", icon='VERTEXSEL')


# -----------------------------------------------------------------------------
# Functions used by panels
# -----------------------------------------------------------------------------
def display_object_info(obj, lightsheet_collection, layout):
    """Draw info and stats about lightsheet related objects."""
    def is_caustic(obj):
        return bool(obj.caustic_info.path)

    def is_lighsheet(obj):
        if lightsheet_collection is None:
            return False

        if obj.name in lightsheet_collection.objects:
            return obj is lightsheet_collection.objects.get(obj.name)
        return False

    layout.label(text="Object Info:")
    layout.label(text=f"Object is {obj.type.lower()}", icon='OBJECT_DATA')
    if obj.type == 'LIGHT':
        # object is light, does it have a lightsheet?
        lightsheets = [child for child in obj.children if is_lighsheet(child)]
        layout.label(text=f"Light has {len(lightsheets)} lightsheets:",
                     icon='LIGHTPROBE_PLANAR')
        for ls in lightsheets:
            layout.label(text=f"    {ls.name}")
    elif obj.type == 'MESH':
        # object is mesh, is it related to lightsheets or caustics?
        if is_caustic(obj):
            # object is caustic
            state = "(finalized)" if obj.caustic_info.finalized else ""
            layout.label(text=f"Object is caustic {state}", icon='SHADERFX')
        elif obj.parent is not None and is_caustic(obj.parent):
            # object is probably a visualized raypath
            layout.label(text="Object is raypath visualization",
                         icon='CURVE_PATH')
        elif obj.parent is not None and obj.parent.type == 'LIGHT':
            # object is or can be lightsheet
            verb = "is" if is_lighsheet(obj) else "can be"
            layout.label(text=f"Object {verb} lightsheet of {obj.parent.name}",
                         icon='LIGHTPROBE_PLANAR')
        else:
            caustics = [child for child in obj.children if is_caustic(child)]
            if caustics:
                # object is parent of caustics
                msg = f"Object is parent for {len(caustics)} caustics"
                layout.label(text=msg, icon='CON_CHILDOF')
            else:
                # object is unrelated
                layout.label(text="Object is unrelated", icon='X')


def display_mesh_info(data, layout):
    """Display info about given mesh datablock."""
    layout.label(text="Mesh Info:")
    layout.label(text=f"Verts: {len(data.vertices):,}", icon='VERTEXSEL')
    layout.label(text=f"Edges: {len(data.edges):,}", icon='EDGESEL')
    layout.label(text=f"Faces: {len(data.polygons):,}", icon='FACESEL')
    layout.label(text=f"UV-Layers: {len(data.uv_layers)}", icon='GROUP_UVS')


def display_materials_info(materials, layout):
    """Display lightsheet interactions with given materials."""
    layout.label(text=f"Materials Info: ({len(materials)} materials)")
    for mat in materials:
        mat_name = "<None>" if mat is None else mat.name
        layout.label(text=f"Material: {mat_name}", icon='MATERIAL')

        # get shader and record interaction.kind
        try:
            surface_shader, volume_params = material.get_material_shader(mat)
        except ValueError as err:
            # display error message, split into in several lines if necessary
            lines = textwrap.wrap(str(err), width=30)
            layout.label(text=lines[0], icon='ERROR')
            for line in lines[1:]:
                layout.label(text=f"        {line}")
        else:
            incoming, normal = Vector((0, 0, -1)), Vector((0, 0, 1))
            interactions = surface_shader(incoming, normal)
            inter_kinds = [intr.kind.capitalize() for intr in interactions]
            layout.label(text=f"    Interactions: {inter_kinds}")

        use_volume = volume_params is not None
        layout.label(text=f"    Volume Absorption: {use_volume}")

    # clear material cache
    material.cache_clear()


def display_caustic_raypath(caustic_info, layout):
    """Display info about caustic raypath."""
    layout.label(text=f"Raypath ({len(caustic_info.path)-1} bounces):")

    # lightsheet info
    lightsheet = caustic_info.lightsheet
    if lightsheet is not None:
        layout.label(text=f"Source: {lightsheet.name}")
    else:
        layout.label(text="Source: <lightsheet not found>", icon='ORPHAN_DATA')

    # list contents of caustic_info.path
    for link in caustic_info.path:
        interaction = link.kind.capitalize()

        if link.object:
            layout.label(text=f"{interaction}: {link.object.name}")
        else:
            layout.label(text=f"{interaction}: <object not found>",
                         icon='ORPHAN_DATA')


def display_caustic_refinement(caustic_info, layout):
    """Display info about refinement steps."""
    layout.label(
        text=f"Refinements ({len(caustic_info.refinements)} steps):")

    for idx, step in enumerate(caustic_info.refinements):
        if step.adaptive_subdivision:
            adaptive = f"{step.error_threshold:.2f}"
        else:
            adaptive = "none"
        txt = f"Step {idx+1}: error={adaptive}, grow={step.grow_boundary}"
        layout.label(text=txt)


def display_caustic_finalization(caustic_info, layout):
    """Display finalization settings."""
    layout.label(text=f"Finalized: {caustic_info.finalized}")

    if caustic_info.finalized:
        layout.label(text=f"Fade out boundary: {caustic_info.fade_boundary}")
        layout.label(text=f"Remove dim faces: {caustic_info.remove_dim_faces}")
        if caustic_info.remove_dim_faces:
            cutoff = caustic_info.emission_cutoff
            layout.label(text=f"    Cutoff: {cutoff:.4f} W/m^2")

        layout.label(text=f"Fix overlap: {caustic_info.fix_overlap}")
        if caustic_info.fix_overlap > 0:
            offset = caustic_info.shrinkwrap_offset
            layout.label(text=f"    Shrinkwrap: {offset:.4f}m")
