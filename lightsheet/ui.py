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
LIGHTSHEET_PT_caustic: Summary of caustic information.
The tool panel shows some information about the active object.
"""

from bpy.types import Panel


class LIGHTSHEET_PT_tools(Panel):
    """Create a lightsheet tool panel in the 3d-view sidebar."""
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

        # info about the active object and selected objects
        box = layout.box()
        if context.object:
            ls_coll_name = f"Lightsheets in {context.scene.name}"
            ls_coll = context.scene.collection.children.get(ls_coll_name)
            display_obj_info(context.object, ls_coll, box)
        box.label(text=f"{len(context.selected_objects)} objects are selected")


def display_obj_info(obj, lightsheet_collection, layout):
    """Draw info and stats about lightsheet related objects."""
    def is_lighsheet(obj):
        if lightsheet_collection is None:
            return False

        if obj.name in lightsheet_collection.objects:
            if obj is lightsheet_collection.objects.get(obj.name):
                return True
        return False

    layout.label(text="Active object:")
    layout.label(text=f"{obj.name} ({obj.type})", icon='OBJECT_DATA')
    if obj.type == 'LIGHT':
        # object is light, does it have a lightsheet?
        has_lightsheet = any(is_lighsheet(child) for child in obj.children)
        if has_lightsheet:
            layout.label(text="Light has a lightsheet", icon='CHECKBOX_HLT')
        else:
            layout.label(text="Light has no lightsheet", icon='CHECKBOX_DEHLT')
    elif obj.type == 'MESH':
        # object is mesh, is it a lightsheet, caustic or parent of caustic?
        if obj.parent is not None and obj.parent.type == 'LIGHT':
            # object is or can be lightsheet
            verb = "is" if is_lighsheet(obj) else "can be"
            layout.label(text=f"Object {verb} lightsheet",
                         icon='LIGHTPROBE_PLANAR')
        elif obj.caustic_info.path:
            # object is caustic
            state = "(finalized)" if obj.caustic_info.finalized else ""
            layout.label(text=f"Object is caustic {state}", icon='SHADERFX')
        elif any(chd.caustic_info.path for chd in obj.children):
            # object is parent of caustics
            layout.label(text="Object is parent for caustics",
                         icon='CON_CHILDOF')
        elif obj.parent is not None and obj.parent.caustic_info.path:
            # object is probably a visualized raypath
            layout.label(text="Object is raypath visualization",
                         icon='CURVE_PATH')
        else:
            # object is unrelated
            layout.label(text="Object is unrelated", icon='X')


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

        # display mesh stats
        box = layout.box().column(align=True)
        display_mesh_info(caustic.data, box)


def display_caustic_raypath(caustic_info, layout):
    """Display info about caustic raypath."""
    layout.label(text=f"Raypath info: ({len(caustic_info.path)-1} bounces)")

    # lightsheet info
    lightsheet = caustic_info.lightsheet
    if lightsheet is not None:
        layout.label(text=f"Source: {lightsheet.name}")
    else:
        layout.label(text="Lightsheet not found", icon='ORPHAN_DATA')

    # list contents of caustic_info.path
    for link in caustic_info.path:
        interaction = link.kind.capitalize()
        if interaction == "Diffuse":
            interaction = "Target"

        if link.object:
            obj_name = link.object.name
        else:
            obj_name = "<object not found>"
        layout.label(text=f"{interaction}: {obj_name}")


def display_caustic_refinement(caustic_info, layout):
    """Display info about refinement steps."""
    layout.label(
        text=f"Refinement info: ({len(caustic_info.refinements)} steps)")

    for idx, step in enumerate(caustic_info.refinements):
        if step.adaptive_subdivision:
            adaptive = f"{step.error_threshold:.2f}"
        else:
            adaptive = "none"
        span = "span" if step.span_faces else "    "
        grow = "grow" if step.grow_boundary else "    "
        txt = f"Refinement {idx+1}: {adaptive}, {span}, {grow}"
        layout.label(text=txt)


def display_caustic_finalization(caustic_info, layout):
    """Display finalization settings."""
    layout.label(text=f"Finalized: {caustic_info.finalized}")

    if caustic_info.finalized:
        layout.label(text=f"Fade out boundary: {caustic_info.fade_boundary}")
        layout.label(text=f"Remove dim faces: {caustic_info.remove_dim_faces}")
        if caustic_info.remove_dim_faces:
            cutoff = caustic_info.emission_cutoff
            layout.label(text=f"    Cutoff: {cutoff:.5f}")
        layout.label(text=f"Fix overlap: {caustic_info.fix_overlap}")


def display_mesh_info(data, layout):
    """Display number of vertices, edges and faces of given mesh datablock."""
    layout.label(text="Mesh info:")
    layout.label(text=f"Verts: {len(data.vertices):,}", icon='VERTEXSEL')
    layout.label(text=f"Edges: {len(data.edges):,}", icon='EDGESEL')
    layout.label(text=f"Faces: {len(data.polygons):,}", icon='FACESEL')


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
