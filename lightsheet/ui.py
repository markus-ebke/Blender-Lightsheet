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
"""

from bpy.types import Panel

print("lightsheet ui.py")


class LIGHTSHEET_PT_tools(Panel):
    """Create a lightsheet tool panel in the 3d-view sidebar."""
    bl_label = "Lightsheet Tools"
    bl_context = 'objectmode'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Lightsheet"

    def draw(self, context):
        layout = self.layout

        # info about active object
        obj = context.object
        if obj:
            col = layout.column()
            col.label(text=f"Active object: '{obj.name}'", icon='OBJECT_DATA')
            if obj.type == 'LIGHT':
                col.label(text=f"Active object is light", icon='LIGHT')
            elif obj.type == 'MESH':
                if obj.parent is not None and obj.parent.type == 'LIGHT':
                    col.label(text=f"Active object is lightsheet",
                              icon='LIGHTPROBE_PLANAR')
                elif obj.caustic_info.path:
                    col.label(text="Active object is caustic", icon='SHADERFX')

        # prepare layout for important operators
        col = layout.column(align=True)
        col.scale_y = 1.5

        # here come the important operators
        col.operator("lightsheet.create", icon='LIGHTPROBE_PLANAR')
        col.operator("lightsheet.trace", icon='HIDE_OFF')


class LIGHTSHEET_PT_caustic(Panel):
    """Create a caustic info panel in the 3d-view sidebar."""
    bl_label = "Caustic Raypath"
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
        col = layout.column()

        # get caustic info from active object
        assert context.object is not None and context.object.caustic_info.path
        caustic_info = context.object.caustic_info

        # print lightsheet
        lightsheet = caustic_info.lightsheet
        if lightsheet is not None:
            col.label(text=f"Source: {lightsheet.name}")
        else:
            col.label(text="Lightsheet not found", icon='ORPHAN_DATA')

        # list contents of lightsheet_path
        for link in caustic_info.path:
            interaction = link.kind.capitalize()
            if link.object:
                obj_name = link.object.name
            else:
                obj_name = "<object not found>"
            col.label(text=f"{interaction}: {obj_name}")
