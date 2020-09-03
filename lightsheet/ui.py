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
    """Create a tool panel in the 3d-view sidebar."""
    bl_label = "Lightsheet Tools"
    bl_context = 'objectmode'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Lightsheet"

    def draw(self, context):
        layout = self.layout

        # here come the important operators
        col = layout.column(align=True)
        col.scale_y = 1.5
        col.operator("lightsheet.create", icon='LIGHTPROBE_PLANAR')
        col.operator("lightsheet.trace", icon='HIDE_OFF')

        # debug info
        layout.separator()
        col = layout.column()
        col.label(text="Debug info:", icon='INFO')
        obj = context.object
        if obj is not None:
            col.prop(obj, "name")
            col.prop(obj, "type")

            if obj.type == 'LIGHT':
                col.prop(obj.data, "type")
