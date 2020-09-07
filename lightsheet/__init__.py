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

# <pep8 compliant>

bl_info = {
    "name": "Lightsheet",
    "author": "Markus Ebke",
    "version": (0, 1),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > Lightsheet Tab",
    "description": "Create fake caustics renderable in Cycles and EEVEE",
    "warning": "",
    "doc_url": "",
    "tracker_url": "https://github.com/markus-ebke/Blender-Lightsheet/issues",
    "category": "Lighting",
}

print("lightsheet __init__.py")

# support reloading scripts and addons
if "bpy" in locals():
    import importlib

    importlib.reload(operators)
    importlib.reload(properties)
    importlib.reload(ui)

    print("lighsheet reloaded")
else:
    from lightsheet import operators, properties, ui

    print("lightsheet loaded")

import bpy
from bpy.utils import register_class, unregister_class


# registration
classes = (
    ui.LIGHTSHEET_PT_tools,
    ui.LIGHTSHEET_PT_caustic,
    operators.LIGHTSHEET_OT_create_lightsheet,
    operators.LIGHTSHEET_OT_trace_lightsheet,
    properties.CausticPathLink,
    properties.CausticInfo,
)


def register():
    print("register lightsheet")
    for cls in classes:
        print("register", cls)
        register_class(cls)
    bpy.types.Object.caustic_info = bpy.props.PointerProperty(type=properties.CausticInfo)


def unregister():
    print("unregister lightsheet")
    del bpy.types.Object.caustic_info
    for cls in reversed(classes):
        print("unregister", cls)
        unregister_class(cls)


if __name__ == "__main__":
    register()
