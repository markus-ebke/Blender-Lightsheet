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

# <pep8 compliant>


"""
# support reloading scripts and addons
if "bpy" in locals():
    import importlib

    importlib.reload(create_lightsheet)
    importlib.reload(trace_lightsheet)
    importlib.reload(refine_caustic)
    importlib.reload(finalize_caustic)
    importlib.reload(visualize_raypath)
    importlib.reload(animated_trace)
    importlib.reload(properties)
    importlib.reload(ui)

    # print("Lighsheet: Addon reloaded")
"""

import bpy
from bpy.utils import register_class, unregister_class

from . import (create_lightsheet,
               trace_lightsheet,
               refine_caustic,
               finalize_caustic,
               visualize_raypath,
               animated_trace,
               properties,
               ui)
# print("Lightsheet: Addon loaded")


# registration
classes = (
    create_lightsheet.LIGHTSHEET_OT_create_lightsheets,
    trace_lightsheet.LIGHTSHEET_OT_trace_lightsheets,
    refine_caustic.LIGHTSHEET_OT_refine_caustics,
    finalize_caustic.LIGHTSHEET_OT_finalize_caustics,
    visualize_raypath.LIGHTSHEET_OT_visualize_raypath,
    animated_trace.LIGHTSHEET_OT_animated_trace,
    properties.CausticPathLink,
    properties.CausticRefinementSetting,
    properties.CausticInfo,
    ui.LIGHTSHEET_PT_tools,
    ui.LIGHTSHEET_PT_object,
    ui.LIGHTSHEET_PT_caustic,
    ui.LIGHTSHEET_PT_raypath,
)


def register():
    for cls in classes:
        register_class(cls)

    bpy.types.Object.caustic_info = bpy.props.PointerProperty(
        type=properties.CausticInfo)

    # print("Lightsheet: Addon registered")


def unregister():
    del bpy.types.Object.caustic_info

    for cls in reversed(classes):
        unregister_class(cls)

    # print("Lightsheet: Addon unregistered")


if __name__ == "__main__":
    register()
