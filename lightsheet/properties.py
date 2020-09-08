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
"""Custom Properties to keep track of certain information."""

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty,
                       PointerProperty)


class CausticPathLink(bpy.types.PropertyGroup):
    """Link in the raypath of a caustic."""
    object: PointerProperty(
        type=bpy.types.Object,
        name="Object",
        description="The object that was hit by a lightsheet ray")
    kind: EnumProperty(
        name="Kind",
        description="The kind of interaction with the object",
        items=[
            ('DIFFUSE', "Diffuse", ""),
            ('REFLECT', "Reflect", ""),
            ('REFRACT', "Refract", ""),
            ('TRANSPARENT', "Transparent", ""),
        ])


class CausticInfo(bpy.types.PropertyGroup):
    """Information about the raypath to a caustic."""
    lightsheet: PointerProperty(
        type=bpy.types.Object,
        name="Lightsheet",
        description="Lightsheet object that send out this caustic")
    path: CollectionProperty(
        type=CausticPathLink,
        name="Lightsheet path",
        description="Path of the lightrays to this caustic")
    finalized: BoolProperty(
        name="Finalized", description="If caustic has been finalized",
        default=False)
