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
"""Custom properties to keep track of caustic information.

CausticInfo is a property for objects, if the path variable not empty then the
object is a caustic. The entries of path are of type CausticPathLink and
specify the object that was hit and the kind of interaction. The target object
is the last one in this path and the only entry with a diffuse interaction.
Caustics that are finalized should not be refined anymore.
"""

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty,
                       PointerProperty)


class CausticPathLink(bpy.types.PropertyGroup):
    """Link in the raypath of a caustic."""
    object: PointerProperty(
        name="Object",
        description="The object that was hit by a lightsheet ray",
        type=bpy.types.Object
    )
    kind: EnumProperty(
        name="Kind",
        description="The kind of interaction with the object",
        items=[
            ('DIFFUSE', "Diffuse", ""),
            ('REFLECT', "Reflect", ""),
            ('REFRACT', "Refract", ""),
            ('TRANSPARENT', "Transparent", ""),
        ])


class CausticRefinementSetting(bpy.types.PropertyGroup):
    """Settings for one refinement via the lightsheet.refine operator."""
    adaptive_subdivision: bpy.props.BoolProperty(name="Adaptive Subdivision")
    error_threshold: bpy.props.FloatProperty(name="Error Threshold")
    grow_boundary: bpy.props.BoolProperty(name="Grow Boundary")


class CausticInfo(bpy.types.PropertyGroup):
    """Information about raypath, refinement and finalization of a caustic."""
    # raypath, set by lightsheet.create operator
    lightsheet: PointerProperty(
        name="Lightsheet",
        description="Lightsheet object that send out this caustic",
        type=bpy.types.Object
    )
    path: CollectionProperty(
        name="Lightsheet Path",
        description="Path of the lightrays to this caustic",
        type=CausticPathLink
    )
    # refinement settings, extended by lightsheet.refine operator
    refinements: CollectionProperty(
        type=CausticRefinementSetting,
        name="Refinements",
        description="Settings used for refinements of this caustic"
    )
    # finalization, set by lightsheet.finalize operator
    finalized: BoolProperty(
        name="Finalized",
        description="If caustic has been finalized",
        default=False
    )
    fade_boundary: bpy.props.BoolProperty(name="Fade Out Boundary")
    remove_dim_faces: bpy.props.BoolProperty(name="Remove Dim Faces")
    emission_cutoff: bpy.props.FloatProperty(name="Emission Cutoff")
    fix_overlap: bpy.props.BoolProperty(name="Merge Overlapping Faces")
