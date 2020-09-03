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
"""Functions for handling materials (for tracing and caustic node tree)."""

from collections import namedtuple

from mathutils import Color

print("lightsheet material.py")

Interaction = namedtuple("Interaction", ["kind", "outgoing", "tint"])
Interaction.__doc__ = """Type of ray after surface interaction.

    kind (str): one of "diffuse", "reflect", "refract" or "transparent"
    outgoing (mathutils.Vector or None): new ray direction
    tint (mathutils.Color or None): tint from surface in this direction

    If kind == "diffuse" then outgoing and tint will not be used and are None.
    """

# -----------------------------------------------------------------------------
# Material handling for tracing
# -----------------------------------------------------------------------------
# cache material to shader mapping for faster access, materials_cache is a dict
# of form {material: (surface shader function}
materials_cache = dict()


def get_material_shader(mat):
    """Returns a function that models the material interaction.

    The function models the surface interaction like this:
    func(ray_direction, normal) = list of interactions
    """
    # check cache
    shader = materials_cache.get(mat)
    if shader is not None:
        return shader

    # dummy surface shader that tries out all interactions
    white = Color((1.0, 1.0, 1.0))

    def surface_shader(ray_direction, normal):
        interactions = []

        interactions.append(("diffuse", None, None))

        refle = ray_direction.reflect(normal)
        interactions.append(("reflect", refle, 0.25*white))

        refra = ray_direction
        interactions.append(("refract", refra, 0.25*white))

        interactions.append(("transparent", ray_direction, 0.25*white))

        return interactions

    materials_cache[mat] = surface_shader
    return surface_shader


# -----------------------------------------------------------------------------
# Setup caustic material node tree
# -----------------------------------------------------------------------------
def setup_caustic_material(mat):
    """Setup node tree for caustic material."""
    # see https://blender.stackexchange.com/a/23446
    pass
