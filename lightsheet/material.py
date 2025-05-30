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
"""Handle materials for tracing lightsheets and setup of caustic node trees.

The main function that handles materials for tracing is get_material_shader,
it uses the setup_<node>-functions to process the active surface node.

A caustic material is created via get_caustic_material, it will add nodes for
Cycles, EEVEE and add drivers for light strength and color.

Note that after tracing you should cleanup the cached material shaders via
material.cache_clear()
"""

from collections import namedtuple
from functools import lru_cache
from math import sqrt

import bpy
from mathutils import Color

from . import utils

# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
# organize outcome of interactions
Interaction = namedtuple("Interaction", ["kind", "outgoing", "tint"])
Interaction.__doc__ = """Type of ray after surface interaction.

    kind (str): one of 'DIFFUSE', 'REFLECT', 'REFRACT' or 'TRANSPARENT'
    outgoing (mathutils.Vector or None): new ray direction
    tint (mathutils.Color or None): tint from surface in this direction

    If kind == 'DIFFUSE' then outgoing and tint will not be used and are None.
    """


# -----------------------------------------------------------------------------
# Material handling for tracing
# -----------------------------------------------------------------------------
# helper functions ------------------------------------------------------------
def fresnel(ray_direction, normal, ior):
    """Fresnel mix factor."""
    # cosine of angle between incident and normal directions
    cos_in = -ray_direction.dot(normal)

    # if ray comes from inside, swap normal and ior
    if cos_in < 0:
        cos_in *= -1  # swapped normal
        ior_inside = 1.0
        ior_outside = ior
    else:
        ior_inside = ior
        ior_outside = 1.0

    # square of sine of angle between transmitted direction and normal
    gamma = ior_outside / ior_inside
    sin_out2 = gamma**2 * (1 - cos_in**2)
    if sin_out2 >= 1.0:
        # total internal reflection here
        return 1.0

    cos_out = sqrt(1 - sin_out2)

    # compute reflectivity from Fresnel's equations
    ii = ior_inside * cos_in
    oo = ior_outside * cos_out
    r_s = (ii - oo) / (ii + oo)

    io = ior_inside * cos_out
    oi = ior_outside * cos_in
    r_p = (io - oi) / (oi + io)

    return (r_s**2 + r_p**2) / 2


def refract(ray_direction, normal, ior):
    """Return direction of refracted ray where normal points outside."""
    # cosine of angle between incident and normal directions
    cos_in = -ray_direction.dot(normal)

    # if ray comes from inside, swap normal and ior
    if cos_in < 0:
        normal = -normal
        cos_in *= -1
        gamma = ior  # gamma = ior_outside / ior_inside = ior / 1
    else:
        gamma = 1 / ior  # gamma = ior_outside / ior_inside = 1 / ior

    # square of sine of angle between transmitted direction and normal
    sin_out2 = gamma**2 * (1 - cos_in**2)
    if sin_out2 >= 1:
        # total internal reflection => no refraction
        return None

    # compute refraction direction
    cos_out = sqrt(1 - sin_out2)
    return gamma * ray_direction + (gamma * cos_in - cos_out) * normal


def diffuse_surface_shader(ray_direction, normal):
    # caustic on object, but no tracing of further rays
    return [Interaction('DIFFUSE', None, None)]


# setup surface interaction from BSDF node ------------------------------------
def setup_diffuse(node):
    # assert node.type == 'BSDF_DIFFUSE'
    return diffuse_surface_shader


def setup_glossy(node):
    # assert node.type == 'BSDF_GLOSSY', node.type

    # don't handle reflection if roughness is nonzero
    if (node.inputs['Roughness'].default_value > 0.0 and
            node.distribution != 'SHARP'):
        return None

    # node settings
    color_srgb = node.inputs['Color'].default_value[:3]
    color = Color(utils.srgb_to_linear(color_srgb))

    def surface_shader(ray_direction, normal):
        refle = ray_direction.reflect(normal)
        return [Interaction('REFLECT', refle, color)]

    return surface_shader


def setup_transparent(node):
    # assert node.type == 'BSDF_TRANSPARENT', node.type

    # node settings
    color_srgb = node.inputs['Color'].default_value[:3]
    color = Color(utils.srgb_to_linear(color_srgb))

    def surface_shader(ray_direction, normal):
        return [Interaction('TRANSPARENT', ray_direction, color)]

    return surface_shader


def setup_refraction(node):
    # assert node.type == 'BSDF_REFRACTION', node.type

    # don't handle refraction if roughness is nonzero
    if (node.inputs['Roughness'].default_value > 0.0 and
            node.distribution != 'SHARP'):
        return None

    # node settings
    color_srgb = node.inputs['Color'].default_value[:3]
    color = Color(utils.srgb_to_linear(color_srgb))
    ior = node.inputs['IOR'].default_value

    def surface_shader(ray_direction, normal):
        refra = refract(ray_direction, normal, ior)
        if refra is not None:
            return [Interaction('REFRACT', refra, color)]
        return []

    return surface_shader


def setup_glass(node):
    # assert node.type == 'BSDF_GLASS', node.type

    # don't handle reflection or refraction for rough glass
    if (node.inputs['Roughness'].default_value > 0.0 and
            node.distribution != 'SHARP'):
        return None

    # node settings
    color_srgb = node.inputs['Color'].default_value[:3]
    color = Color(utils.srgb_to_linear(color_srgb))
    ior = node.inputs['IOR'].default_value

    def surface_shader(ray_direction, normal):
        # outgoing vectors
        refle = ray_direction.reflect(normal)
        refra = refract(ray_direction, normal, ior)

        if refra is None:
            # total internal reflection
            return [Interaction('REFLECT', refle, color)]

        # reflection and refraction
        reflectivity = fresnel(ray_direction, normal, ior)
        interactions = [
            Interaction('REFLECT', refle, reflectivity * color),
            Interaction('REFRACT', refra, (1 - reflectivity) * color)
        ]
        return interactions

    return surface_shader


def setup_principled(node):
    # assert node.type == 'BSDF_PRINCIPLED', node.type

    # node settings
    color_srgb = node.inputs['Base Color'].default_value[:3]
    color = Color(utils.srgb_to_linear(color_srgb))
    metallic = node.inputs['Metallic'].default_value
    if bpy.app.version < (4, 0, 0):
        specular = node.inputs['Specular'].default_value
        transmission = node.inputs['Transmission'].default_value
    else:
        specular = node.inputs['Specular IOR Level'].default_value
        transmission = node.inputs['Transmission Weight'].default_value
    ior = node.inputs['IOR'].default_value

    # convert specular to ior for fresnel i.e. invert
    # specular = ((ior - 1) / (ior + 1))**2 / 0.08
    x = sqrt(specular * 0.08)
    specular_ior = (1 + x) / (1 - x)

    # have diffuse caustic if material is not a perfect mirror or glass
    handle_diffuse = metallic < 1 and transmission < 1

    # don't handle reflection and refraction if they have roughness
    roughness = node.inputs['Roughness'].default_value
    if not (roughness == 0 or node.distribution == 'SHARP'):
        # no reflection and refraction tracing, but maybe diffuse
        if handle_diffuse:
            return diffuse_surface_shader

        return None  # no diffuse => no caustic or caustic tracing at all

    # don't handle refraction if it has roughness
    if bpy.app.version < (4, 0, 0):
        extra_roughness = node.inputs['Transmission Roughness'].default_value
    else:
        extra_roughness = roughness
    handle_refraction = extra_roughness == 0 or node.distribution == 'SHARP'

    white = Color((1.0, 1.0, 1.0))

    # diffuse, reflection and possibly refraction
    def surface_shader(ray_direction, normal):
        # mix_1: diffuse <-> tinted refraction via transmission
        refra_tint = color * transmission

        # mix_2: mix_1 <-> tinted reflection via metallic
        refle_tint = color * metallic
        refra_tint *= 1 - metallic

        # mix_3: mix_2 <-> facing*metallic tinted reflection via fresnel
        reflectivity = fresnel(ray_direction, normal, specular_ior)
        refle_tint = (1 - reflectivity) * refle_tint
        refra_tint *= 1 - reflectivity

        # TODO is tint via facing neccessary or is white (glossy = 1) enough?
        incoming = -ray_direction
        facing = 1 - abs(normal.dot(incoming))  # layer weight, blend = 0.5
        glossy = (1 - metallic) * 1 + metallic * facing  # mix white <-> facing
        # glossy = 1 + metallic * abs(normal.dot(ray_direction))
        refle_tint += white * reflectivity * glossy

        interactions = []
        if handle_diffuse:
            interactions.append(Interaction('DIFFUSE', None, None))

        if refle_tint.v > 0:
            # some light is reflected
            refle = ray_direction.reflect(normal)
            interactions.append(Interaction('REFLECT', refle, refle_tint))

        if refra_tint.v > 0 and handle_refraction:
            # some light is transmitted
            refra = refract(ray_direction, normal, ior)
            interactions.append(Interaction('REFRACT', refra, refra_tint))

        return interactions

    return surface_shader


def setup_metallic(node):
    # assert node.type == 'BSDF_METALLIC', node.type

    # node settings
    color_srgb = node.inputs['Base Color'].default_value[:3]
    color = Color(utils.srgb_to_linear(color_srgb))

    # TODO understand how the Metallic BSDF works and properly model
    # node.fresnel_type, node.inputs['IOR'], node.inputs['Extinction'] and
    # node.inputs['Edge Tint']. Here I just guess based on the Principled BSDF.
    specular = 0.5

    # convert specular to ior for fresnel i.e. invert
    # specular = ((ior - 1) / (ior + 1))**2 / 0.08
    x = sqrt(specular * 0.08)
    specular_ior = (1 + x) / (1 - x)

    # don't handle reflection and refraction if they have roughness
    roughness = node.inputs['Roughness'].default_value
    if roughness != 0:
        return None  # no diffuse => no caustic or caustic tracing at all

    white = Color((1.0, 1.0, 1.0))

    # diffuse, reflection and possibly refraction
    def surface_shader(ray_direction, normal):
        # facing * metallic tinted reflection via fresnel
        reflectivity = fresnel(ray_direction, normal, specular_ior)
        refle_tint = (1 - reflectivity) * color

        # TODO is tint via facing neccessary or is white (facing = 1) enough?
        incoming = -ray_direction
        facing = 1 - abs(normal.dot(incoming))  # layer weight, blend = 0.5
        refle_tint += white * reflectivity * facing

        interactions = []

        if refle_tint.v > 0:
            # some light is reflected
            refle = ray_direction.reflect(normal)
            interactions.append(Interaction('REFLECT', refle, refle_tint))

        return interactions

    return surface_shader


def setup_scalar_node(node, from_socket_identifier=None):
    # assert from_socket_identifier in {'Fresnel', 'Facing'}

    if node.type == 'FRESNEL':  # fresnel node
        ior = node.inputs['IOR'].default_value

        def fac(incoming, normal):
            return fresnel(incoming, normal, ior)
    elif node.type == 'LAYER_WEIGHT':  # layer weight node
        blend = node.inputs['Blend'].default_value

        # formulas from cycles/kernel/svm/svm_fresnel.h
        if from_socket_identifier == 'Fresnel':
            ior = 1 / max(1 - blend, 1e-5)

            def fac(incoming, normal):
                return fresnel(incoming, normal, ior)
        else:
            # from_socket_identifier == 'Facing'
            def fac(incoming, normal):
                dot = abs(incoming.dot(normal))
                if blend <= 0.5:
                    return 1 - pow(dot, 2 * blend)
                blend_clamped = max(blend, 1 - 1e-5)  # blend < 1
                return 1 - pow(dot, 0.5 / (1 - blend_clamped))
    else:
        # none of the above, will later raise ValueError
        return None

    return fac


def setup_mix(node):
    # assert node.type == 'MIX_SHADER', node.type

    # mix factor
    if node.inputs[0].links:
        link = node.inputs[0].links[0]
        fac = setup_scalar_node(link.from_node, link.from_socket.identifier)
        if fac is None:
            supported = "only 'FRESNEL' and 'LAYER_WEIGHT'"
            msg = f"Factor input of mix node is not supported, {supported}"
            raise ValueError(msg)
    else:
        fac_value = node.inputs[0].default_value

        def fac(incoming, normal):
            return fac_value

    # get the two connected shader nodes and setup shader functions
    shader1 = None
    if node.inputs[1].links:
        node1 = node.inputs[1].links[0].from_node
        setup = setup_shader_from_node.get(node1.type)
        if setup is not None:
            shader1 = setup(node1)
    if shader1 is None:
        raise ValueError("First shader input of Mix Shader node is invalid")

    shader2 = None
    if node.inputs[2].links:
        node2 = node.inputs[2].links[0].from_node
        setup = setup_shader_from_node.get(node2.type)
        if setup is not None:
            shader2 = setup(node2)
    if shader2 is None:
        raise ValueError("Second shader input of Mix Shader node is invalid")

    def surface_shader(ray_direction, normal):
        # evaluate shader1
        interactions_map1 = dict()
        for kind, vec, col in shader1(ray_direction, normal):
            interactions_map1[kind] = (vec, col)

        # evaluate shader1
        interactions_map2 = dict()
        for kind, vec, col in shader2(ray_direction, normal):
            interactions_map2[kind] = (vec, col)

        # build up interactions for mixed shader
        weight2 = fac(ray_direction, normal)
        weight1 = 1 - weight2
        interactions = []

        # diffuse
        if 'DIFFUSE' in interactions_map1 or 'DIFFUSE' in interactions_map2:
            interactions.append(Interaction('DIFFUSE', None, None))

        # all other interactions
        for kind in ['REFLECT', 'REFRACT', 'TRANSPARENT']:
            vec1, col1 = interactions_map1.get(kind, (None, None))
            vec2, col2 = interactions_map2.get(kind, (None, None))
            if vec1 is not None:
                col = weight1 * col1
                if vec2 is not None:
                    col += weight2 * col2
                interactions.append(Interaction(kind, vec1, col))
                # note: with two refraction shaders we might have the problem,
                # that vec1 and vec2 point in different directions.
                # In such a case we still follow vec1 and do not add another
                # REFRACT interaction, because otherwise the caustics can't be
                # uniquely identified by their raypath
            elif vec2 is not None:
                interactions.append(Interaction(kind, vec2, weight2*col2))

        return interactions

    return surface_shader


# the add shader is similar to the mix shader, except that both weights are 1
def setup_add(node):
    # assert node.type == 'ADD_SHADER', node.type

    # get the two connected shader nodes and setup shader functions
    shader1 = None
    if node.inputs[0].links:
        node1 = node.inputs[0].links[0].from_node
        setup = setup_shader_from_node.get(node1.type)
        if setup is not None:
            shader1 = setup(node1)

        # default: diffuse
        if shader1 is None:
            shader1 = diffuse_surface_shader
    else:
        raise ValueError("First shader input of Add Shader node is invalid")

    shader2 = None
    if node.inputs[1].links:
        node2 = node.inputs[1].links[0].from_node
        setup = setup_shader_from_node.get(node2.type)
        if setup is not None:
            shader2 = setup(node2)

        # default: diffuse
        if shader2 is None:
            shader2 = diffuse_surface_shader
    else:
        raise ValueError("Second shader input of Add Shader node is invalid")

    def surface_shader(ray_direction, normal):
        # evaluate shader1
        interactions_map1 = dict()
        for kind, vec, col in shader1(ray_direction, normal):
            interactions_map1[kind] = (vec, col)

        # evaluate shader1
        interactions_map2 = dict()
        for kind, vec, col in shader2(ray_direction, normal):
            interactions_map2[kind] = (vec, col)

        # build up interactions for added shader
        interactions = []

        # diffuse
        if 'DIFFUSE' in interactions_map1 or 'DIFFUSE' in interactions_map2:
            interactions.append(Interaction('DIFFUSE', None, None))

        # all other interactions
        for kind in ['REFLECT', 'REFRACT', 'TRANSPARENT']:
            vec1, col1 = interactions_map1.get(kind, (None, None))
            vec2, col2 = interactions_map2.get(kind, (None, None))
            if vec1 is not None:
                if vec2 is not None:
                    col1 += col2
                interactions.append(Interaction(kind, vec1, col))
            elif vec2 is not None:
                interactions.append(Interaction(kind, vec2, col2))

        return interactions

    return surface_shader


def setup_shadow(node):
    # assert node.type == 'HOLDOUT' or node.type == 'EMISSION', node.type

    def nothing_surface_shader(ray_direction, normal):
        # no caustics at all
        return []

    return nothing_surface_shader


# mapping from node type to setup of surface interaction function
setup_shader_from_node = {
    'BSDF_DIFFUSE': setup_diffuse,
    'BSDF_GLOSSY': setup_glossy,
    'BSDF_TRANSPARENT': setup_transparent,
    'BSDF_REFRACTION': setup_refraction,
    'BSDF_GLASS': setup_glass,
    'BSDF_PRINCIPLED': setup_principled,
    'BSDF_METALLIC': setup_metallic,
    'MIX_SHADER': setup_mix,
    'ADD_SHADER': setup_add,
    # holdout and emission shader stop raypath, but don't show caustics
    'HOLDOUT': setup_shadow,
    'EMISSION': setup_shadow,
}
# surface shader nodes that will be treated as diffuse:
#     BSDF_ANISOTROPIC
#     BSDF_HAIR
#     BSDF_HAIR_PRINCIPLED
#     BSDF_SHEEN
#     BSDF_TOON
#     BSDF_TRANSLUCENT
#     SUBSURFACE_SCATTERING
#     BSDF_VELVET
#     EEVEE_SPECULAR
#     PRINCIPLED_VOLUME
#     VOLUME_ABSORPTION
#     VOLUME_SCATTER
#     BSDF_RAY_PORTAL


# main function to process materials ------------------------------------------
@lru_cache(maxsize=None)
def get_material_shader(mat):
    """Returns a function and a tuple that model the material interaction.

    The function models the surface interaction like this:
    func(ray_direction, normal) = list of interactions
    The second return value is a tuple (or None) that summarizes the parameters
    for volume absorption. If not None (i.e. there is absorption) then it is of
    the form (color: 3-tuple, density: float).
    """
    # check if material is valid and uses nodes
    if mat is None or not mat.use_nodes:
        # default material: diffuse, no caustic rays
        return (diffuse_surface_shader, None)

    # find the active output node
    outnode = None
    for node in mat.node_tree.nodes:
        if node.type == 'OUTPUT_MATERIAL' and node.is_active_output:
            outnode = node
            # TODO distinguish between EEVEE and Cycles output

    if outnode is None:
        # no output node: invalid material
        raise ValueError(f"Material {mat.name} has no active output node")

    # find linked shader for volume, if any
    volume_params = None
    volume_links = outnode.inputs['Volume'].links
    if volume_links:
        volume_node = volume_links[0].from_node
        if volume_node.type == 'VOLUME_ABSORPTION':
            color_srgb = volume_node.inputs['Color'].default_value[:3]
            # freeze color because it will be put into a link for a chain
            # which will be used as a key for a dict and the key must be
            # hashable
            volume_params = (
                Color(utils.srgb_to_linear(color_srgb)).freeze(),
                volume_node.inputs['Density'].default_value
            )

    # find linked shader for surface (if none: invalid)
    surface_links = outnode.inputs['Surface'].links
    if not surface_links:
        raise ValueError(f"Material {mat.name} has no active surface shader")

    # get node and setup shader for surface
    node = surface_links[0].from_node
    setup = setup_shader_from_node.get(node.type)
    if setup is None:
        # node type not in dict, assume it's a diffuse shader
        return (diffuse_surface_shader, volume_params)

    # setup surface shader
    surface_shader = setup(node)
    if surface_shader is None:
        # can't process node e.g. rough glass/glossy, fall back to diffuse
        return (diffuse_surface_shader, volume_params)

    # surface and volume setup complete
    return (surface_shader, volume_params)


def cache_clear():
    """Clear the cache used by get_material_shader."""
    get_material_shader.cache_clear()


# -----------------------------------------------------------------------------
# Setup caustic material node tree
# -----------------------------------------------------------------------------
def get_caustic_material(light, parent_obj):
    """Get or setup caustic material for given light/parent combination."""
    # caustic material name
    parent_mat = parent_obj.active_material
    if parent_mat is None:
        parent_mat_name = "<Default Material>"
    else:
        parent_mat_name = parent_mat.name
    mat_name = f"Caustic of {light.name} for {parent_mat_name}"

    # lookup if it already exists, if yes we are done
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]

    # setup material
    mat = bpy.data.materials.new(mat_name)
    mat.cycles.sample_as_light = False  # disable multiple-importance sampling
    mat.use_nodes = True
    mat.node_tree.nodes.clear()

    # for Cycles
    add_nodes_for_cycles(mat.node_tree, light)

    # for EEVEE
    if bpy.app.version < (4, 2, 0):
        mat.blend_method = 'BLEND'  # alpha blending
        mat.shadow_method = 'NONE'  # no shadows needed
    else:
        # settings for EEVEE_NEXT
        mat.surface_render_method = 'BLENDED'  # alpha blending
        # shadow method is now an object visibility setting
    add_nodes_for_eevee(mat.node_tree, light, parent_obj.data.uv_layers.active)

    return mat


# for node handling see https://blender.stackexchange.com/a/23446
def add_nodes_for_cycles(node_tree, light):
    # add nodes
    nodes = node_tree.nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    geometry = nodes.new(type='ShaderNodeNewGeometry')
    add_shader = nodes.new(type='ShaderNodeAddShader')
    transparent = nodes.new(type='ShaderNodeBsdfTransparent')
    emission = nodes.new(type='ShaderNodeEmission')
    mix_rgb = nodes.new(type='ShaderNodeMixRGB')
    math = nodes.new(type='ShaderNodeMath')
    color = nodes.new(type='ShaderNodeRGB')
    vertex_color = nodes.new(type='ShaderNodeVertexColor')
    strength = nodes.new(type='ShaderNodeValue')
    sep_xyz = nodes.new(type='ShaderNodeSeparateXYZ')
    uv_squeeze = nodes.new(type='ShaderNodeUVMap')

    # change location
    output.location = (300, 700)
    mix_shader.location = (60, 700)
    geometry.location = (-180, 700)
    add_shader.location = (-180, 575)
    transparent.location = (-420, 650)
    emission.location = (-420, 400)
    mix_rgb.location = (-660, 550)
    math.location = (-660, 250)
    color.location = (-900, 600)
    vertex_color.location = (-900, 380)
    strength.location = (-900, 220)
    sep_xyz.location = (-900, 100)
    uv_squeeze.location = (-1140, 100)

    # change settings
    output.target = 'CYCLES'
    for option in geometry.outputs:
        # hide unused output options
        if option.name != 'Backfacing':
            option.hide = True
    mix_rgb.blend_type = 'MULTIPLY'
    mix_rgb.inputs['Fac'].default_value = 1.0
    math.operation = 'MULTIPLY'
    color.label = "Light Color"
    vertex_color.layer_name = 'Caustic Tint'
    strength.label = "Light Strength"
    uv_squeeze.uv_map = 'Caustic Squeeze'

    # add links
    links = node_tree.links
    links.new(mix_shader.outputs[0], output.inputs[0])
    links.new(geometry.outputs['Backfacing'], mix_shader.inputs[0])
    links.new(transparent.outputs[0], mix_shader.inputs[1])
    links.new(add_shader.outputs[0], mix_shader.inputs[2])
    links.new(transparent.outputs[0], add_shader.inputs[0])
    links.new(emission.outputs[0], add_shader.inputs[1])
    links.new(mix_rgb.outputs[0], emission.inputs[0])
    links.new(math.outputs[0], emission.inputs[1])
    links.new(color.outputs[0], mix_rgb.inputs[1])
    links.new(vertex_color.outputs[0], mix_rgb.inputs[2])
    links.new(strength.outputs[0], math.inputs[0])
    links.new(sep_xyz.outputs['Y'], math.inputs[1])
    links.new(uv_squeeze.outputs[0], sep_xyz.inputs[0])

    # add driver
    add_drivers_from_light(color, strength, light)


def add_nodes_for_eevee(node_tree, light, uv_layer=None):
    # add nodes
    nodes = node_tree.nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    add_shader = nodes.new(type='ShaderNodeAddShader')
    transparent = nodes.new(type='ShaderNodeBsdfTransparent')
    emission = nodes.new(type='ShaderNodeEmission')
    mix_rgb_diffuse = nodes.new(type='ShaderNodeMixRGB')
    math = nodes.new(type='ShaderNodeMath')
    mix_rgb_light = nodes.new(type='ShaderNodeMixRGB')
    strength = nodes.new(type='ShaderNodeValue')
    sep_xyz = nodes.new(type='ShaderNodeSeparateXYZ')
    color = nodes.new(type='ShaderNodeRGB')
    vertex_color = nodes.new(type='ShaderNodeVertexColor')
    uv_squeeze = nodes.new(type='ShaderNodeUVMap')

    # change location
    output.location = (300, -600)
    add_shader.location = (60, -600)
    transparent.location = (-180, -550)
    emission.location = (-180, -700)
    mix_rgb_diffuse.location = (-420, -550)
    math.location = (-420, -850)
    mix_rgb_light.location = (-660, -650)
    strength.location = (-660, -880)
    sep_xyz.location = (-660, -1000)
    color.location = (-900, -600)
    vertex_color.location = (-900, -820)
    uv_squeeze.location = (-900, -1000)

    # change settings
    output.target = 'EEVEE'
    mix_rgb_diffuse.blend_type = 'MULTIPLY'
    mix_rgb_diffuse.inputs['Fac'].default_value = 1.0
    mix_rgb_diffuse.inputs['Color1'].default_value = (0.8, 0.8, 0.8, 1.0)
    mix_rgb_diffuse.label = "Diffuse Mix"
    math.operation = 'MULTIPLY'
    mix_rgb_light.blend_type = 'MULTIPLY'
    mix_rgb_light.inputs['Fac'].default_value = 1.0
    strength.label = "Light Strength"
    color.label = "Light Color"
    vertex_color.layer_name = 'Caustic Tint'
    uv_squeeze.uv_map = 'Caustic Squeeze'

    # add links
    links = node_tree.links
    links.new(add_shader.outputs[0], output.inputs[0])
    links.new(transparent.outputs[0], add_shader.inputs[0])
    links.new(emission.outputs[0], add_shader.inputs[1])
    links.new(mix_rgb_diffuse.outputs[0], emission.inputs[0])
    links.new(math.outputs[0], emission.inputs[1])
    links.new(mix_rgb_light.outputs[0], mix_rgb_diffuse.inputs[2])
    links.new(strength.outputs[0], math.inputs[0])
    links.new(sep_xyz.outputs['Y'], math.inputs[1])
    links.new(color.outputs[0], mix_rgb_light.inputs[1])
    links.new(vertex_color.outputs[0], mix_rgb_light.inputs[2])
    links.new(uv_squeeze.outputs[0], sep_xyz.inputs[0])

    # add example texture and uvmap node for default diffuse color
    texture = nodes.new(type='ShaderNodeTexChecker')
    uv_map = nodes.new(type='ShaderNodeUVMap')
    texture.location = (-660, -400)
    uv_map.location = (-900, -400)
    texture.inputs['Color1'].default_value = (0.0, 1.0, 0.0, 1.0)
    texture.inputs['Color2'].default_value = (1.0, 0.0, 1.0, 1.0)
    texture.inputs['Scale'].default_value = 42.0
    texture.label = "Original Diffuse Texture"
    texture.use_custom_color = True
    texture.color = (1.0, 0.0, 0.0)
    uv_map.uv_map = "UVMap" if uv_layer is None else uv_layer.name
    links.new(texture.outputs[0], mix_rgb_diffuse.inputs[1])
    links.new(uv_map.outputs[0], texture.inputs[0])

    # add driver
    add_drivers_from_light(color, strength, light)


def add_drivers_from_light(color, strength, light):
    """Setup drivers for color and light strength nodes."""
    # setup driver for light color node
    for idx in range(3):
        fcurve = color.outputs[0].driver_add("default_value", idx)
        driver = fcurve.driver
        driver.type = 'AVERAGE'

        # color from light
        var = driver.variables.new()
        var.name = "color"
        var.targets[0].id_type = 'LIGHT'
        var.targets[0].id = light.data
        var.targets[0].data_path = f'color[{idx}]'

    # setup driver for light strength node
    fcurve = strength.outputs[0].driver_add("default_value")
    driver = fcurve.driver
    driver.type = 'SCRIPTED'

    # energy from light
    var = driver.variables.new()
    var.name = "energy"
    var.targets[0].id_type = 'LIGHT'
    var.targets[0].id = light.data
    var.targets[0].data_path = 'energy'

    # caustic emission shader strength = light strength (in W/m^2) / pi
    # light strength = light energy / area over which light is emitted
    if light.data.type == 'SUN':
        # strength of sun light is already in W/m^2
        driver.expression = "energy / pi"
    else:
        # assert light.data.type in {'SPOT', 'POINT'}, light
        # point and spot lights emit their energy over a sphere (area = 4*pi)
        driver.expression = "energy / (4 * pi * pi)"

    # Note: I don't know why we need to divide light strength by pi to get the
    # emission shader strength, I found this out by rendering some test scenes:
    # - Sun light with strength 3.14159 shines perpendicular onto diffuse plane
    #   with diffuse color (1.0, 1.0, 1.0) => rendered pixel values are 1.0.
    # - Point light with strength 39.4784 W is at center of a unit sphere
    #   diffuse color (1.0, 1.0, 1.0) => rendered pixel values are 1.0.
    # When replacing the sun light or point light by appropriate meshes with
    # emission shaders the pixel values rendered by Cycles are 1.0 when the
    # emission strength is 1.0.
