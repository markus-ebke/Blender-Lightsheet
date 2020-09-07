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
"""Functions for handling materials for tracing and setup of caustic node tree.

The main function that handles materials for tracing is get_material_shader,
it uses the setup_node_<...>-functions to process the active surface node.
"""

from collections import namedtuple
from math import sqrt

import bpy
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
# of form {material: (surface shader function, volume parameters tuple)}
materials_cache = dict()


# helper functions ------------------------------------------------------------
def fresnel(ray_direction, normal, ior):
    """Fresnel mix factor."""
    # TODO simplify all of this?
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


def invalid_surface_shader(ray_direction, normal):
    # no caustics at all
    return []


def diffuse_surface_shader(ray_direction, normal):
    # caustic on object, but no tracing of further rays
    return [("diffuse", None, None)]


# setup surface interaction from BSDF node ------------------------------------
def setup_node_diffuse(node):
    assert node.type == 'BSDF_DIFFUSE'
    return diffuse_surface_shader


def setup_node_glossy(node):
    assert node.type == 'BSDF_GLOSSY', node.type

    # don't handle reflection if roughness is nonzero
    if (node.inputs['Roughness'].default_value > 0.0 and
            node.distribution != 'SHARP'):
        return None

    # node settings
    color = Color(node.inputs['Color'].default_value[:3])

    def surface_shader(ray_direction, normal):
        refle = ray_direction.reflect(normal)
        return [("reflect", refle, color)]

    return surface_shader


def setup_node_transparent(node):
    assert node.type == 'BSDF_TRANSPARENT', node.type

    # node settings
    color = Color(node.inputs['Color'].default_value[:3])

    def surface_shader(ray_direction, normal):
        return [("transparent", ray_direction, color)]

    return surface_shader


def setup_node_refraction(node):
    assert node.type == 'BSDF_REFRACTION', node.type

    # don't handle refraction if roughness is nonzero
    if (node.inputs['Roughness'].default_value > 0.0 and
            node.distribution != 'SHARP'):
        return None

    # node settings
    color = Color(node.inputs['Color'].default_value[:3])
    ior = node.inputs['IOR'].default_value

    def surface_shader(ray_direction, normal):
        refra = refract(ray_direction, normal, ior)
        if refra is not None:
            return [("refract", refra, color)]
        else:
            return []

    return surface_shader


def setup_node_glass(node):
    assert node.type == 'BSDF_GLASS', node.type

    # don't handle reflection or refraction for rough glass
    if (node.inputs['Roughness'].default_value > 0.0 and
            node.distribution != 'SHARP'):
        return None

    # node settings
    color = Color(node.inputs['Color'].default_value[:3])
    ior = node.inputs['IOR'].default_value

    def surface_shader(ray_direction, normal):
        # outgoing vectors
        refle = ray_direction.reflect(normal)
        refra = refract(ray_direction, normal, ior)

        if refra is None:
            # total internal reflection
            return [("reflect", refle, color)]

        # reflection and refraction
        reflectivity = fresnel(ray_direction, normal, ior)
        interactions = [
            ("reflect", refle, reflectivity * color),
            ("refract", refra, (1 - reflectivity) * color)
        ]
        return interactions

    return surface_shader


def setup_node_principled(node):
    assert node.type == 'BSDF_PRINCIPLED', node.type

    # node settings
    color = Color(node.inputs['Base Color'].default_value[:3])
    metallic = node.inputs['Metallic'].default_value
    specular = node.inputs['Specular'].default_value
    ior = node.inputs['IOR'].default_value
    transmission = node.inputs['Transmission'].default_value

    # convert specular to ior for fresnel i.e. invert
    # specular = ((ior - 1) / (ior + 1))**2 / 0.08
    temp = sqrt(specular*0.08)
    specular_ior = (1 + temp) / (1 - temp)

    # have diffuse caustic if material is not a perfect mirror or glass
    handle_diffuse = metallic < 1 and transmission < 1

    # don't handle reflection and refraction if they have roughness
    roughness = node.inputs['Roughness'].default_value
    if not (roughness == 0 or node.distribution == 'SHARP'):
        # no reflection and refraction tracing, but maybe diffuse
        if handle_diffuse:
            return diffuse_surface_shader
        else:
            return None  # no diffuse => no caustic or caustic tracing at all

    # don't handle refraction if it has roughness
    extra_roughness = node.inputs['Transmission Roughness'].default_value
    handle_refraction = extra_roughness == 0 or node.distribution == 'SHARP'

    white = Color((1.0, 1.0, 1.0))

    # diffuse, reflection and possibly refraction
    def surface_shader(ray_direction, normal):
        # mix_1: diffuse <-> refra tinted via transmission
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
            interactions.append(("diffuse", None, None))

        if refle_tint.v > 0:
            # some light is reflected
            refle = ray_direction.reflect(normal)
            interactions.append(("reflect", refle, refle_tint))

        if refra_tint.v > 0 and handle_refraction:
            # some light is transmitted
            refra = refract(ray_direction, normal, ior)
            interactions.append(("refract", refra, refra_tint))

        return interactions

    return surface_shader


def setup_scalar_node(node, from_socket_identifier=None):
    if node.type == 'FRESNEL':  # fresnel node
        ior = node.inputs['IOR'].default_value

        def fac(incoming, normal):
            return fresnel(incoming, normal, ior)
    elif node.type == 'LAYER_WEIGHT':  # layer weight node
        blend = node.inputs['Blend'].default_value

        # formulas from cycles/kernel/svm_fresnel.h
        if from_socket_identifier == 'Fresnel':
            ior = 1 / max(1 - blend, 1e-5)
            print(f"IOR: {ior}")

            def fac(incoming, normal):
                return fresnel(incoming, normal, ior)
        else:
            assert from_socket_identifier == 'Facing'

            def fac(incoming, normal):
                dot = abs(incoming.dot(normal))
                if blend <= 0.5:
                    return 1 - pow(dot, 2 * blend)
                blend_clamped = max(blend, 1 - 1e-5)  # blend < 1
                return 1 - pow(dot, 0.5 / (1 - blend_clamped))
    else:  # none of the above
        print(f"scalar node {node.type} not supported")

        def fac(incoming, normal):
            return 0.0

    return fac


def setup_node_mix(node):
    assert node.type == 'MIX_SHADER', node.type

    # mix factor
    if node.inputs[0].links:
        link = node.inputs[0].links[0]
        fac = setup_scalar_node(link.from_node, link.from_socket.identifier)
    else:
        fac_value = node.inputs[0].default_value

        def fac(incoming, normal):
            return fac_value

    # get the two connected shader nodes and setup shader functions
    if node.inputs[1].links:
        node1 = node.inputs[1].links[0].from_node
        setup = setup_node_interactions.get(node1.type)
        if setup is not None:
            shader1 = setup(node1)
        else:
            shader1 = invalid_surface_shader
    else:
        shader1 = invalid_surface_shader

    if node.inputs[2].links:
        node2 = node.inputs[2].links[0].from_node
        setup = setup_node_interactions.get(node2.type)
        if setup is not None:
            shader2 = setup(node2)
        else:
            shader2 = invalid_surface_shader
    else:
        shader2 = invalid_surface_shader

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
        if "diffuse" in interactions_map1 or "diffuse" in interactions_map2:
            interactions.append(("diffuse", None, None))

        # all other interactions
        for kind in ["reflect", "refract", "transparent"]:
            vec1, col1 = interactions_map1.get(kind, (None, None))
            vec2, col2 = interactions_map2.get(kind, (None, None))
            if vec1 is not None:
                vec = vec1
                col = weight1 * col1
                if vec2 is not None:
                    col += weight2 * col2
                interactions.append((kind, vec, col))
            elif vec2 is not None:
                vec = vec2
                col = weight2 * col2
                interactions.append((kind, vec, col))

        return interactions

    return surface_shader


# mapping from node type to setup of surface interaction function
setup_node_interactions = {
    'BSDF_DIFFUSE': setup_node_diffuse,
    'BSDF_GLOSSY': setup_node_glossy,
    'BSDF_TRANSPARENT': setup_node_transparent,
    'BSDF_REFRACTION': setup_node_refraction,
    'BSDF_GLASS': setup_node_glass,
    'BSDF_PRINCIPLED': setup_node_principled,
    'MIX_SHADER': setup_node_mix,
}


# main function to process materials ------------------------------------------
def get_material_shader(mat):
    """Returns a function and a tuple that model the material interaction.

    The function models the surface interaction like this:
    func(ray_direction, normal) = list of interactions
    The second return value is a tuple (or None) that summarizes the parameters
    for volume absorption. If not None (i.e. there is absorption) then it is of
    the form (color: mathutils.Color, density: float).
    """
    # check cache
    shader = materials_cache.get(mat)
    if shader is not None:
        return shader

    # check if material is valid and uses nodes
    if mat is None or not mat.use_nodes:
        # default material: diffuse, no caustic rays
        print(f"{mat.name}: material doesn't use nodes => diffuse")

        materials_cache[mat] = (diffuse_surface_shader, None)
        return (diffuse_surface_shader, None)

    # find the active output node
    outnode = None
    for node in mat.node_tree.nodes:
        if node.type == 'OUTPUT_MATERIAL' and node.is_active_output:
            outnode = node
            # TODO distinguish between EEVEE and Cycles output

    if outnode is None:
        # no output node: invalid material
        print(f"{mat.name}: no active output node => black")

        materials_cache[mat] = (invalid_surface_shader, None)
        return (invalid_surface_shader, None)

    # find linked shader for volume, if any
    volume_params = None
    volume_links = outnode.inputs['Volume'].links
    if volume_links:
        volume_node = volume_links[0].from_node
        if volume_node.type == 'VOLUME_ABSORPTION':
            volume_params = (
                Color(volume_node.inputs['Color'].default_value[:3]),
                volume_node.inputs['Density'].default_value
            )

    # find linked shader for surface (if none: invalid)
    surface_links = outnode.inputs['Surface'].links
    if not surface_links:
        print(f"{mat.name}: no connected surface shader node => black")

        materials_cache[mat] = (invalid_surface_shader, volume_params)
        return (invalid_surface_shader, volume_params)

    # get node and setup shader for surface
    node = surface_links[0].from_node
    setup = setup_node_interactions.get(node.type)
    if setup is None:
        # node type not in dict, i.e. we don't know how to handle this node
        print(f"{mat.name}: don't know {node.type} => black")

        materials_cache[mat] = (invalid_surface_shader, volume_params)
        return (invalid_surface_shader, volume_params)

    # setup surface shader
    surface_shader = setup(node)
    if surface_shader is None:
        # e.g. rough glass/glossy => no tracing, no diffuse caustic
        print(f"{mat.name}: can't handle settings of {node.type} => black")

        materials_cache[mat] = (invalid_surface_shader, None)
        return (invalid_surface_shader, None)

    # surface and volume setup complete
    print(f"{mat.name}: setup {node.type} shader")

    materials_cache[mat] = (surface_shader, volume_params)
    return (surface_shader, volume_params)


# -----------------------------------------------------------------------------
# Setup caustic material node tree
# -----------------------------------------------------------------------------
def get_caustic_material(light, parent_obj):
    """Get or setup caustic material for given light/parent combination."""
    # caustic material name
    mat_name = f"Caustic of {light.name} on {parent_obj.name}"

    # lookup if it already exists, if yes we are done
    mat = bpy.data.materials.get(mat_name)
    if mat is not None:
        return mat

    # setup material
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()

    # for Cycles
    add_nodes_for_cycles(mat.node_tree, light)

    # for EEVEE
    mat.blend_method = 'BLEND'  # alpha blending
    mat.shadow_method = 'NONE'  # no shadows needed
    add_nodes_for_eevee(mat.node_tree, light, parent_obj)

    return mat


# for node handling see https://blender.stackexchange.com/a/23446
def add_nodes_for_cycles(node_tree, light):
    # add nodes
    nodes = node_tree.nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    path = nodes.new(type='ShaderNodeLightPath')
    add_shader = nodes.new(type='ShaderNodeAddShader')
    transparent = nodes.new(type='ShaderNodeBsdfTransparent')
    emission = nodes.new(type='ShaderNodeEmission')
    mix_rgb = nodes.new(type='ShaderNodeMixRGB')
    math = nodes.new(type='ShaderNodeMath')
    color = nodes.new(type='ShaderNodeRGB')
    vertex_color = nodes.new(type='ShaderNodeVertexColor')
    energy = nodes.new(type='ShaderNodeValue')
    sep_xyz = nodes.new(type='ShaderNodeSeparateXYZ')
    uv_squeeze = nodes.new(type='ShaderNodeUVMap')

    # change location
    output.location = (300, 700)
    mix_shader.location = (60, 700)
    path.location = (-180, 700)
    add_shader.location = (-180, 575)
    transparent.location = (-420, 650)
    emission.location = (-420, 400)
    mix_rgb.location = (-660, 550)
    math.location = (-660, 250)
    color.location = (-900, 600)
    vertex_color.location = (-900, 380)
    energy.location = (-900, 220)
    sep_xyz.location = (-900, 100)
    uv_squeeze.location = (-1140, 100)

    # change settings
    output.target = 'CYCLES'
    for option in path.outputs:
        # hide unused output options
        if option.name != 'Is Diffuse Ray':
            option.hide = True
    mix_rgb.blend_type = 'MULTIPLY'
    mix_rgb.inputs['Fac'].default_value = 1.0
    math.operation = 'MULTIPLY'
    color.label = "Light Color"
    vertex_color.layer_name = 'Caustic Tint'
    energy.label = "Light Energy"
    uv_squeeze.uv_map = 'Caustic Squeeze'

    # add links
    links = node_tree.links
    links.new(mix_shader.outputs[0], output.inputs[0])
    links.new(path.outputs['Is Diffuse Ray'], mix_shader.inputs[0])
    links.new(transparent.outputs[0], mix_shader.inputs[1])
    links.new(add_shader.outputs[0], mix_shader.inputs[2])
    links.new(transparent.outputs[0], add_shader.inputs[0])
    links.new(emission.outputs[0], add_shader.inputs[1])
    links.new(mix_rgb.outputs[0], emission.inputs[0])
    links.new(math.outputs[0], emission.inputs[1])
    links.new(color.outputs[0], mix_rgb.inputs[1])
    links.new(vertex_color.outputs[0], mix_rgb.inputs[2])
    links.new(energy.outputs[0], math.inputs[0])
    links.new(sep_xyz.outputs['Y'], math.inputs[1])
    links.new(uv_squeeze.outputs[0], sep_xyz.inputs[0])

    # add driver
    add_drivers_from_light(color, energy, light)


def add_nodes_for_eevee(node_tree, light, parent_obj):
    # add nodes
    nodes = node_tree.nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    add_shader = nodes.new(type='ShaderNodeAddShader')
    transparent = nodes.new(type='ShaderNodeBsdfTransparent')
    emission = nodes.new(type='ShaderNodeEmission')
    mix_rgb_diffuse = nodes.new(type='ShaderNodeMixRGB')
    math = nodes.new(type='ShaderNodeMath')
    checker = nodes.new(type='ShaderNodeTexChecker')
    mix_rgb_light = nodes.new(type='ShaderNodeMixRGB')
    energy = nodes.new(type='ShaderNodeValue')
    sep_xyz = nodes.new(type='ShaderNodeSeparateXYZ')
    uv_map = nodes.new(type='ShaderNodeUVMap')
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
    checker.location = (-660, -400)
    mix_rgb_light.location = (-660, -650)
    energy.location = (-660, -880)
    sep_xyz.location = (-660, -1000)
    uv_map.location = (-900, -400)
    color.location = (-900, -600)
    vertex_color.location = (-900, -820)
    uv_squeeze.location = (-900, -1000)

    # change settings
    output.target = 'EEVEE'
    mix_rgb_diffuse.blend_type = 'MULTIPLY'
    mix_rgb_diffuse.inputs['Fac'].default_value = 1.0
    math.operation = 'MULTIPLY'
    checker.inputs['Color1'].default_value = (1.0, 0.0, 1.0, 1.0)
    checker.inputs['Color2'].default_value = (0.0, 1.0, 0.0, 1.0)
    checker.inputs['Scale'].default_value = 42.0
    checker.label = "Original Diffuse Color"
    mix_rgb_light.blend_type = 'MULTIPLY'
    mix_rgb_light.inputs['Fac'].default_value = 1.0
    energy.label = "Light Energy"
    uv_map.uv_map = parent_obj.data.uv_layers.active.name
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
    links.new(checker.outputs[0], mix_rgb_diffuse.inputs[1])
    links.new(mix_rgb_light.outputs[0], mix_rgb_diffuse.inputs[2])
    links.new(energy.outputs[0], math.inputs[0])
    links.new(sep_xyz.outputs['Y'], math.inputs[1])
    links.new(uv_map.outputs[0], checker.inputs[0])
    links.new(color.outputs[0], mix_rgb_light.inputs[1])
    links.new(vertex_color.outputs[0], mix_rgb_light.inputs[2])
    links.new(uv_squeeze.outputs[0], sep_xyz.inputs[0])

    # add driver
    add_drivers_from_light(color, energy, light)


def add_drivers_from_light(color, energy, light):
    # light color
    for idx in range(3):
        driver = color.outputs[0].driver_add("default_value", idx)
        driver.driver.type = 'AVERAGE'

        var = driver.driver.variables.new()
        var.name = "color"
        var.targets[0].id_type = 'LIGHT'
        var.targets[0].id = light.data
        var.targets[0].data_path = f'color[{idx}]'

    # light energy
    driver = energy.outputs[0].driver_add("default_value")
    driver.driver.type = 'AVERAGE'

    var = driver.driver.variables.new()
    var.name = "energy"
    var.targets[0].id_type = 'LIGHT'
    var.targets[0].id = light.data
    var.targets[0].data_path = 'energy'
