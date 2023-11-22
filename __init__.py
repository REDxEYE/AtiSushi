import copy
from pathlib import Path

import bpy
import numpy as np

from ...common_api import *
from ...plugins.AtiSushi.so2 import load_so2, So2PixelFormat


def plugin_init():
    pass


def so2_init():
    pass


def so2_load(operator, filepath: str, files: list[str]):
    collection = get_or_create_collection("Main", bpy.context.scene.collection)
    base_path = Path(filepath).parent
    for file in files:
        filepath = base_path / file
        with FileBuffer(filepath, 'rb') as f:
            so2 = load_so2(file, f)

        for texture in so2.textures:
            if texture.size == 0:
                continue
            tmp = {
                So2PixelFormat.R8: PixelFormat.R8,
                So2PixelFormat.RGBA8888: PixelFormat.RGBA8888,
                So2PixelFormat.BGRA8888: PixelFormat.RGBA8888,
                So2PixelFormat.BC1: PixelFormat.BC1,
                So2PixelFormat.BC2: PixelFormat.BC2,
                So2PixelFormat.BC3: PixelFormat.BC3,
                So2PixelFormat.RG1616_SNORM: PixelFormat.RG16_SIGNED,
                So2PixelFormat.RGBA1010102: PixelFormat.RGBA1010102,
            }
            data_size = get_buffer_size_from_texture_format(texture.width, texture.height * texture.layers,
                                                            tmp[texture.pixel_format])
            image = create_image_from_data(Path(texture.name).stem, texture.data[:data_size], texture.width,
                                           texture.height * texture.layers,
                                           tmp[texture.pixel_format],
                                           True)
            if image is None:
                operator.report({"ERROR"}, f"Failed to load {Path(texture.name).stem}")
            image.use_fake_user = True
        for render_group in so2.render_groups:
            for i, stream_buffer in enumerate(render_group.stream_buffers):
                for vertex_buffer_index, index_buffer_index in zip(stream_buffer.stream_vertex_index_array,
                                                                   stream_buffer.stream_index_index_array):
                    vertex_buffer = render_group.vertex_buffers[vertex_buffer_index]
                    index_buffer = render_group.index_buffers[index_buffer_index]

                    vertex_format = render_group.shader_header.vertex_formats[stream_buffer.vertex_format]
                    vertices = np.frombuffer(vertex_buffer.data, vertex_format.get_dtype())
                    mesh_data = bpy.data.meshes.new(render_group.name + f"_MESH")
                    mesh_obj = bpy.data.objects.new(render_group.name, mesh_data)
                    mesh_data.from_pydata(vertices["Position"], [], index_buffer.data.reshape((-1, 3)))
                    mesh_data.update(calc_edges=True, calc_edges_loose=True)
                    for attribute in vertex_format.attributes:
                        if attribute.name.startswith("UV"):
                            add_uv_layer(attribute.name, vertices[attribute.name], mesh_data, flip_uv=False)
                        elif attribute.name.startswith("VertexColor"):
                            add_vertex_color_layer(attribute.name,
                                                   vertices[attribute.name].copy().astype(np.float32) / 255,
                                                   mesh_data)
                    if "Normal" in vertices.dtype.names:
                        if vertices["Normal"].dtype == np.float32:
                            add_custom_normals(vertices["Normal"], mesh_data)
                    material_indices_array = np.zeros(index_buffer.count // 3, np.uint32)
                    mat_id = 0
                    texture_map = {}
                    materials = []
                    for command in render_group.command_buffers[i].data:
                        if command.id == 0:  # Draw indexes
                            material = create_material(f"{render_group.name}_{mat_id}", mesh_obj)
                            index_start, index_end, _ = command.args
                            material_indices_array[index_start // 3:index_end // 3] = mat_id
                            mat_id += 1
                            materials.append((material, copy.deepcopy(texture_map)))
                        elif command.id == 2:  # Bind texture
                            texture_slot, texture_id, _ = command.args
                            texture_map[texture_slot] = so2.textures[texture_id]
                    mesh_data.polygons.foreach_set('material_index', material_indices_array)
                    for material, texture_mapping in materials:
                        material.use_nodes = True
                        for texture in texture_mapping.values():
                            if texture.name and Path(texture.name).stem in bpy.data.images:
                                image = bpy.data.images[Path(texture.name).stem]
                                create_texture_node(material, image)

                    collection.objects.link(mesh_obj)
    return {"FINISHED"}


plugin_info = {
    "name": "ATI Sushi Chef",
    "id": "ATISushiChef",
    "description": "Import ATI Sushi engine assets",
    "version": (0, 1, 0),
    "loaders": [
        {
            "name": "Load .so2 file",
            "id": "asc_so2",
            "exts": ("*.so2",),
            "init_fn": so2_init,
            "import_fn": so2_load,
            "properties": [
            ]
        },
    ],
    "init_fn": plugin_init
}
