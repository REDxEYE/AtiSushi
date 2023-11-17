from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

from ....common_api import Buffer
import re


@dataclass(slots=True)
class Versions:
    sushi_version: tuple[int, int, int]
    engine_version: tuple[int, int]
    material_version: int


class So2PixelFormat(IntEnum):
    INVALID = -1
    R8 = 0
    RGBA8888 = 3
    BGRA8888 = 4
    BC1 = 6
    BC2 = 7
    BC3 = 8
    RG1616_SNORM = 12
    RGBA1010102 = 14
    UNK = 16


@dataclass(slots=True)
class TextureEntry:
    name: str
    name2: str
    name3: str
    pixel_format: So2PixelFormat
    size: int
    width: int
    height: int
    layers: int
    unk_data: list[int]
    data: Optional[bytes] = field(default=None, repr=False)

    @classmethod
    def from_buffer(cls, buffer: Buffer, versions: Versions):
        name = buffer.read_ascii_string(128)
        name2 = buffer.read_ascii_string(128)
        pixel_format = So2PixelFormat(buffer.read_int32())
        unk_data = []
        unk_data.extend(buffer.read_fmt("2I"))
        name3 = buffer.read_ascii_string(64)
        unk_data.extend(buffer.read_fmt("30I"))
        size = buffer.read_uint32()
        unk_data.extend(buffer.read_fmt("2I"))
        width, height, layers = buffer.read_fmt("2HB")
        unk_data.extend(buffer.read_fmt("6I"))
        return cls(name, name2, name3, pixel_format, size, width, height, layers, unk_data)


class AttributeType(IntEnum):
    FLOAT32_VEC2 = 1
    FLOAT32_VEC3 = 2
    FLOAT32_VEC4 = 3
    UINT32_VEC1 = 4
    UINT8_VEC4 = 5
    UNK_UINT8_VEC4 = 14


@dataclass(slots=True)
class VertexAttribute:
    name: str
    type: AttributeType
    semantic: int
    slot: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, versions: Versions):
        return cls(buffer.read_ascii_string(buffer.read_uint32()), AttributeType(buffer.read_uint32()),
                   buffer.read_uint32(), buffer.read_uint8())

    @property
    def dtype(self):
        if self.type == AttributeType.FLOAT32_VEC2:
            return self.name, np.float32, 2
        elif self.type == AttributeType.FLOAT32_VEC3:
            return self.name, np.float32, 3
        elif self.type == AttributeType.FLOAT32_VEC4:
            return self.name, np.float32, 4
        elif self.type == AttributeType.UINT32_VEC1:
            return self.name, np.uint8, 4
        elif self.type == AttributeType.UINT8_VEC4:
            return self.name, np.uint8, 4
        elif self.type == AttributeType.UNK_UINT8_VEC4:
            return self.name, np.uint8, 4
        raise NotImplementedError(f"Attribute type \"{self.type.name}\" not supported")


@dataclass(slots=True)
class VertexFormat:
    name: str
    type: str
    attributes: list[VertexAttribute]
    unk: tuple[int, int]

    @classmethod
    def from_buffer(cls, buffer: Buffer, versions: Versions):
        name = buffer.read_ascii_string(buffer.read_uint32())
        name2 = buffer.read_ascii_string(buffer.read_uint32())
        attributes = [VertexAttribute.from_buffer(buffer, versions) for _ in range(buffer.read_uint32())]

        return cls(name, name2, attributes, buffer.read_fmt("2i"))

    def get_dtype(self):
        return np.dtype([d.dtype for d in self.attributes])


@dataclass(slots=True)
class TextureBinding:
    name: str
    type: str

    @classmethod
    def from_buffer(cls, buffer: Buffer, versions: Versions):
        return cls(
            buffer.read_ascii_string(buffer.read_uint32()),
            buffer.read_ascii_string(buffer.read_uint32()))


@dataclass(slots=True)
class ShaderModel:
    name: str
    unk: list[int]

    @classmethod
    def from_buffer(cls, buffer: Buffer, versions: Versions):
        name = buffer.read_ascii_string(buffer.read_uint32())
        return cls(name, buffer.read_fmt(f"{buffer.read_uint32()}I"))


@dataclass(slots=True)
class ShaderHeader:
    version: int
    key_values: dict[str, str]
    vertex_formats: list[VertexFormat]
    shader_models: list[ShaderModel]
    texture_bindings: list[TextureBinding]

    @classmethod
    def from_buffer(cls, buffer: Buffer, versions: Versions):
        version_str = buffer.read_ascii_string(32)
        version_match = re.match(r"\*\sATI\sShader\sFile\sv(\d+)\s\*", version_str)
        if version_match is None:
            raise ValueError(f"Got invalid engine and material version: {version_str!r}")
        shader_version = int(version_match.group(1))
        del version_str, version_match
        key_values = {}
        for _ in range(buffer.read_uint32()):
            key = buffer.read_ascii_string(buffer.read_uint32())
            key_values[key] = buffer.read_ascii_string(buffer.read_uint32())
        formats = [VertexFormat.from_buffer(buffer, versions) for _ in range(buffer.read_uint32())]
        shader_models = [ShaderModel.from_buffer(buffer, versions) for _ in range(buffer.read_uint32())]
        texture_bindings = [TextureBinding.from_buffer(buffer, versions) for _ in range(buffer.read_uint32())]
        return cls(shader_version, key_values, formats, shader_models, texture_bindings)


@dataclass(slots=True)
class VertexBuffer:
    unk0: int
    unk1: int
    vertex_size: int
    vertex_count: int
    unk2: int
    data: bytes = field(default=None)

    @classmethod
    def from_buffer(cls, buffer: Buffer, versions: Versions):
        unk0, unk1, vertex_size, vertex_count, _, unk3 = buffer.read_fmt("6I")
        return cls(unk0, unk1, vertex_size, vertex_count, unk3)


@dataclass(slots=True)
class IndexBuffer:
    unk0: int
    count: int
    unk1: int
    data: np.ndarray = field(init=False)

    @classmethod
    def from_buffer(cls, buffer: Buffer, versions: Versions):
        unk0, count, _, unk1 = buffer.read_fmt("4I")
        return cls(unk0, count, unk1)


class CommandId(IntEnum):
    DRAW_INDEXED = 0
    UNK1 = 1
    BIND_TEXTURE = 2
    UNK3 = 3
    UNK4 = 4


@dataclass(slots=True)
class Command:
    id: CommandId
    args: tuple[int, int, int]

    @classmethod
    def from_buffer(cls, buffer: Buffer, versions: Versions):
        c_id, *args = buffer.read_fmt("4I")
        return cls(CommandId(c_id), args)


@dataclass(slots=True)
class CommandBuffer:
    count: int
    data: list[Command] = field(init=False)

    @classmethod
    def from_buffer(cls, buffer: Buffer, versions: Versions):
        count = buffer.read_uint32()
        buffer.skip(4)
        return cls(count)


@dataclass(slots=True)
class StreamMap:
    count: int
    data: list[int] = field(init=False)

    @classmethod
    def from_buffer(cls, buffer: Buffer, versions: Versions):
        count = buffer.read_uint32()
        buffer.skip(4)
        return cls(count)


@dataclass(slots=True)
class UnkData:
    name: str
    unk: tuple[int, ...]
    unk1: int
    unk2: int
    count: int
    unk3: int
    data: list[tuple[float, ...]] = field(init=False)

    @classmethod
    def from_buffer(cls, buffer: Buffer, versions: Versions):
        name = buffer.read_ascii_string(32)
        unk = buffer.read_fmt("8I")
        unk1 = buffer.read_uint32()
        unk2 = buffer.read_uint32()
        count = buffer.read_uint32()
        unk3 = buffer.read_uint32()
        return cls(name, unk, unk1, unk2, count, unk3)


@dataclass(slots=True)
class StreamBuffer:
    unk0: int
    vertex_format: int
    vertex_index_array_count: int
    element_stream_count: int
    stream_vertex_index_array: list[int] = field(init=False)
    stream_index_index_array: list[int] = field(init=False)
    stream_element_array: list[int] = field(init=False)

    @classmethod
    def from_buffer(cls, buffer: Buffer, versions: Versions):
        unk0, vertex_index_array_count, unk1, unk2, vertex_format, element_stream_count, unk3 = buffer.read_fmt("i6I")

        return cls(unk0, vertex_format, vertex_index_array_count, element_stream_count)


@dataclass(slots=True)
class RenderGroup:
    unk0: int
    unk1: int
    name: str
    unk3_0: int
    unk3_1: int
    unk3_3: int
    unk3_4: int
    unk3_6: int
    unk3_8: int
    unk3_10: int
    unk3_12: int
    unk3_14: int
    unk3_16: int
    shader_header: ShaderHeader

    time_array: list[int]
    vertex_buffers: list[VertexBuffer]
    index_buffers: list[IndexBuffer]
    command_buffers: list[CommandBuffer]
    stream_buffers: list[StreamBuffer]
    stream_map_array: list[StreamMap]
    unk_array: list[UnkData]

    @classmethod
    def from_buffer(cls, buffer: Buffer, versions: Versions):
        unk0, unk1 = buffer.read_fmt("2H")
        name = buffer.read_ascii_string(128)
        (unk3_0, unk3_1, time_array_count, unk3_3, unk3_4, vertex_buffer_count, unk3_6,
         index_buffer_count, unk3_8, command_buffer_count, unk3_10, stream_buffer_count, unk3_12, stream_map_count,
         unk3_14, unk_count, unk3_16) = buffer.read_fmt("17I")
        shader_header = ShaderHeader.from_buffer(buffer, versions)
        time_array = [buffer.read_uint32() for _ in range(time_array_count)]

        vertex_buffers = [VertexBuffer.from_buffer(buffer, versions) for _ in range(vertex_buffer_count)]
        for vertex_buffer in vertex_buffers:
            vertex_buffer.data = buffer.read(vertex_buffer.vertex_size * vertex_buffer.vertex_count)

        index_buffers = [IndexBuffer.from_buffer(buffer, versions) for _ in range(index_buffer_count)]
        for index_buffer in index_buffers:
            index_buffer.data = np.frombuffer(buffer.read(2 * index_buffer.count), np.uint16)

        command_buffers = [CommandBuffer.from_buffer(buffer, versions) for _ in range(command_buffer_count)]
        for command_buffer in command_buffers:
            command_buffer.data = [Command.from_buffer(buffer, versions) for _ in range(command_buffer.count)]

        stream_buffers = [StreamBuffer.from_buffer(buffer, versions) for _ in range(stream_buffer_count)]
        for stream_buffer in stream_buffers:
            stream_buffer.stream_vertex_index_array = buffer.read_fmt(f"{stream_buffer.vertex_index_array_count}I")
            stream_buffer.stream_index_index_array = buffer.read_fmt(f"{stream_buffer.vertex_index_array_count}I")
            stream_buffer.stream_element_array = buffer.read_fmt(f"{stream_buffer.element_stream_count}I")

        stream_map_array = [StreamMap.from_buffer(buffer, versions) for _ in range(stream_map_count)]
        for steam_map in stream_map_array:
            steam_map.data = buffer.read_fmt(f"{steam_map.count}I")

        unk_data = [UnkData.from_buffer(buffer, versions) for _ in range(unk_count)]
        for unk_obj in unk_data:
            unk_obj.data = [buffer.read_fmt("4f") for _ in range(unk_obj.count)]

        return cls(unk0, unk1, name, unk3_0, unk3_1, unk3_3, unk3_4, unk3_6, unk3_8, unk3_10, unk3_12, unk3_14,
                   unk3_16, shader_header, time_array, vertex_buffers,
                   index_buffers, command_buffers, stream_buffers, stream_map_array, unk_data)


@dataclass(slots=True)
class So2:
    textures: list[TextureEntry]
    render_groups: list[RenderGroup]


def load_so2(name: str, buffer: Buffer) -> So2:
    sushi_version_str = buffer.read_ascii_string(24)
    sushi_version_match = re.match(r"SOL\sRUN00\sv(\d{2})\.(\d{2})\.(\d{2})", sushi_version_str)
    if sushi_version_match is None:
        raise ValueError("Not a valid Sushi SO2 file.")
    sushi_version = (int(sushi_version_match.group(1)),
                     int(sushi_version_match.group(2)),
                     int(sushi_version_match.group(3)))
    del sushi_version_str, sushi_version_match
    material_str = buffer.read_ascii_string(36)
    material_match = re.match(r"\*\sATI\sSushi\s(\d)\.(\d)\sMaterial\sv(\d+)\s\*", material_str)
    if material_match is None:
        raise ValueError("Got invalid enigne and material version")
    *engine_version, material_version = tuple(map(int, material_match.groups()))
    del material_str, material_match
    print(
        f"Got SO2 version {'.'.join(map(str, sushi_version))} with engine version {'.'.join(map(str, engine_version))} and material version {material_version} ")
    versions = Versions(sushi_version, engine_version, material_version)
    textures = [TextureEntry.from_buffer(buffer, versions) for _ in range(buffer.read_uint32())]
    for texture in textures:
        if texture.size > 0:
            texture.data = buffer.read(texture.size)
    render_groups = [RenderGroup.from_buffer(buffer, versions) for _ in range(buffer.read_uint32())]
    return So2(textures, render_groups)
