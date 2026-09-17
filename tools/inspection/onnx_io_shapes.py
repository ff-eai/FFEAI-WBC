#!/usr/bin/env python3
"""Minimal, dependency-free ONNX input/output shape reader.

Parses just enough of the ONNX protobuf (ModelProto -> GraphProto ->
ValueInfoProto -> TypeProto.Tensor -> TensorShapeProto) to print the name,
element type, and dims of every graph input and output. Uses only the Python
standard library so it runs without onnx / onnxruntime installed.

Read-only inspection tool (Phase 1). Usage:
    python tools/inspection/onnx_io_shapes.py <model.onnx> [<model2.onnx> ...]
"""

import sys


def _read_varint(buf, i):
    shift = 0
    result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7


def _iter_fields(buf):
    """Yield (field_number, wire_type, value) for a protobuf message.

    value is an int for varint/fixed, or a memoryview/bytes for length-delimited.
    """
    i = 0
    n = len(buf)
    while i < n:
        tag, i = _read_varint(buf, i)
        field = tag >> 3
        wire = tag & 0x7
        if wire == 0:  # varint
            val, i = _read_varint(buf, i)
            yield field, wire, val
        elif wire == 2:  # length-delimited
            ln, i = _read_varint(buf, i)
            yield field, wire, buf[i:i + ln]
            i += ln
        elif wire == 1:  # 64-bit
            yield field, wire, buf[i:i + 8]
            i += 8
        elif wire == 5:  # 32-bit
            yield field, wire, buf[i:i + 4]
            i += 4
        else:
            raise ValueError(f"Unsupported wire type {wire} for field {field}")


ELEM_TYPES = {1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64",
              9: "bool", 10: "float16", 11: "float64", 12: "uint32", 13: "uint64"}


def _parse_value_info(buf):
    name = None
    elem_type = None
    dims = []
    for field, wire, val in _iter_fields(buf):
        if field == 1 and wire == 2:  # name
            name = bytes(val).decode("utf-8", "replace")
        elif field == 2 and wire == 2:  # type (TypeProto)
            for f2, w2, v2 in _iter_fields(val):
                if f2 == 1 and w2 == 2:  # tensor_type (TypeProto.Tensor)
                    for f3, w3, v3 in _iter_fields(v2):
                        if f3 == 1 and w3 == 0:  # elem_type
                            elem_type = ELEM_TYPES.get(v3, f"type_{v3}")
                        elif f3 == 2 and w3 == 2:  # shape (TensorShapeProto)
                            for f4, w4, v4 in _iter_fields(v3):
                                if f4 == 1 and w4 == 2:  # dim (Dimension)
                                    dv = None
                                    for f5, w5, v5 in _iter_fields(v4):
                                        if f5 == 1 and w5 == 0:  # dim_value
                                            dv = v5
                                        elif f5 == 2 and w5 == 2:  # dim_param
                                            dv = bytes(v5).decode("utf-8", "replace")
                                    dims.append(dv if dv is not None else "?")
    return name, elem_type, dims


def inspect(path):
    with open(path, "rb") as f:
        data = f.read()
    graph = None
    for field, wire, val in _iter_fields(data):
        if field == 7 and wire == 2:  # ModelProto.graph
            graph = val
            break
    if graph is None:
        print(f"{path}: no graph found")
        return
    inputs, outputs = [], []
    for field, wire, val in _iter_fields(graph):
        if field == 11 and wire == 2:  # GraphProto.input
            inputs.append(_parse_value_info(val))
        elif field == 12 and wire == 2:  # GraphProto.output
            outputs.append(_parse_value_info(val))
    print(f"=== {path} ===")
    for nm, et, dims in inputs:
        print(f"  IN   {nm:32s} {et:8s} {dims}")
    for nm, et, dims in outputs:
        print(f"  OUT  {nm:32s} {et:8s} {dims}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        inspect(p)
