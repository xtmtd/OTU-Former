"""ONNX files that must survive output cleanup."""

from pathlib import Path


def onnx_input_paths(path: Path) -> list[Path]:
    import onnx

    graph = Path(path).expanduser()
    if not graph.is_absolute():
        graph = Path.cwd() / graph
    model = onnx.load(str(graph), load_external_data=False)
    root = graph.parent.resolve()
    paths = [graph]

    def tensors(message):
        if isinstance(message, onnx.TensorProto):
            yield message
        for field, value in message.ListFields():
            if field.message_type is None:
                continue
            for child in value if field.is_repeated else [value]:
                yield from tensors(child)

    for tensor in tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        locations = [item.value for item in tensor.external_data if item.key == "location"]
        if len(locations) != 1 or not locations[0]:
            raise ValueError(f"external tensor {tensor.name!r} must declare exactly one location")
        entry = graph.parent / locations[0]
        target = entry.resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"external tensor {tensor.name!r} resolves outside the graph directory")
        if not entry.is_file():
            raise FileNotFoundError(f"ONNX external tensor file is missing: {entry}")
        paths.append(entry)
    return list(dict.fromkeys(paths))
