import ast
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


AGENT_PATH = (
    Path(__file__).resolve().parents[2]
    / 'internnav'
    / 'agent'
    / 'internvla_n1_agent_realworld.py'
)


def _load_overlay_function():
    tree = ast.parse(AGENT_PATH.read_text(encoding='utf-8'), filename=str(AGENT_PATH))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == '_load_system2_checkpoint'
    )
    namespace = {'json': json, 'Path': Path, 'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(AGENT_PATH), 'exec'), namespace)
    return namespace['_load_system2_checkpoint']


def _checkpoint(tmp_path, tensors):
    tmp_path.mkdir(parents=True, exist_ok=True)
    shard = tmp_path / 'model-00001-of-00001.safetensors'
    save_file(tensors, shard)
    (tmp_path / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {name: shard.name for name in tensors}}),
        encoding='utf-8',
    )
    return tmp_path


def test_system2_overlay_replaces_only_named_subset(tmp_path):
    load = _load_overlay_function()
    model = torch.nn.Module()
    model.system2 = torch.nn.Linear(2, 2, bias=False)
    model.system1 = torch.nn.Linear(2, 2, bias=False)
    system1_before = model.system1.weight.detach().clone()
    checkpoint = _checkpoint(
        tmp_path,
        {'system2.weight': torch.full_like(model.system2.weight, 3.0)},
    )

    metadata = load(model, checkpoint)

    assert metadata['tensor_count'] == 1
    assert torch.equal(model.system2.weight, torch.full_like(model.system2.weight, 3.0))
    assert torch.equal(model.system1.weight, system1_before)


def test_system2_overlay_rejects_unknown_or_mismatched_tensor(tmp_path):
    load = _load_overlay_function()
    model = torch.nn.Linear(2, 2, bias=False)
    with pytest.raises(RuntimeError, match='unknown tensors'):
        load(model, _checkpoint(tmp_path / 'unknown', {'unknown.weight': torch.ones(2, 2)}))

    mismatch = tmp_path / 'mismatch'
    with pytest.raises(RuntimeError, match='shape mismatch'):
        load(model, _checkpoint(mismatch, {'weight': torch.ones(3, 2)}))


def test_system2_overlay_rejects_dtype_mismatch(tmp_path):
    load = _load_overlay_function()
    model = torch.nn.Linear(2, 2, bias=False)
    with pytest.raises(RuntimeError, match='dtype mismatch'):
        load(model, _checkpoint(tmp_path, {'weight': torch.ones(2, 2, dtype=torch.float64)}))
