import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCH_PATH = ROOT / 'internnav/model/basemodel/internvla_n1/internvla_n1_arch.py'
NAVDP_PATH = ROOT / 'internnav/model/basemodel/internvla_n1/navdp.py'


def test_navdp_uses_the_shared_depthanything_checkpoint_resolver():
    arch_source = ARCH_PATH.read_text(encoding='utf-8')
    navdp_source = NAVDP_PATH.read_text(encoding='utf-8')

    ast.parse(arch_source)
    ast.parse(navdp_source)
    assert 'rgbd_checkpoint=_resolve_depthanything_checkpoint()' in arch_source
    assert 'rgbd_checkpoint=None' in navdp_source
    assert 'checkpoint=rgbd_checkpoint or "checkpoints/depth_anything_v2_vits.pth"' in navdp_source
    assert 'from internnav.model.encoder.diffusion_policy.model.diffusion.positional_embedding' in navdp_source
