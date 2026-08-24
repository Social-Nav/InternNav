"""Pure tests for the InternNav debug-overlay presentation helpers."""

import ast
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont


CLIENT_PATH = (
    Path(__file__).resolve().parents[2]
    / 'scripts'
    / 'realworld'
    / 'http_internvla_client.py'
)
DEJAVU_SANS = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
HELPERS = {
    '_semantic_llm_output',
    '_font_supports_overlay_glyphs',
    '_load_overlay_font',
    '_text_width',
    '_fit_prefix_by_pixel_width',
    '_fit_text_with_ellipsis',
    '_wrap_text_by_pixel_width',
}


@pytest.fixture(scope='module')
def overlay_helpers():
    """Load only pure shipped helpers, without importing ROS dependencies."""
    tree = ast.parse(CLIENT_PATH.read_text(encoding='utf-8'), filename=str(CLIENT_PATH))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in HELPERS
    ]
    assert {node.name for node in functions} == HELPERS
    namespace = {
        'os': os,
        'ImageFont': ImageFont,
        'OVERLAY_FONT_CANDIDATES': (str(DEJAVU_SANS),),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(CLIENT_PATH), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize(
    ('raw_output', 'expected'),
    (
        ('←←←←', 'TURN_LEFT (←←←←)'),
        ('command: →', 'TURN_RIGHT (command: →)'),
        (' ↑   ↑ ', 'GO_STRAIGHT (↑ ↑)'),
        ('↓', 'GO_BACK (↓)'),
        ('STOP', 'STOP'),
        (' 86   427 ', 'PIXEL_TARGET=86 427'),
        ('←→', 'UNKNOWN (←→)'),
        ('turn around', 'UNKNOWN (turn around)'),
        ('', 'UNKNOWN (empty)'),
    ),
)
def test_semantic_llm_output_is_explicit_and_preserves_useful_raw_text(
    overlay_helpers, raw_output, expected
):
    assert overlay_helpers['_semantic_llm_output'](raw_output) == expected


def test_overlay_font_is_concrete_and_has_distinct_arrow_glyphs(overlay_helpers):
    assert DEJAVU_SANS.is_file(), 'the production InternNav font is missing'
    font = overlay_helpers['_load_overlay_font'](size=12, candidates=(str(DEJAVU_SANS),))
    assert Path(font.path).resolve() == DEJAVU_SANS.resolve()
    assert overlay_helpers['_font_supports_overlay_glyphs'](font)

    rasters = [(font.getmask(glyph).size, bytes(font.getmask(glyph))) for glyph in '←↑→↓']
    assert all(font.getmask(glyph).getbbox() is not None for glyph in '←↑→↓')
    assert len(set(rasters)) == 4

    with pytest.raises(RuntimeError, match='no usable InternNav overlay TTF'):
        overlay_helpers['_load_overlay_font'](
            size=12,
            candidates=('/definitely/missing/internnav-overlay-font.ttf',),
        )


def test_instruction_wrap_uses_pixel_width_normalizes_whitespace_and_ellipsizes(
    overlay_helpers,
):
    font = overlay_helpers['_load_overlay_font'](size=12, candidates=(str(DEJAVU_SANS),))
    draw = ImageDraw.Draw(Image.new('RGB', (640, 200)))
    max_width = 180
    lines = overlay_helpers['_wrap_text_by_pixel_width'](
        draw,
        '  Instruction:  walk through the central aisle\nthen turn left near the shelves '
        'and continue until the distant service counter is visible  ',
        font,
        max_width=max_width,
        max_lines=3,
    )

    assert len(lines) == 3
    assert lines[-1].endswith('…')
    assert all('\n' not in line and '  ' not in line for line in lines)
    assert all(
        overlay_helpers['_text_width'](draw, line, font) <= max_width
        for line in lines
    )
