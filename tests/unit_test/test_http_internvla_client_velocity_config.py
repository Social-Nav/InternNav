"""Proof that the robot output velocity settings are wired through, not just read.

This stack has repeatedly shipped settings that parsed and then had no effect, so
the load-bearing test here is the read-back one: a changed limit must change what
the client publishes, measured with the real solver.

Everything runs against the *shipped source*. ``parse_args`` and
``apply_velocity_config`` are extracted from ``http_internvla_client.py`` with
:mod:`ast` and executed rather than re-implemented, because the client cannot be
imported outside its ROS environment.
"""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REALWORLD = Path(__file__).resolve().parents[2] / 'scripts' / 'realworld'
CLIENT_PATH = REALWORLD / 'http_internvla_client.py'

#: The values this client ships, hard-coded on purpose: if a default moves, this
#: test fails rather than following along.  These are *our* defaults, not the
#: upstream constructor ones -- upstream is v_max 0.4 / w_max 0.4 for the MPC and
#: max_v 1.0 / max_w 1.2 for the PID.
SHIPPED_DEFAULTS = {
    'mpc_reference_velocity': 0.3,
    'mpc_max_linear_velocity': 0.4,
    'mpc_max_angular_velocity': 1.0,
    'pid_max_linear_velocity': 0.6,
    'pid_max_angular_velocity': 1.0,
}

#: Limits enforced downstream of this client by Isaac's diff-drive graph, taken
#: from arena_isaac SpawnUsdRobot.py and isaac_utils/graphs/control/differential.py.
#: A default that crosses one of these silently puts a second limiter in the loop.
ISAAC_MAX_ANGULAR_SPEED = 1.5
ISAAC_MAX_WHEEL_SPEED = 10.0
ISAAC_WHEEL_RADIUS = 0.085
ISAAC_HALF_WHEEL_DISTANCE = 0.208

#: Module-level names the extracted surface needs, kept explicit so an unrelated
#: assignment can never be dragged in.
_WANTED_ASSIGNS = frozenset(
    {
        'MPC_REFERENCE_VELOCITY',
        'MPC_MAX_LINEAR_VELOCITY',
        'MPC_MAX_ANGULAR_VELOCITY',
        'PID_MAX_LINEAR_VELOCITY',
        'PID_MAX_ANGULAR_VELOCITY',
        'VELOCITY_SETTINGS',
        'mpc_reference_velocity',
        'mpc_max_linear_velocity',
        'mpc_max_angular_velocity',
        'pid_max_linear_velocity',
        'pid_max_angular_velocity',
    }
)
_WANTED_FUNCS = frozenset({'parse_args', 'apply_velocity_config'})


@pytest.fixture
def client():
    """The client's velocity settings, parser and applier, as shipped."""
    tree = ast.parse(CLIENT_PATH.read_text())
    kept = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in _WANTED_FUNCS)
        or (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in _WANTED_ASSIGNS
        )
    ]
    found_funcs = {n.name for n in kept if isinstance(n, ast.FunctionDef)}
    assert found_funcs == _WANTED_FUNCS, f'missing from client source: {_WANTED_FUNCS - found_funcs}'

    import argparse

    namespace = {'argparse': argparse, 'pid': SimpleNamespace(max_v=None, max_w=None)}
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(CLIENT_PATH), 'exec'), namespace)
    return namespace


def _parse(client, argv):
    """Parse ``argv`` through the client's own parser."""
    saved, sys.argv = sys.argv, ['http_internvla_client.py'] + argv
    try:
        args, _ros = client['parse_args']()
    finally:
        sys.argv = saved
    return args


def test_defaults_unchanged_when_nothing_is_set(client):
    """Negative control: no velocity flag reproduces the shipped values."""
    resolved = client['apply_velocity_config'](_parse(client, []))
    assert resolved == pytest.approx(SHIPPED_DEFAULTS)
    # The applier really wrote them onto the live PID controller, too.
    assert client['pid'].max_v == pytest.approx(SHIPPED_DEFAULTS['pid_max_linear_velocity'])
    assert client['pid'].max_w == pytest.approx(SHIPPED_DEFAULTS['pid_max_angular_velocity'])


def test_cli_flag_overrides_the_stored_default(client):
    """A flag reaches the values the MPC and PID are built from."""
    resolved = client['apply_velocity_config'](
        _parse(client, ['--mpc-max-linear-velocity', '0.137', '--pid-max-angular-velocity', '0.911'])
    )
    assert resolved['mpc_max_linear_velocity'] == pytest.approx(0.137)
    assert resolved['pid_max_angular_velocity'] == pytest.approx(0.911)
    assert client['pid'].max_w == pytest.approx(0.911)
    # Settings that were not passed must not drift.
    assert resolved['mpc_reference_velocity'] == pytest.approx(SHIPPED_DEFAULTS['mpc_reference_velocity'])
    assert resolved['mpc_max_angular_velocity'] == pytest.approx(SHIPPED_DEFAULTS['mpc_max_angular_velocity'])


def test_setting_reaches_the_published_command(client):
    """READ-BACK PROOF, against the real solver.

    ``control_thread`` publishes ``opt_u_controls[0, 0]``.  With the robot behind
    its reference -- the production regime, since the client drops the first 3
    points of every returned trajectory -- that value sits on ``v_max``.  So
    changing ``--mpc-max-linear-velocity`` must move the published number to the
    value asked for.  This simultaneously confirms the two-regime note in the
    client: ``min(reference, max_linear)`` describes only zero tracking error,
    while under a catch-up gap the ceiling is what is commanded.
    """
    pytest.importorskip('casadi', reason='the real solver is only installed in the internnav container')
    sys.path.insert(0, str(REALWORLD))
    from controllers import Mpc_controller

    # A straight reference starting 0.5 m ahead of the robot: a catch-up gap well
    # beyond the 0.05 m that already saturates the bound in production.
    reference = np.array([[0.5 + 0.1 * i, 0.0] for i in range(12)], dtype=float)
    x0 = np.array([0.0, 0.0, 0.0])

    published = {}
    for asked in (SHIPPED_DEFAULTS['mpc_max_linear_velocity'], 0.137, 0.9):
        resolved = client['apply_velocity_config'](_parse(client, ['--mpc-max-linear-velocity', str(asked)]))
        mpc = Mpc_controller(
            reference,
            desired_v=resolved['mpc_reference_velocity'],
            v_max=resolved['mpc_max_linear_velocity'],
            w_max=resolved['mpc_max_angular_velocity'],
        )
        # Note the order: solve() returns (controls, states), and the client
        # publishes opt_u_controls[0, 0] at http_internvla_client.py:238-239.
        opt_u_controls, _opt_x_states = mpc.solve(x0)
        published[asked] = float(opt_u_controls[0, 0])

    for asked, got in published.items():
        assert got == pytest.approx(asked, abs=1e-3), (
            f'asked for v_max={asked} but the solver published {got}: '
            f'the setting did not reach the published command'
        )
    # Distinct values, so this cannot be passing on a constant.
    assert len({round(v, 3) for v in published.values()}) == len(published)


def test_help_text_expands_for_every_velocity_flag(client):
    """argparse %-formats help strings, so a literal '%' there crashes --help.

    The velocity help text quotes measured percentages, which makes this a live
    hazard: a bare '%' raises TypeError the first time anyone runs --help, and
    nothing else in this suite would notice because parsing itself still works.
    Escaped as '%%' it is fine.  Formatting the whole help is the check.
    """
    saved, sys.argv = sys.argv, ['http_internvla_client.py', '--help']
    try:
        # --help makes argparse expand every help string and then exit, which is
        # exactly the path that a stray '%' breaks.
        with pytest.raises(SystemExit) as exc:
            client['parse_args']()
        assert exc.value.code == 0
    finally:
        sys.argv = saved


def test_angular_defaults_stay_below_downstream_limiters():
    """The angular defaults must not hand control to Isaac's limiters.

    Isaac clamps angular at max_angular_speed and clips the wheel command at
    maxWheelSpeed, where wheel = (v + half_wheel_distance * w) / wheel_radius.
    If an angular default crosses either, the value this client publishes stops
    being the value that reaches the wheels and every commanded-vs-achieved
    reading becomes unattributable.  The wheel budget is the tighter of the two
    and it couples to the *linear* default, which is why this is checked as a
    pair rather than against the angular clamp alone.

    The linear term is the *largest* linear default rather than the MPC one,
    because the wheel clip does not care which code path issued the command and
    the batch runs raise the MPC linear ceiling by flag.  Pairing the largest
    linear with the largest angular is the only combination that bounds every
    invocation; using the MPC default alone left this check with no power, which
    a deliberate 1.3 rad/s sabotage exposed.
    """
    v = max(
        SHIPPED_DEFAULTS['mpc_max_linear_velocity'],
        SHIPPED_DEFAULTS['pid_max_linear_velocity'],
    )
    for name in ('mpc_max_angular_velocity', 'pid_max_angular_velocity'):
        w = SHIPPED_DEFAULTS[name]
        assert w < ISAAC_MAX_ANGULAR_SPEED, (
            f'{name}={w} is at or above Isaac max_angular_speed='
            f'{ISAAC_MAX_ANGULAR_SPEED}; Isaac would silently clamp it'
        )
        wheel = (v + ISAAC_HALF_WHEEL_DISTANCE * w) / ISAAC_WHEEL_RADIUS
        assert wheel < ISAAC_MAX_WHEEL_SPEED, (
            f'{name}={w} with linear={v} asks for {wheel:.3f} rad/s of wheel '
            f'speed, at or above maxWheelSpeed={ISAAC_MAX_WHEEL_SPEED}; the '
            f'wheel command would be clipped'
        )
