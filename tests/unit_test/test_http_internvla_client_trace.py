import ast
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


CLIENT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "realworld"
    / "http_internvla_client.py"
)
APPROVED_CASES = (
    (
        "case_1",
        "travels across the open office to deliver items while avoiding tighter chair clusters",
        85,
        "fc4da22ea418bca6c7ea53fdb7957d1ad2614d42664915e19059b1edd4863574",
    ),
    (
        "case_2",
        "navigates through the central open aisle for a short delivery task while avoiding shelf edges",
        93,
        "bde0356ff2fdd1b5b7ade6c58c26077517ca97a4973f339a290e7f34af4a8309",
    ),
    (
        "case_3",
        "service robot crosses the central upper aisle between open circulation points",
        77,
        "8dcc7675122de32d72a3bc0f9c6995fd5d61579b7c1a9c72a7563a4a0d86c13e",
    ),
)
_, APPROVED_CASE_1_INSTRUCTION, APPROVED_CASE_1_UTF8_BYTES, APPROVED_CASE_1_SHA256 = APPROVED_CASES[0]


class FakeResponse:
    def __init__(self, payload):
        self.text = json.dumps(payload)

    def raise_for_status(self):
        return None


def _source_tree():
    source = CLIENT_PATH.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(CLIENT_PATH))


def _load_actual_functions(*names, namespace):
    _, tree = _source_tree()
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert set(names) <= functions.keys()
    selected = ast.Module(body=[functions[name] for name in names], type_ignores=[])
    ast.fix_missing_locations(selected)
    compiled = compile(selected, str(CLIENT_PATH), "exec")
    exec(compiled, namespace)
    return functions


def _base_namespace(post, write_trace):
    return {
        "json": json,
        "hashlib": hashlib,
        "requests": SimpleNamespace(post=post),
        "time": SimpleNamespace(time=lambda: 1234.5),
        "_write_trace": write_trace,
        "policy_init": True,
        "http_idx": -1,
        "first_running_time": 0.0,
        "print": lambda *args, **kwargs: None,
    }


def _expected_payload(
    *, instruction, pose, camera_pose, intrinsic, look_down, reset=True, http_idx=-1
):
    return {
        "reset": reset,
        "idx": http_idx,
        "request_id": http_idx + 1,
        "instruction": instruction,
        "pose": pose,
        "camera_pose": camera_pose,
        "intrinsic": intrinsic,
        "look_down": bool(look_down),
        "client": "internnav_realworld_ros2_http_client",
    }


def _invoke(dual_sys_eval, *, instruction=APPROVED_CASE_1_INSTRUCTION):
    return dual_sys_eval(
        b"image-secret-" + (b"A" * 128),
        b"depth-secret-" + (b"B" * 128),
        None,
        url="http://trace-must-not-leak.invalid/eval?token=secret-token",
        instruction=instruction,
        pose=[1.25, -2.5, 0.75],
        camera_pose=[[1.0, 0.0, 0.0, 1.25], [0.0, 1.0, 0.0, -2.5]],
        intrinsic=[[320.0, 0.0, 160.0], [0.0, 320.0, 120.0], [0.0, 0.0, 1.0]],
        look_down=True,
        timeout=17,
        planning_trace_context={
            "planning_period_sec": 0.3,
            "odom": [1.25, -2.5, 0.75],
            "rgb_time": 123.25,
        },
    )


@pytest.mark.parametrize(
    "case_id,instruction,expected_bytes,expected_sha256",
    APPROVED_CASES,
    ids=[case[0] for case in APPROVED_CASES],
)
def test_approved_instruction_metadata_is_pinned_and_mutation_sensitive(
    case_id, instruction, expected_bytes, expected_sha256
):
    instruction_utf8 = instruction.encode("utf-8")
    assert case_id in {"case_1", "case_2", "case_3"}
    assert len(instruction_utf8) == expected_bytes
    assert hashlib.sha256(instruction_utf8).hexdigest() == expected_sha256

    mutated = instruction[:-1] + "X"
    assert len(mutated) == len(instruction)
    assert hashlib.sha256(mutated.encode("utf-8")).hexdigest() != expected_sha256


def test_actual_dual_sys_eval_preserves_posted_json_and_emits_safe_trace_metadata():
    call_order = []
    posted = {}
    traced = {}

    def post(url, *, files, data, timeout):
        call_order.append("post")
        posted.update(url=url, files=files, data=data, timeout=timeout)
        return FakeResponse({"model_output": "SECRET_MODEL_OUTPUT", "request_id": 0})

    def write_trace(event, **fields):
        call_order.append("trace")
        traced.update(event=event, **fields)

    namespace = _base_namespace(post, write_trace)
    functions = _load_actual_functions("dual_sys_eval", namespace=namespace)
    result = _invoke(namespace["dual_sys_eval"])

    assert namespace["dual_sys_eval"].__code__.co_filename == str(CLIENT_PATH)
    assert namespace["dual_sys_eval"].__code__.co_firstlineno == functions["dual_sys_eval"].lineno
    assert call_order == ["trace", "post"]
    assert result == {"model_output": "SECRET_MODEL_OUTPUT", "request_id": 0}

    expected_payload = _expected_payload(
        instruction=APPROVED_CASE_1_INSTRUCTION,
        pose=[1.25, -2.5, 0.75],
        camera_pose=[[1.0, 0.0, 0.0, 1.25], [0.0, 1.0, 0.0, -2.5]],
        intrinsic=[[320.0, 0.0, 160.0], [0.0, 320.0, 120.0], [0.0, 0.0, 1.0]],
        look_down=True,
    )
    expected_json = json.dumps(expected_payload)
    assert posted["data"] == {"json": expected_json}
    assert posted["data"]["json"].encode("utf-8") == expected_json.encode("utf-8")
    assert json.loads(posted["data"]["json"]) == expected_payload
    assert json.loads(posted["data"]["json"])["instruction"] == APPROVED_CASE_1_INSTRUCTION
    assert posted["files"] == {
        "image": ("rgb_image", b"image-secret-" + (b"A" * 128), "image/jpeg"),
        "depth": ("depth_image", b"depth-secret-" + (b"B" * 128), "image/png"),
    }
    assert posted["timeout"] == 17

    assert traced == {
        "event": "planning_request_started",
        "planning_period_sec": 0.3,
        "odom": [1.25, -2.5, 0.75],
        "rgb_time": 123.25,
        "request_id": expected_payload["request_id"],
        "instruction_utf8_sha256": APPROVED_CASE_1_SHA256,
        "instruction_utf8_bytes": APPROVED_CASE_1_UTF8_BYTES,
    }
    trace_json = json.dumps(traced, sort_keys=True)
    for forbidden in (
        APPROVED_CASE_1_INSTRUCTION,
        expected_json,
        posted["url"],
        "secret-token",
        "image-secret",
        "depth-secret",
        "SECRET_MODEL_OUTPUT",
    ):
        assert forbidden not in trace_json
    assert not ({"instruction", "json", "payload", "image", "depth", "url", "token", "model"} & traced.keys())
    assert not re.search(r"[A-Za-z0-9+/]{96,}={0,2}", trace_json)
    assert len(trace_json.encode("utf-8")) < 512
    assert namespace["policy_init"] is False
    assert namespace["http_idx"] == 0
    assert traced["request_id"] == json.loads(posted["data"]["json"])["request_id"]


def test_long_lived_client_partitions_three_cases_across_reset_request_ids():
    posted = []
    written_traces = []

    class MonotonicClock:
        def __init__(self):
            self.value = 1000.0

        def time(self):
            self.value += 0.25
            return self.value

    class TraceHandle:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def write(self, text):
            written_traces.append(text)

    def post(url, *, files, data, timeout):
        payload = json.loads(data["json"])
        posted.append({"url": url, "files": files, "data": data, "timeout": timeout})
        return FakeResponse({"request_id": payload["request_id"]})

    namespace = _base_namespace(post, write_trace=None)
    namespace.update(
        trace_path="/controlled/trace.jsonl",
        desired_v=0.0,
        desired_w=0.0,
        time=MonotonicClock(),
        os=SimpleNamespace(
            path=SimpleNamespace(dirname=lambda path: "/controlled"),
            makedirs=lambda *args, **kwargs: None,
        ),
        open=lambda *args, **kwargs: TraceHandle(),
    )
    _load_actual_functions("_json_default", "_write_trace", "dual_sys_eval", namespace=namespace)

    expected_runs = []
    results = []
    for _ in range(2):
        namespace["policy_init"] = True
        namespace["http_idx"] = -1
        for case_id, instruction, expected_bytes, expected_sha256 in APPROVED_CASES:
            expected_payload = _expected_payload(
                instruction=instruction,
                pose=[1.25, -2.5, 0.75],
                camera_pose=[[1.0, 0.0, 0.0, 1.25], [0.0, 1.0, 0.0, -2.5]],
                intrinsic=[[320.0, 0.0, 160.0], [0.0, 320.0, 120.0], [0.0, 0.0, 1.0]],
                look_down=True,
                reset=namespace["policy_init"],
                http_idx=namespace["http_idx"],
            )
            expected_runs.append(
                (case_id, instruction, expected_bytes, expected_sha256, expected_payload)
            )
            results.append(_invoke(namespace["dual_sys_eval"], instruction=instruction))

    traces = [json.loads(line) for line in written_traces]
    assert len(traces) == len(posted) == len(results) == 6
    assert [trace["request_id"] for trace in traces] == [0, 1, 2, 0, 1, 2]
    assert [trace["instruction_utf8_sha256"] for trace in traces] == [
        case[3] for case in APPROVED_CASES
    ] * 2
    assert {trace["instruction_utf8_sha256"] for trace in traces} == {
        case[3] for case in APPROVED_CASES
    }

    for trace, post_call, result, expected_run in zip(
        traces, posted, results, expected_runs, strict=True
    ):
        case_id, instruction, expected_bytes, expected_sha256, expected_payload = expected_run
        assert case_id in {"case_1", "case_2", "case_3"}
        assert trace["event"] == "planning_request_started"
        assert trace["instruction_utf8_sha256"] == expected_sha256
        assert trace["instruction_utf8_bytes"] == expected_bytes
        assert trace["request_id"] == expected_payload["request_id"] == result["request_id"]
        assert post_call["data"] == {"json": json.dumps(expected_payload)}
        assert post_call["data"]["json"].encode("utf-8") == json.dumps(expected_payload).encode(
            "utf-8"
        )
        assert json.loads(post_call["data"]["json"])["instruction"] == instruction
        assert instruction not in json.dumps(trace, sort_keys=True)

    hash_and_request_id = {
        (trace["instruction_utf8_sha256"], trace["request_id"]) for trace in traces
    }
    partition_keys = {
        (trace["instruction_utf8_sha256"], trace["request_id"], trace["time"])
        for trace in traces
    }
    assert len(hash_and_request_id) == 3
    assert len(partition_keys) == 6

    namespace["policy_init"] = True
    namespace["http_idx"] = -1
    mutated_instruction = APPROVED_CASE_1_INSTRUCTION[:-1] + "X"
    mutated_result = _invoke(namespace["dual_sys_eval"], instruction=mutated_instruction)
    mutated_trace = json.loads(written_traces[-1])
    mutated_payload = json.loads(posted[-1]["data"]["json"])
    expected_mutated_sha256 = hashlib.sha256(mutated_instruction.encode("utf-8")).hexdigest()

    assert mutated_trace["request_id"] == mutated_payload["request_id"] == mutated_result["request_id"] == 0
    assert mutated_trace["instruction_utf8_sha256"] == expected_mutated_sha256
    assert mutated_trace["instruction_utf8_sha256"] != APPROVED_CASE_1_SHA256
    assert mutated_trace["instruction_utf8_sha256"] not in {case[3] for case in APPROVED_CASES}
    assert mutated_trace["instruction_utf8_bytes"] == APPROVED_CASE_1_UTF8_BYTES
    assert mutated_instruction not in json.dumps(mutated_trace, sort_keys=True)


def test_disabled_trace_path_uses_actual_writer_without_changing_payload():
    posted = {}

    def post(url, *, files, data, timeout):
        posted.update(url=url, files=files, data=data, timeout=timeout)
        return FakeResponse({"request_id": 0})

    namespace = _base_namespace(post, write_trace=None)
    namespace.update(
        trace_path="",
        desired_v=0.0,
        desired_w=0.0,
    )
    _load_actual_functions("_json_default", "_write_trace", "dual_sys_eval", namespace=namespace)
    result = _invoke(namespace["dual_sys_eval"])

    expected = _expected_payload(
        instruction=APPROVED_CASE_1_INSTRUCTION,
        pose=[1.25, -2.5, 0.75],
        camera_pose=[[1.0, 0.0, 0.0, 1.25], [0.0, 1.0, 0.0, -2.5]],
        intrinsic=[[320.0, 0.0, 160.0], [0.0, 320.0, 120.0], [0.0, 0.0, 1.0]],
        look_down=True,
    )
    assert result == {"request_id": 0}
    assert posted["data"] == {"json": json.dumps(expected)}
    assert namespace["http_idx"] == 0


def test_http_exception_preserves_payload_request_id_and_existing_state_transitions():
    posted = {}
    traces = []

    class HttpFailure(Exception):
        pass

    failure = HttpFailure("controlled post failure")

    def post(url, *, files, data, timeout):
        posted.update(url=url, files=files, data=data, timeout=timeout)
        raise failure

    def write_trace(event, **fields):
        traces.append({"event": event, **fields})

    namespace = _base_namespace(post, write_trace)
    _load_actual_functions("dual_sys_eval", namespace=namespace)

    with pytest.raises(HttpFailure) as raised:
        _invoke(namespace["dual_sys_eval"])

    expected = _expected_payload(
        instruction=APPROVED_CASE_1_INSTRUCTION,
        pose=[1.25, -2.5, 0.75],
        camera_pose=[[1.0, 0.0, 0.0, 1.25], [0.0, 1.0, 0.0, -2.5]],
        intrinsic=[[320.0, 0.0, 160.0], [0.0, 320.0, 120.0], [0.0, 0.0, 1.0]],
        look_down=True,
    )
    assert raised.value is failure
    assert posted["data"] == {"json": json.dumps(expected)}
    assert traces[0]["request_id"] == expected["request_id"]
    assert namespace["policy_init"] is False
    assert namespace["http_idx"] == -1
    assert namespace["first_running_time"] == 0.0


def test_actual_trace_writer_failure_is_non_fatal_to_post_and_payload():
    posted = {}
    printed = []

    def post(url, *, files, data, timeout):
        posted.update(url=url, files=files, data=data, timeout=timeout)
        return FakeResponse({"request_id": 0})

    def fail_makedirs(*args, **kwargs):
        raise OSError("controlled trace failure")

    namespace = _base_namespace(post, write_trace=None)
    namespace.update(
        trace_path="/controlled/not-written/trace.jsonl",
        desired_v=0.0,
        desired_w=0.0,
        os=SimpleNamespace(
            path=SimpleNamespace(dirname=lambda path: "/controlled/not-written"),
            makedirs=fail_makedirs,
        ),
        print=lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )
    _load_actual_functions("_json_default", "_write_trace", "dual_sys_eval", namespace=namespace)
    result = _invoke(namespace["dual_sys_eval"])

    expected = _expected_payload(
        instruction=APPROVED_CASE_1_INSTRUCTION,
        pose=[1.25, -2.5, 0.75],
        camera_pose=[[1.0, 0.0, 0.0, 1.25], [0.0, 1.0, 0.0, -2.5]],
        intrinsic=[[320.0, 0.0, 160.0], [0.0, 320.0, 120.0], [0.0, 0.0, 1.0]],
        look_down=True,
    )
    assert result == {"request_id": 0}
    assert posted["data"] == {"json": json.dumps(expected)}
    assert any("failed to write InternNav client trace" in message for message in printed)


def test_status_publication_and_source_correlation_remain_partitioned():
    published = []
    traces = []
    manager = SimpleNamespace(
        request_cnt=7,
        odom_cnt=11,
        publish_status=lambda payload: published.append(payload),
    )
    namespace = {
        "manager": manager,
        "http_idx": -1,
        "desired_v": 0.0,
        "desired_w": 0.0,
        "current_control_mode": SimpleNamespace(name="MPC_Mode"),
        "policy_init": True,
        "_write_trace": lambda event, **fields: traces.append({"event": event, **fields}),
    }
    _load_actual_functions("_publish_status", namespace=namespace)
    namespace["_publish_status"](
        "planning_request_started",
        emit_trace=False,
        request_id=0,
        planning_period_sec=0.3,
        odom=[1.25, -2.5, 0.75],
        rgb_time=123.25,
    )

    assert traces == []
    assert published[0]["status"] == "planning_request_started"
    assert published[0]["debug"] == {
        "request_id": 0,
        "planning_period_sec": 0.3,
        "odom": [1.25, -2.5, 0.75],
        "rgb_time": 123.25,
    }

    _, tree = _source_tree()
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    dual_body = functions["dual_sys_eval"].body

    trace_index = next(
        index
        for index, statement in enumerate(dual_body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_write_trace"
        and isinstance(statement.value.args[0], ast.Constant)
        and statement.value.args[0].value == "planning_request_started"
    )
    post_index = next(
        index
        for index, statement in enumerate(dual_body)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and isinstance(statement.value.func.value, ast.Name)
        and statement.value.func.value.id == "requests"
        and statement.value.func.attr == "post"
    )
    assert post_index == trace_index + 1

    planning_calls = [node for node in ast.walk(functions["planning_thread"]) if isinstance(node, ast.Call)]
    started_status = next(
        call
        for call in planning_calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "_publish_status"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "planning_request_started"
    )
    assert next(keyword.value.value for keyword in started_status.keywords if keyword.arg == "emit_trace") is False

    dual_call = next(
        call
        for call in planning_calls
        if isinstance(call.func, ast.Name) and call.func.id == "dual_sys_eval"
    )
    context = next(keyword.value for keyword in dual_call.keywords if keyword.arg == "planning_trace_context")
    assert isinstance(context, ast.Dict)
    assert {key.value for key in context.keys} == {"planning_period_sec", "odom", "rgb_time"}

    response_trace = next(
        call
        for call in planning_calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "_write_trace"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "planning_response_received"
    )
    response_request_id = next(keyword.value for keyword in response_trace.keywords if keyword.arg == "request_id")
    assert isinstance(response_request_id, ast.Name)
    assert response_request_id.id == "next_request_id"
