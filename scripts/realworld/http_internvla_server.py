import argparse
import json
import os
import threading
import time
import traceback
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

app = Flask(__name__)
idx = 0
start_time = time.time()
output_dir = ''
agent_lock = threading.Lock()


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return _jsonable(value.tolist())
    return str(value)


def _normalize_output(output, *, request_id=None, elapsed_sec=None):
    result = {'status': 'internvla_realworld_http_command', 'debug': {}}
    if request_id is not None:
        result['request_id'] = request_id
        result['debug']['request_id'] = request_id
    if elapsed_sec is not None:
        result['debug']['server_compute_sec'] = elapsed_sec

    action = getattr(output, 'output_action', None)
    trajectory = getattr(output, 'output_trajectory', None)
    pixel = getattr(output, 'output_pixel', None)
    if action is not None:
        result['discrete_action'] = _jsonable(action)
    if trajectory is not None:
        result['output_trajectory'] = _jsonable(trajectory)
        # Backward-compatible field for the upstream realworld client.
        result['trajectory'] = result['output_trajectory']
    if pixel is not None:
        result['output_pixel'] = _jsonable(pixel)
        # Backward-compatible field for the upstream realworld client.
        result['pixel_goal'] = result['output_pixel']
    llm_output = getattr(agent, 'llm_output', '')
    if llm_output:
        result['debug']['llm_output'] = str(llm_output)
    result['debug']['system2_episode_idx'] = getattr(agent, 'episode_idx', None)
    result['debug']['system1_output_latent_pending'] = getattr(agent, 'output_latent', None) is not None
    result['debug']['system1_output_action_pending'] = getattr(agent, 'output_action', None) is not None
    return _jsonable(result)


def _write_server_error(stage, exc, *, request_id=None):
    record = {
        'time': time.time(),
        'stage': stage,
        'request_id': request_id,
        'idx': idx,
        'error': repr(exc),
        'traceback': traceback.format_exc(),
    }
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'server_errors.jsonl'), 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + '\n')
    except Exception as log_exc:
        print(f"failed to write InternVLA server error log: {log_exc!r}", flush=True)
    print(f"InternVLA HTTP server error at {stage}: {record['error']}\n{record['traceback']}", flush=True)


def _error_response(stage, exc, *, request_id=None, status_code=200):
    _write_server_error(stage, exc, request_id=request_id)
    return jsonify(_jsonable({
        'status': 'internvla_realworld_http_error',
        'request_id': request_id,
        'discrete_action': [0],
        'debug': {
            'stage': stage,
            'error': repr(exc),
            'traceback': traceback.format_exc(),
        },
    })), status_code


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ready',
        'service': 'internvla_n1_realworld_http',
        'idx': idx,
        'model_path': getattr(args, 'model_path', ''),
        'device': getattr(args, 'device', ''),
    })


@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, output_dir, start_time
    start_time = time.time()
    request_id = None

    try:
        image_file = request.files['image']
        depth_file = request.files['depth']
        json_data = request.form['json']
        data = json.loads(json_data)
        request_id = data.get('request_id', data.get('idx'))

        image = Image.open(image_file.stream)
        image = image.convert('RGB')
        image = np.asarray(image)

        depth = Image.open(depth_file.stream)
        depth = depth.convert('I')
        depth = np.asarray(depth)
        depth = depth.astype(np.float32) / 10000.0
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'invalid RGB image shape: {getattr(image, "shape", None)}')
        if depth.ndim != 2:
            raise ValueError(f'invalid depth image shape: {getattr(depth, "shape", None)}')
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0.0] = 0.0
        print(f"read http data cost {time.time() - start_time}")
    except Exception as exc:
        return _error_response('decode_request', exc, request_id=request_id, status_code=400)

    camera_pose = np.asarray(
        data.get('camera_pose', [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        dtype=np.float32,
    )
    instruction = data.get(
        'instruction',
        "Turn around and walk out of this office. Turn towards your slight right at the chair. Move forward to the walkway and go near the red bin. You can see an open door on your right side, go inside the open door. Stop at the computer monitor",
    )
    intrinsic = np.asarray(data.get('intrinsic', args.camera_intrinsic), dtype=np.float32)
    policy_init = bool(data.get('reset', False))
    look_down = bool(data.get('look_down', False))

    t0 = time.time()
    with agent_lock:
        if policy_init:
            start_time = time.time()
            idx = 0
            output_dir = os.path.join(args.output_dir, 'runs' + datetime.now().strftime('%m-%d-%H%M'))
            os.makedirs(output_dir, exist_ok=True)
            print("init reset model!!!")
            agent.reset()

        try:
            idx += 1
            dual_sys_output = agent.step(
                image, depth, camera_pose, instruction, intrinsic=intrinsic, look_down=look_down
            )
            if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
                look_down = True
                dual_sys_output = agent.step(
                    image, depth, camera_pose, instruction, intrinsic=intrinsic, look_down=look_down
                )
        except Exception as exc:
            return _error_response('agent_step', exc, request_id=request_id, status_code=200)

    t1 = time.time()
    generate_time = t1 - t0
    json_output = _normalize_output(dual_sys_output, request_id=request_id, elapsed_sec=generate_time)
    json_output['debug']['look_down'] = look_down
    json_output['debug']['instruction_length'] = len(str(instruction))
    print(f"dual sys step {generate_time}")
    print(f"json_output {json_output}")
    return jsonify(json_output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1")
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=4)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5801)
    parser.add_argument("--output_dir", type=str, default="/tmp/internvla_http_output")
    args = parser.parse_args()

    args.camera_intrinsic = np.array(
        [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    agent = InternVLAN1AsyncAgent(args)
    agent.step(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640), dtype=np.float32),
        np.eye(4),
        "hello",
        args.camera_intrinsic,
    )
    agent.reset()

    app.run(host=args.host, port=args.port, threaded=True)
