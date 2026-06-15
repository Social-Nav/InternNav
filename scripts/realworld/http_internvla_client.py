import copy
import argparse
import io
import json
import math
import os
import threading
import time
from collections import deque
from enum import Enum

import numpy as np
import rclpy
import requests
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from PIL import Image as PIL_Image
from PIL import ImageDraw
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Int16, String

frame_data = {}
frame_idx = 0
# user-specific
from controllers import Mpc_controller, PID_controller
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from thread_utils import ReadWriteLock


class ControlMode(Enum):
    PID_Mode = 1
    MPC_Mode = 2


# global variable
policy_init = True
mpc = None
pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.6, max_w=0.5)
http_idx = -1
first_running_time = 0.0
last_pixel_goal = None
last_s2_step = -1
manager = None
current_control_mode = ControlMode.MPC_Mode
trajs_in_world = None
http_url = 'http://127.0.0.1:5801/eval_dual'
latest_instruction = ''
latest_intrinsic = None
force_look_down = False
last_readiness_log_time = 0.0
trace_path = ''
last_overlay_publish_time = 0.0

desired_v, desired_w = 0.0, 0.0
rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
mpc_rw_lock = ReadWriteLock()


def dual_sys_eval(
    image_bytes,
    depth_bytes,
    front_image_bytes,
    url='http://127.0.0.1:5801/eval_dual',
    *,
    instruction='',
    pose=None,
    camera_pose=None,
    intrinsic=None,
    look_down=False,
    timeout=100,
):
    global policy_init, http_idx, first_running_time
    data = {
        "reset": policy_init,
        "idx": http_idx,
        "request_id": http_idx + 1,
        "instruction": instruction,
        "pose": pose,
        "camera_pose": camera_pose,
        "intrinsic": intrinsic,
        "look_down": bool(look_down),
        "client": "internnav_realworld_ros2_http_client",
    }
    json_data = json.dumps(data)

    policy_init = False
    files = {
        'image': ('rgb_image', image_bytes, 'image/jpeg'),
        'depth': ('depth_image', depth_bytes, 'image/png'),
    }
    start = time.time()
    response = requests.post(url, files=files, data={'json': json_data}, timeout=timeout)
    response.raise_for_status()
    print(f"response {response.text}")
    http_idx += 1
    if http_idx == 0:
        first_running_time = time.time()
    print(f"idx: {http_idx} after http {time.time() - start}")

    return json.loads(response.text)


def control_thread():
    global desired_v, desired_w
    while True:
        global current_control_mode
        if manager is None or not getattr(manager, 'episode_started', False):
            if desired_v != 0.0 or desired_w != 0.0:
                desired_v, desired_w = 0.0, 0.0
                manager.move(0.0, 0.0, 0.0)
            time.sleep(0.1)
            continue

        if current_control_mode == ControlMode.MPC_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            if mpc is not None and manager is not None and odom is not None:
                local_mpc = mpc
                opt_u_controls, opt_x_states = local_mpc.solve(np.array(odom))
                v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]

                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)
        elif current_control_mode == ControlMode.PID_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
            vel = manager.vel.copy() if manager.vel is not None else None
            homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None

            if homo_odom is not None and vel is not None and homo_goal is not None:
                v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
                if v < 0.0:
                    v = 0.0
                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)

        time.sleep(0.1)


def _reset_policy_state(reason='reset'):
    global policy_init, mpc, http_idx, first_running_time, last_pixel_goal, last_s2_step
    global current_control_mode, trajs_in_world, desired_v, desired_w, frame_data
    policy_init = True
    mpc = None
    http_idx = -1
    first_running_time = 0.0
    last_pixel_goal = None
    last_s2_step = -1
    current_control_mode = ControlMode.PID_Mode
    trajs_in_world = None
    desired_v, desired_w = 0.0, 0.0
    frame_data = {}
    print(f"official InternNav client reset policy state: {reason}")


def _stop_robot(reason='stop'):
    global mpc, current_control_mode, desired_v, desired_w, trajs_in_world
    desired_v, desired_w = 0.0, 0.0
    trajs_in_world = None
    mpc_rw_lock.acquire_write()
    try:
        mpc = None
    finally:
        mpc_rw_lock.release_write()
    current_control_mode = ControlMode.PID_Mode
    if manager is not None:
        manager.stop_motion(reason)


def _readiness_missing(odom_infer, rgb_bytes, depth_bytes):
    missing = []
    if manager is None or not getattr(manager, 'episode_started', False):
        missing.append('eval_ready_episode')
    if odom_infer is None:
        missing.append('odom')
    if rgb_bytes is None:
        missing.append('rgb')
    if depth_bytes is None:
        missing.append('depth')
    if latest_intrinsic is None:
        missing.append('camera_info')
    if not str(latest_instruction or '').strip():
        missing.append('instruction')
    return missing


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _write_trace(event, **fields):
    if not trace_path:
        return
    record = {
        'time': time.time(),
        'event': event,
        'http_idx': http_idx,
        'desired_v': desired_v,
        'desired_w': desired_w,
        **fields,
    }
    try:
        os.makedirs(os.path.dirname(trace_path) or '.', exist_ok=True)
        with open(trace_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, default=_json_default, ensure_ascii=False) + '\n')
    except Exception as exc:
        print(f"failed to write InternNav client trace: {exc!r}")


def _publish_status(status, **debug):
    if manager is None:
        return
    payload = {
        'status': status,
        'http_idx': http_idx,
        'request_cnt': getattr(manager, 'request_cnt', 0),
        'odom_cnt': getattr(manager, 'odom_cnt', 0),
        'desired_v': desired_v,
        'desired_w': desired_w,
        'control_mode': current_control_mode.name,
        'policy_init': policy_init,
        'debug': debug,
    }
    manager.publish_status(payload)
    _write_trace(status, **debug)


def _short_json(value, limit=96):
    try:
        text = json.dumps(value, default=_json_default, ensure_ascii=False)
    except Exception:
        text = str(value)
    return text if len(text) <= limit else text[: max(0, limit - 3)] + '...'


def _action_label(actions):
    if actions is None:
        return 'none'
    labels = {0: 'STOP', 1: 'FORWARD', 2: 'TURN_LEFT', 3: 'TURN_RIGHT', 5: 'NOOP', 9: 'NOOP'}
    if isinstance(actions, (list, tuple)):
        return '[' + ', '.join(labels.get(int(a), str(a)) for a in actions if isinstance(a, (int, float, np.integer, np.floating))) + ']'
    try:
        return labels.get(int(actions), str(actions))
    except Exception:
        return str(actions)


def _extract_xy_pairs(values):
    points = []
    if not isinstance(values, (list, tuple)):
        return points
    for item in values:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                points.append((float(item[0]), float(item[1])))
            except Exception:
                continue
    return points


def _draw_polyline(draw, points, origin, scale, color, width=3):
    if len(points) < 2:
        return
    projected = []
    ox, oy = origin
    for x, y in points:
        # Robot frame: +x forward. Draw forward upward and +y to the left.
        projected.append((int(ox - y * scale), int(oy - x * scale)))
    try:
        draw.line(projected, fill=color, width=width)
        for px, py in projected[-3:]:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
    except Exception:
        pass


def _publish_debug_overlay(response, *, odom_infer=None, rgb_image=None, trajectory=None, mode='unknown'):
    global last_overlay_publish_time
    if manager is None or manager.debug_overlay_pub is None:
        return
    now = time.time()
    min_period = 1.0 / max(float(getattr(manager, 'visualization_rate_hz', 5.0) or 5.0), 0.1)
    if now - last_overlay_publish_time < min_period:
        return
    if rgb_image is None:
        return

    try:
        base = np.asarray(rgb_image, dtype=np.uint8)
        image = PIL_Image.fromarray(base).convert('RGB')
        draw = ImageDraw.Draw(image, 'RGBA')
        width, height = image.size
        panel_h = 132
        draw.rectangle((0, 0, width, panel_h), fill=(0, 0, 0, 184), outline=(0, 220, 255, 255), width=2)

        debug = response.get('debug') if isinstance(response, dict) else {}
        trajectory_points = _extract_xy_pairs(trajectory or response.get('output_trajectory') or response.get('trajectory')) if isinstance(response, dict) else []
        discrete_action = response.get('discrete_action') if isinstance(response, dict) else None
        pixel_goal = response.get('output_pixel', response.get('pixel_goal')) if isinstance(response, dict) else None

        lines = [
            'InternNav real HTTP async agent overlay',
            f"System-1/action: mode={mode} control={current_control_mode.name} v={desired_v:.3f} w={desired_w:.3f}",
            f"System-1/traj: len={len(trajectory_points)} endpoint={_short_json(trajectory_points[-1] if trajectory_points else None, 64)}",
            f"System-2/output: discrete={_action_label(discrete_action)} pixel={_short_json(pixel_goal, 64)}",
            f"HTTP idx={http_idx} request_cnt={getattr(manager, 'request_cnt', 0)} server={float((debug or {}).get('server_compute_sec') or 0.0):.2f}s",
        ]
        if debug and debug.get('llm_output'):
            lines.append('LLM: ' + str(debug.get('llm_output'))[:110])
        if debug:
            lines.append(
                'pending: '
                f"S1_latent={bool(debug.get('system1_output_latent_pending'))} "
                f"S1_action={bool(debug.get('system1_output_action_pending'))} "
                f"S2_ep={debug.get('system2_episode_idx')}"
            )
        for idx, line in enumerate(lines[:7]):
            draw.text((12, 10 + idx * 18), line, fill=(255, 255, 255, 255))

        # Draw the latest System-1 local trajectory in a small robot-frame inset.
        inset = (width - 170, panel_h + 10, width - 10, panel_h + 170)
        draw.rectangle(inset, fill=(0, 0, 0, 148), outline=(255, 200, 0, 255), width=2)
        cx = (inset[0] + inset[2]) // 2
        cy = inset[3] - 18
        draw.line((cx, inset[1] + 10, cx, inset[3] - 8), fill=(80, 80, 80, 255), width=1)
        draw.line((inset[0] + 10, cy, inset[2] - 10, cy), fill=(80, 80, 80, 255), width=1)
        draw.polygon([(cx, cy - 10), (cx - 7, cy + 7), (cx + 7, cy + 7)], fill=(255, 80, 80, 255))
        draw.text((inset[0] + 8, inset[1] + 6), 'S1 traj', fill=(255, 220, 0, 255))
        _draw_polyline(draw, trajectory_points, (cx, cy), 28.0, (0, 220, 255, 255), width=3)

        if pixel_goal and isinstance(pixel_goal, (list, tuple)) and len(pixel_goal) >= 2:
            try:
                px, py = float(pixel_goal[0]), float(pixel_goal[1])
                # Accept normalized [0,1] or image-space pixel coordinates.
                if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0:
                    px, py = px * width, py * height
                draw.ellipse((px - 8, py - 8, px + 8, py + 8), outline=(255, 255, 0, 255), width=3)
                draw.line((px - 14, py, px + 14, py), fill=(255, 255, 0, 255), width=2)
                draw.line((px, py - 14, px, py + 14), fill=(255, 255, 0, 255), width=2)
            except Exception:
                pass

        msg = manager.cv_bridge.cv2_to_imgmsg(np.asarray(image, dtype=np.uint8), encoding='rgb8')
        msg.header.stamp = manager.get_clock().now().to_msg()
        msg.header.frame_id = 'internnav_debug_overlay'
        manager.debug_overlay_pub.publish(msg)
        if manager.action_overlay_pub is not None:
            manager.action_overlay_pub.publish(msg)
        last_overlay_publish_time = now
    except Exception as exc:
        print(f"failed to publish InternNav debug overlay: {exc!r}")


def planning_thread():
    global trajs_in_world, last_readiness_log_time

    while True:
        start_time = time.time()
        DESIRED_TIME = 0.3
        time.sleep(0.05)

        if not manager.new_image_arrived:
            time.sleep(0.01)
            continue
        manager.new_image_arrived = False
        rgb_depth_rw_lock.acquire_read()
        rgb_bytes = copy.deepcopy(manager.rgb_bytes)
        depth_bytes = copy.deepcopy(manager.depth_bytes)
        infer_rgb = copy.deepcopy(manager.rgb_image)
        infer_depth = copy.deepcopy(manager.depth_image)
        rgb_time = manager.rgb_time
        rgb_depth_rw_lock.release_read()
        odom_rw_lock.acquire_read()
        min_diff = 1e10
        # time_diff = 1e10
        odom_infer = None
        for odom in manager.odom_queue:
            diff = abs(odom[0] - rgb_time)
            if diff < min_diff:
                min_diff = diff
                odom_infer = copy.deepcopy(odom[1])
                # time_diff = odom[0] - rgb_time
        # odom_time = manager.odom_timestamp
        odom_rw_lock.release_read()

        missing = _readiness_missing(odom_infer, rgb_bytes, depth_bytes)
        if not missing:
            global frame_data
            frame_data[http_idx] = {
                'infer_rgb': copy.deepcopy(infer_rgb),
                'infer_depth': copy.deepcopy(infer_depth),
                'infer_odom': copy.deepcopy(odom_infer),
            }
            if len(frame_data) > 100:
                del frame_data[min(frame_data.keys())]
            camera_pose = None
            if odom_infer is not None:
                x_, y_, yaw_ = odom_infer[0], odom_infer[1], odom_infer[2]
                camera_pose = [
                    [float(np.cos(yaw_)), float(-np.sin(yaw_)), 0.0, float(x_)],
                    [float(np.sin(yaw_)), float(np.cos(yaw_)), 0.0, float(y_)],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            try:
                _publish_status('inference_started', odom=odom_infer, rgb_time=rgb_time)
                response = dual_sys_eval(
                    rgb_bytes,
                    depth_bytes,
                    None,
                    url=http_url,
                    instruction=latest_instruction,
                    pose=odom_infer,
                    camera_pose=camera_pose,
                    intrinsic=latest_intrinsic,
                    look_down=force_look_down,
                )
            except Exception as exc:
                print(f"skip planning after HTTP inference error: {exc!r}")
                _publish_status('exception', error=repr(exc))
                _stop_robot('http_inference_error')
                time.sleep(0.5)
                continue

            manager.publish_model_output(response)

            global current_control_mode
            traj_len = 0.0
            if 'trajectory' in response or 'output_trajectory' in response:
                trajectory = response.get('output_trajectory', response.get('trajectory'))
                if not trajectory or len(trajectory) <= 3:
                    print(f"skip invalid/short trajectory response: {trajectory}")
                    _publish_status('invalid_or_short_trajectory', trajectory=trajectory, raw_response=response)
                    _stop_robot('invalid_or_short_trajectory')
                    time.sleep(0.1)
                    continue
                trajs_in_world = []
                odom = odom_infer
                traj_len = np.linalg.norm(trajectory[-1][:2])
                print(f"traj len {traj_len}")
                for i, traj in enumerate(trajectory):
                    if i < 3:
                        continue
                    x_, y_, yaw_ = odom[0], odom[1], odom[2]

                    w_T_b = np.array(
                        [
                            [np.cos(yaw_), -np.sin(yaw_), 0, x_],
                            [np.sin(yaw_), np.cos(yaw_), 0, y_],
                            [0.0, 0.0, 1.0, 0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    )
                    w_P = (w_T_b @ (np.array([traj[0], traj[1], 0.0, 1.0])).T)[:2]
                    trajs_in_world.append(w_P)
                trajs_in_world = np.array(trajs_in_world)
                print(f"{time.time()} update traj")

                manager.last_trajs_in_world = trajs_in_world
                mpc_rw_lock.acquire_write()
                global mpc
                if mpc is None:
                    mpc = Mpc_controller(np.array(trajs_in_world))
                else:
                    mpc.update_ref_traj(np.array(trajs_in_world))
                manager.request_cnt += 1
                mpc_rw_lock.release_write()
                current_control_mode = ControlMode.MPC_Mode
                _publish_debug_overlay(
                    response,
                    odom_infer=odom_infer,
                    rgb_image=infer_rgb,
                    trajectory=trajectory,
                    mode='trajectory',
                )
                _publish_status(
                    'trajectory',
                    trajectory_len=len(trajectory),
                    trajectory_endpoint=trajectory[-1],
                    trajectory_norm=traj_len,
                )
            elif 'discrete_action' in response:
                actions = response['discrete_action']
                if actions == [0] or actions == 0:
                    print('official InternNav client received STOP/no-action; publishing zero velocity')
                    _publish_debug_overlay(
                        response,
                        odom_infer=odom_infer,
                        rgb_image=infer_rgb,
                        trajectory=None,
                        mode='stop',
                    )
                    _publish_status('stop', discrete_action=actions, raw_response=response)
                    _stop_robot('internnav_stop')
                elif actions != [5] and actions != [9]:
                    manager.incremental_change_goal(actions)
                    current_control_mode = ControlMode.PID_Mode
                    _publish_debug_overlay(
                        response,
                        odom_infer=odom_infer,
                        rgb_image=infer_rgb,
                        trajectory=None,
                        mode='discrete_action',
                    )
                    _publish_status('discrete_action', discrete_action=actions)
            else:
                _publish_debug_overlay(
                    response,
                    odom_infer=odom_infer,
                    rgb_image=infer_rgb,
                    trajectory=None,
                    mode='unknown_response',
                )
                _publish_status('unknown_response', raw_response=response)
        else:
            now = time.time()
            if now - last_readiness_log_time > 2.0:
                print(f"skip planning until real inputs are ready; missing={missing}")
                _publish_status('required_inputs_not_ready', missing_inputs=missing)
                last_readiness_log_time = now
            time.sleep(0.1)

        time.sleep(max(0, DESIRED_TIME - (time.time() - start_time)))


class Go2Manager(Node):
    def __init__(self, args):
        super().__init__('go2_manager')

        rgb_down_sub = Subscriber(self, Image, args.rgb_topic)
        depth_down_sub = Subscriber(self, Image, args.depth_topic)

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.syncronizer = ApproximateTimeSynchronizer([rgb_down_sub, depth_down_sub], 1, 0.1)
        self.syncronizer.registerCallback(self.rgb_depth_down_callback)
        self.odom_sub = self.create_subscription(Odometry, args.odom_topic, self.odom_callback, qos_profile)
        instruction_qos = QoSProfile(depth=1)
        instruction_qos.reliability = ReliabilityPolicy.RELIABLE
        instruction_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.instruction_sub = self.create_subscription(String, args.instruction_topic, self.instruction_callback, instruction_qos)
        self.camera_info_sub = self.create_subscription(CameraInfo, args.camera_info_topic, self.camera_info_callback, 10)
        goal_qos = QoSProfile(depth=1)
        goal_qos.reliability = ReliabilityPolicy.RELIABLE
        goal_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.navigation_goal_sub = self.create_subscription(
            PoseStamped,
            args.navigation_goal_topic,
            self.navigation_goal_callback,
            goal_qos,
        )
        eval_ready_qos = QoSProfile(depth=1)
        eval_ready_qos.reliability = ReliabilityPolicy.RELIABLE
        eval_ready_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.eval_ready_sub = self.create_subscription(String, args.eval_ready_topic, self.eval_ready_callback, eval_ready_qos)
        self.task_reset_sub = self.create_subscription(Int16, args.task_reset_topic, self.task_reset_callback, 10)
        self.scenario_reset_sub = None
        if args.scenario_reset_topic and args.scenario_reset_topic != args.task_reset_topic:
            self.scenario_reset_sub = self.create_subscription(Int16, args.scenario_reset_topic, self.task_reset_callback, 10)

        # publisher
        self.control_pub = self.create_publisher(Twist, args.cmd_vel_topic, 5)
        status_qos = QoSProfile(depth=1)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.status_pub = self.create_publisher(String, args.status_topic, status_qos) if args.status_topic else None
        self.model_output_pub = self.create_publisher(String, args.model_output_topic, 10) if args.model_output_topic else None
        self.visualization_rate_hz = max(float(args.visualization_rate_hz), 0.1)
        self.debug_overlay_pub = None
        self.action_overlay_pub = None
        if args.enable_visualization and args.visualization_topic:
            self.debug_overlay_pub = self.create_publisher(Image, args.visualization_topic, 10)
        if args.enable_visualization and args.action_visualization_topic:
            self.action_overlay_pub = self.create_publisher(Image, args.action_visualization_topic, 10)

        # class member variable
        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.rgb_bytes = None
        self.depth_image = None
        self.depth_bytes = None
        self.rgb_forward_image = None
        self.rgb_forward_bytes = None
        self.new_image_arrived = False
        self.new_vis_image_arrived = False
        self.rgb_time = 0.0

        self.odom = None
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.request_cnt = 0
        self.odom_cnt = 0
        self.odom_queue = deque(maxlen=50)
        self.odom_timestamp = 0.0

        self.last_s2_step = -1
        self.last_trajs_in_world = None
        self.last_all_trajs_in_world = None
        self.homo_odom = None
        self.homo_goal = None
        self.vel = None
        self.last_instruction = ''
        self.episode_started = False
        self.eval_ready_episode = None

        self.publish_status(
            {
                'status': 'client_ready',
                'http_idx': http_idx,
                'request_cnt': self.request_cnt,
                'odom_cnt': self.odom_cnt,
                'desired_v': desired_v,
                'desired_w': desired_w,
                'control_mode': current_control_mode.name,
                'policy_init': policy_init,
                'debug': {
                    'ready': True,
                    'episode_started': self.episode_started,
                    'rgb_topic': args.rgb_topic,
                    'depth_topic': args.depth_topic,
                    'odom_topic': args.odom_topic,
                    'instruction_topic': args.instruction_topic,
                    'task_reset_topic': args.task_reset_topic,
                    'scenario_reset_topic': args.scenario_reset_topic,
                    'navigation_goal_topic': args.navigation_goal_topic,
                    'eval_ready_topic': args.eval_ready_topic,
                    'visualization_topic': args.visualization_topic,
                    'action_visualization_topic': args.action_visualization_topic,
                    'visualization_rate_hz': self.visualization_rate_hz,
                },
            }
        )
        _write_trace(
            'client_ready',
            ready=True,
            episode_started=self.episode_started,
            rgb_topic=args.rgb_topic,
            depth_topic=args.depth_topic,
            odom_topic=args.odom_topic,
            instruction_topic=args.instruction_topic,
            task_reset_topic=args.task_reset_topic,
            scenario_reset_topic=args.scenario_reset_topic,
            navigation_goal_topic=args.navigation_goal_topic,
            eval_ready_topic=args.eval_ready_topic,
            visualization_topic=args.visualization_topic,
            action_visualization_topic=args.action_visualization_topic,
            visualization_rate_hz=self.visualization_rate_hz,
        )

    def reset_runtime_state(self, reason):
        _reset_policy_state(reason)
        rgb_depth_rw_lock.acquire_write()
        try:
            self.rgb_image = None
            self.rgb_bytes = None
            self.depth_image = None
            self.depth_bytes = None
            self.rgb_forward_image = None
            self.rgb_forward_bytes = None
            self.new_image_arrived = False
            self.new_vis_image_arrived = False
            self.rgb_time = 0.0
        finally:
            rgb_depth_rw_lock.release_write()
        odom_rw_lock.acquire_write()
        try:
            self.odom = None
            self.odom_queue.clear()
            self.odom_timestamp = 0.0
            self.linear_vel = 0.0
            self.angular_vel = 0.0
            self.homo_odom = None
            self.homo_goal = None
            self.vel = None
        finally:
            odom_rw_lock.release_write()
        self.request_cnt = 0
        self.odom_cnt = 0
        self.last_s2_step = -1
        self.last_trajs_in_world = None
        self.last_all_trajs_in_world = None
        self.stop_motion(reason)

    def stop_motion(self, reason='stop'):
        self.homo_goal = self.homo_odom.copy() if self.homo_odom is not None else None
        print(f"publish zero cmd_vel: {reason}")
        self.move(0.0, 0.0, 0.0)

    def publish_status(self, payload):
        if self.status_pub is None:
            return
        msg = String()
        msg.data = json.dumps(payload, default=_json_default, ensure_ascii=False)
        self.status_pub.publish(msg)

    def publish_model_output(self, response):
        if self.model_output_pub is None:
            return
        msg = String()
        msg.data = json.dumps(response, default=_json_default, ensure_ascii=False)
        self.model_output_pub.publish(msg)

    def rgb_forward_callback(self, rgb_msg):
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        self.rgb_forward_image = raw_image
        image = PIL_Image.fromarray(self.rgb_forward_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        self.rgb_forward_bytes = image_bytes
        self.new_vis_image_arrived = True
        self.new_image_arrived = True

    def rgb_depth_down_callback(self, rgb_msg, depth_msg):
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        self.rgb_image = raw_image
        image = PIL_Image.fromarray(self.rgb_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        raw_depth[np.isnan(raw_depth)] = 0
        raw_depth[np.isinf(raw_depth)] = 0
        if raw_depth.dtype == np.uint16:
            self.depth_image = raw_depth.astype(np.float32) / 1000.0
        else:
            self.depth_image = raw_depth.astype(np.float32)
        self.depth_image -= 0.0
        self.depth_image[np.where(self.depth_image < 0)] = 0
        depth = (np.clip(self.depth_image * 10000.0, 0, 65535)).astype(np.uint16)
        depth = PIL_Image.fromarray(depth)
        depth_bytes = io.BytesIO()
        depth.save(depth_bytes, format='PNG')
        depth_bytes.seek(0)

        rgb_depth_rw_lock.acquire_write()
        self.rgb_bytes = image_bytes

        self.rgb_time = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec / 1.0e9
        self.last_rgb_time = self.rgb_time

        self.depth_bytes = depth_bytes
        self.depth_time = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec / 1.0e9
        self.last_depth_time = self.depth_time

        rgb_depth_rw_lock.release_write()

        self.new_vis_image_arrived = True
        self.new_image_arrived = True

    def instruction_callback(self, msg):
        global latest_instruction
        instruction = str(msg.data or '')
        if instruction and instruction != self.last_instruction:
            self.last_instruction = instruction
            latest_instruction = instruction
            self.reset_runtime_state('instruction_changed')
        else:
            latest_instruction = instruction

    def task_reset_callback(self, msg):
        reset_episode = getattr(msg, 'data', None)
        # /task_reset and /eval_ready are published back-to-back by Arena on
        # different subscriptions.  DDS can deliver eval_ready(stage=episode,
        # ready=true) before the matching task_reset sample.  In that ordering,
        # clearing episode_started here leaves the client permanently gated
        # until the next episode.  Treat a task_reset for the already-ready
        # episode as a late boundary notification and keep the ready state.
        if self.episode_started and self.eval_ready_episode == reset_episode:
            _publish_status(
                'task_reset_after_episode_ready',
                ready=True,
                episode_started=True,
                task_reset=reset_episode,
            )
            return

        self.episode_started = False
        self.reset_runtime_state(f'task_reset:{getattr(msg, "data", "")})')
        _publish_status(
            'resetting',
            ready=False,
            episode_started=self.episode_started,
            task_reset=reset_episode,
        )

    def navigation_goal_callback(self, msg):
        if not self.episode_started:
            return
        _publish_status(
            'navigation_goal_seen',
            ready=True,
            episode_started=self.episode_started,
            goal={
                'x': float(msg.pose.position.x),
                'y': float(msg.pose.position.y),
            },
        )

    def eval_ready_callback(self, msg):
        try:
            payload = json.loads(msg.data or '{}')
        except Exception:
            return
        if payload.get('stage') != 'episode':
            return
        ready = bool(payload.get('ready'))
        episode = payload.get('episode')
        reason = str((payload.get('details') or {}).get('reason', ''))
        if not ready:
            self.episode_started = False
            self.eval_ready_episode = None
            self.reset_runtime_state(f'eval_ready:false:{reason}')
            _publish_status('resetting', ready=False, episode_started=False, reason=reason, episode=episode)
            return
        self.reset_runtime_state(f'eval_ready:true:{reason}')
        self.eval_ready_episode = episode
        self.episode_started = True
        _publish_status('episode_ready', ready=True, episode_started=True, reason=reason, episode=episode)

    def camera_info_callback(self, msg):
        global latest_intrinsic
        if len(msg.k) >= 9:
            latest_intrinsic = [
                [float(msg.k[0]), float(msg.k[1]), float(msg.k[2])],
                [float(msg.k[3]), float(msg.k[4]), float(msg.k[5])],
                [float(msg.k[6]), float(msg.k[7]), float(msg.k[8])],
            ]

    def odom_callback(self, msg):
        self.odom_cnt += 1
        odom_rw_lock.acquire_write()
        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * zz * ww, 1 - 2 * zz * zz)
        self.odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1.0e9
        self.odom_queue.append((odom_stamp, copy.deepcopy(self.odom)))
        self.odom_timestamp = odom_stamp
        self.linear_vel = msg.twist.twist.linear.x
        self.angular_vel = msg.twist.twist.angular.z
        odom_rw_lock.release_write()

        R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

        if self.odom_cnt == 1:
            self.homo_goal = self.homo_odom.copy()

    def incremental_change_goal(self, actions):
        if self.homo_goal is None:
            raise ValueError("Please initialize homo_goal before change it!")
        homo_goal = self.homo_odom.copy()
        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action == 2:
                angle = math.radians(15)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:
                angle = -math.radians(15.0)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
        self.homo_goal = homo_goal

    def move(self, vx, vy, vyaw):
        request = Twist()
        request.linear.x = vx
        request.linear.y = 0.0
        request.angular.z = vyaw

        self.control_pub.publish(request)


def parse_args():
    parser = argparse.ArgumentParser(description='InternVLA-N1 realworld ROS2 HTTP client')
    parser.add_argument('--url', default='http://127.0.0.1:5801/eval_dual')
    parser.add_argument('--rgb-topic', default='/camera/camera/color/image_raw')
    parser.add_argument('--depth-topic', default='/camera/camera/aligned_depth_to_color/image_raw')
    parser.add_argument('--odom-topic', default='/odom_bridge')
    parser.add_argument('--cmd-vel-topic', default='/cmd_vel_bridge')
    parser.add_argument('--instruction-topic', default='/task_generator_node/vln_instruction')
    parser.add_argument('--camera-info-topic', default='/camera/camera/color/camera_info')
    parser.add_argument('--navigation-goal-topic', default='/task_generator_node/Ai2_Bot2/episode_goal_pose')
    parser.add_argument('--eval-ready-topic', default='/task_generator_node/eval_ready')
    parser.add_argument('--task-reset-topic', default='/task_generator_node/task_reset')
    parser.add_argument('--scenario-reset-topic', default='')
    parser.add_argument('--status-topic', default='')
    parser.add_argument('--model-output-topic', default='')
    parser.add_argument('--visualization-topic', default='')
    parser.add_argument('--action-visualization-topic', default='')
    parser.add_argument('--visualization-rate-hz', type=float, default=5.0)
    parser.add_argument('--enable-visualization', action='store_true')
    parser.add_argument('--disable-visualization', dest='enable_visualization', action='store_false')
    parser.set_defaults(enable_visualization=False)
    parser.add_argument('--trace-path', default='')
    parser.add_argument('--look-down', action='store_true')
    args, ros_args = parser.parse_known_args()
    return args, ros_args


if __name__ == '__main__':
    args, ros_args = parse_args()
    http_url = args.url
    force_look_down = bool(args.look_down)
    trace_path = args.trace_path
    control_thread_instance = threading.Thread(target=control_thread)
    planning_thread_instance = threading.Thread(target=planning_thread)
    control_thread_instance.daemon = True
    planning_thread_instance.daemon = True
    rclpy.init(args=ros_args)

    try:
        manager = Go2Manager(args)

        control_thread_instance.start()
        planning_thread_instance.start()

        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    finally:
        manager.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
