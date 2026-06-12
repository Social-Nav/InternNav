import copy
import itertools
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from collections import OrderedDict

from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLProcessor,
    Qwen2VLImageProcessor,
)
try:
    from transformers import Qwen2VLVideoProcessor
except ImportError:
    Qwen2VLVideoProcessor = None

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM, InternVLAN1ModelConfig
from internnav.model.utils.vln_utils import S2Output, split_and_clean, traj_to_actions

DEFAULT_IMAGE_TOKEN = "<image>"
QWEN_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


def _render_messages_with_qwen_image_tokens(messages):
    rendered = []
    for message in messages:
        role = message.get('role', 'user')
        content = message.get('content', '')
        parts = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'image':
                    parts.append(QWEN_IMAGE_TOKEN)
                elif isinstance(item, dict):
                    text = item.get('text', '')
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            content_text = ' '.join(part for part in parts if part)
        else:
            content_text = str(content)
        rendered.append(f"<|im_start|>{role}\n{content_text}<|im_end|>")
    rendered.append("<|im_start|>assistant\n")
    return "\n".join(rendered)


def _register_internvla_transformers_classes():
    """Teach Transformers Auto* loaders about the InternVLA checkpoint type.

    The released DualVLN checkpoint declares ``model_type=internvla_n1``.  Newer
    Transformers versions reject that custom type unless it is registered before
    processor/model loading.  The actual processor is still the Qwen2.5-VL
    processor used by InternVLA-N1.
    """
    try:
        AutoConfig.register('internvla_n1', InternVLAN1ModelConfig, exist_ok=True)
    except Exception:
        pass
    try:
        AutoModelForCausalLM.register(InternVLAN1ModelConfig, InternVLAN1ForCausalLM, exist_ok=True)
    except Exception:
        pass
    try:
        AutoProcessor.register(InternVLAN1ModelConfig, Qwen2_5_VLProcessor, exist_ok=True)
    except Exception:
        pass


def _resolve_runtime_device(requested_device):
    requested = str(requested_device or 'cpu').strip() or 'cpu'
    strict_device = os.environ.get('INTERNNAV_STRICT_DEVICE', '').strip().lower() in {'1', 'true', 'yes', 'on'}
    try:
        device = torch.device(requested)
    except Exception as exc:
        if strict_device and requested != 'cpu':
            raise RuntimeError(f"Failed to validate requested device '{requested}': {exc}") from exc
        device = torch.device('cpu')
        requested = 'cpu'

    if device.type == 'cuda' and not torch.cuda.is_available():
        if strict_device:
            raise RuntimeError(f"Requested device '{requested}' but CUDA is unavailable")
        return torch.device('cpu'), requested, 'cpu_fallback_no_cuda'
    return device, requested, ('cpu' if device.type == 'cpu' else 'native')


def _load_qwen25_vl_processor(model_path):
    """Load Qwen2.5-VL processor across Transformers processor API variants.

    Newer Transformers versions require a video processor component even when
    the caller only uses image inputs.  The InternVLA-N1 real-world agent calls
    the processor with ``images=...`` only, but the constructor still validates
    that ``video_processor`` is a BaseVideoProcessor.
    """
    try:
        return Qwen2_5_VLProcessor.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
        processor_kwargs = {
            'image_processor': image_processor,
            'tokenizer': tokenizer,
        }
        if Qwen2VLVideoProcessor is not None:
            try:
                processor_kwargs['video_processor'] = Qwen2VLVideoProcessor.from_pretrained(model_path)
            except Exception:
                try:
                    processor_kwargs['video_processor'] = Qwen2VLVideoProcessor()
                except Exception:
                    pass

        try:
            return Qwen2_5_VLProcessor(**processor_kwargs)
        except TypeError:
            processor_kwargs.pop('video_processor', None)
            return Qwen2_5_VLProcessor(**processor_kwargs)


def _model_load_kwargs(device):
    if device.type == 'cpu':
        cpu_dtype_name = os.environ.get('ARENA_INTERNNAV_CPU_DTYPE', 'float32').strip().lower()
        cpu_dtype = torch.bfloat16 if cpu_dtype_name in {'bf16', 'bfloat16'} else torch.float32
        return {
            'torch_dtype': cpu_dtype,
            'attn_implementation': 'eager',
            'low_cpu_mem_usage': True,
        }
    try:
        import flash_attn  # noqa: F401
        attn_implementation = 'flash_attention_2'
    except Exception:
        # Keep real GPU inference usable in lean runtime environments where the
        # checkpoint and CUDA PyTorch are installed but flash-attn is not.
        attn_implementation = 'sdpa'
    return {
        'torch_dtype': torch.bfloat16,
        'attn_implementation': attn_implementation,
        'device_map': {'': str(device)},
    }


def _env_int(name, default, minimum=None, maximum=None):
    try:
        value = int(str(os.environ.get(name, '')).strip() or default)
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    if maximum is not None:
        value = min(int(maximum), value)
    return value


class InternVLAN1AsyncAgent:
    def __init__(self, args):
        _register_internvla_transformers_classes()
        self.device, self.requested_device, self.runtime_mode = _resolve_runtime_device(args.device)
        self._save_root = self._resolve_save_root()
        self._reset_save_dir()
        self.load_kwargs = _model_load_kwargs(self.device)
        config = InternVLAN1ModelConfig.from_pretrained(args.model_path)
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            **self.load_kwargs,
        )
        self.model.eval()
        self.model.to(self.device)

        self.processor = _load_qwen25_vl_processor(args.model_path)
        self.processor.tokenizer.padding_side = 'left'
        if not getattr(self.processor, 'chat_template', None):
            self.processor.chat_template = getattr(self.processor.tokenizer, 'chat_template', None)

        self.resize_w = args.resize_w
        self.resize_h = args.resize_h
        self.num_history = args.num_history
        self.PLAN_STEP_GAP = getattr(args, 'plan_step_gap', 4)
        # Keep the default aligned with the official HabitatVlnEvaluator call
        # path, which decodes up to 128 new tokens.  Runtime deployments can
        # still lower ARENA_INTERNNAV_MAX_NEW_TOKENS if latency dominates.
        self.max_new_tokens = _env_int('ARENA_INTERNNAV_MAX_NEW_TOKENS', 128, minimum=1, maximum=128)

        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.last_generated_token_ids = []
        self.last_digit_groups = []
        self.last_symbolic_action_seq = []
        self.last_output_mode = ""
        self.past_key_values = None
        self.last_s2_idx = -100

        # output
        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None

    def _resolve_save_root(self) -> Path:
        root = os.environ.get('ARENA_INTERNNAV_SAVE_ROOT', '').strip()
        if not root:
            root = os.environ.get('ARENA_INTERNNAV_WORK_DIR', '/tmp/arena_internnav_work').strip()
        return Path(root).expanduser().resolve()

    def _reset_save_dir(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self._save_root / 'test_data' / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = str(save_dir)

    def reset(self):
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.last_generated_token_ids = []
        self.last_digit_groups = []
        self.last_symbolic_action_seq = []
        self.last_output_mode = ""
        self.past_key_values = None

        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        self._reset_save_dir()

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def step_no_infer(self, rgb, depth, pose):
        image = Image.fromarray(rgb).convert('RGB')
        image = image.resize((self.resize_w, self.resize_h))
        self.rgb_list.append(image)
        image.save(f"{self.save_dir}/debug_raw_{self.episode_idx:04d}.jpg")
        self.episode_idx += 1

    def trajectory_tovw(self, trajectory, kp=1.0):
        subgoal = trajectory[-1]
        linear_vel, angular_vel = kp * np.linalg.norm(subgoal[:2]), kp * subgoal[2]
        linear_vel = np.clip(linear_vel, 0, 0.5)
        angular_vel = np.clip(angular_vel, -0.5, 0.5)
        return linear_vel, angular_vel

    def step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        dual_sys_output = S2Output()
        no_output_flag = self.output_action is None and self.output_latent is None
        if (self.episode_idx - self.last_s2_idx > self.PLAN_STEP_GAP) or look_down or no_output_flag:
            self.output_action, self.output_latent, self.output_pixel = self.step_s2(
                rgb, depth, pose, instruction, intrinsic, look_down
            )
            self.last_s2_idx = self.episode_idx
            dual_sys_output.output_pixel = self.output_pixel
            self.pixel_goal_rgb = copy.deepcopy(rgb)
            self.pixel_goal_depth = copy.deepcopy(depth)
        else:
            self.step_no_infer(rgb, depth, pose)

        if self.output_action is not None:
            dual_sys_output.output_action = copy.deepcopy(self.output_action)
            self.output_action = None
        elif self.output_latent is not None:
            processed_pixel_rgb = np.array(Image.fromarray(self.pixel_goal_rgb).resize((224, 224))) / 255
            processed_pixel_depth = np.array(Image.fromarray(self.pixel_goal_depth).resize((224, 224)))
            processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255
            processed_depth = np.array(Image.fromarray(depth).resize((224, 224)))
            rgbs = (
                torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)])
                .unsqueeze(0)
                .to(self.device)
            )
            depths = (
                torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)])
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self.device)
            )
            trajectories = self.step_s1(self.output_latent, rgbs, depths)

            dual_sys_output.output_trajectory = traj_to_actions(trajectories, use_discrate_action=False)

        return dual_sys_output

    def step_s2(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        image = Image.fromarray(rgb).convert('RGB')
        if not look_down:
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
            image.save(f"{self.save_dir}/debug_raw_{self.episode_idx:04d}.jpg")
        else:
            image.save(f"{self.save_dir}/debug_raw_{self.episode_idx:04d}_look_down.jpg")
        if not look_down:
            self.conversation_history = []
            self.past_key_values = None

            sources = copy.deepcopy(self.conversation)
            route_instruction = str(instruction or '').strip().rstrip('.')
            sources[0]["value"] = sources[0]["value"].replace('<instruction>', route_instruction)
            cur_images = self.rgb_list[-1:]
            if self.episode_idx == 0:
                history_id = []
            else:
                history_id = np.unique(np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)).tolist()
                placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'

            history_id = sorted(history_id)
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
            self.episode_idx += 1
        else:
            self.input_images.append(image)
            input_img_id = -1
            assert self.llm_output != "", "Last llm_output should not be empty when look down"
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
            )

        prompt = self.conjunctions[0] + DEFAULT_IMAGE_TOKEN
        sources[0]["value"] += f" {prompt}."
        prompt_instruction = copy.deepcopy(sources[0]["value"])
        parts = split_and_clean(prompt_instruction)

        content = []
        for i in range(len(parts)):
            if parts[i] == "<image>":
                content.append({"type": "image", "image": self.input_images[input_img_id]})
                input_img_id += 1
            else:
                content.append({"type": "text", "text": parts[i]})

        self.conversation_history.append({'role': 'user', 'content': content})

        try:
            text = self.processor.apply_chat_template(
                self.conversation_history,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Some converted InternVLA checkpoints only keep the tokenizer chat
            # template, which does not understand Qwen-VL list content blocks.
            # Flatten the message blocks while preserving image placeholders so
            # the processor still aligns `images=self.input_images` correctly.
            text = _render_messages_with_qwen_image_tokens(self.conversation_history)

        if text.count('<|image_pad|>') != len(self.input_images):
            # The checkpoint tokenizer may ship a text-only chat template; in
            # that case `apply_chat_template` succeeds but drops image blocks,
            # causing Qwen2.5-VL to see zero image tokens while the processor
            # supplies image features.  Re-render explicitly with Qwen's image
            # sentinel tokens so image tokens and image features stay aligned.
            text = _render_messages_with_qwen_image_tokens(self.conversation_history)

        t_processor0 = time.time()
        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt").to(self.device)
        t_processor1 = time.time()
        print(
            f"InternNav step_s2 processor episode={self.episode_idx} images={len(self.input_images)} "
            f"input_tokens={int(inputs.input_ids.shape[1])} cost={t_processor1 - t_processor0:.3f}s",
            file=sys.stderr,
            flush=True,
        )
        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=False,
                # raw_input_ids=copy.deepcopy(inputs.input_ids),
            )

        t1 = time.time()
        self.llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        self.last_generated_token_ids = output_ids[0][inputs.input_ids.shape[1] :].detach().cpu().tolist()
        self.last_digit_groups = [int(c) for c in re.findall(r'\d+', self.llm_output)]
        self.last_symbolic_action_seq = []
        self.last_output_mode = "pixel_goal" if self.last_digit_groups else "symbolic_action"
        with open(f"{self.save_dir}/llm_output_{self.episode_idx:04d}.txt", 'w') as f:
            f.write(self.llm_output)
        self.last_output_ids = copy.deepcopy(output_ids[0])
        self.past_key_values = None
        print(
            f"output {self.episode_idx} {self.llm_output!r} generate_cost={t1 - t0:.3f}s "
            f"max_new_tokens={self.max_new_tokens}",
            file=sys.stderr,
            flush=True,
        )
        if bool(re.search(r'\d', self.llm_output)):
            coord = self.last_digit_groups
            if len(coord) < 2:
                self.last_output_mode = "invalid_digit_output"
                return [], None, None
            pixel_goal = [int(coord[1]), int(coord[0])]
            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
            pixel_values = inputs.pixel_values
            t0 = time.time()
            with torch.no_grad():
                traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)
                print(
                    f"InternNav generate_latents episode={self.episode_idx} cost={time.time() - t0:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )
                return None, traj_latents, pixel_goal

        else:
            action_seq = self.parse_actions(self.llm_output)
            self.last_symbolic_action_seq = list(action_seq)
            return action_seq, None, None

    def step_s1(self, latent, rgb, depth):
        all_trajs = self.model.generate_traj(latent, rgb, depth)
        return all_trajs
