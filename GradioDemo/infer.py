# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import json
import math
import base64
import requests

import torch
import decord
import numpy as np
from PIL import Image, ImageSequence
from torchvision.io import read_image, encode_jpeg
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode


class ConversationModeI18N:
    G = "General"
    D = "Deep Thinking"


class ConversationModeCN:
    G = "常规"
    D = "深度思考"


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def get_resized_hw_for_Navit(
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    max_ratio: int = 200,
    factor: int = 28,
):
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return int(h_bar), int(w_bar)


class SeedVLInfer:

    def __init__(
        self,
        api_key: str,
        base_url:
        str = 'https://ark.cn-beijing.volces.com/api/v3/chat/completions',
        model_id: str = 'doubao-1-5-thinking-vision-pro-250428',
        min_pixels: int = 4 * 28 * 28,
        max_pixels: int = 5120 * 28 * 28,
        video_sampling_strategy: dict = {
            'sampling_fps':
            1,
            'min_n_frames':
            16,
            'max_video_length':
            81920,
            'max_pixels_choices': [
                640 * 28 * 28, 512 * 28 * 28, 384 * 28 * 28, 256 * 28 * 28,
                160 * 28 * 28, 128 * 28 * 28
            ],
            'use_timestamp':
            True,
        },
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model_id = model_id
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.sampling_fps = video_sampling_strategy.get('sampling_fps', 1)
        self.min_n_frames = video_sampling_strategy.get('min_n_frames', 16)
        self.max_video_length = video_sampling_strategy.get(
            'max_video_length', 81920)
        self.max_pixels_choices = video_sampling_strategy.get(
            'max_pixels_choices', [
                640 * 28 * 28, 512 * 28 * 28, 384 * 28 * 28, 256 * 28 * 28,
                160 * 28 * 28, 128 * 28 * 28
            ])
        self.use_timestamp = video_sampling_strategy.get('use_timestamp', True)

    def preprocess_video(self, video_path: str):
        try:
            video_reader = decord.VideoReader(video_path, num_threads=2)
            fps = video_reader.get_avg_fps()
        except decord._ffi.base.DECORDError:
            video_reader = [
                frame.convert('RGB')
                for frame in ImageSequence.Iterator(Image.open(video_path))
            ]
            fps = 1

        length = len(video_reader)
        n_frames = min(
            max(math.ceil(length / fps * self.sampling_fps),
                self.min_n_frames), length)
        frame_indices = np.linspace(0, length - 1,
                                    n_frames).round().astype(int).tolist()
        max_pixels = self.max_pixels
        for round_idx, max_pixels in enumerate(self.max_pixels_choices):
            is_last_round = round_idx == len(self.max_pixels_choices) - 1
            if len(frame_indices
                   ) * max_pixels / 28 / 28 > self.max_video_length:
                if is_last_round:
                    max_frame_num = int(self.max_video_length / max_pixels *
                                        28 * 28)
                    select_ids = np.linspace(
                        0,
                        len(frame_indices) - 1,
                        max_frame_num).round().astype(int).tolist()
                    frame_indices = [
                        frame_indices[select_id] for select_id in select_ids
                    ]
                else:
                    continue
            else:
                break

        if hasattr(video_reader, "get_batch"):
            video_clip = torch.from_numpy(
                video_reader.get_batch(frame_indices).asnumpy()).permute(
                    0, 3, 1, 2)
        else:
            video_clip_array = torch.stack(
                [np.array(video_reader[i]) for i in frame_indices], dim=0)
            video_clip = torch.from_numpy(video_clip_array).permute(0, 3, 1, 2)

        height, width = video_clip.shape[-2:]
        resized_height, resized_width = get_resized_hw_for_Navit(
            height,
            width,
            min_pixels=self.min_pixels,
            max_pixels=max_pixels,
        )
        resized_video_clip = resize(video_clip,
                                    (resized_height, resized_width),
                                    interpolation=InterpolationMode.BICUBIC,
                                    antialias=True)
        if self.use_timestamp:
            resized_video_clip = [
                (round(i / fps, 1), f)
                for i, f in zip(frame_indices, resized_video_clip)
            ]
        return resized_video_clip

    def preprocess_streaming_frame(self, frame: torch.Tensor):
        height, width = frame.shape[-2:]
        resized_height, resized_width = get_resized_hw_for_Navit(
            height,
            width,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels_choices[0],
        )
        resized_frame = resize(frame[None], (resized_height, resized_width),
                               interpolation=InterpolationMode.BICUBIC,
                               antialias=True)[0]
        return resized_frame

    def encode_image(self, image: torch.Tensor) -> str:
        encoded = encode_jpeg(image)
        return base64.b64encode(encoded.numpy()).decode('utf-8')

    def construct_messages(self,
                           inputs: dict,
                           streaming_timestamp: int = None) -> list[dict]:
        content = []
        for i, path in enumerate(inputs.get('files', [])):
            if path.endswith('.mp4'):
                video = self.preprocess_video(video_path=path)
                for frame in video:
                    if self.use_timestamp:
                        timestamp, frame = frame
                        content.append({
                            "type": "text",
                            "text": f'[{timestamp} second]',
                        })
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url":
                            f"data:image/jpeg;base64,{self.encode_image(frame)}",
                            "detail": "high"
                        },
                    })
            else:
                image = read_image(path)
                if path.endswith('.webp'):
                    streaming_timestamp = i
                if streaming_timestamp is not None:
                    image = self.preprocess_streaming_frame(frame=image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url":
                        f"data:image/jpeg;base64,{self.encode_image(image)}",
                        "detail": "high"
                    },
                })
                if streaming_timestamp is not None:
                    content.insert(
                        0, {
                            "type": "text",
                            "text": f'[{streaming_timestamp} second]',
                        })
        query = inputs.get('text', '')
        if query:
            content.append({
                "type": "text",
                "text": query,
            })
        messages = [{
            "role": "user",
            "content": content,
        }]
        if streaming_timestamp == 0:
            messages.insert(
                0, {
                    'role': 'system',
                    'content': self.system_prompts[ConversationModeI18N.P]
                })
        return messages

    def request(self,
                messages,
                thinking: bool = True,
                temperature: float = 1.0):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
            "thinking": {
                "type": "enabled" if thinking else "disabled",
            },
            "temperature": temperature,
        }
        for _ in range(3):
            try:
                requested = requests.post(self.base_url,
                                          headers=headers,
                                          json=payload,
                                          stream=True)
                break
            except Exception as e:
                print(e)
        content, reasoning_content = '', ''
        for line in requested.iter_lines():
            if not line:
                continue
            if line.startswith(b'data:'):
                data = line[len("data: "):]
                if data == b"[DONE]":
                    break
                delta = json.loads(data)['choices'][0]['delta']
                content += delta['content']
                reasoning_content += delta.get('reasoning_content', '')
                yield content, reasoning_content

    def __call__(self,
                 inputs: dict,
                 history: list[dict] = [],
                 mode: str = ConversationModeI18N.D,
                 temperature: float = 1.0):
        messages = self.construct_messages(inputs=inputs)
        updated_history = history + messages
        for response, reasoning in self.request(
                messages=updated_history,
                thinking=mode == ConversationModeI18N.D,
                temperature=temperature):
            if mode == ConversationModeI18N.D:
                response = '<think>' + reasoning + '</think>' + response
            yield response, updated_history + [{
                'role':
                'assistant',
                'content': [{
                    'type': 'text',
                    'text': response
                }]
            }]
