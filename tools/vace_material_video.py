"""Full-timeline media adapter for the masked VACE material-edit leaf."""
from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

import av
import cv2
import numpy as np


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames or fps <= 0:
        raise ValueError(f'Cannot decode video: {path}')
    return frames, fps


def write_video(path, frames, fps):
    with av.open(str(path), mode='w') as container:
        stream = container.add_stream('libx264', rate=Fraction(fps).limit_denominator(100000))
        stream.width = frames[0].shape[1]
        stream.height = frames[0].shape[0]
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '16', 'preset': 'medium'}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='bgr24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def prepare_material_video(graph, input_root):
    """Sample aligned source/mask across their whole duration, never a prefix."""
    entries = [(node_id, node) for node_id, node in graph.items()
               if node.get('class_type') == 'WanVaceToVideo'
               and node.get('_meta', {}).get('full_timeline_material_edit')]
    if not entries:
        return graph, None
    if len(entries) != 1:
        raise ValueError('Full-timeline material adapter requires one VACE edit')
    node_id, node = entries[0]
    settings = node['_meta']['full_timeline_material_edit']
    source_id, mask_id = settings['source_loader'], settings['mask_loader']
    source_path = input_root / graph[source_id]['inputs']['file']
    mask_path = input_root / graph[mask_id]['inputs']['file']
    source, fps = read_video(source_path)
    masks, mask_fps = read_video(mask_path)
    if len(source) != len(masks) or abs(fps - mask_fps) > .01:
        raise ValueError('VACE source and mask must have equal frame counts and FPS')
    if any(a.shape != b.shape for a, b in zip(source, masks)):
        raise ValueError('VACE source and mask must have matching geometry')
    inputs = node['inputs']
    count = int(inputs['length'])
    if count < 5 or count % 4 != 1:
        raise ValueError('VACE inference length must be 4n+1 and at least 5')
    width, height = int(inputs['width']), int(inputs['height'])
    indices = np.rint(np.linspace(0, len(source) - 1, count)).astype(int)
    inference_fps = count * fps / len(source)
    token = uuid4().hex
    source_name, mask_name = f'vace_{token}_source.mp4', f'vace_{token}_mask.mp4'
    write_video(input_root / source_name,
                [cv2.resize(source[i], (width, height)) for i in indices], inference_fps)
    write_video(input_root / mask_name,
                [cv2.resize(masks[i], (width, height), interpolation=cv2.INTER_NEAREST)
                 for i in indices], inference_fps)
    prepared = copy.deepcopy(graph)
    prepared[source_id]['inputs']['file'] = source_name
    prepared[mask_id]['inputs']['file'] = mask_name
    # The neutral control must match the VACE batch geometry and length.
    clear_id = settings['neutral_image']
    prepared[clear_id]['inputs'].update(width=width, height=height, batch_size=count)
    return prepared, {
        'source': str(source_path), 'mask': str(mask_path),
        'source_frames': len(source), 'source_fps': fps,
        'inference_frames': count, 'inference_fps': inference_fps,
        'source_indices': indices.tolist(), 'covers_entire_source': True,
    }


def restore_material_video(sampled_output, context, target):
    """Interpolate edited appearance; composite through each original mask."""
    source, fps = read_video(context['source'])
    masks, _ = read_video(context['mask'])
    edited, _ = read_video(sampled_output)
    count = context['inference_frames']
    if len(edited) != count:
        raise ValueError(f'VACE returned {len(edited)} frames, expected {count}')
    full = []
    for i, frame in enumerate(source):
        position = i * (count - 1) / max(1, len(source) - 1)
        lo = int(position)
        hi = min(lo + 1, count - 1)
        edit = cv2.addWeighted(edited[lo], 1 - (position - lo), edited[hi], position - lo, 0)
        edit = cv2.resize(edit, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
        mask = (masks[i][:, :, 2] / 255.0).astype(np.float32)
        mask[mask < .02] = 0
        mask = cv2.GaussianBlur(mask, (5, 5), .8)[:, :, None]
        full.append(np.clip(frame * (1 - mask) + edit * mask, 0, 255).astype(np.uint8))
    write_video(target, full, fps)
    decoded, output_fps = read_video(target)
    if len(decoded) != len(source) or abs(output_fps - fps) > .01:
        raise ValueError('Restored VACE output timing differs from the source')
    return target
