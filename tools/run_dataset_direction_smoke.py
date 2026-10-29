"""Bounded media sampling shared with the execution/diagnosis runtime."""
import json, os, subprocess, sys, tempfile
from pathlib import Path

def sample_video(path: Path, label: str, frame_limit: int = 5) -> tuple[list[str], dict, str | None]:
    """Mirror ComfyBench: inspect metadata and sample at most five ordered frames."""
    if path.suffix.lower() == ".gif":
        try:
            from PIL import Image
            image = Image.open(path)
            count = int(getattr(image, "n_frames", 1))
            interval = max(6, count // frame_limit)
            folder = Path(tempfile.mkdtemp(prefix="comfyrouter-video-"))
            frames: list[str] = []
            duration_ms = 0
            for index in range(count):
                image.seek(index)
                duration_ms += int(image.info.get("duration", 0))
                if index % interval == 0:
                    target = folder / f"{label}_{len(frames):02d}.png"
                    image.convert("RGB").save(target, "PNG")
                    frames.append(str(target))
            if not frames or duration_ms <= 0:
                return [], {}, "GIF has no readable timed frames"
            return frames, {"width": image.width, "height": image.height, "num_frames": count,
                            "frame_rate": count / (duration_ms / 1000),
                            "duration": duration_ms / 1000, "sample_interval": interval,
                            "sampled_frames": len(frames)}, None
        except (OSError, EOFError):
            return [], {}, "GIF is unreadable"
    try:
        import cv2
    except ImportError:
        return _sample_video_with_pyav(path, label, frame_limit)
    video = cv2.VideoCapture(str(path))
    try:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(video.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0 or count <= 0 or fps <= 0:
            return [], {}, "video metadata is invalid or unreadable"
        folder = Path(tempfile.mkdtemp(prefix="comfyrouter-video-"))
        interval = max(6, count // frame_limit)
        frames: list[str] = []
        index = 0
        while video.isOpened():
            ok, frame = video.read()
            if not ok:
                break
            if index % interval == 0:
                target = folder / f"{label}_{len(frames):02d}.png"
                if cv2.imwrite(str(target), frame):
                    frames.append(str(target))
            index += 1
        return frames, {"width": width, "height": height, "num_frames": count,
                        "frame_rate": fps, "duration": count / fps,
                        "sample_interval": interval, "sampled_frames": len(frames)}, None
    finally:
        video.release()

def _sample_video_with_pyav(path: Path, label: str, frame_limit: int) -> tuple[list[str], dict, str | None]:
    """Decode non-GIF video with PyAV when the runner lacks OpenCV.

    ComfyUI's virtual environment supplies PyAV even when the lightweight
    router environment does not. The helper preserves ComfyBench's bounded,
    ordered-frame evaluation rather than treating an MP4 as uninspectable.
    """
    interpreter = os.getenv("COMFYROUTER_MEDIA_PYTHON", sys.executable)
    if not Path(interpreter).is_file():
        return [], {}, "OpenCV is unavailable and no PyAV media interpreter is configured"
    folder = Path(tempfile.mkdtemp(prefix="comfyrouter-video-"))
    helper = r'''
import json
import sys
from pathlib import Path
import av

source, target_dir, label, frame_limit = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
container = av.open(source)
stream = container.streams.video[0]
width, height = stream.codec_context.width, stream.codec_context.height
fps = float(stream.average_rate) if stream.average_rate else 0.0
count = stream.frames or 0
frames = list(container.decode(stream))
container.close()
if not count:
    count = len(frames)
if width <= 0 or height <= 0 or count <= 0 or fps <= 0:
    raise RuntimeError("video metadata is invalid or unreadable")
interval = max(6, count // frame_limit)
paths = []
for index, frame in enumerate(frames):
    if index % interval == 0:
        output = target_dir / f"{label}_{len(paths):02d}.png"
        frame.to_image().save(output)
        paths.append(str(output))
print(json.dumps({"frames": paths, "metadata": {"width": width, "height": height,
      "num_frames": count, "frame_rate": fps, "duration": count / fps,
      "sample_interval": interval, "sampled_frames": len(paths)}}))
'''
    try:
        result = subprocess.run([interpreter, "-c", helper, str(path), str(folder), label, str(frame_limit)],
                                check=True, capture_output=True, text=True, timeout=120)
        payload = json.loads(result.stdout)
        frames, metadata = payload.get("frames"), payload.get("metadata")
        if not isinstance(frames, list) or not isinstance(metadata, dict) or not frames:
            return [], {}, "PyAV did not produce readable video frames"
        return [str(item) for item in frames], metadata, None
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return [], {}, f"PyAV fallback could not decode video: {exc}"
