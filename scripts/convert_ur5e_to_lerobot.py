#!/usr/bin/env python3
"""Convert recorded UR5e episodes into a LeRobotDataset.

Expected raw episode layout:

    episode_000/
      numeric/
        TCP_POSE.csv
        GRIPPER.csv
        GRIPPER_TARGET.csv
      videos/
        CAMERA_1.avi
        CAMERA_1_frames.csv
        CAMERA_2.avi
        CAMERA_2_frames.csv

Legacy image-directory input is also supported:

    episode_000/
      images/
        CAMERA_1/step_000000_rgb.png
        CAMERA_2/step_000000_rgb.png

The converter keeps all modalities aligned by row/frame order. If cameras are
shorter than numeric data, only the common prefix is converted. Actions are
TCP delta commands from the previous frame to the current frame plus an absolute
gripper target opening. UR rx/ry/rz deltas default to a relative SO(3) rotation
vector instead of direct component subtraction, which avoids jumps between
equivalent rotvecs.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import statistics
import struct
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TCP_COLUMNS = [
    "TCP_POSE_1",
    "TCP_POSE_2",
    "TCP_POSE_3",
    "TCP_POSE_4",
    "TCP_POSE_5",
    "TCP_POSE_6",
]
GRIPPER_COLUMNS = ["GRIPPER_1", "GRIPPER_2"]
GRIPPER_TARGET_COLUMNS = ["GRIPPER_TARGET_1"]

STATE_NAMES = [
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_rx",
    "tcp_ry",
    "tcp_rz",
    "gripper_opening",
    "gripper_current",
]
ACTION_NAMES = [
    "delta_tcp_x",
    "delta_tcp_y",
    "delta_tcp_z",
    "delta_tcp_rx",
    "delta_tcp_ry",
    "delta_tcp_rz",
    "gripper_target_opening",
]

IMAGE_RE = re.compile(r"step_(\d+)_rgb\.(png|jpg|jpeg)$", re.IGNORECASE)
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


@dataclass(frozen=True)
class CameraInfo:
    raw_name: str
    feature_key: str
    source_kind: str
    files: list[Path]
    video_path: Path | None
    frame_indices: list[int]
    timestamps: list[float]
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class EpisodeInfo:
    path: Path
    tcp: list[list[float]]
    tcp_timestamps: list[float]
    gripper: list[list[float]]
    gripper_timestamps: list[float]
    gripper_target: list[list[float]]
    gripper_target_timestamps: list[float]
    gripper_target_source: str
    cameras: list[CameraInfo]
    length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert UR5e CSV + camera videos/images to a LeRobot ACT dataset."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/test2"),
        help="Raw dataset root or a single episode directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/lerobot/ur5e_cylinder_to_box"),
        help="Directory where the LeRobotDataset will be written.",
    )
    parser.add_argument(
        "--repo-id",
        default="local/ur5e_cylinder_to_box",
        help="LeRobot dataset repo_id stored in metadata.",
    )
    parser.add_argument(
        "--task",
        default="Pick up the cylinder and place it into the box",
        help="Task string stored in the LeRobot episode.",
    )
    parser.add_argument(
        "--robot-type",
        default="ur5e",
        help="Robot type stored in metadata.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Dataset FPS. If omitted, inferred from TCP timestamps and rounded.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Camera names to include. Defaults to all cameras under videos/ or images/.",
    )
    parser.add_argument(
        "--camera-source",
        choices=["auto", "videos", "images"],
        default="auto",
        help="Raw camera input source. auto prefers videos/ when present, otherwise images/.",
    )
    parser.add_argument(
        "--image-storage",
        choices=["video", "image"],
        default="video",
        help="Store visual observations as LeRobot videos or image files.",
    )
    parser.add_argument(
        "--vcodec",
        default="auto",
        help=(
            "Video codec passed to LeRobot when --image-storage=video. "
            "Use auto to prefer hardware encoders such as h264_nvenc; use h264 for CPU encoding."
        ),
    )
    parser.add_argument(
        "--batch-encoding-size",
        type=int,
        default=1,
        help="Number of episodes to accumulate before LeRobot batch-encodes videos.",
    )
    parser.add_argument(
        "--streaming-encoding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Encode video while frames are added instead of first writing temporary PNGs. "
            "This can be faster with GPU encoders, but keep the queue large to avoid dropped frames."
        ),
    )
    parser.add_argument(
        "--encoder-queue-maxsize",
        type=int,
        default=1024,
        help="Per-camera frame queue size used only with --streaming-encoding.",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=None,
        help="Threads per encoder instance. Leave unset for hardware encoders.",
    )
    parser.add_argument(
        "--rotation-delta-mode",
        choices=["relative-rotvec", "raw-rotvec"],
        default="relative-rotvec",
        help=(
            "How to compute the rotational part of delta actions. "
            "relative-rotvec uses log(R_prev.T @ R_curr); raw-rotvec subtracts UR rx/ry/rz directly."
        ),
    )
    parser.add_argument(
        "--include-current-in-action",
        action="store_true",
        help="Append gripper current delta to action. Usually leave this off.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-root first if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect episodes and print the conversion plan without writing.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Threads for LeRobot's async image writer.",
    )
    parser.add_argument(
        "--no-parallel-encoding",
        action="store_true",
        help="Disable parallel per-camera video encoding in save_episode().",
    )
    return parser.parse_args()


def discover_episodes(raw_root: Path) -> list[Path]:
    raw_root = raw_root.expanduser().resolve()
    if is_episode_dir(raw_root):
        return [raw_root]

    episodes = [path for path in sorted(raw_root.iterdir()) if path.is_dir() and is_episode_dir(path)]
    if not episodes:
        raise FileNotFoundError(
            f"No episode directories found under {raw_root}. "
            "Expected numeric/ plus videos/ or images/."
        )
    return episodes


def is_episode_dir(path: Path) -> bool:
    return (path / "numeric" / "TCP_POSE.csv").is_file() and (path / "numeric" / "GRIPPER.csv").is_file()


def read_numeric_csv(path: Path, columns: Iterable[str]) -> tuple[list[list[float]], list[float]]:
    rows: list[list[float]] = []
    timestamps: list[float] = []
    with path.open("r", newline="") as stream:
        reader = csv.DictReader(stream)
        required = ["timestamp", *columns]
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        for row in reader:
            timestamps.append(float(row["timestamp"]))
            rows.append([float(row[name]) for name in columns])
    return rows, timestamps


def camera_feature_key(camera_name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", camera_name).strip("_").lower()
    return f"observation.images.{normalized}"


def list_camera_files(camera_dir: Path) -> list[Path]:
    indexed: list[tuple[int, Path]] = []
    for path in camera_dir.iterdir():
        if not path.is_file():
            continue
        match = IMAGE_RE.fullmatch(path.name)
        if match:
            indexed.append((int(match.group(1)), path))

    if not indexed:
        raise FileNotFoundError(f"No step_XXXXXX_rgb image files found in {camera_dir}")

    indexed.sort(key=lambda item: item[0])
    indices = [idx for idx, _ in indexed]
    expected = list(range(len(indices)))
    if indices != expected:
        missing = sorted(set(range(indices[-1] + 1)) - set(indices))
        raise ValueError(
            f"{camera_dir} has non-contiguous frame indices. "
            f"Missing examples: {missing[:10]}"
        )
    return [path for _, path in indexed]


def read_video_frame_csv(path: Path) -> tuple[list[int], list[float]]:
    frame_indices: list[int] = []
    timestamps: list[float] = []
    with path.open("r", newline="") as stream:
        reader = csv.DictReader(stream)
        required = ["frame", "timestamp"]
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        previous_frame = -1
        for row in reader:
            frame_index = int(row["frame"])
            if frame_index <= previous_frame:
                raise ValueError(
                    f"{path} frame indices must be strictly increasing. "
                    f"Got {frame_index} after {previous_frame}."
                )
            previous_frame = frame_index
            frame_indices.append(frame_index)
            timestamps.append(float(row["timestamp"]))

    if not frame_indices:
        raise ValueError(f"{path} contains no frame rows")
    return frame_indices, timestamps


def probe_image_shape(path: Path) -> tuple[int, int, int]:
    if path.suffix.lower() == ".png":
        with path.open("rb") as stream:
            header = stream.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
            width, height = struct.unpack(">II", header[16:24])
            return (height, width, 3)

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            f"Cannot inspect non-PNG image {path} without Pillow installed."
        ) from exc

    with Image.open(path) as image:
        width, height = image.size
    return (height, width, 3)


def probe_video_shape(path: Path) -> tuple[int, int, int]:
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            f"Cannot inspect video {path} without PyAV installed. "
            "Install the repo requirements in the LeRobot environment."
        ) from exc

    with av.open(str(path)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"{path} has no video stream")

        width = int(stream.codec_context.width or stream.width or 0)
        height = int(stream.codec_context.height or stream.height or 0)
        if width > 0 and height > 0:
            return (height, width, 3)

        for frame in container.decode(stream):
            return (frame.height, frame.width, 3)

    raise ValueError(f"Cannot determine video shape for {path}")


def load_camera_infos(
    path: Path,
    selected_cameras: list[str] | None,
    camera_source: str,
) -> list[CameraInfo]:
    videos_dir = path / "videos"
    images_dir = path / "images"

    use_videos = False
    if camera_source == "videos":
        use_videos = True
    elif camera_source == "images":
        use_videos = False
    elif videos_dir.is_dir() and any(
        item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
        for item in videos_dir.iterdir()
    ):
        use_videos = True

    if use_videos:
        return load_video_cameras(videos_dir, selected_cameras)
    return load_image_cameras(images_dir, selected_cameras)


def load_image_cameras(
    images_dir: Path,
    selected_cameras: list[str] | None,
) -> list[CameraInfo]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No camera image directory found at {images_dir}")

    camera_dirs = [cam for cam in sorted(images_dir.iterdir()) if cam.is_dir()]
    if selected_cameras is not None:
        wanted = set(selected_cameras)
        camera_dirs = [cam for cam in camera_dirs if cam.name in wanted]
        missing = sorted(wanted - {cam.name for cam in camera_dirs})
        if missing:
            raise FileNotFoundError(f"{images_dir.parent} is missing requested cameras: {missing}")

    if not camera_dirs:
        raise FileNotFoundError(f"No camera directories found in {images_dir}")

    cameras: list[CameraInfo] = []
    for camera_dir in camera_dirs:
        files = list_camera_files(camera_dir)
        cameras.append(
            CameraInfo(
                raw_name=camera_dir.name,
                feature_key=camera_feature_key(camera_dir.name),
                source_kind="images",
                files=files,
                video_path=None,
                frame_indices=list(range(len(files))),
                timestamps=[],
                shape=probe_image_shape(files[0]),
            )
        )
    return cameras


def load_video_cameras(
    videos_dir: Path,
    selected_cameras: list[str] | None,
) -> list[CameraInfo]:
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"No camera video directory found at {videos_dir}")

    video_files = [
        path for path in sorted(videos_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if selected_cameras is not None:
        wanted = set(selected_cameras)
        video_files = [path for path in video_files if path.stem in wanted or path.name in wanted]
        found = {path.stem for path in video_files} | {path.name for path in video_files}
        missing = sorted(wanted - found)
        if missing:
            raise FileNotFoundError(f"{videos_dir.parent} is missing requested cameras: {missing}")

    if not video_files:
        raise FileNotFoundError(f"No camera video files found in {videos_dir}")

    cameras: list[CameraInfo] = []
    for video_path in video_files:
        frames_csv = video_path.with_name(f"{video_path.stem}_frames.csv")
        if not frames_csv.is_file():
            raise FileNotFoundError(
                f"{video_path} is missing its frame index/timestamp file: {frames_csv.name}"
            )
        frame_indices, timestamps = read_video_frame_csv(frames_csv)
        cameras.append(
            CameraInfo(
                raw_name=video_path.stem,
                feature_key=camera_feature_key(video_path.stem),
                source_kind="videos",
                files=[],
                video_path=video_path,
                frame_indices=frame_indices,
                timestamps=timestamps,
                shape=probe_video_shape(video_path),
            )
        )
    return cameras


def read_gripper_target(path: Path, gripper: list[list[float]], gripper_ts: list[float]):
    target_path = path / "numeric" / "GRIPPER_TARGET.csv"
    if target_path.is_file():
        rows, timestamps = read_numeric_csv(target_path, GRIPPER_TARGET_COLUMNS)
        return rows, timestamps, "GRIPPER_TARGET.csv"

    rows = [[row[0]] for row in gripper]
    return rows, list(gripper_ts), "GRIPPER.csv fallback opening"


def load_episode(
    path: Path,
    selected_cameras: list[str] | None,
    camera_source: str,
) -> EpisodeInfo:
    tcp, tcp_ts = read_numeric_csv(path / "numeric" / "TCP_POSE.csv", TCP_COLUMNS)
    gripper, gripper_ts = read_numeric_csv(path / "numeric" / "GRIPPER.csv", GRIPPER_COLUMNS)
    gripper_target, gripper_target_ts, gripper_target_source = read_gripper_target(
        path,
        gripper,
        gripper_ts,
    )
    cameras = load_camera_infos(path, selected_cameras, camera_source)

    length = min(
        len(tcp),
        len(gripper),
        len(gripper_target),
        *(len(camera.frame_indices) for camera in cameras),
    )
    if length <= 0:
        raise ValueError(f"{path} has no aligned frames")

    return EpisodeInfo(
        path=path,
        tcp=tcp,
        tcp_timestamps=tcp_ts,
        gripper=gripper,
        gripper_timestamps=gripper_ts,
        gripper_target=gripper_target,
        gripper_target_timestamps=gripper_target_ts,
        gripper_target_source=gripper_target_source,
        cameras=cameras,
        length=length,
    )


def infer_fps(episodes: list[EpisodeInfo]) -> int:
    intervals: list[float] = []
    for episode in episodes:
        timestamps = episode.tcp_timestamps[: episode.length]
        intervals.extend(
            b - a for a, b in zip(timestamps, timestamps[1:]) if b > a
        )

    if not intervals:
        return 30

    median_dt = statistics.median(intervals)
    if median_dt <= 0:
        return 30
    return max(1, round(1.0 / median_dt))


def print_plan(episodes: list[EpisodeInfo], fps: int, args: argparse.Namespace) -> None:
    print(f"Raw root: {args.raw_root}")
    print(f"Output root: {args.output_root}")
    print(f"repo_id: {args.repo_id}")
    print(f"fps: {fps}")
    print(f"raw camera source: {args.camera_source}")
    print(f"image storage: {args.image_storage}")
    print(f"requested video codec: {args.vcodec}")
    print(f"streaming encoding: {args.streaming_encoding}")
    print(f"batch encoding size: {args.batch_encoding_size}")
    print("action mode: TCP delta from previous frame + absolute gripper target opening")
    print(f"rotation delta mode: {args.rotation_delta_mode}")
    print(f"episodes: {len(episodes)}")

    for episode in episodes:
        numeric_len = min(len(episode.tcp), len(episode.gripper), len(episode.gripper_target))
        camera_counts = {camera.raw_name: len(camera.frame_indices) for camera in episode.cameras}
        dropped = {
            "tcp_rows": len(episode.tcp) - episode.length,
            "gripper_rows": len(episode.gripper) - episode.length,
            "gripper_target_rows": len(episode.gripper_target) - episode.length,
            **{name: count - episode.length for name, count in camera_counts.items()},
        }
        camera_shapes = {camera.raw_name: camera.shape for camera in episode.cameras}
        camera_sources = {camera.raw_name: camera.source_kind for camera in episode.cameras}
        print(
            f"- {episode.path.name}: length={episode.length}, "
            f"numeric_min={numeric_len}, cameras={camera_counts}, "
            f"sources={camera_sources}, shapes={camera_shapes}, "
            f"gripper_target={episode.gripper_target_source}, dropped_tail={dropped}"
        )


def import_lerobot_dataset():
    try:
        from lerobot.datasets import LeRobotDataset

        return LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            return LeRobotDataset
        except ImportError:
            try:
                from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

                return LeRobotDataset
            except ImportError as exc:
                raise ImportError(
                    "Cannot import LeRobotDataset. Activate an environment with "
                    "LeRobot installed, for example: conda activate LeRobot"
                ) from exc


def build_features(
    cameras: list[CameraInfo],
    image_storage: str,
    include_current_in_action: bool,
) -> dict:
    action_names = ACTION_NAMES + (["delta_gripper_current"] if include_current_in_action else [])
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": {"state": STATE_NAMES},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": {"action": action_names},
        },
    }

    for camera in cameras:
        features[camera.feature_key] = {
            "dtype": image_storage,
            "shape": camera.shape,
            "names": ["height", "width", "channel"],
        }
    return features


def make_state(episode: EpisodeInfo, index: int):
    import numpy as np

    return np.asarray([*episode.tcp[index], *episode.gripper[index]], dtype=np.float32)


def make_action(
    episode: EpisodeInfo,
    index: int,
    rotation_delta_mode: str,
    include_current: bool,
):
    import numpy as np

    if index == 0:
        values = [0.0] * 6
    else:
        values = pose_delta(
            previous=episode.tcp[index - 1],
            current=episode.tcp[index],
            rotation_delta_mode=rotation_delta_mode,
        )

    values.append(episode.gripper_target[index][0])
    if include_current:
        if index == 0:
            values.append(0.0)
        else:
            values.append(episode.gripper[index][1] - episode.gripper[index - 1][1])
    return np.asarray(values, dtype=np.float32)


def pose_delta(
    previous: list[float],
    current: list[float],
    rotation_delta_mode: str,
) -> list[float]:
    translation_delta = [current[i] - previous[i] for i in range(3)]

    if rotation_delta_mode == "raw-rotvec":
        rotation_delta = [current[i] - previous[i] for i in range(3, 6)]
    elif rotation_delta_mode == "relative-rotvec":
        rotation_delta = relative_rotvec_delta(previous[3:6], current[3:6])
    else:
        raise ValueError(f"Unsupported rotation delta mode: {rotation_delta_mode}")

    return [*translation_delta, *rotation_delta]


def relative_rotvec_delta(previous: list[float], current: list[float]) -> list[float]:
    import numpy as np

    previous_rot = rotvec_to_matrix(np.asarray(previous, dtype=np.float64))
    current_rot = rotvec_to_matrix(np.asarray(current, dtype=np.float64))
    delta_rot = previous_rot.T @ current_rot
    return matrix_to_rotvec(delta_rot).tolist()


def rotvec_to_matrix(rotvec):
    import numpy as np

    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)

    axis = rotvec / theta
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(theta) * skew
        + (1.0 - math.cos(theta)) * (skew @ skew)
    )


def matrix_to_rotvec(matrix):
    import numpy as np

    cos_theta = (float(np.trace(matrix)) - 1.0) / 2.0
    cos_theta = min(1.0, max(-1.0, cos_theta))
    theta = math.acos(cos_theta)

    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)

    vector = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )

    if abs(math.pi - theta) < 1e-5:
        axis = np.empty(3, dtype=np.float64)
        axis[0] = math.sqrt(max(0.0, (matrix[0, 0] + 1.0) / 2.0))
        axis[1] = math.sqrt(max(0.0, (matrix[1, 1] + 1.0) / 2.0))
        axis[2] = math.sqrt(max(0.0, (matrix[2, 2] + 1.0) / 2.0))
        axis[1] = math.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
        axis[2] = math.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return theta * axis / norm

    return theta * vector / (2.0 * math.sin(theta))


class CameraFrameReader:
    def __init__(self, camera: CameraInfo):
        self.camera = camera
        self._container = None
        self._stream = None
        self._frames = None
        self._decoded_index = -1

    def __enter__(self):
        if self.camera.source_kind == "videos":
            try:
                import av
            except ImportError as exc:
                raise ImportError(
                    f"Cannot decode video {self.camera.video_path} without PyAV installed."
                ) from exc

            if self.camera.video_path is None:
                raise ValueError(f"{self.camera.raw_name} has no video path")
            self._container = av.open(str(self.camera.video_path))
            self._stream = next(
                (item for item in self._container.streams if item.type == "video"),
                None,
            )
            if self._stream is None:
                raise ValueError(f"{self.camera.video_path} has no video stream")
            self._frames = self._container.decode(self._stream)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._container is not None:
            self._container.close()
        return False

    def read(self, index: int):
        if self.camera.source_kind == "images":
            return self._read_image(index)
        if self.camera.source_kind == "videos":
            return self._read_video_frame(index)
        raise ValueError(f"Unsupported camera source: {self.camera.source_kind}")

    def _read_image(self, index: int):
        import numpy as np

        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to load camera images for conversion.") from exc

        with Image.open(self.camera.files[index]) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _read_video_frame(self, index: int):
        if self._frames is None:
            raise RuntimeError(f"Video reader for {self.camera.raw_name} is not open")

        target_frame = self.camera.frame_indices[index]
        if target_frame < self._decoded_index:
            raise ValueError(
                f"{self.camera.raw_name} requested frame {target_frame} after "
                f"already decoding frame {self._decoded_index}"
            )

        for frame in self._frames:
            self._decoded_index += 1
            if self._decoded_index == target_frame:
                return frame.to_ndarray(format="rgb24")

        raise EOFError(
            f"{self.camera.video_path} ended before requested frame {target_frame}"
        )


def create_dataset(args: argparse.Namespace, episodes: list[EpisodeInfo], fps: int):
    LeRobotDataset = import_lerobot_dataset()

    features = build_features(
        cameras=episodes[0].cameras,
        image_storage=args.image_storage,
        include_current_in_action=args.include_current_in_action,
    )

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_root} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    use_videos = args.image_storage == "video"
    create_kwargs = {
        "repo_id": args.repo_id,
        "root": output_root,
        "fps": fps,
        "robot_type": args.robot_type,
        "features": features,
        "use_videos": use_videos,
        "vcodec": args.vcodec,
        "batch_encoding_size": args.batch_encoding_size,
        "streaming_encoding": args.streaming_encoding,
        "encoder_queue_maxsize": args.encoder_queue_maxsize,
        "encoder_threads": args.encoder_threads,
        "image_writer_threads": args.image_writer_threads,
    }

    try:
        dataset = LeRobotDataset.create(**create_kwargs)
    except TypeError:
        create_kwargs.pop("vcodec", None)
        create_kwargs.pop("batch_encoding_size", None)
        create_kwargs.pop("streaming_encoding", None)
        create_kwargs.pop("encoder_queue_maxsize", None)
        create_kwargs.pop("encoder_threads", None)
        create_kwargs.pop("image_writer_threads", None)
        dataset = LeRobotDataset.create(**create_kwargs)

    if use_videos:
        print(f"Resolved video codec: {getattr(dataset, 'vcodec', args.vcodec)}")

    for episode in episodes:
        validate_feature_compatibility(episodes[0], episode)
        print(f"Converting {episode.path.name}: {episode.length} frames")
        with ExitStack() as stack:
            readers = {
                camera.feature_key: stack.enter_context(CameraFrameReader(camera))
                for camera in episode.cameras
            }
            for frame_index in range(episode.length):
                frame = {
                    "observation.state": make_state(episode, frame_index),
                    "action": make_action(
                        episode,
                        frame_index,
                        args.rotation_delta_mode,
                        args.include_current_in_action,
                    ),
                    "task": args.task,
                }

                for camera in episode.cameras:
                    frame[camera.feature_key] = readers[camera.feature_key].read(frame_index)

                dataset.add_frame(frame)

        try:
            dataset.save_episode(parallel_encoding=not args.no_parallel_encoding)
        except TypeError:
            dataset.save_episode()

    if hasattr(dataset, "finalize"):
        dataset.finalize()

    return output_root


def validate_feature_compatibility(reference: EpisodeInfo, episode: EpisodeInfo) -> None:
    ref_cameras = [(cam.raw_name, cam.feature_key, cam.shape) for cam in reference.cameras]
    cur_cameras = [(cam.raw_name, cam.feature_key, cam.shape) for cam in episode.cameras]
    if cur_cameras != ref_cameras:
        raise ValueError(
            f"{episode.path} camera set or image shapes differ from first episode. "
            f"Expected {ref_cameras}, got {cur_cameras}"
        )


def main() -> int:
    args = parse_args()
    episodes = [
        load_episode(path, args.cameras, args.camera_source)
        for path in discover_episodes(args.raw_root)
    ]
    fps = args.fps if args.fps is not None else infer_fps(episodes)

    print_plan(episodes, fps, args)
    if args.dry_run:
        return 0

    output_root = create_dataset(args, episodes, fps)
    print(f"Done. LeRobot dataset written to: {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
