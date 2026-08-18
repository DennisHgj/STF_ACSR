"""CSPM prompt construction and command-line demo."""

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, conint

from .supportprompt_position import build_position_support_set
from .supportprompt_shape import build_shape_support_set
from .util import get_keyframes


DEFAULT_MODEL = "gpt-4o"

BACKGROUND_PROMPT = """
Classify Mandarin Cued Speech hand keyframes. Hand position means the location
pointed to by the straight fingers. Hand shape means which fingers are straight.
Use the supplied labeled support images before classifying the test images.

Position labels:
0 Eye
1 Right side of the head
2 Cheek
3 Chin
4 Below the head

Shape labels:
0 Index straight only
1 Index and middle straight and together
2 Middle, ring and pinky straight
3 Index, middle, ring and pinky straight; thumb bent
4 All five fingers straight
5 Thumb and index straight
6 Thumb, index and middle straight
7 Index and middle straight and apart

For every test image, return its exact supplied frame_id, one position label,
one shape label and a concise visual verification rationale. Compare easily
confused support categories before finalizing the labels.
""".strip()


class Frame(BaseModel):
    frame_id: conint(ge=0)  # type: ignore[valid-type]
    hand_position: conint(ge=0, le=4)  # type: ignore[valid-type]
    hand_shape: conint(ge=0, le=7)  # type: ignore[valid-type]
    reasoning_process: str


class HandRecognition(BaseModel):
    recog_results: List[Frame]


def _default_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI Python package is required for live CSPM inference. "
            "Install requirements.txt first."
        ) from exc
    return OpenAI()


def _image_part(encoded_jpeg: str) -> Dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encoded_jpeg}",
            "detail": "high",
        },
    }


def build_messages(
    hand_frames: Sequence[str],
    frame_ids: Sequence[int],
    support_set_path: str,
) -> List[Dict[str, Any]]:
    if not hand_frames:
        raise ValueError("At least one hand keyframe is required.")
    if len(hand_frames) != len(frame_ids):
        raise ValueError("hand_frames and frame_ids must have the same length.")

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": BACKGROUND_PROMPT},
        {"type": "text", "text": "Position support set:"},
    ]
    content.extend(build_position_support_set(support_set_path))
    content.append({"type": "text", "text": "Shape support set:"})
    content.extend(build_shape_support_set(support_set_path))

    for frame_id, encoded_frame in zip(frame_ids, hand_frames):
        content.append(
            {
                "type": "text",
                "text": f"Test hand keyframe with frame_id={int(frame_id)}:",
            }
        )
        content.append(_image_part(encoded_frame))

    content.append(
        {
            "type": "text",
            "text": (
                "Return exactly one result for every supplied frame_id, in the "
                "same order. Do not add or omit frames."
            ),
        }
    )
    return [
        {
            "role": "system",
            "content": (
                "You classify Mandarin Cued Speech hand position and shape and "
                "return the requested structured result."
            ),
        },
        {"role": "user", "content": content},
    ]


def _model_dump(value: BaseModel) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


def generate_recognition_single(
    hand_frames: Sequence[str],
    seed: Optional[int] = 3702,
    support_set_path: str = "",
    *,
    frame_ids: Optional[Sequence[int]] = None,
    model: Optional[str] = None,
    client: Any = None,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Recognize one keyframe sequence while preserving the released API shape."""
    resolved_ids = (
        [int(index) for index in frame_ids]
        if frame_ids is not None
        else list(range(len(hand_frames)))
    )
    messages = build_messages(hand_frames, resolved_ids, support_set_path)
    request: Dict[str, Any] = {
        "model": model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        "messages": messages,
        "response_format": HandRecognition,
    }
    if seed is not None:
        request["seed"] = seed

    api_client = client if client is not None else _default_client()
    response = api_client.chat.completions.parse(**request)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("The model returned no parsed hand-recognition result.")

    recognition = [_model_dump(item) for item in parsed.recog_results]
    returned_ids = [int(item["frame_id"]) for item in recognition]
    if returned_ids != resolved_ids:
        raise ValueError(
            "Returned frame IDs do not match the request: "
            f"expected {resolved_ids}, received {returned_ids}."
        )

    usage = getattr(response, "usage", None)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    return recognition, [total_tokens]


def process_video(video_path: str) -> List[str]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to read CSPM videos.") from exc

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    encoded_frames: List[str] = []
    while True:
        success, frame = video.read()
        if not success:
            break
        encoded, buffer = cv2.imencode(".jpg", frame)
        if not encoded:
            video.release()
            raise RuntimeError(f"Failed to encode a frame from {video_path}.")
        encoded_frames.append(base64.b64encode(buffer).decode("ascii"))
    video.release()
    if not encoded_frames:
        raise ValueError(f"Video contains no readable frames: {video_path}")
    return encoded_frames


def run_cspm(
    video_path: str,
    position_path: str,
    support_set_path: str,
    output_path: str,
    *,
    model: Optional[str] = None,
    seed: Optional[int] = 3702,
    speed_threshold: float = 6.0,
    index_distance_threshold: int = 2,
    client: Any = None,
) -> Dict[str, Any]:
    frames = process_video(video_path)
    keyframe_indices = get_keyframes(
        position_path, speed_threshold, index_distance_threshold
    )
    if not keyframe_indices:
        raise ValueError("The keyframe filter selected no frames.")
    if keyframe_indices[-1] >= len(frames):
        raise ValueError(
            "Hand-position sequence and video are misaligned: "
            f"selected frame {keyframe_indices[-1]} from {len(frames)} frames."
        )

    recognition, token_usage = generate_recognition_single(
        [frames[index] for index in keyframe_indices],
        seed=seed,
        support_set_path=support_set_path,
        frame_ids=keyframe_indices,
        model=model,
        client=client,
    )
    result = {
        "source_video": str(video_path),
        "total_frames": len(frames),
        "frame_index": keyframe_indices,
        "keyframe_count": len(keyframe_indices),
        "speed_threshold": speed_threshold,
        "index_distance_threshold": index_distance_threshold,
        "model": model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        "total_tokens": sum(token_usage),
        "recog_results": recognition,
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    module_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the STF-ACSR CSPM demo.")
    parser.add_argument("--video", default=str(module_dir / "HS-0001.mp4"))
    parser.add_argument("--positions", default=str(module_dir / "HS-0001.npy"))
    parser.add_argument("--support-set", default=str(module_dir / "support_set"))
    parser.add_argument("--output", default="outputs/cspm/HS-0001.json")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--seed", type=int, default=3702)
    parser.add_argument("--speed-threshold", type=float, default=6.0)
    parser.add_argument("--index-distance-threshold", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_cspm(
        video_path=args.video,
        position_path=args.positions,
        support_set_path=args.support_set,
        output_path=args.output,
        model=args.model,
        seed=args.seed,
        speed_threshold=args.speed_threshold,
        index_distance_threshold=args.index_distance_threshold,
    )
    print(
        f"Wrote {result['keyframe_count']} CSPM predictions to {args.output} "
        f"using {result['total_tokens']} tokens."
    )


if __name__ == "__main__":
    main()
