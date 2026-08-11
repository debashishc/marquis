from __future__ import annotations

import os
from collections.abc import Callable


def create_note_taking_fn(
    backend: str = "openai_vision",
    local_model: str = "Qwen/Qwen3.5-0.8B",
    local_download_dir: str = "",
    local_fps: float = 1.0,
    local_max_frames: int | None = None,
    local_max_tokens: int = 2048,
    local_allowed_media_path: str | None = None,
    openai_model: str = "gpt-5",
    openai_api_key: str | None = None,
    openai_max_tokens: int = 2048,
) -> Callable[[str, str], str]:
    if backend == "local_qwen":
        return _build_local_qwen(
            model=local_model,
            download_dir=local_download_dir,
            fps=local_fps,
            max_frames=local_max_frames,
            max_tokens=local_max_tokens,
            allowed_local_media_path=local_allowed_media_path,
        )
    elif backend == "openai_vision":
        return _build_openai_vision(
            model=openai_model,
            api_key=openai_api_key,
            max_tokens=openai_max_tokens,
        )
    else:
        raise ValueError(f"Unknown VLM backend: {backend!r}. Use 'local_qwen' or 'openai_vision'.")


def _build_local_qwen(
    model: str,
    download_dir: str,
    fps: float,
    max_frames: int | None,
    max_tokens: int,
    allowed_local_media_path: str | None,
) -> Callable[[str, str], str]:
    _vlm_instance = None

    def note_taking(video_path: str, perception_query: str) -> str:
        nonlocal _vlm_instance
        if _vlm_instance is None:
            from marquis.common.model_backends import Qwen3_5_VL

            _vlm_instance = Qwen3_5_VL(
                model=model,
                download_dir=download_dir,
                fps=fps,
                max_frames=max_frames,
                max_tokens=max_tokens,
                allowed_local_media_path=allowed_local_media_path,
            )
        return _vlm_instance.infer(video_path, perception_query)

    return note_taking


def _build_openai_vision(
    model: str,
    api_key: str | None,
    max_tokens: int,
) -> Callable[[str, str], str]:

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError("OPENAI_API_KEY required for openai_vision backend")

    from openai import OpenAI

    client = OpenAI(api_key=resolved_key)

    def note_taking(video_path: str, perception_query: str) -> str:
        frames_b64 = _extract_frames_as_base64(video_path, max_frames=8)
        content = [{"type": "text", "text": perception_query}]
        for frame_b64 in frames_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                }
            )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content

    return note_taking


def _extract_frames_as_base64(video_path: str, max_frames: int = 8) -> list[str]:
    import base64

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python (cv2) is required for openai_vision backend frame extraction"
        ) from exc

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            _, buf = cv2.imencode(".jpg", frame)
            frames_b64.append(base64.b64encode(buf).decode("utf-8"))
    cap.release()
    return frames_b64
