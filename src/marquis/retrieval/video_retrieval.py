"""OmniEmbed video retrieval for the QA system (the ``retrieve`` command).

A self-contained retriever over a corpus of videos: it builds a dense index by
transcribing (Whisper) and embedding (OmniEmbed) each video, then scores
sub-queries against that index by cosine similarity and returns the top videos.

This is the **QA-system retrieval path**, and is deliberately *separate* from the
tevatron first-stage search / fusion pipeline (steps 3-6 in ``README.md``):

* tevatron path — external dense search → ``rank-subqueries.txt`` → fusion / eval.
* this path     — OmniEmbed → ``retrieved_videos.json`` → answer generation.

The ``retrieve`` command reads ``subqueries.jsonl`` plus a video corpus and
writes ``data.retrieved_videos`` (top videos per sub-query). It does retrieval
only — **no** answer generation; the generator consumes its output. The
functions are also importable as an engine. ``torch`` / ``transformers`` /
``whisper`` / ``yt_dlp`` are imported lazily so importing this module stays
cheap::

    python -m marquis.retrieval.cli retrieve runtime.retrieve.top_k=4

    from marquis.retrieval.video_retrieval import (
        load_retrieval_models, build_index, retrieve,
    )
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig

from marquis.retrieval._common import build_config

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

SIM_THRESHOLD = 0.1
TOP_K = 4
MAX_FRAMES = 32

# Use environment-driven cache (matches the SLURM launch environment).
CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

VIDEO_DIR = os.environ.get("VIDEO_DIR", "videos")
AUDIO_DIR = os.environ.get("AUDIO_DIR", "audio")


# ─────────────────────────────────────────────
# Model loading (retrieval only)
# ─────────────────────────────────────────────


def load_embedder(
    model_name: str = "Qwen/Qwen2.5-Omni-7B",
    adapter: str = "Tevatron/OmniEmbed-v0.1",
    cache_dir: str | None = None,
):
    """Load the OmniEmbed processor + model used to embed videos and queries."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    cache_dir = cache_dir or CACHE_DIR

    print("Loading OmniEmbed...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
    ).to("cuda")

    model = (
        PeftModel.from_pretrained(
            base,
            adapter,
            cache_dir=cache_dir,
        )
        .merge_and_unload()
        .eval()
    )

    return processor, model


def load_transcriber(model_name: str = "medium.en"):
    """Load the Whisper model used to transcribe videos before embedding."""
    import whisper

    print("Loading Whisper...")
    whisper_cache = os.path.join(os.environ.get("XDG_CACHE_HOME", "/tmp"), "whisper")
    os.makedirs(whisper_cache, exist_ok=True)
    return whisper.load_model(model_name, download_root=whisper_cache)


def load_retrieval_models(
    model_name: str = "Qwen/Qwen2.5-Omni-7B",
    adapter: str = "Tevatron/OmniEmbed-v0.1",
    whisper_model: str = "medium.en",
    cache_dir: str | None = None,
):
    """Load both retrieval models: ``(processor, embed_model, whisper_model)``."""
    processor, embed_model = load_embedder(model_name, adapter, cache_dir)
    transcriber = load_transcriber(whisper_model)
    return processor, embed_model, transcriber


# ─────────────────────────────────────────────
# Video utilities
# ─────────────────────────────────────────────


def download_youtube(url: str, out_path: str, cookiefile: str = "/workspace/cookies.txt") -> None:
    import yt_dlp

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": out_path,
        "noplaylist": True,
        "cookiefile": cookiefile,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def extract_audio(video_path: str, audio_dir: str = AUDIO_DIR) -> str:
    """Extract a 16kHz mono WAV track from a video."""
    wav = Path(audio_dir) / (Path(video_path).stem + ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", str(wav)],
        check=True,
    )
    return str(wav)


def transcribe(audio_or_video_path: str, whisper_model) -> str:
    """Transcribe a media file to text with Whisper."""
    return whisper_model.transcribe(audio_or_video_path)["text"]


def downsample_video(video_path: str, num_frames: int = 32) -> str:
    """Downsample a video to ~``num_frames`` frames at 224px width."""
    out = video_path.replace(".mp4", "_ds.mp4")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path],
        capture_output=True,
        text=True,
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    step = max(1, int(duration * 30 / num_frames))  # assumes ~30fps source

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vf",
            f"select='not(mod(n\\,{step}))',scale=224:-1",
            "-vsync",
            "vfr",
            "-an",
            out,
        ],
        check=True,
    )
    return out


# ─────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────


def encode_message(message, processor, model):
    """Embed a chat-formatted multimodal message into a normalized vector."""
    import torch
    import torch.nn.functional as F
    from qwen_omni_utils import process_mm_info

    texts = (
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        + "<|endoftext|>"
    )

    audio, images, video = process_mm_info(message, use_audio_in_video=False)
    if video is not None:
        video = video[:MAX_FRAMES]

    inputs = processor(
        text=texts,
        audio=audio,
        images=images,
        videos=video,
        return_tensors="pt",
        padding="longest",
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    reps = outputs.hidden_states[-1][:, -1]
    return F.normalize(reps.float(), dim=-1).cpu()


def embed_video(path: str, processor, model):
    msg = [{"role": "user", "content": [{"type": "video", "video": path}]}]
    return encode_message(msg, processor, model)


def embed_query(text: str, processor, model):
    msg = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return encode_message(msg, processor, model)


# ─────────────────────────────────────────────
# Index building
# ─────────────────────────────────────────────


def build_index(
    video_links,
    processor,
    embed_model,
    whisper_model,
    *,
    video_dir: str = VIDEO_DIR,
    audio_dir: str = AUDIO_DIR,
    download: bool = False,
    downsample: bool = False,
):
    """Build a dense index ``{vid: {path, transcript, embedding}}`` over videos.

    By default this expects pre-downloaded, pre-downsampled assets on disk
    (``<vid>_ds.mp4`` under ``video_dir`` and ``<vid>.wav`` under ``audio_dir``).
    Set ``download`` / ``downsample`` to fetch and preprocess assets on the fly.
    """
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    index = {}
    print("Building index...")
    for url in video_links:
        vid = url.split("=")[-1]

        if download or downsample:
            raw_path = os.path.join(video_dir, vid + ".mp4")
            if download and not os.path.exists(raw_path):
                download_youtube(url, raw_path)
            audio_path = raw_path
            video_path = downsample_video(raw_path) if downsample else raw_path
        else:
            video_path = os.path.join(video_dir, vid + "_ds.mp4")
            audio_path = os.path.join(audio_dir, vid + ".wav")

        transcript = transcribe(audio_path, whisper_model)
        emb = embed_video(video_path, processor, embed_model)

        index[vid] = {
            "path": video_path,
            "transcript": transcript,
            "embedding": emb,
        }

    return index


# ─────────────────────────────────────────────
# Dense retrieval
# ─────────────────────────────────────────────


def retrieve_scored(
    subquery: str,
    index,
    processor,
    embed_model,
    *,
    top_k: int = TOP_K,
    sim_threshold: float = SIM_THRESHOLD,
    exclude: set | None = None,
) -> list[tuple[str, float]]:
    """Return the ``top_k`` ``(vid, score)`` pairs most similar to ``subquery``.

    Entries scoring below ``sim_threshold`` are dropped, and any ``vid`` in
    ``exclude`` is skipped (useful for iterative retrieval that avoids
    re-surfacing already-seen videos).
    """
    import torch.nn.functional as F

    exclude = exclude or set()
    q_emb = embed_query(subquery, processor, embed_model)

    scored = []
    for vid, data in index.items():
        if vid in exclude:
            continue
        sim = F.cosine_similarity(q_emb, data["embedding"], dim=-1).item()
        if sim >= sim_threshold:
            scored.append((sim, vid))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(vid, sim) for sim, vid in scored[:top_k]]


def retrieve(subquery: str, index, processor, embed_model, **kwargs):
    """Convenience wrapper returning the ranked index *entries* (dicts).

    Same ranking as :func:`retrieve_scored`; accepts the same keyword args
    (``top_k`` / ``sim_threshold`` / ``exclude``).
    """
    return [
        index[vid] for vid, _ in retrieve_scored(subquery, index, processor, embed_model, **kwargs)
    ]


# ─────────────────────────────────────────────
# `retrieve` command
# ─────────────────────────────────────────────


def load_subqueries(path: str) -> list[tuple[str, str]]:
    """Load ``subqueries.jsonl`` into ``[(query_id, query_text), ...]``."""
    subqueries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                subqueries.append((obj["query_id"], obj["query"]))
    print(f"Loaded {len(subqueries)} sub-queries")
    return subqueries


def run(cfg: DictConfig) -> None:
    """OmniEmbed retrieval for the QA system → ``retrieved_videos.json``.

    This is the QA-system retrieval path. It is independent of the tevatron
    first-stage search / fusion pipeline: it reads the same ``subqueries.jsonl``
    but writes its own artifact (``data.retrieved_videos``) — the top videos per
    sub-query that the generation system answers over.
    """
    rc = cfg.runtime.retrieve
    emb = cfg.model.embedder

    print("=== QA retrieval (OmniEmbed) ===")
    subqueries = load_subqueries(cfg.data.subqueries_jsonl)
    with open(cfg.data.video_links, encoding="utf-8") as f:
        video_links = json.load(f)
    print(f"Loaded {len(video_links)} videos")

    processor, embed_model, whisper_model = load_retrieval_models(
        emb.model,
        emb.adapter,
        emb.whisper,
        emb.cache_dir or None,
    )

    index = build_index(
        video_links,
        processor,
        embed_model,
        whisper_model,
        video_dir=cfg.data.video_dir,
        audio_dir=cfg.data.audio_dir,
        download=rc.download,
        downsample=rc.downsample,
    )

    results = {}
    for sq_id, text in subqueries:
        hits = retrieve_scored(
            text,
            index,
            processor,
            embed_model,
            top_k=rc.top_k,
            sim_threshold=rc.sim_threshold,
        )
        results[sq_id] = {
            "query": text,
            "videos": [
                {"vid": vid, "score": score, "path": index[vid]["path"]} for vid, score in hits
            ],
        }
        print(f"  {sq_id}: {len(hits)} videos")

    out_path = cfg.data.retrieved_videos
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved retrieved videos to {out_path}")


def main(argv: list[str] | None = None) -> int:
    """Entry point: remaining argv are Hydra ``key=value`` overrides."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = build_config(overrides)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
