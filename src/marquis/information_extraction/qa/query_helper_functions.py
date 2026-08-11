"""Shared helpers for the QA runners: model loading, transcription, VLM QA.

Model/whisper/video-dir settings are passed in by the callers (driven by the
Hydra config) rather than read from a config module.
"""

from __future__ import annotations

import json
import os

from marquis.information_extraction.prompts import QA_ASK_PROMPT, QA_COMBINE_PROMPT


class Transcriber:
    """Serves transcripts to the QA runners, by video id.

    Backed either by a precomputed ``{video_id: transcript}`` cache (produced by
    the ``prepare-transcripts`` command, which runs Whisper in its *own* process so
    it never shares the GPU with the QA VLM) or, as a fallback, by a live Whisper
    model that transcribes the audio file on demand.
    """

    def __init__(self, whisper_model=None, cache: dict | None = None):
        self._whisper = whisper_model
        self._cache = cache or {}

    def get(self, vid: str, audio_path: str) -> str:
        if vid in self._cache:
            return self._cache[vid]
        if self._whisper is None:
            print(f"[warn] no cached transcript for {vid!r} and no Whisper model loaded")
            return ""
        return self._whisper.transcribe(audio_path)["text"]


def _load_whisper(whisper_model: str):
    """Load a Whisper model into the local Whisper cache dir.

    NOTE: this opens a CUDA context in *this* process. When the QA VLM (vLLM) is
    also live, that is a second context on the same card; on a GPU in
    EXCLUSIVE_PROCESS compute mode the two clash and Whisper's load fails with
    ``cudaErrorDevicesUnavailable``. Run ``prepare-transcripts`` first (separate
    process) and point ``data.transcripts`` at its output to avoid loading Whisper
    here at all.
    """
    import whisper

    whisper_cache = os.path.join(os.environ.get("XDG_CACHE_HOME", "/tmp"), "whisper")
    os.makedirs(whisper_cache, exist_ok=True)
    return whisper.load_model(whisper_model, download_root=whisper_cache)


def load_transcriber(whisper_model: str, transcripts: str | None = None) -> Transcriber:
    """Build a :class:`Transcriber`: precomputed cache when available, else Whisper.

    When ``transcripts`` points at an existing ``{video_id: transcript}`` JSON the
    QA runners read from it and Whisper is never loaded (so no GPU-context clash
    with the VLM). Otherwise we fall back to a live Whisper model.
    """
    if transcripts and os.path.exists(transcripts):
        with open(transcripts, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"[transcripts] loaded {len(cache)} cached transcripts from {transcripts}")
        return Transcriber(cache=cache)
    if transcripts:
        print(
            f"[warn] data.transcripts={transcripts!r} not found; falling back to live "
            "Whisper (run `prepare-transcripts` first to avoid loading Whisper here)"
        )
    print("Loading Whisper...")
    return Transcriber(whisper_model=_load_whisper(whisper_model))


def load_models(
    qa_model: str,
    cache_dir: str | None,
    whisper_model: str,
    video_root: str | None = None,
    transcripts: str | None = None,
    fps: float = 1.0,
    max_frames: int | None = 128,
):
    """Load the QA VLM + a transcript source (cache or Whisper).

    The QA model is the same Qwen3.5-VL vLLM backend used by Step-1 extraction, so
    it actually samples the video frames (not just the transcript) when answering.
    ``video_root`` is handed to vLLM as ``allowed_local_media_path`` so it may read
    the materialized ``file://`` mp4s. ``transcripts`` is an optional path to a
    precomputed transcript cache (see :func:`load_transcriber`).

    ``fps``/``max_frames`` control video frame sampling and must match Step-1
    extraction (``fps=1.0``, ``max_frames=128``): with ``max_frames=None`` the Qwen
    processor packs many low-resolution frames into a fixed pixel budget, blurring
    fine detail (on-screen text, distant figures) the model needs to answer, so it
    falls back to "I don't know" on videos extraction handled fine.
    """
    from marquis.common.model_backends import Qwen3_5_VL

    print("Loading Qwen VLM...")
    vlm = Qwen3_5_VL(
        model=qa_model,
        download_dir=cache_dir or None,
        allowed_local_media_path=video_root,
        fps=fps,
        max_frames=max_frames,
    )

    transcriber = load_transcriber(whisper_model, transcripts)
    return vlm, transcriber


def build_topic_videos(
    video_ids, models, video_dir: str, audio_dir: str, audio_ext: str = ".m4a"
):
    """Resolve a topic's video IDs to {id, path, transcript} records.

    No embedding/retrieval: the query_video_mapping already tells us which videos
    belong to the query. Transcripts come from Whisper run on the materialized
    audio file (``audio_dir/{chunk_id}{audio_ext}``); ``path`` is the video file
    (``video_dir/{chunk_id}.mp4``) the QA VLM samples frames from. Both are
    materialized up front by ``prepare-videos`` / ``prepare-audio``.
    """
    _vlm, transcriber = models
    videos = []
    for vid in video_ids:
        video_path = os.path.join(video_dir, vid + ".mp4")
        audio_path = os.path.join(audio_dir, vid + audio_ext)
        if not os.path.exists(audio_path):
            print(f"[warn] missing audio file, skipping: {audio_path}")
            continue
        transcript = transcriber.get(vid, audio_path)
        videos.append({"id": vid, "path": video_path, "transcript": transcript})
    return videos


def _ask_qwen_query(transcript, question):
    """Build the per-video QA prompt: Whisper transcript + the question."""
    prompt = QA_ASK_PROMPT.format(question=question)
    return f"Transcript:\n{transcript}\n\n{prompt}" if transcript else prompt


def ask_qwen_batch(videos, question, vlm):
    """Answer ``question`` about each video, batched into one VLM call.

    Each video is an independent request — its own frames (``video_path``) and
    Whisper transcript, grounded in both vision and audio — so an answer is
    identical to asking that video on its own; the videos are merely submitted
    together so the VLM runs them concurrently. Returns answers in input order.
    """
    items = [(v["path"], _ask_qwen_query(v["transcript"], question)) for v in videos]
    return vlm.infer_batch(items)


def is_valid_answer(ans: str) -> bool:
    return ans not in ("I don't know", "I don't know.")


def combine_answers(subquery, video_answer_pairs, vlm):
    print("PAIRS:", [ans for _, ans in video_answer_pairs])
    valid_pairs = [(v, ans) for v, ans in video_answer_pairs if is_valid_answer(ans)]
    print("VALID PAIRS:", [ans for _, ans in valid_pairs])

    if not valid_pairs:
        return "I don't know", []

    valid_answers = [ans for _, ans in valid_pairs]

    prompt = QA_COMBINE_PROMPT.format(subquery=subquery, valid_answers=valid_answers)
    final_answer = vlm.infer(video_path=None, query=prompt)

    sources = [v["path"] for v, _ in valid_pairs]
    print(final_answer, sources)
    return final_answer, sources


# ─────────────────────────────────────────────
# Optional dense-retrieval path (used when query_video_mapping is unavailable)
# ─────────────────────────────────────────────
# Video embeddings are precomputed (loaded from a file); at query time we only
# embed the query (reusing the retrieval branch's OmniEmbed engine) and score it
# against the loaded video vectors. Transcripts — needed by the VLM, not by
# retrieval — are produced lazily for the few retrieved videos. File convention:
# ``<vid>.mp4`` under ``video_dir`` (matching build_topic_videos).


def load_embedder(embed_model: str, embed_adapter: str, cache_dir: str | None):
    """Load the OmniEmbed processor + model used to embed sub-queries."""
    from marquis.retrieval.video_retrieval import load_embedder as _load_embedder

    return _load_embedder(embed_model, embed_adapter, cache_dir)


def load_video_embeddings(path: str) -> dict:
    """Load precomputed video embeddings into ``{vid: tensor[1, dim]}``.

    Accepts a torch ``.pt`` (a ``{vid: vector}`` dict, or ``{"ids", "embeddings"}``),
    ``.npz`` (arrays ``ids`` + ``embeddings``), ``.jsonl`` (``{"id", "embedding"}``
    per line) or ``.json`` (``{vid: vector}``).
    """
    import torch

    def _as_row(vec):
        return torch.as_tensor(vec, dtype=torch.float).reshape(1, -1)

    if path.endswith((".pt", ".pth")):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "embeddings" in obj and "ids" in obj:
            emb = torch.as_tensor(obj["embeddings"]).float()
            return {str(vid): emb[i].reshape(1, -1) for i, vid in enumerate(obj["ids"])}
        if isinstance(obj, dict):
            return {str(k): _as_row(v) for k, v in obj.items()}
        raise SystemExit(f"Unsupported .pt structure in {path}")
    if path.endswith(".npz"):
        import numpy as np

        z = np.load(path, allow_pickle=True)
        emb = torch.as_tensor(z["embeddings"]).float()
        return {str(vid): emb[i].reshape(1, -1) for i, vid in enumerate(z["ids"])}
    if path.endswith(".jsonl"):
        out = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    vid = str(o.get("id") or o.get("vid") or o.get("docid"))
                    out[vid] = _as_row(o.get("embedding") or o.get("vector"))
        return out
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    return {str(k): _as_row(v) for k, v in obj.items()}


def build_retrieval_index(video_embeddings_path: str, video_dir: str):
    """Build a dense index from precomputed video embeddings (no video encoding).

    Returns ``{vid: {id, path, transcript=None, embedding}}``; transcripts are
    filled in lazily by :func:`retrieve_videos` for retrieved videos only.
    """
    embeddings = load_video_embeddings(video_embeddings_path)
    print(f"Loaded {len(embeddings)} video embeddings from {video_embeddings_path}")
    return {
        vid: {
            "id": vid,
            "path": os.path.join(video_dir, vid + ".mp4"),
            "transcript": None,
            "embedding": emb,
        }
        for vid, emb in embeddings.items()
    }


def retrieve_videos(
    query,
    index,
    embedder,
    transcriber,
    *,
    top_k: int,
    sim_threshold: float,
    audio_dir: str,
    audio_ext: str = ".m4a",
):
    """Embed ``query`` and return the top videos, transcribing the hits on demand.

    Transcripts come from the :class:`Transcriber` (precomputed cache or live
    Whisper) keyed by video id; the audio file (``audio_dir/{chunk_id}{audio_ext}``)
    is only read when Whisper has to transcribe a cache miss.
    """
    from marquis.retrieval.video_retrieval import retrieve

    processor, embed_model = embedder
    hits = retrieve(query, index, processor, embed_model, top_k=top_k, sim_threshold=sim_threshold)
    out = []
    for v in hits:
        audio_path = os.path.join(audio_dir, v["id"] + audio_ext)
        if not os.path.exists(audio_path):
            print(f"[warn] missing audio file, skipping: {audio_path}")
            continue
        if v.get("transcript") is None:
            v["transcript"] = transcriber.get(v["id"], audio_path)
        out.append(v)
    return out


def build_query_video_mapping(cfg, queries, mode: str) -> dict:
    """Resolve each query's videos as ``{query_id: [video_id, ...]}``.

    Reads the query-keyed ``data.query_video_mapping`` file directly (no
    event-name matching at run time). Tolerates absence outside ``topic`` mode
    (dense retrieval fills in).
    """
    from marquis.common.contracts import load_query_video_mapping

    path = cfg.data.get("query_video_mapping")
    if path and os.path.exists(path):
        return load_query_video_mapping(path)

    if mode == "topic":
        raise SystemExit(
            f"topic mode requires data.query_video_mapping (not found: {path!r})"
        )
    print(f"[info] no query_video_mapping at {path!r}; will use dense retrieval")
    return {}


def prepare_video_selection(cfg, queries):
    """Resolve the QA video-selection mode and (lazily) the retrieval context.

    Returns ``(query_video_mapping, mode, embedder, video_embeddings, top_k,
    sim_threshold)``. ``query_video_mapping`` is keyed by ``query_id``. The
    OmniEmbed query embedder is loaded only when retrieval is actually needed
    (``mode=retrieval``, or ``mode=auto`` and some query has no mapped videos);
    ``video_embeddings`` is the path to the precomputed video-embedding file.
    """
    rc = cfg.runtime.qa.retrieval
    mode = rc.mode
    mapping = build_query_video_mapping(cfg, queries, mode)

    def videos_missing(q):
        return not mapping.get(str(q.get("query_id")))

    need_retrieval = mode == "retrieval" or (
        mode == "auto" and any(videos_missing(q) for q in queries)
    )

    embedder, video_embeddings = None, None
    if need_retrieval:
        video_embeddings = rc.video_embeddings
        embedder = load_embedder(rc.embed_model, rc.embed_adapter, cfg.model.download_dir or None)

    return mapping, mode, embedder, video_embeddings, rc.top_k, rc.sim_threshold


def run_qa(
    queries,
    models,
    answer_one,
    *,
    query_video_mapping,
    video_dir,
    audio_dir,
    audio_ext=".m4a",
    mode="auto",
    embedder=None,
    video_embeddings=None,
    top_k=4,
    sim_threshold=0.1,
    output_path=None,
):
    """Drive QA over each query, choosing videos by query mapping or dense retrieval.

    For each query a ``get_videos(question)`` callable is built and handed to
    ``answer_one(subquery, get_videos, vlm)``:

    * mapped mode — ``get_videos`` ignores the question and returns the query's
      relevant videos (from ``query_video_mapping``, keyed by ``query_id``),
      transcribed once and shared across the query's sub-queries;
    * retrieval mode — ``get_videos`` dense-retrieves for the *given* question,
      so callers that re-ask (iterative QA) retrieve afresh every step.

    ``answer_one`` returns a list of result dicts, to which query_id/topic are
    attached.

    If ``output_path`` is given, the accumulated results are written there as
    JSONL after each query finishes, so progress survives a later failure. On a
    re-run, queries already present in ``output_path`` are skipped (resume), and
    a ``{output_path}.done`` marker is written once every query has been handled.
    """
    vlm, transcriber = models
    index = None
    results = []
    done_qids: set[str] = set()

    # Resume: reload any results from a previous, interrupted run and skip those
    # queries. A query only reaches the output file after all its sub-queries
    # finish, so a half-done query is never marked complete.
    if output_path and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                results.append(rec)
                done_qids.add(str(rec.get("query_id")))
        if done_qids:
            print(f"[resume] {len(done_qids)} queries already in {output_path}; skipping them")

    for q in queries:
        qid = str(q.get("query_id"))
        if qid in done_qids:
            print(f"[resume] skipping already-completed query {qid}")
            continue
        topic = q.get("title", "").title()
        video_ids = query_video_mapping.get(qid, []) if query_video_mapping else []
        use_mapped = mode != "retrieval" and bool(video_ids)

        if use_mapped:
            topic_vids = build_topic_videos(video_ids, models, video_dir, audio_dir, audio_ext)
            if not topic_vids:
                print(f"[warn] no usable video files for query {qid}; skipping")
                continue

            def get_videos(_question, _topic_vids=topic_vids):
                return _topic_vids

        elif mode == "topic":
            print(f"[warn] no videos mapped for query_id={qid}; skipping")
            continue
        else:
            if embedder is None or video_embeddings is None:
                raise SystemExit("retrieval mode requires an embedder and video_embeddings")
            if index is None:
                index = build_retrieval_index(video_embeddings, video_dir)
            retrieval_index = index

            def get_videos(question, _index=retrieval_index):
                return retrieve_videos(
                    question,
                    _index,
                    embedder,
                    transcriber,
                    top_k=top_k,
                    sim_threshold=sim_threshold,
                    audio_dir=audio_dir,
                    audio_ext=audio_ext,
                )

        for sub in q["sub_queries"]:
            for rec in answer_one(sub, get_videos, vlm):
                rec["query_id"] = q.get("query_id")
                rec["topic"] = topic
                results.append(rec)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"[ok] query {qid} done -> {len(results)} results so far -> {output_path}")

    # Every query handled: write a non-empty completion marker so callers (and
    # the dj_test step_done guard) can tell a finished run from an interrupted one.
    if output_path:
        with open(output_path + ".done", "w", encoding="utf-8") as f:
            f.write(f"done: {len(results)} results\n")
    return results
