"""
Evaluate VLMs on video QA tasks using vLLM offline inference.

Usage:
    python eval_vlms.py \
        --video_dir /path/to/videos \
        --input_json /path/to/qa.json \
        --output_dir results/ \
        --model Qwen/Qwen3.5-9B
"""

import os
from abc import ABC, abstractmethod


class VLM(ABC):
    """Base class for Vision-Language Models."""

    @abstractmethod
    def infer(self, video_path: str | None, query: str, **kwargs) -> str:
        """Run inference on a video and query. Returns the raw model output."""
        pass

    def infer_batch(self, items: list[tuple], **kwargs) -> list[str]:
        """Run a batch of independent ``(video_path, query)`` requests.

        Each item is answered exactly as a standalone :meth:`infer` call would
        — independent conversation, independent output — and results come back
        in the same order as ``items``. This default loops :meth:`infer`;
        backends with native batched inference (e.g. vLLM) override it to submit
        all requests in one call so they run concurrently.
        """
        return [self.infer(video_path=vp, query=q, **kwargs) for vp, q in items]


class Qwen3_5_VL(VLM):
    """Inference class for Qwen3.5-VL models via vLLM."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3.5-9B",
        download_dir: str | None = None,
        fps: float = 1.0,
        max_frames: int = None,
        enable_thinking: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_tokens: int = 1024,
        seed: int = 42,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        allowed_local_media_path: str = None,
    ):
        # Some cluster Python builds omit the stdlib sqlite extension. vLLM's
        # structured-output stack imports diskcache, which requires sqlite3.
        try:
            import sqlite3  # noqa: F401
        except ModuleNotFoundError:
            import sys

            import pysqlite3

            sys.modules["sqlite3"] = pysqlite3

        # vLLM launches its EngineCore in a subprocess. The default start
        # method is "fork", which fails ("Cannot re-initialize CUDA in forked
        # subprocess") whenever CUDA is already initialized in this parent
        # process -- which happens when a transformers model (device_map="auto")
        # was loaded earlier in the same run, or just from vLLM's own
        # device-capability probing. Force "spawn" so the engine subprocess can
        # safely initialize CUDA. setdefault keeps any user-provided override.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from vllm import LLM, SamplingParams

        self.fps = fps
        self.max_frames = max_frames
        self.enable_thinking = enable_thinking

        # Shard the model across every GPU visible to this process so a big VLM
        # that doesn't fit on one card loads (tensor parallel). Auto-detected
        # from the device count -- 1 card => 1 (normal single-GPU load).
        import torch

        tensor_parallel_size = max(1, torch.cuda.device_count())

        llm_kwargs = dict(
            model=model,
            seed=seed,
            tensor_parallel_size=tensor_parallel_size,
        )
        # Only pass download_dir when explicitly set. An empty string makes vLLM
        # treat the cache as a relative path (== cwd), so models would download
        # into the repo root; omitting it lets HF use its default (~/.cache/huggingface).
        if download_dir:
            llm_kwargs["download_dir"] = download_dir
        if allowed_local_media_path is not None:
            llm_kwargs["allowed_local_media_path"] = allowed_local_media_path
        print(f"[vllm] loading {model} with tensor_parallel_size={tensor_parallel_size}")
        self.llm = LLM(**llm_kwargs)

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens * 5 if enable_thinking else max_tokens,
            seed=seed,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
        )

    def infer(
        self,
        video_path: str | None = None,
        query: str = "",
        fps: float = None,
        max_frames: int = None,
        enable_thinking: bool = None,
        video_first: bool = True,
    ) -> str:
        """
        Run inference on an optional video and query.

        Args:
            video_path: Path or file:// URL to the video. When None, runs
                text-only inference (fps, max_frames, video_first are ignored).
            query: The text prompt to send alongside the video.
            fps: Frames per second for video sampling (overrides instance default).
            max_frames: Max frames to sample (overrides instance default).
            enable_thinking: Whether to enable thinking traces (overrides instance default).
            video_first: If True, video appears before the text query; if False, after.

        Returns:
            The raw model output as a string.
        """
        enable_thinking = enable_thinking if enable_thinking is not None else self.enable_thinking

        if video_path is None:
            # Text-only inference
            content = [{"type": "text", "text": query}]
            conversation = [{"role": "user", "content": content}]
            outputs = self.llm.chat(
                conversation,
                sampling_params=self.sampling_params,
                chat_template_kwargs={"enable_thinking": enable_thinking},
            )
            return outputs[0].outputs[0].text

        fps = fps if fps is not None else self.fps
        max_frames = max_frames if max_frames is not None else self.max_frames

        if not video_path.startswith("file://"):
            video_path = f"file://{video_path}"

        video_content = {"type": "video_url", "video_url": {"url": video_path}}
        text_content = {"type": "text", "text": query}
        content = [video_content, text_content] if video_first else [text_content, video_content]

        conversation = [{"role": "user", "content": content}]

        mm_processor_kwargs = {"fps": fps}
        if max_frames is not None:
            mm_processor_kwargs["max_frames"] = max_frames

        outputs = self.llm.chat(
            conversation,
            sampling_params=self.sampling_params,
            mm_processor_kwargs=mm_processor_kwargs,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        return outputs[0].outputs[0].text

    def infer_batch(
        self,
        items: list[tuple],
        fps: float = None,
        max_frames: int = None,
        enable_thinking: bool = None,
        video_first: bool = True,
    ) -> list[str]:
        """Batch of independent ``(video_path, query)`` requests in one vLLM call.

        Each item is its own conversation (its own video frames + prompt), so the
        result for a given item is identical to calling :meth:`infer` on it alone
        — only the scheduling changes: vLLM runs the whole batch concurrently
        instead of one round-trip at a time. Outputs are returned in input order.

        All items share one ``mm_processor_kwargs`` (fps/max_frames), so a batch
        must use the same sampling settings; mixing text-only and video items is
        fine (the video kwargs are simply ignored by the text-only ones).
        """
        if not items:
            return []
        enable_thinking = enable_thinking if enable_thinking is not None else self.enable_thinking
        fps = fps if fps is not None else self.fps
        max_frames = max_frames if max_frames is not None else self.max_frames

        conversations = []
        any_video = False
        for video_path, query in items:
            if video_path is None:
                conversations.append(
                    [{"role": "user", "content": [{"type": "text", "text": query}]}]
                )
                continue
            any_video = True
            if not video_path.startswith("file://"):
                video_path = f"file://{video_path}"
            video_content = {"type": "video_url", "video_url": {"url": video_path}}
            text_content = {"type": "text", "text": query}
            if video_first:
                content = [video_content, text_content]
            else:
                content = [text_content, video_content]
            conversations.append([{"role": "user", "content": content}])

        chat_kwargs = dict(
            sampling_params=self.sampling_params,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        if any_video:
            mm_processor_kwargs = {"fps": fps}
            if max_frames is not None:
                mm_processor_kwargs["max_frames"] = max_frames
            chat_kwargs["mm_processor_kwargs"] = mm_processor_kwargs

        outputs = self.llm.chat(conversations, **chat_kwargs)
        return [o.outputs[0].text for o in outputs]


class APIVLM(VLM):
    """VLM backend that calls a remote OpenAI-compatible API server.

    Drop-in replacement for Qwen3_5_VL: same infer() signature, but requests
    are shipped to a vLLM/LiteLLM server (e.g. http://rack3n07:4000/v1)
    instead of loading weights onto local GPUs.

    Videos are sent inline as base64 ``data:`` URIs rather than ``file://``
    paths. The remote server reads file:// paths from its own filesystem,
    which fails when the video lives on node-local scratch ($TMPDIR) the
    server can't see, or outside its --allowed-local-media-path. Embedding
    the bytes in the request sidesteps both. Encodings are cached so repeated
    queries against the same video read/encode it only once.

    Verify the server is up with: ``curl <api_base>/models``
    """

    def __init__(
        self,
        api_base: str = "http://rack3n07:4000/v1",
        api_model: str = "qwen-27b",
        api_key: str = "dummy",
        fps: float = 1.0,
        enable_thinking: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_tokens: int = 1024,
        seed: int = 42,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
    ):
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai is required for the API backend. Install it with: pip install openai"
            ) from exc

        # api_key is required by the client library but the LiteLLM proxy
        # doesn't validate it -- any non-empty string works.
        self._client = openai.OpenAI(base_url=api_base, api_key=api_key)
        self._api_model = api_model
        self.fps = fps
        self.enable_thinking = enable_thinking
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._max_tokens = max_tokens
        self._seed = seed
        self._repetition_penalty = repetition_penalty
        self._presence_penalty = presence_penalty
        # sent in the OpenAI `user` field so the proxy can apply per-user backoff
        import getpass

        self._user = os.environ.get("USER") or getpass.getuser()
        self._video_uri_cache: dict[str, str] = {}

    def _video_data_uri(self, video_path: str) -> str:
        """Read a local video and return an inline base64 data: URI (cached)."""
        import base64

        path = video_path[len("file://") :] if video_path.startswith("file://") else video_path
        cached = self._video_uri_cache.get(path)
        if cached is None:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            cached = f"data:video/mp4;base64,{b64}"
            self._video_uri_cache[path] = cached
        return cached

    def _chat(self, content, enable_thinking: bool) -> str:
        response = self._client.chat.completions.create(
            model=self._api_model,
            messages=[{"role": "user", "content": content}],
            temperature=self._temperature,
            top_p=self._top_p,
            # In thinking mode the model emits a reasoning trace before the
            # answer; multiply max_tokens by 5 to leave room for both
            # (mirrors Qwen3_5_VL).
            max_tokens=self._max_tokens * 5 if enable_thinking else self._max_tokens,
            seed=self._seed,
            presence_penalty=self._presence_penalty,
            user=self._user,
            extra_body={
                # vLLM-specific sampling knobs not in the OpenAI schema
                "top_k": self._top_k,
                "repetition_penalty": self._repetition_penalty,
                # required: the proxy defaults to thinking=True for Qwen3;
                # content comes back None without this
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
        )
        msg = response.choices[0].message
        # when thinking is on, the reply is in reasoning_content rather than content
        return msg.content or getattr(msg, "reasoning_content", None) or ""

    def infer(
        self,
        video_path: str | None = None,
        query: str = "",
        fps: float = None,
        max_frames: int = None,
        enable_thinking: bool = None,
        video_first: bool = True,
    ) -> str:
        """
        Run inference on an optional video and query via the remote API.

        Mirrors Qwen3_5_VL.infer(). max_frames is accepted for signature
        compatibility but ignored (frame sampling is controlled server-side
        via the fps hint).
        """
        enable_thinking = enable_thinking if enable_thinking is not None else self.enable_thinking

        if video_path is None:
            # Text-only inference
            return self._chat([{"type": "text", "text": query}], enable_thinking)

        fps = fps if fps is not None else self.fps
        data_uri = self._video_data_uri(video_path)

        # Video first by default: keeps the (large) video prefix identical
        # across queries about the same video so the server can reuse its
        # cached prefix.
        video_content = {"type": "video_url", "video_url": {"url": data_uri}, "fps": fps}
        text_content = {"type": "text", "text": query}
        content = [video_content, text_content] if video_first else [text_content, video_content]
        return self._chat(content, enable_thinking)


class UNLI(VLM):
    """UNLI video+claim scorer.

    Supports two loading modes:

    - **Merged checkpoint** (default):
        ``UNLI(model="AdoptedIrelia/UNLI")``

    - **Base + LoRA adapter**:
        ``UNLI(base_model="Qwen/Qwen2.5-Omni-3B", lora_path="AdoptedIrelia/UNLI/lora")``
        (requires ``peft``; ``model`` is ignored in this mode)
    """

    def __init__(
        self,
        model: str = "AdoptedIrelia/UNLI",
        base_model: str | None = None,
        lora_path: str | None = None,
        download_dir: str = "",
        fps: float = 0.5,
        resized_height: int = 256,
        resized_width: int = 256,
        new_token_num: int = 100,
        new_token_prefix: str = "<CON_{idx}>",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: str = "sdpa",
    ):
        # --- validate load-mode -----------------------------------------
        if base_model is not None and lora_path is None:
            raise ValueError(
                "base_model was provided without lora_path. "
                "Either provide both base_model and lora_path for LoRA mode, "
                "or omit both for merged-checkpoint mode."
            )
        if lora_path is not None and base_model is None:
            raise ValueError(
                "lora_path was provided without base_model. "
                "LoRA mode requires both base_model and lora_path."
            )
        # if base_model is not None and lora_path is not None and model != "AdoptedIrelia/UNLI":
        #     raise ValueError(
        #         "When using base_model + lora_path, the model parameter must "
        #         "remain at its default ('AdoptedIrelia/UNLI') because it is "
        #         "ignored in LoRA mode. Passing a custom model value alongside "
        #         "base_model/lora_path is ambiguous."
        #     )

        # --- store config -----------------------------------------------
        self.fps = fps
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.new_token_num = new_token_num
        self.new_token_prefix = new_token_prefix

        # --- lazy imports (avoid top-level heavy deps) ------------------
        from qwen_omni_utils import process_mm_info
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        self.process_mm_info = process_mm_info

        # --- determine load path ----------------------------------------
        use_lora = lora_path is not None

        if use_lora:
            checkpoint = model
        else:
            checkpoint = model

        pretrained_kwargs = dict(
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        if download_dir:
            pretrained_kwargs["cache_dir"] = download_dir

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            checkpoint,
            **({"cache_dir": download_dir} if download_dir else {}),
        )
        self._model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            checkpoint,
            **pretrained_kwargs,
        )

        if use_lora:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "peft is required for LoRA mode. Install it with: pip install peft"
                ) from exc
            normalized_lora_path = lora_path
            # Accept HF adapter paths written as namespace/repo/subfolder and
            # resolve the adapter subfolder to a local path before giving it
            # to peft. This avoids model_id/subfolder handling differences
            # across peft/huggingface_hub versions.
            if (
                isinstance(lora_path, str)
                and not os.path.exists(lora_path)
                and lora_path.count("/") >= 2
            ):
                from huggingface_hub import snapshot_download

                parts = lora_path.split("/")
                repo_id = "/".join(parts[:2])
                subfolder = "/".join(parts[2:])
                snapshot_path = snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[f"{subfolder}/*"],
                    cache_dir=download_dir or None,
                )
                normalized_lora_path = os.path.join(snapshot_path, subfolder)
            self._model = PeftModel.from_pretrained(
                self._model,
                normalized_lora_path,
            )

        self._model.eval()

    def score(self, video_path: str, claim: str) -> float:
        """Score a single video+claim pair.

        Returns a float in [0, 1] representing the probability that the claim
        is supported by the video.
        """
        import torch

        from marquis.common.model_prompts import NLI_PROMPT, NLI_SYSTEM_PROMPT
        from marquis.common.model_utils import extract_score

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": NLI_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": NLI_PROMPT.format(text=claim)},
                    {
                        "type": "video",
                        "video": video_path,
                        "resized_height": self.resized_height,
                        "resized_width": self.resized_width,
                        "fps": self.fps,
                    },
                ],
            },
        ]

        # Try with audio first; fall back to no-audio on failure.
        for use_audio in (True, False):
            try:
                text = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                audios, images, videos = self.process_mm_info(
                    conversation,
                    use_audio_in_video=use_audio,
                )
                inputs = self.processor(
                    text=text,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=use_audio,
                )
                break
            except Exception:
                if not use_audio:
                    raise

        # The UNLI jobs run on a single GPU. Move processor tensors onto the
        # model's active device before the forward pass to avoid CPU/CUDA
        # index_select mismatches inside the Qwen2.5-Omni stack.
        target_device = next(self._model.parameters()).device
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(target_device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        return extract_score(
            logits,
            self.processor,
            new_token_num=self.new_token_num,
            new_token_prefix=self.new_token_prefix,
        )

    def infer(self, video_path: str, query: str, **kwargs) -> str:
        """Run inference on a single video and claim.

        Delegates to ``score()`` and returns the result as a string.
        """
        return str(self.score(video_path, query))
