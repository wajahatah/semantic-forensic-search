"""Gemma 4 scorer (spec-vlm-followups E6, D-032): the same relevance score as `sfs.teacher.vlm` —
log p("yes") - log p("no") of the next token — for Google's Gemma 4 vision-language models, with the
same frames (`prepare_frames`, our frame metadata, no resampling) and the same question text
(INSTRUCT_PROMPT), put through Gemma's own chat template with the video first.

Fitting gemma-4-E4B-it (16 GB in bf16) on the 8 GB laptop GPU, without quantisation:
- the per-layer embedding table (5.8 GB) stays in CPU memory and is looked up there; only the rows
  for the prompt's tokens go to the GPU (`CpuLookup`). A table lookup, so the result is exact;
- the audio tower stays on the CPU and is never called (video only);
- the vision tower, token embeddings, final norm and as many decoder layers as fit in `gpu_budget_gb`
  live on the GPU; the remaining decoder layers are kept in CPU memory and copied to the GPU for each
  pass (`stream_weights`). Exact; only slower.
The vision encoder runs once per clip; the language model makes one full pass per question (no
KV-cache shortcut yet: `score_clip` equals `score_full`, `shortcut = False`).
`last_mass` holds p(yes) + p(no) of the last call (G0 reports it: does the model answer yes/no?).
"""
from __future__ import annotations

import json
import os

import numpy as np

from sfs.teacher.offload import CpuLookup, _nbytes, place_layers, stream_weights  # noqa: F401
from sfs.teacher.vlm import INSTRUCT_PROMPT, NO, YES, _metadata, prepare_frames

GEMMA_TYPES = ("gemma4", "gemma4_unified")


def attention(name: str = "sdpa") -> dict:
    """Attention kernel per sub-model: `name` for text and vision; the audio tower supports only eager
    (it is never called for video, but it is built when the model loads)."""
    return {"": name, "text_config": name, "vision_config": name, "audio_config": "eager"}


def model_type(path: str) -> str:
    return json.load(open(os.path.join(path, "config.json"))).get("model_type", "")


def place(model, device: str, gpu_budget_gb: float | None) -> dict:
    """Split a CPU-loaded Gemma 4 between CPU and `device` as described in the module docstring.
    Returns the placement summary (recorded in G0)."""
    inner = model.model
    lm = inner.language_model
    cpu_only = [m for m in (getattr(inner, "audio_tower", None), getattr(inner, "embed_audio", None)) if m is not None]
    ple = getattr(lm, "embed_tokens_per_layer", None)
    summary = place_layers(model, lm.layers, device, gpu_budget_gb, cpu_only + ([ple] if ple is not None else []), pin=True)
    if ple is not None:
        lm.embed_tokens_per_layer = CpuLookup(ple, device)
    return {**summary, "per_layer_table_on_cpu": ple is not None, "audio_on_cpu": bool(cpu_only)}


class GemmaScorer:
    shortcut = False

    def __init__(self, path: str, kind: str = "instruct", device: str = "cuda", gpu_budget_gb: float | None = 6.0):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if kind != "instruct":
            raise ValueError("Gemma 4 is scored as an instruct model only")
        if model_type(path) not in GEMMA_TYPES:
            raise ValueError(f"{path} is not a Gemma 4 model")
        self.kind, self.device = kind, device
        self.processor = AutoProcessor.from_pretrained(path)
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=attention())
        self.placement = place(model.eval(), device, gpu_budget_gb if device != "cpu" else None)
        self.model = model
        self._tokens()
        src = os.path.join(path, "_source.json")
        self.revision = json.load(open(src))["revision"] if os.path.exists(src) else "unknown"
        self.model_id = f"{os.path.basename(os.path.normpath(path))}@{self.revision[:12]}"

    @classmethod
    def from_parts(cls, model, processor, device: str = "cpu", model_id: str = "test", gpu_budget_gb=None):
        """Wrap an already-built model + processor (tests use a tiny random-weight Gemma 4)."""
        self = cls.__new__(cls)
        self.kind, self.device, self.processor = "instruct", device, processor
        self.placement = place(model.eval(), device, gpu_budget_gb if device != "cpu" else None)
        self.model, self.revision, self.model_id = model, "test", model_id
        self._tokens()
        return self

    def _tokens(self):
        tok = self.processor.tokenizer
        self.yes = [tok.convert_tokens_to_ids(t) for t in YES["instruct"]]
        self.no = [tok.convert_tokens_to_ids(t) for t in NO["instruct"]]
        self.last_mass = np.empty(0, np.float32)

    def prompt(self, query: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "video"},
                                             {"type": "text", "text": INSTRUCT_PROMPT.format(query=query)}]}]
        return self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def score_full(self, frames, frame_numbers, fps, queries) -> np.ndarray:
        """One pass per question. The vision encoder runs once per clip and its output is passed to the
        model through `mm_encoder_outputs` (a supported input): in transformers 5.20.2
        `Gemma4Model.forward` does not hand `num_frames_per_video` to `get_video_features`, so video
        given as pixels always fails there. The encoder output depends only on the frames, so this is
        the same computation, done once instead of once per question."""
        import torch
        vid = prepare_frames(frames)
        meta = _metadata(frame_numbers, fps, len(vid))
        out, mass = np.empty(len(queries), np.float32), np.empty(len(queries), np.float32)
        video = None
        with torch.no_grad():
            for i, q in enumerate(queries):
                enc = self.processor(text=[self.prompt(q)], videos=[vid], video_metadata=[meta],
                                     do_sample_frames=False, return_tensors="pt").to(self.device)
                pix = enc.pop("pixel_values_videos")
                pos, nfr = enc.pop("video_position_ids"), enc.pop("num_frames_per_video")
                if video is None:
                    video = self.model.model.get_video_features(pix, pos, nfr, return_dict=True)
                logits = self.model(**enc, mm_encoder_outputs={"video": video}, use_cache=False,
                                    logits_to_keep=1).logits[:, -1]
                lp = torch.log_softmax(logits.float(), dim=-1)[0]
                y, n = torch.logsumexp(lp[self.yes], 0), torch.logsumexp(lp[self.no], 0)
                out[i] = float(y - n)
                mass[i] = float(torch.exp(y) + torch.exp(n))
        self.last_mass = mass
        return out

    def score_clip(self, frames, frame_numbers, fps, queries) -> np.ndarray:
        return self.score_full(frames, frame_numbers, fps, queries)      # no shortcut (see docstring)

    # ------------------------------------------------------------ device hooks (as VLMScorer)
    def sync(self):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def reset_peak_memory(self):
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def peak_memory_gb(self) -> float:
        import torch
        return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    def set_attn(self, name: str):
        self.model.set_attn_implementation({"text_config": name, "vision_config": name})
