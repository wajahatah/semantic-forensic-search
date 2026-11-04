"""VLM scorer: relevance of a (clip, text) pair as log p("yes") - log p("no") of the next token
(spec-rerank-cascade D2; VQAScore written as a margin, sigma(margin) = p_yes / (p_yes + p_no)).

Qwen-VL models, loaded from local weights through transformers' auto class (Qwen3-VL resolves to
Qwen3VLForConditionalGeneration as before; Qwen2.5-VL, spec-vlm-followups E7, to
Qwen2_5_VLForConditionalGeneration). `fp8_at_load` stores the language model in FP8 (128x128 blocks,
vision tower and lm_head kept bf16), for checkpoints that do not fit the GPU in bf16; such runs are
named `<model>+fp8load`. `gpu_budget_gb` loads the model into CPU memory and keeps only what fits on
the GPU (token embeddings looked up on the CPU, decoder layers that do not fit streamed per pass;
`sfs.teacher.offload`, exact): E4's Qwen3-VL-32B-FP8 does not fit 16 GB. `clip_batch` = questions per
cached batch (each copies the video's KV cache; smaller for big models). Roles:
- "instruct" (Qwen3-VL-2B-Instruct): the video comes FIRST and the question last, so the video's
  key/value cache is computed once per clip and reused for every question (`score_clip`).
- "reranker" (Qwen3-VL-Reranker-2B): its trained format puts the query BEFORE the document, so
  every pair is a full forward pass (`score_full`). Its score head is the yes-minus-no direction
  of the LM head, which is exactly the margin used here.
`score_full` also serves as the uncached reference for the instruct model (tests compare them).
A quantised checkpoint (E4: Qwen3-VL-8B-Instruct-FP8, FP8 language model, bf16 vision tower) is loaded
straight onto the device; unquantised ones keep the recorded load path. The device hooks (`sync`,
`reset_peak_memory`, `peak_memory_gb`, `set_attn`) let G0 and the scoring service treat a local scorer
and `sfs.teacher.remote.RemoteScorer` alike.

Every frame is resized so its longest side is FRAME_SIDE (fixed token budget per frame for both
views); an odd number of frames is padded with the last frame (temporal patch = 2).
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np

FRAME_SIDE = 448
INSTRUCT_PROMPT = "Does this video show the following: {query}? Answer yes or no."
RERANK_SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the Instruct '
                 'provided. Note that the answer can only be "yes" or "no".')
RERANK_INSTRUCTION = "Retrieve surveillance videos that show the activity described in the query."
YES = {"instruct": ("yes", "Yes"), "reranker": ("yes",)}
NO = {"instruct": ("no", "No"), "reranker": ("no",)}


def prepare_frames(frames: list[np.ndarray], side: int = FRAME_SIDE) -> np.ndarray:
    import cv2
    out = []
    for f in frames:
        h, w = f.shape[:2]
        s = side / max(h, w)
        size = (max(1, round(w * s)), max(1, round(h * s)))
        out.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC))
    # A frame with no actor box nearby is stored as the full frame (sfs.embed.run.regions_for), so one
    # clip can mix square crops with a 16:9 frame. Letterbox every frame, centred, onto one canvas;
    # a clip whose frames share a shape is returned unchanged.
    h, w = max(f.shape[0] for f in out), max(f.shape[1] for f in out)
    for n, f in enumerate(out):
        if f.shape[:2] != (h, w):
            canvas = np.zeros((h, w, 3), f.dtype)
            y, x = (h - f.shape[0]) // 2, (w - f.shape[1]) // 2
            canvas[y:y + f.shape[0], x:x + f.shape[1]] = f
            out[n] = canvas
    if len(out) % 2:
        out.append(out[-1])
    return np.stack(out)


def _metadata(frame_numbers: list[int], fps: float, n: int) -> dict:
    nums = list(frame_numbers) + [frame_numbers[-1]] * (n - len(frame_numbers))
    rel = [x - nums[0] for x in nums]
    return {"fps": float(fps), "frames_indices": rel, "total_num_frames": rel[-1] + 1,
            "duration": (rel[-1] + 1) / fps}


def instruct_prefix() -> str:
    return "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"


def instruct_suffix(query: str) -> str:
    return INSTRUCT_PROMPT.format(query=query) + "<|im_end|>\n<|im_start|>assistant\n"


def reranker_text(query: str) -> str:
    return (f"<|im_start|>system\n{RERANK_SYSTEM}<|im_end|>\n<|im_start|>user\n<Instruct>: {RERANK_INSTRUCTION}"
            f"<Query>:{query}\n<Document>:<|vision_start|><|video_pad|><|vision_end|><|im_end|>\n"
            "<|im_start|>assistant\n")


QWEN_TYPES = ("qwen3_vl", "qwen2_5_vl")


def load_scorer(path: str, kind: str, gpu_budget_gb: float | None = None, fp8_at_load: bool = False,
                pin: bool | None = None):
    """The local scorer for a weights directory, chosen by its config: Gemma 4 -> GemmaScorer
    (spec-vlm-followups E6; uses gpu_budget_gb), Qwen-VL -> VLMScorer (uses fp8_at_load)."""
    from sfs.teacher.gemma import GEMMA_TYPES, GemmaScorer, model_type
    if model_type(path) in GEMMA_TYPES:
        return GemmaScorer(path, kind, **({} if gpu_budget_gb is None else {"gpu_budget_gb": gpu_budget_gb}))
    return VLMScorer(path, kind, fp8_at_load=fp8_at_load, gpu_budget_gb=gpu_budget_gb, pin=pin)


class VLMScorer:
    shortcut = True                     # instruct: video-once KV cache (score_clip)

    def __init__(self, path: str, kind: str, device: str = "cuda", fp8_at_load: bool = False,
                 gpu_budget_gb: float | None = None, clip_batch: int = 16, pin: bool | None = None):
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, FineGrainedFP8Config
        if kind not in YES:
            raise ValueError(f"kind must be one of {sorted(YES)}")
        cfg = AutoConfig.from_pretrained(path)
        if cfg.model_type not in QWEN_TYPES:
            raise ValueError(f"{path}: model type {cfg.model_type} is not one of {QWEN_TYPES}")
        self.kind, self.device = kind, device
        self.processor = AutoProcessor.from_pretrained(path)
        kw = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
        if fp8_at_load:
            if getattr(cfg, "quantization_config", None) is not None:
                raise ValueError(f"{path} is already quantised")
            kw["quantization_config"] = FineGrainedFP8Config(weight_block_size=(128, 128),
                                                             modules_to_not_convert=["visual", "lm_head"])
        self.clip_batch, self.placement = clip_batch, None
        if gpu_budget_gb is not None:
            self.model = AutoModelForImageTextToText.from_pretrained(path, device_map="cpu", **kw).eval()
            self.split(gpu_budget_gb, pin=pin)
        elif fp8_at_load or getattr(cfg, "quantization_config", None) is not None:
            self.model = AutoModelForImageTextToText.from_pretrained(path, device_map=device, **kw).eval()
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(path, **kw).to(device).eval()
        tok = self.processor.tokenizer
        self.yes = [tok.convert_tokens_to_ids(t) for t in YES[kind]]
        self.no = [tok.convert_tokens_to_ids(t) for t in NO[kind]]
        src = os.path.join(path, "_source.json")
        self.revision = json.load(open(src))["revision"] if os.path.exists(src) else "unknown"
        self.model_id = f"{os.path.basename(os.path.normpath(path))}{'+fp8load' if fp8_at_load else ''}@{self.revision[:12]}"

    @classmethod
    def from_parts(cls, model, processor, kind: str, device: str = "cpu", model_id: str = "test"):
        """Wrap an already-built model + processor (tests use a tiny random-weight Qwen3-VL)."""
        self = cls.__new__(cls)
        self.kind, self.device, self.model, self.processor = kind, device, model.eval(), processor
        self.clip_batch, self.placement = 16, None
        tok = processor.tokenizer
        self.yes = [tok.convert_tokens_to_ids(t) for t in YES[kind]]
        self.no = [tok.convert_tokens_to_ids(t) for t in NO[kind]]
        self.revision, self.model_id = "test", model_id
        return self

    def split(self, gpu_budget_gb: float | None, pin: bool | None = None) -> dict:
        """CPU-loaded model -> token embeddings looked up on the CPU (unless tied to lm_head), decoder layers that do not fit
        `gpu_budget_gb` streamed, everything else on `self.device` (sfs.teacher.offload; exact)."""
        from sfs.teacher.offload import CpuLookup, place_layers
        lm = self.model.model.language_model
        emb = lm.embed_tokens
        head = getattr(self.model, "lm_head", None)
        tied = head is not None and head.weight is emb.weight          # one table: it must stay with lm_head
        self.placement = place_layers(self.model, lm.layers, self.device, gpu_budget_gb, [] if tied else [emb], pin=pin)
        if not tied:
            lm.embed_tokens = CpuLookup(emb, self.device)
        self.placement["token_embeddings_on_cpu"] = not tied
        return self.placement

    # ------------------------------------------------------------ device hooks
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
        self.model.set_attn_implementation(name)

    def _margin(self, logits):
        import torch
        lp = torch.log_softmax(logits.float(), dim=-1)
        return torch.logsumexp(lp[..., self.yes], -1) - torch.logsumexp(lp[..., self.no], -1)

    def _video(self, text: str, frames: np.ndarray, meta: dict):
        return self.processor(text=[text], videos=[frames], video_metadata=[meta], do_sample_frames=False,
                              return_tensors="pt").to(self.device)

    # ------------------------------------------------------------ full forward (reference / reranker)
    def score_full(self, frames: list[np.ndarray], frame_numbers: list[int], fps: float,
                   queries: list[str]) -> np.ndarray:
        import torch
        vid = prepare_frames(frames)
        meta = _metadata(frame_numbers, fps, len(vid))
        out = np.empty(len(queries), np.float32)
        with torch.no_grad():
            for i, q in enumerate(queries):
                text = reranker_text(q) if self.kind == "reranker" else instruct_prefix() + instruct_suffix(q)
                logits = self.model(**self._video(text, vid, meta), use_cache=False,
                                    logits_to_keep=1).logits[:, -1]
                out[i] = float(self._margin(logits)[0])
        return out

    # ------------------------------------------------------------ cached: one video pass, many questions
    def score_clip(self, frames: list[np.ndarray], frame_numbers: list[int], fps: float,
                   queries: list[str], batch: int | None = None) -> np.ndarray:
        import torch
        batch = batch or self.clip_batch
        if self.kind != "instruct":
            raise ValueError("score_clip needs the video before the question (instruct format)")
        vid = prepare_frames(frames)
        inputs = self._video(instruct_prefix(), vid, _metadata(frame_numbers, fps, len(vid)))
        tok = self.processor.tokenizer
        out = np.empty(len(queries), np.float32)
        with torch.no_grad():
            self.model.model.rope_deltas = None
            pre = self.model(**inputs, use_cache=True, logits_to_keep=1)
            cache, delta = pre.past_key_values, self.model.model.rope_deltas      # delta: (1, 1)
            p = inputs["input_ids"].shape[1]
            for s in range(0, len(queries), batch):
                qs = queries[s:s + batch]
                enc = tok([instruct_suffix(q) for q in qs], add_special_tokens=False, padding=True,
                          padding_side="right", return_tensors="pt").to(self.device)
                b, length = enc["input_ids"].shape
                kv = copy.deepcopy(cache)
                kv.batch_repeat_interleave(b)
                attn = torch.cat([torch.ones(b, p, dtype=enc["attention_mask"].dtype, device=self.device),
                                  enc["attention_mask"]], dim=1)
                pos = (torch.arange(p, p + length, device=self.device) + delta.to(self.device).view(-1)[0])
                pos = pos.view(1, 1, -1).expand(3, b, -1)
                logits = self.model(input_ids=enc["input_ids"], attention_mask=attn, past_key_values=kv,
                                    position_ids=pos, cache_position=torch.arange(p, p + length, device=self.device),
                                    use_cache=True).logits
                last = enc["attention_mask"].sum(1) - 1
                out[s:s + b] = self._margin(logits[torch.arange(b), last]).cpu().numpy()
                del kv, logits
        return out
