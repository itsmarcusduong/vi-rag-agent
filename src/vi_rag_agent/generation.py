"""Prompting and answer generation for RAG QA (M5, docs/protocols/m5_generation.md).

The prompts are fixed by the protocol; changing them means a new protocol. ``HFGenerator`` loads an
open-weight chat model in 4-bit NF4 and decodes greedily with thinking disabled.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

SYSTEM_RAG = (
    "Bạn là trợ lý hỏi đáp tiếng Việt. Chỉ dùng thông tin trong các đoạn văn được cung cấp. "
    "Trả lời bằng một cụm từ ngắn, trích nguyên văn từ đoạn văn nếu có thể, không giải thích."
)
SYSTEM_CLOSED = "Bạn là trợ lý hỏi đáp tiếng Việt. Trả lời bằng một cụm từ ngắn, không giải thích."
MAX_PASSAGE_CHARS = 1200
MAX_NEW_TOKENS = 48


def build_messages(question: str, passages: Sequence[str] | None) -> list[dict]:
    """Chat messages for one question; ``passages=None`` means closed-book."""
    if passages is None:
        return [
            {"role": "system", "content": SYSTEM_CLOSED},
            {"role": "user", "content": f"Câu hỏi: {question}\nTrả lời:"},
        ]
    blocks = [f"[{i}] {p[:MAX_PASSAGE_CHARS]}" for i, p in enumerate(passages, start=1)]
    context = "\n\n".join(blocks)
    return [
        {"role": "system", "content": SYSTEM_RAG},
        {"role": "user", "content": f"Đoạn văn:\n{context}\n\nCâu hỏi: {question}\nTrả lời:"},
    ]


def clean_answer(text: str) -> str:
    """First non-empty line, without a leading 'Trả lời:' label or surrounding quotes."""
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("trả lời:"):
            line = line[len("trả lời:") :].strip()
        return line.strip("\"'“”‘’ ")
    return ""


class HFGenerator:  # pragma: no cover - needs a GPU and model weights
    """4-bit NF4 chat model with greedy decoding and thinking disabled."""

    def __init__(self, model_name: str, batch_size: int = 8):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        try:
            from transformers import AutoModelForMultimodalLM as AutoModel
        except ImportError:
            from transformers import AutoModelForCausalLM as AutoModel

        self.model_name = model_name
        self.batch_size = batch_size
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            model_name, quantization_config=quant, device_map={"": 0}, dtype=torch.float16
        )
        self.model.eval()
        self.load_s = time.perf_counter() - t0
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def _render(self, messages: list[dict]) -> str:
        parts = [
            {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            for m in messages
        ]
        return self.processor.apply_chat_template(
            parts, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

    def generate(self, batch_messages: list[list[dict]]) -> tuple[list[str], list[int]]:
        """Answers (raw text) and generated token counts, in input order."""
        import torch

        texts = [self._render(m) for m in batch_messages]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        outputs: list[str] = [""] * len(texts)
        n_tokens: list[int] = [0] * len(texts)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self.tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True).to(0)
            with torch.inference_mode():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new = out[:, enc["input_ids"].shape[1] :]
            for j, i in enumerate(idx):
                ids = new[j]
                ids = ids[ids != self.tokenizer.pad_token_id]
                n_tokens[i] = int(ids.shape[0])
                outputs[i] = self.tokenizer.decode(ids, skip_special_tokens=True)
        return outputs, n_tokens

    def close(self) -> None:
        import gc

        import torch

        del self.model
        gc.collect()
        torch.cuda.empty_cache()
