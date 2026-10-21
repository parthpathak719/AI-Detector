"""Frozen MuRIL with the original study's masked mean pooling and token limit."""
import os
from pathlib import Path
import numpy as np

MODEL_ID = "google/muril-base-cased"
REVISION = "afd9f36c7923d54e97903922ff1b260d091d202f"
MODEL_DIR = Path(__file__).resolve().parents[1] / ".cache" / "muril"


class MuRIL:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))
        self.torch = torch
        path = os.environ.get("MURIL_MODEL_PATH", str(MODEL_DIR))
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).eval()

    def token_count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def encode(self, texts):
        results = []
        for offset in range(0, len(texts), 16):
            encoded = self.tokenizer(texts[offset:offset + 16], padding=True,
                                     truncation=True, max_length=128, return_tensors="pt")
            with self.torch.inference_mode():
                h = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(h.dtype)
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            results.append(pooled.cpu().numpy())
        return np.concatenate(results).astype(np.float32)
