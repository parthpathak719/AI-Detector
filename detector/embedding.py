"""Pinned MuRIL with optional task-specific fine-tuned final transformer blocks."""
import os
import copy
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
        self.task_layers = {}

    def configure_task(self, task, layer_path):
        """Load a trusted, checksum-verified task-specific final transformer block."""
        from safetensors.torch import load_file
        block = copy.deepcopy(self.model.encoder.layer[-1])
        block.load_state_dict(load_file(str(layer_path)), strict=True)
        self.task_layers[task] = block.eval()

    def token_count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def chunks(self, text):
        """Partition all cleaned words into independently valid encoder windows."""
        words = text.split()
        encoded = self.tokenizer(words, add_special_tokens=False,
                                 return_attention_mask=False, return_token_type_ids=False)
        result, current, tokens = [], [], 0
        for word, ids in zip(words, encoded["input_ids"]):
            if len(ids) > 126:
                raise ValueError("The document contains an unusually long unbroken word. Separate joined words and try again.")
            if current and tokens + len(ids) > 126:
                result.append(" ".join(current))
                current, tokens = [], 0
            current.append(word)
            tokens += len(ids)
        if current:
            result.append(" ".join(current))
        if len(result) > 1024:
            raise ValueError("This document produces too many analysis segments. Please split it into smaller documents.")
        # Do not silently truncate if tokenizer behavior changes.
        if any(self.token_count(chunk) > 128 for chunk in result):
            raise ValueError("Unable to segment this text without truncation. Please split it into smaller passages.")
        return result

    def encode(self, texts, task=None):
        results = []
        for offset in range(0, len(texts), 16):
            encoded = self.tokenizer(texts[offset:offset + 16], padding=True,
                                     truncation=True, max_length=128, return_tensors="pt")
            with self.torch.inference_mode():
                block = self.task_layers.get(task)
                output = self.model(**encoded, output_hidden_states=block is not None)
                h = output.last_hidden_state
                if block is not None:
                    hidden = output.hidden_states[-2]
                    attention = (1. - encoded["attention_mask"][:, None, None, :].to(hidden.dtype)) * self.torch.finfo(hidden.dtype).min
                    h = block(hidden, attention_mask=attention)[0]
                mask = encoded["attention_mask"].unsqueeze(-1).to(h.dtype)
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            results.append(pooled.cpu().numpy())
        return np.concatenate(results).astype(np.float32)
