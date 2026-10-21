"""Download the pinned public encoder; normal TLS and Hub integrity checks apply."""
from huggingface_hub import snapshot_download
from detector.embedding import MODEL_ID, REVISION, MODEL_DIR

if __name__ == "__main__":
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=MODEL_DIR,
                      allow_patterns=["config.json", "pytorch_model.bin", "vocab.txt",
                                      "special_tokens_map.json", "tokenizer_config.json"])
    print(f"MuRIL saved to {MODEL_DIR}")
