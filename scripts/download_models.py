"""Download the small local models used in the project (all open weights, run fully on-premises)."""
from huggingface_hub import snapshot_download
MODELS = ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B",
          "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2",
          "cross-encoder/ms-marco-MiniLM-L-6-v2",
          "Qwen/Qwen3-Embedding-4B", "intfloat/e5-large-v2", "thenlper/gte-large",
          "Qwen/Qwen3-4B", "Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-3.5-mini-instruct"]
if __name__ == "__main__":
    for r in MODELS:
        print(r, snapshot_download(r, ignore_patterns=["*onnx*", "*openvino*", "*.h5", "*.bin", "*.msgpack", "*.ot", "*tflite*", "*.gguf"]), flush=True)
