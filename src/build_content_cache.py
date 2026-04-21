import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Embedding import Qwen3VLEmbedder


def resolve_image_path(project_root: Path, raw_path: Optional[str]) -> Optional[str]:
    if raw_path is None:
        return raw_path
    candidates = [
        project_root / raw_path,
        project_root / "data" / raw_path,
        project_root / "src" / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(project_root / raw_path)


def normalize_prompt(project_root: Path, prompt: Dict, text_max_chars: int) -> Dict:
    normalized = dict(prompt)
    if normalized.get("text") is not None:
        normalized["text"] = str(normalized["text"])[:text_max_chars]
    if normalized.get("image") is not None:
        normalized["image"] = resolve_image_path(project_root, normalized["image"])
    normalized.setdefault("instruction", "Represent the item for sequential recommendation.")
    return normalized


def resolve_prompt_path(data_dir: Path, data_name: str) -> Path:
    nested_prompt_path = data_dir / data_name / f"{data_name}_prompt.json"
    flat_prompt_path = data_dir / f"{data_name}_prompt.json"

    if nested_prompt_path.exists():
        return nested_prompt_path
    if flat_prompt_path.exists():
        return flat_prompt_path

    # If caller already passes the dataset directory, support that too.
    direct_prompt_path = data_dir / f"{data_name}_prompt.json"
    if direct_prompt_path.parent.name == data_name:
        return direct_prompt_path
    return nested_prompt_path


def load_prompt_map(project_root: Path, data_dir: Path, data_name: str, text_max_chars: int) -> Dict[int, Dict]:
    prompt_path = resolve_prompt_path(data_dir, data_name)
    with open(prompt_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        int(item_id): normalize_prompt(project_root, prompt, text_max_chars)
        for item_id, prompt in raw.items()
    }


def encode_prompt_batch(embedder: Qwen3VLEmbedder, prompts: List[Dict]) -> torch.Tensor:
    conversations = [
        embedder.format_model_input(
            text=prompt.get("text"),
            image=prompt.get("image"),
            video=prompt.get("video"),
            instruction=prompt.get("instruction"),
        )
        for prompt in prompts
    ]
    model_inputs = embedder._preprocess_inputs(conversations)
    device = next(embedder.model.parameters()).device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    with torch.no_grad():
        outputs = embedder.model(**model_inputs)
        pooled = embedder._pooling_last(outputs.last_hidden_state, model_inputs["attention_mask"])
    return pooled.detach().cpu()


def normalize_loaded_cache_keys(data_name: str, embedding_cache: Dict) -> Dict[int, torch.Tensor]:
    normalized = {}
    for raw_key, embedding in embedding_cache.items():
        if isinstance(raw_key, int):
            normalized[raw_key] = embedding
        elif isinstance(raw_key, str):
            if raw_key.isdigit():
                normalized[int(raw_key)] = embedding
            elif ":" in raw_key:
                prefix, item_id = raw_key.split(":", 1)
                if prefix == data_name and item_id.isdigit():
                    normalized[int(item_id)] = embedding
    return normalized


def load_existing_cache(cache_path: Path, data_name: str) -> Dict[int, torch.Tensor]:
    if not cache_path.exists():
        return {}
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")

    if isinstance(payload, dict) and "embedding_cache" in payload:
        return normalize_loaded_cache_keys(data_name, payload["embedding_cache"])
    if isinstance(payload, dict):
        return normalize_loaded_cache_keys(data_name, payload)
    if isinstance(payload, torch.Tensor):
        return {idx + 1: emb for idx, emb in enumerate(payload)}
    return {}


def build_or_update_cache(args):
    project_root = Path(__file__).resolve().parent.parent

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    if getattr(args, "cache_path", None):
        cache_path = Path(args.cache_path)
        if not cache_path.is_absolute():
            cache_path = project_root / cache_path
    else:
        cache_path = project_root / "output" / args.data_name / f"{args.data_name}_raw_cache.pt"

    prompt_map = load_prompt_map(project_root, data_dir, args.data_name, args.text_max_chars)
    embedding_cache = load_existing_cache(cache_path, args.data_name)

    missing_item_ids = sorted(set(prompt_map.keys()) - set(embedding_cache.keys()))
    if not missing_item_ids:
        print(f"Cache already complete: {cache_path}")
        return

    device_dtype = torch.float32 if args.cpu or not torch.cuda.is_available() else torch.bfloat16
    embedder = Qwen3VLEmbedder(
        model_name_or_path=str(model_path),
        dtype=device_dtype,
        low_cpu_mem_usage=True,
    )
    for param in embedder.model.parameters():
        param.requires_grad = False
    embedder.model.eval()

    for start in tqdm(range(0, len(missing_item_ids), args.cache_batch_size), desc="Caching", dynamic_ncols=True):
        batch_ids = missing_item_ids[start:start + args.cache_batch_size]
        batch_prompts = [prompt_map[item_id] for item_id in batch_ids]
        batch_emb = encode_prompt_batch(embedder, batch_prompts)
        for item_id, emb in zip(batch_ids, batch_emb):
            embedding_cache[item_id] = emb

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "data_name": args.data_name,
            "embedding_cache": embedding_cache,
        },
        cache_path,
    )
    print(f"Saved cache to {cache_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="beauty", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--model_path", default="Embedding", type=str)
    parser.add_argument("--cache_path", default=None, type=str)
    parser.add_argument("--cache_batch_size", default=4, type=int)
    parser.add_argument("--text_max_chars", default=160, type=int)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    build_or_update_cache(args)


if __name__ == "__main__":
    main()
