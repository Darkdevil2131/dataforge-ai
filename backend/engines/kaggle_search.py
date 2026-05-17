"""
kaggle_search.py
================
Searches Kaggle + HuggingFace + OpenML for real datasets.
Uses Kaggle public API (requires kaggle.json credentials).
Falls back to curated metadata if no credentials.
"""
import os
import json
import urllib.request
import urllib.parse
import urllib.error

KAGGLE_META_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_metadata.json")


def search_kaggle_api(query: str, limit: int = 5) -> list:
    """Search Kaggle API for real datasets. Requires KAGGLE_USERNAME + KAGGLE_KEY env vars."""
    username = os.environ.get("KAGGLE_USERNAME")
    api_key  = os.environ.get("KAGGLE_KEY")
    if not username or not api_key:
        return []

    import base64
    creds = base64.b64encode(f"{username}:{api_key}".encode()).decode()
    encoded_q = urllib.parse.quote(query)
    url = f"https://www.kaggle.com/api/v1/datasets/list?search={encoded_q}&page=1&pageSize={limit}&sortBy=voteCount"

    try:
        req = urllib.request.Request(url, headers={
            "Authorization": f"Basic {creds}",
            "Content-Type":  "application/json",
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return []

    results = []
    for ds in data:
        results.append({
            "id":          ds.get("ref", ""),
            "name":        ds.get("title", "Unknown"),
            "source":      "Kaggle",
            "url":         f"https://www.kaggle.com/datasets/{ds.get('ref','')}",
            "description": ds.get("subtitle", "")[:200],
            "tags":        [t.get("name","") for t in ds.get("tags", [])][:5],
            "rows":        ds.get("totalBytes", 0) // 100,  # estimate
            "votes":       ds.get("voteCount", 0),
            "downloads":   ds.get("downloadCount", 0),
            "difficulty":  "intermediate",
            "relevance_score": min(0.99, ds.get("voteCount", 0) / 10000),
        })
    return results


def search_huggingface(query: str, limit: int = 3) -> list:
    """Search HuggingFace Datasets Hub."""
    encoded_q = urllib.parse.quote(query)
    url = f"https://huggingface.co/api/datasets?search={encoded_q}&limit={limit}&sort=likes"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DataForgeAI/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return []

    results = []
    for ds in data[:limit]:
        ds_id = ds.get("id", "")
        results.append({
            "id":          ds_id,
            "name":        ds_id.split("/")[-1].replace("-", " ").replace("_", " ").title(),
            "source":      "HuggingFace",
            "url":         f"https://huggingface.co/datasets/{ds_id}",
            "description": (ds.get("cardData") or {}).get("pretty_name", ds_id),
            "tags":        ds.get("tags", [])[:5],
            "rows":        None,
            "difficulty":  "intermediate",
            "relevance_score": 0.75,
        })
    return results


def search_datasets(query: str, intent: str, limit: int = 8) -> list:
    """
    Search all sources for real datasets matching the query.
    Priority: Kaggle API > HuggingFace > local curated metadata
    """
    all_results = []

    # 1. Kaggle (real API)
    kaggle_results = search_kaggle_api(query, limit=5)
    all_results.extend(kaggle_results)

    # 2. HuggingFace
    if len(all_results) < 4:
        hf_results = search_huggingface(query, limit=3)
        all_results.extend(hf_results)

    # 3. Curated local metadata (always include relevant ones)
    try:
        with open(KAGGLE_META_PATH) as f:
            metadata = json.load(f)
        q_lower = query.lower()
        for ds in metadata:
            score = 0
            if intent in ds.get("task_type", []):           score += 0.4
            if any(t in q_lower for t in ds.get("tags",[])):score += 0.3
            if ds.get("domain","") in q_lower:               score += 0.2
            if score > 0.2:
                ds_copy = ds.copy()
                ds_copy["relevance_score"] = min(0.98, score + ds.get("relevance_score", 0))
                ds_copy["source"] = ds_copy.get("source", "UCI / Kaggle")
                all_results.append(ds_copy)
    except Exception:
        pass

    # Deduplicate by name, sort by relevance
    seen = set()
    unique = []
    for r in sorted(all_results, key=lambda x: x.get("relevance_score", 0), reverse=True):
        key = r.get("name","").lower()[:30]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique[:limit]
