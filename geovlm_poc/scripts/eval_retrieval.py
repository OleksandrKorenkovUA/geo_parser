import argparse
import json
import os
from typing import Any, Dict, List, Optional

from geovlm_poc.semantic_index import EmbeddingClient, VectorIndex, SemanticSearcher
from geovlm_poc.normalize import normalize_label, normalize_roof_color


def _load_cases(path: str) -> List[Dict[str, Any]]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def _passes_filters_hit(hit, flt: Dict[str, Any]) -> bool:
    m = hit.meta or {}
    for k, v in flt.items():
        if k == "kind":
            want = (str(v).strip().lower() if v is not None else "")
            have = (str(hit.kind).strip().lower() if hit.kind is not None else "")
            if not want or have != want:
                return False
        elif k == "roof_color":
            if normalize_roof_color(m.get("roof_color")) != normalize_roof_color(v):
                return False
        elif k == "roof_color_any":
            vals = v if isinstance(v, (list, tuple, set)) else [v]
            want = {normalize_roof_color(x) for x in vals}
            have = normalize_roof_color(m.get("roof_color"))
            if have not in want:
                return False
        elif k == "label":
            if normalize_label(m.get("label")) != normalize_label(v):
                return False
        elif k == "rail_dist_max_m":
            rd = m.get("rail_dist_m")
            try:
                if rd is None or float(rd) > float(v):
                    return False
            except (TypeError, ValueError):
                return False
        elif k == "change_type":
            if (m.get("change_type") or "").strip().lower() != str(v).strip().lower():
                return False
    return True


def _parse_k_list(value: Optional[str]) -> List[int]:
    if not value:
        return [1, 5, 10, 20]
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    out = [k for k in out if k > 0]
    out = sorted(set(out))
    return out


def _recall_at_k(hits, relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = hits[:k]
    found = {h.doc_id for h in top}
    rel = set(relevant_ids)
    return float(len(found & rel)) / float(len(rel))


def _hit_at_k(hits, relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = hits[:k]
    rel = set(relevant_ids)
    return 1.0 if any(h.doc_id in rel for h in top) else 0.0


def _mrr_at_k(hits, relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    rel = set(relevant_ids)
    for i, h in enumerate(hits[:k], start=1):
        if h.doc_id in rel:
            return 1.0 / float(i)
    return 0.0


def _success_at_k(hits, rel_filters: Dict[str, Any], k: int) -> float:
    top = hits[:k]
    return 1.0 if any(_passes_filters_hit(h, rel_filters) for h in top) else 0.0


def _mrr_filter_at_k(hits, rel_filters: Dict[str, Any], k: int) -> float:
    for i, h in enumerate(hits[:k], start=1):
        if _passes_filters_hit(h, rel_filters):
            return 1.0 / float(i)
    return 0.0


def _hit_summary(hit, rel_filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    summary = {
        "doc_id": hit.doc_id,
        "score": float(hit.score),
        "kind": hit.kind,
        "image_id": hit.image_id,
        "meta": hit.meta,
        "preview": hit.preview,
    }
    if rel_filters:
        summary["passes_filters"] = _passes_filters_hit(hit, rel_filters)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--cases", required=True, help="JSONL with query_uk and relevant_doc_ids or relevant_filters")
    ap.add_argument("--k", default=None, help="Comma-separated K list, e.g. 1,5,10,20")
    ap.add_argument("--dump-fails", default=None, help="Write failed cases with top hits to JSONL")
    args = ap.parse_args()

    if not os.path.isdir(args.index):
        raise FileNotFoundError(f"Index directory not found: {args.index}")

    cases = _load_cases(args.cases)
    if not cases:
        raise RuntimeError("No cases loaded")

    base_url = os.environ.get("EMB_BASE_URL")
    api_key = os.environ.get("EMB_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("EMB_BASE_URL and EMB_API_KEY must be set")
    model = os.environ.get("EMB_MODEL", "text-embedding-3-large")
    emb = EmbeddingClient(base_url=base_url, api_key=api_key, model=model)
    ks = _parse_k_list(args.k)
    max_k = max(ks)

    dump_fails = None
    try:
        idx = VectorIndex.load(args.index)
        searcher = SemanticSearcher(idx, emb)
        totals = {k: 0.0 for k in ks}
        doc_recall = {k: 0.0 for k in ks}
        doc_hit = {k: 0.0 for k in ks}
        doc_mrr = {k: 0.0 for k in ks}
        filter_success = {k: 0.0 for k in ks}
        filter_mrr = {k: 0.0 for k in ks}
        used = 0
        used_doc = 0
        used_filter = 0
        failed = 0
        if args.dump_fails:
            dump_fails = open(args.dump_fails, "w", encoding="utf-8")
        for case in cases:
            q = case.get("query_uk") or case.get("query")
            rel_ids = case.get("relevant_doc_ids")
            rel_filters = case.get("relevant_filters")
            if not q or (not rel_ids and not rel_filters):
                continue
            hits = searcher.query(q, top_k=max_k, filters=None)
            if rel_ids:
                used_doc += 1
                for k in ks:
                    r = _recall_at_k(hits, rel_ids, k)
                    totals[k] += r
                    doc_recall[k] += r
                    doc_hit[k] += _hit_at_k(hits, rel_ids, k)
                    doc_mrr[k] += _mrr_at_k(hits, rel_ids, k)
                failed_case = _hit_at_k(hits, rel_ids, max_k) == 0.0
            elif rel_filters:
                used_filter += 1
                for k in ks:
                    top = hits[:k]
                    s = _success_at_k(hits, rel_filters, k)
                    totals[k] += s
                    filter_success[k] += s
                    filter_mrr[k] += _mrr_filter_at_k(hits, rel_filters, k)
                failed_case = _success_at_k(hits, rel_filters, max_k) == 0.0
            else:
                failed_case = False
            used += 1
            if dump_fails and failed_case:
                failed += 1
                case_type = "doc_id" if rel_ids else "filters"
                payload = {
                    "query": q,
                    "case_type": case_type,
                    "case": case,
                    "top_k": max_k,
                    "hits": [_hit_summary(h, rel_filters) for h in hits[:max_k]],
                }
                dump_fails.write(json.dumps(payload, ensure_ascii=False) + "\n")
        n = max(1, used)
        out = {f"recall@{k}": round(totals[k] / n, 4) for k in ks}
        doc_out = {
            "recall": {f"recall@{k}": round(doc_recall[k] / max(1, used_doc), 4) for k in ks},
            "hit": {f"hit@{k}": round(doc_hit[k] / max(1, used_doc), 4) for k in ks},
            "mrr": {f"mrr@{k}": round(doc_mrr[k] / max(1, used_doc), 4) for k in ks},
        }
        filter_out = {
            "success": {f"success@{k}": round(filter_success[k] / max(1, used_filter), 4) for k in ks},
            "mrr": {f"mrr@{k}": round(filter_mrr[k] / max(1, used_filter), 4) for k in ks},
        }
        result = {
            "cases": n,
            "cases_doc_id": used_doc,
            "cases_filter": used_filter,
            "failed": failed,
            "metrics": out,
            "metrics_doc_id": doc_out,
            "metrics_filter": filter_out,
        }
        print(json.dumps(result, ensure_ascii=False))
    finally:
        if dump_fails:
            dump_fails.close()
        emb.close()


if __name__ == "__main__":
    main()
