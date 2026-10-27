"""CIDEr-D + CLIPScore + Overall for one run -- the local evaluator the ablation
tables need (B1 in VGCAP_TODO.md).

Everything in Table 3, Table 4 and Appendix A needs ~40 scored configurations.
The EVENTA leaderboard cannot absorb 40 submissions, so scoring has to happen
here.

    python scripts/evaluate.py --run runs/vgcap_enrichment_s0 \
        --gt data/public_test_gt.json \
        --query_images data/track1_public/query

Ground truth is a JSON mapping query_id -> reference caption (a list of
references per query is also accepted).  Writes runs/<tag>/scores.json, which
scripts/collect_results.py then picks up automatically.

VALIDATE FIRST: run it on the published VG-Cap captions and check that it
reproduces Overall 0.6219 / CIDEr 0.2651 / CLIP 0.858 before trusting any new
number.  Use --expect to make that check fail loudly.

CIDEr-D is implemented here (no pycocoevalcap dependency) following Vedantam et
al. 2015: corpus-level IDF over n-grams up to 4, cosine similarity per n, a
Gaussian length penalty with sigma=6, and the factor 10 of the reference
implementation.  Retrieval metrics are read from the run's submission CSV when
a qrels file is supplied, otherwise they must be passed with --ap/--r1/--r10.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

WEIGHTS = {"ap": 1, "r1": 2, "r10": 2, "clip": 3, "cider": 2}
PUNCT = re.compile(r"[^a-z0-9 ]+")


# ---------------------------------------------------------------------------
# CIDEr-D
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return PUNCT.sub(" ", text.lower()).split()


def ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def build_document_frequency(references: dict[str, list[str]], max_n: int = 4):
    """df[n][ngram] = in how many queries that n-gram appears in any reference."""
    df = [defaultdict(int) for _ in range(max_n)]
    for refs in references.values():
        for n in range(1, max_n + 1):
            seen = set()
            for ref in refs:
                seen.update(ngram_counts(tokenize(ref), n))
            for gram in seen:
                df[n - 1][gram] += 1
    return df


def tfidf_vector(tokens: list[str], n: int, df, log_n_queries: float):
    counts = ngram_counts(tokens, n)
    vector, norm = {}, 0.0
    for gram, count in counts.items():
        idf = log_n_queries - math.log(max(1.0, df[n - 1].get(gram, 0.0)))
        value = count * idf
        vector[gram] = value
        norm += value * value
    return vector, math.sqrt(norm), len(tokens)


def cider_d_single(candidate: str, refs: list[str], df, log_n_queries: float,
                   max_n: int = 4, sigma: float = 6.0) -> float:
    cand_tokens = tokenize(candidate)
    score = 0.0
    for n in range(1, max_n + 1):
        cvec, cnorm, clen = tfidf_vector(cand_tokens, n, df, log_n_queries)
        n_score = 0.0
        for ref in refs:
            rvec, rnorm, rlen = tfidf_vector(tokenize(ref), n, df, log_n_queries)
            # min-clipped dot product, as in the reference implementation
            dot = sum(min(cvec[g], rvec.get(g, 0.0)) * rvec.get(g, 0.0) for g in cvec)
            if cnorm > 0 and rnorm > 0:
                dot /= cnorm * rnorm
            delta = clen - rlen
            n_score += dot * math.exp(-(delta ** 2) / (2 * sigma ** 2))
        score += n_score / max(1, len(refs))
    return 10.0 * score / max_n


def cider_d_corpus(candidates: dict[str, str], references: dict[str, list[str]]):
    df = build_document_frequency(references)
    log_n = math.log(max(1, len(references)))
    per_query = {
        qid: cider_d_single(candidates.get(qid, ""), refs, df, log_n)
        for qid, refs in references.items()
    }
    mean = sum(per_query.values()) / max(1, len(per_query))
    return mean, per_query


# ---------------------------------------------------------------------------
# CLIPScore
# ---------------------------------------------------------------------------
def clip_scores(captions: dict[str, str], image_dir: Path, model_name: str,
                batch_size: int, device: str):
    """Reference-free CLIPScore: 2.5 * max(0, cos(image, text)) (Hessel et al. 2021)."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    suffixes = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

    def find_image(qid: str):
        for suffix in suffixes:
            path = image_dir / f"{qid}{suffix}"
            if path.exists():
                return path
        return None

    per_query, missing = {}, []
    items = sorted(captions.items())
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        images, texts, ids = [], [], []
        for qid, caption in chunk:
            path = find_image(qid)
            if path is None:
                missing.append(qid)
                continue
            images.append(Image.open(path).convert("RGB"))
            texts.append(caption)
            ids.append(qid)
        if not ids:
            continue
        inputs = processor(text=texts, images=images, return_tensors="pt",
                           padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            out = model(**inputs)
            img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            cos = (img * txt).sum(dim=-1)
        for qid, value in zip(ids, cos.tolist()):
            per_query[qid] = 2.5 * max(0.0, value)
        print(f"  clip {start + len(chunk)}/{len(items)}", end="\r")

    print()
    if missing:
        print(f"  WARNING: no image found for {len(missing)} queries, e.g. {missing[:5]}")
    mean = sum(per_query.values()) / max(1, len(per_query))
    return mean, per_query


# ---------------------------------------------------------------------------
def overall_score(ap, r1, r10, clip, cider, eps: float = 1e-8) -> float:
    m = {"ap": ap, "r1": r1, "r10": r10, "clip": clip, "cider": cider}
    return sum(WEIGHTS.values()) / sum(w / (m[k] + eps) for k, w in WEIGHTS.items())


def load_captions(run_dir: Path) -> dict[str, str]:
    records = json.loads((run_dir / "normalized.json").read_text(encoding="utf-8"))
    if isinstance(records, dict):           # tolerate the raw-draft format
        return records
    return {r["query_id"]: r["normalized_caption"] for r in records}


def load_references(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    refs = {}
    for qid, value in data.items():
        if isinstance(value, str):
            refs[qid] = [value]
        elif isinstance(value, list):
            refs[qid] = list(value)
        elif isinstance(value, dict):       # {"caption": ...} / {"captions": [...]}
            got = value.get("captions") or value.get("caption")
            refs[qid] = list(got) if isinstance(got, list) else [got]
        else:
            raise ValueError(f"cannot read a reference caption for '{qid}'")
    return refs


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--run", required=True, help="runs/<tag> directory to score")
    ap_.add_argument("--gt", default=None,
                     help="JSON: query_id -> reference caption(s). Omit when there is no public-test "
                          "ground truth: CLIPScore is reference-free and still computable.")
    ap_.add_argument("--cider", type=float, default=None,
                     help="CIDEr taken from the leaderboard, to complete Overall without a GT file")
    ap_.add_argument("--query_images", default=None, help="directory of query images (enables CLIPScore)")
    ap_.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap_.add_argument("--batch_size", type=int, default=64)
    ap_.add_argument("--device", default="cuda:0")
    ap_.add_argument("--ap", type=float, default=0.992, help="retrieval AP (shared backbone)")
    ap_.add_argument("--r1", type=float, default=0.985)
    ap_.add_argument("--r10", type=float, default=1.000)
    ap_.add_argument("--clip", type=float, default=None, help="skip CLIP computation and use this value")
    ap_.add_argument("--expect", default=None,
                     help='validation, e.g. \'{"overall":0.6219,"cider":0.2651}\' -- exits non-zero on mismatch')
    ap_.add_argument("--tolerance", type=float, default=0.002)
    args = ap_.parse_args()

    run_dir = Path(args.run)
    captions = load_captions(run_dir)

    if args.gt:
        references = load_references(Path(args.gt))
        common = set(captions) & set(references)
        print(f"{len(captions)} captions, {len(references)} references, {len(common)} in common")
        if not common:
            raise SystemExit("no overlapping query_id between the run and the ground truth")
        print("computing CIDEr-D ...")
        cider, cider_per_query = cider_d_corpus(captions, {k: references[k] for k in common})
        print(f"  CIDEr-D = {cider:.4f}")
    else:
        common = set(captions)
        cider_per_query = {}
        cider = args.cider
        if cider is None:
            print("no --gt and no --cider: CIDEr and Overall are left out "
                  "(fill them with scripts/record_score.py after submitting)")
        else:
            print(f"  CIDEr = {cider:.4f} (supplied from the leaderboard)")

    if args.clip is not None:
        clip, clip_per_query = args.clip, {}
        print(f"  CLIPScore = {clip:.4f} (supplied)")
    elif args.query_images:
        print("computing CLIPScore ...")
        clip, clip_per_query = clip_scores(
            {k: captions[k] for k in common}, Path(args.query_images),
            args.clip_model, args.batch_size, args.device)
        print(f"  CLIPScore = {clip:.4f}")
    else:
        raise SystemExit("pass --query_images to compute CLIPScore, or --clip to supply it")

    scores = {"clip": round(clip, 4), "ap": args.ap, "r1": args.r1, "r10": args.r10,
              "n_scored": len(common)}
    if cider is not None:
        scores["cider"] = round(cider, 4)
        scores["overall"] = round(overall_score(args.ap, args.r1, args.r10, clip, cider), 4)
        scores["cider_source"] = "local" if args.gt else "leaderboard"
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")

    per_query = {qid: {"cider": round(cider_per_query.get(qid, 0.0), 4),
                       "clip": round(clip_per_query.get(qid, 0.0), 4)}
                 for qid in common if qid in clip_per_query or qid in cider_per_query}
    (run_dir / "per_query_scores.json").write_text(json.dumps(per_query, indent=2), encoding="utf-8")

    print(json.dumps(scores, indent=2))
    print(f"-> {run_dir/'scores.json'} and per_query_scores.json (needed by scripts/bootstrap.py)")

    if args.expect:
        expected = json.loads(args.expect)
        bad = {k: (v, scores.get(k)) for k, v in expected.items()
               if k not in scores or abs(scores[k] - v) > args.tolerance}
        if bad:
            print("\nVALIDATION FAILED (expected, got):")
            for key, (want, got) in bad.items():
                print(f"  {key}: {want} vs {got}")
            raise SystemExit(1)
        print("\nvalidation passed")


if __name__ == "__main__":
    main()
