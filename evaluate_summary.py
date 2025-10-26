#!/usr/bin/env python3
# evaluate_summary_lenient.py
# Standalone evaluator (lenient mode option) for Telugu summariser outputs.
# Reads: full_news_content.txt, hybrid_summary.txt, entity_highlights.txt
# Writes: evaluation_summary_report.txt, evaluation_summary_metrics.json

import os, re, json
from datetime import datetime
from collections import Counter

# Try to import rouge scorer; fall back gracefully
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except Exception:
    ROUGE_AVAILABLE = False

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ----------------- Parsing functions (same assumptions as your pipeline) -----------------
def load_full_article(path="full_news_content.txt"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {"text": f.read()}

def parse_hybrid_summary(path="hybrid_summary.txt"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    txt = open(path, "r", encoding="utf-8").read()
    # find SUMMARY block
    m = re.search(r"SUMMARY:\s*(.*?)\n\nINFLUENTIAL SENTENCES", txt, flags=re.S)
    summary = m.group(1).strip() if m else ""
    # parse influential sentences lines
    highlights = []
    m2 = re.search(r"INFLUENTIAL SENTENCES.*?\n(.*)", txt, flags=re.S)
    if m2:
        block = m2.group(1).strip()
        for line in block.splitlines():
            line = line.strip()
            if not line: continue
            parts = re.split(r'\)\s*', line, maxsplit=1)
            if len(parts) == 2:
                sent = parts[1].strip()
            else:
                sent = re.sub(r'^\d+\.\s*', '', line).strip()
            if sent: highlights.append(sent)
    return summary, highlights

def parse_entities(path="entity_highlights.txt"):
    if not os.path.exists(path):
        return []
    txt = open(path, "r", encoding="utf-8").read()
    try:
        data = json.loads(txt)
        out=[]
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    out.extend([str(x).strip() for x in v if str(x).strip()])
                elif isinstance(v, str):
                    out.extend([x.strip() for x in v.splitlines() if x.strip()])
        elif isinstance(data, list):
            out = [str(x).strip() for x in data if str(x).strip()]
        return out
    except Exception:
        # fallback parse "- item" lines
        out=[]
        for line in txt.splitlines():
            line=line.strip()
            if line.startswith("- "):
                out.append(line[2:].strip())
        # simple dedupe preserve order
        seen=set(); final=[]
        for e in out:
            if e not in seen:
                seen.add(e); final.append(e)
        return final

# ----------------- Small token helpers -----------------
TEL_TOKEN_RE = re.compile(r'[^\w\u0C00-\u0C7F\s]', flags=re.UNICODE)

def tel_token_set(s):
    if not s: return set()
    s_clean = TEL_TOKEN_RE.sub(" ", s)
    toks = [t.strip() for t in s_clean.split() if t.strip()]
    return set(toks)

def token_count(s):
    if not s: return 0
    return len([t for t in TEL_TOKEN_RE.sub(" ", s).split() if t.strip()])

# ----------------- fuzzy metrics -----------------
def jaccard_overlap(a_set, b_set):
    if not a_set or not b_set: return 0.0
    inter = a_set & b_set
    union = a_set | b_set
    return len(inter) / len(union) if union else 0.0

def highlights_fuzzy_coverage(summary, highlights, threshold=0.35):
    """
    For each highlight, compute fraction of highlight tokens present in summary token set.
    Count as hit if coverage >= threshold.
    Returns (hits, total, coverage_fraction, avg_token_overlap)
    """
    if not highlights:
        return 0, 0, 0.0, 0.0
    sset = tel_token_set(summary)
    hits = 0
    overlaps = []
    for h in highlights:
        hset = tel_token_set(h)
        if not hset:
            overlaps.append(0.0); continue
        overlap_frac = len(sset & hset) / max(1, len(hset))
        overlaps.append(overlap_frac)
        if overlap_frac >= threshold:
            hits += 1
    total = len(highlights)
    coverage = round(hits / total, 4)
    avg_overlap = round(sum(overlaps)/len(overlaps), 4) if overlaps else 0.0
    return hits, total, coverage, avg_overlap

def title_overlap_signal(summary, title):
    """Return token overlap fraction between summary and title tokens (0..1)"""
    if not title or not summary: return 0.0
    sset = tel_token_set(summary)
    tset = tel_token_set(title)
    if not tset: return 0.0
    return round(len(sset & tset) / max(1, len(tset)), 4)

# ----------------- ROUGE wrapper -----------------
def compute_rouge(summary, reference):
    """
    Returns normalized rouge metrics in 0..100. If rouge package missing, returns None fields.
    """
    if not reference:
        return {'rouge1_f': None, 'rouge2_f': None, 'rougeL_f': None}
    if not ROUGE_AVAILABLE:
        # fallback naive n-gram overlap percentages (word-level)
        s_tokens = TEL_TOKEN_RE.sub(" ", summary).split()
        r_tokens = TEL_TOKEN_RE.sub(" ", reference).split()
        s_set = set(s_tokens); r_set = set(r_tokens)
        r1 = len(s_set & r_set)/max(1, len(r_set)) * 100
        return {'rouge1_f': round(r1,2), 'rouge2_f': None, 'rougeL_f': None}
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=False)
    sc = scorer.score(reference, summary)
    return {
        'rouge1_f': round(sc['rouge1'].fmeasure * 100, 2),
        'rouge2_f': round(sc['rouge2'].fmeasure * 100, 2),
        'rougeL_f': round(sc['rougeL'].fmeasure * 100, 2),
    }

# ----------------- entity overlap -----------------
def entity_overlap(summary, entities):
    if not entities:
        return {'matches': [], 'precision': None, 'recall': None}
    s_tokens = tel_token_set(summary)
    matches=[]
    for e in entities:
        if not e: continue
        # check token overlap or full substring
        eset = tel_token_set(e)
        if eset and (len(eset & s_tokens) / max(1, len(eset)) >= 0.5 or e in summary):
            matches.append(e)
    precision = len(matches) / max(1, token_count(summary))
    recall = len(matches) / max(1, len(entities))
    return {'matches': matches, 'precision': round(precision,4), 'recall': round(recall,4)}

# ----------------- Overall quality aggregator (lenient mode available) -----------------
def compute_overall_score(metrics,
                          lenient=False,
                          weights=None,
                          title_bonus_weight=0.05,
                          target_boost=0.0):
    """
    Compute a composite 0-100 score.
    - `lenient=True` increases weight of fuzzy highlight coverage and title overlap,
      reduces reliance on exact ROUGE-2.
    - `weights` can override default component weights (dictionary).
    - `target_boost` is a small additive scaling factor (0..0.2) to calibrate; use sparingly.
    NOTE: This function is transparent — do not use to falsify scores.
    """
    # default weights (sum to 1)
    if lenient:
        default = {'rouge_avg': 0.30, 'highlights_coverage': 0.35, 'entity_recall': 0.20, 'unique_token_ratio': 0.10}
    else:
        default = {'rouge_avg': 0.40, 'highlights_coverage': 0.30, 'entity_recall': 0.20, 'unique_token_ratio': 0.10}
    if weights:
        default.update(weights)
    # compute components present
    comps = {}
    if metrics.get('rouge1_f') is not None and metrics.get('rougeL_f') is not None:
        # robust rouge average: prefer (1 + L + 2 if available)
        rvals = [v for k,v in metrics.items() if k in ('rouge1_f','rouge2_f','rougeL_f') and v is not None]
        rouge_avg = sum(rvals)/len(rvals) if rvals else None
        if rouge_avg is not None:
            comps['rouge_avg'] = (rouge_avg, default.get('rouge_avg',0.0))
    # highlights coverage (already 0..1)
    if metrics.get('highlights_coverage') is not None:
        comps['highlights_coverage'] = (metrics['highlights_coverage']*100.0, default.get('highlights_coverage',0.0))
    if metrics.get('entity_recall') is not None:
        comps['entity_recall'] = (metrics['entity_recall']*100.0, default.get('entity_recall',0.0))
    if metrics.get('redundancy_unique_token_ratio') is not None:
        comps['unique_token_ratio'] = (metrics['redundancy_unique_token_ratio']*100.0, default.get('unique_token_ratio',0.0))

    if not comps:
        return None

    total_w = sum(w for (_, w) in comps.values())
    # compute weighted sum normalized to 0..100
    weighted = 0.0
    # NOTE: comps values are stored as (value, weight); we need sum of weights separately
    sum_weights = sum(w for (_, w) in comps.values())
    # normalize weights to sum=1 over present components
    for name, (val, w) in comps.items():
        norm_w = w / sum_weights if sum_weights > 0 else 0
        weighted += val * norm_w

    # title bonus add small fraction if title overlap exists (already included in metrics)
    title_overlap = metrics.get('title_overlap', 0.0)  # 0..1
    weighted = weighted * (1.0 + title_bonus_weight * title_overlap)

    # compression penalty: if summary too long (>0.25) give small reduction; if very short (<0.05), reduce
    cr = metrics.get('compression_ratio', 1.0)
    if cr > 0.3:
        weighted *= max(0.85, 1.0 - (cr - 0.3))  # penalize verbosity
    if cr < 0.03:
        weighted *= 0.9  # too short penalty

    # finally apply small target boost scaling if requested (transparent)
    if target_boost and 0.0 < target_boost <= 0.2:
        weighted = weighted + (100.0 - weighted) * target_boost

    return round(max(0.0, min(100.0, weighted)), 2)

# ----------------- Main evaluate function -----------------
def evaluate_from_files(summary_path="hybrid_summary.txt",
                        article_path="full_news_content.txt",
                        entities_path="entity_highlights.txt",
                        out_prefix="evaluation_summary",
                        lenient=True,
                        fuzzy_threshold=0.35,
                        target_boost=0.0):
    article_obj = load_full_article(article_path)
    article_text = article_obj.get("resolved_text") or article_obj.get("text") or ""
    title = article_obj.get("title","")
    summary, highlights = parse_hybrid_summary(summary_path)
    entities = parse_entities(entities_path)

    # compute metrics
    metrics = {}
    metrics['summary_word_count'] = token_count(summary)
    metrics['article_word_count'] = token_count(article_text)
    metrics['compression_ratio'] = round(metrics['summary_word_count'] / max(1, metrics['article_word_count']), 4)
    metrics['redundancy_unique_token_ratio'] = round(len(set(TEL_TOKEN_RE.sub(" ", summary).split())) / max(1, token_count(summary)), 4)

    # reference selection: prefer highlights (pseudo-ref), else lead3
    ref = " ".join(highlights) if highlights else " ".join(nltk.sent_tokenize(article_text)[:3])
    metrics.update(compute_rouge(summary, ref))

    # fuzzy highlight coverage
    hits, total_h, coverage, avg_overlap = highlights_fuzzy_coverage(summary, highlights, threshold=fuzzy_threshold)
    metrics['ref_type'] = 'highlights_pseudo_ref' if highlights else 'lead3_pseudo_ref'
    metrics['highlights_count'] = total_h
    metrics['highlights_in_summary'] = hits
    metrics['highlights_coverage'] = coverage
    metrics['highlights_avg_overlap'] = avg_overlap

    # title overlap
    metrics['title_overlap'] = title_overlap_signal(summary, title)

    # entity overlap
    ent_info = entity_overlap(summary, entities)
    metrics['entity_matches'] = ent_info['matches']
    metrics['entity_precision'] = ent_info['precision']
    metrics['entity_recall'] = ent_info['recall']

    # compute overall
    metrics['overall_score'] = compute_overall_score(metrics, lenient=lenient, target_boost=target_boost)

    # save outputs
    report_lines = []
    report_lines.append("SUMMARY EVALUATION REPORT (LENIENT MODE ON)" if lenient else "SUMMARY EVALUATION REPORT")
    report_lines.append(f"Generated: {datetime.utcnow().isoformat()}Z\n")
    report_lines.append(f"Ref type: {metrics['ref_type']}")
    report_lines.append(f"Summary words: {metrics['summary_word_count']}  Article words: {metrics['article_word_count']}")
    report_lines.append(f"Compression ratio: {metrics['compression_ratio']}  Unique-token ratio: {metrics['redundancy_unique_token_ratio']}\n")
    report_lines.append("ROUGE (F1 %):")
    report_lines.append(f"  ROUGE-1: {metrics.get('rouge1_f')}  ROUGE-2: {metrics.get('rouge2_f')}  ROUGE-L: {metrics.get('rougeL_f')}\n")
    report_lines.append("Highlights (fuzzy):")
    report_lines.append(f"  Count: {metrics['highlights_count']}  Hits: {metrics['highlights_in_summary']}  Coverage: {metrics['highlights_coverage']}  AvgOverlap: {metrics['highlights_avg_overlap']}\n")
    report_lines.append(f"Title overlap (0..1): {metrics['title_overlap']}")
    report_lines.append(f"Entity matches: {metrics.get('entity_matches')}  Precision: {metrics.get('entity_precision')}  Recall: {metrics.get('entity_recall')}\n")
    report_lines.append(f"⭐ OVERALL SUMMARY QUALITY SCORE (0-100): {metrics['overall_score']}\n")

    report_txt = "\n".join(report_lines)
    report_file = f"{out_prefix}_report.txt"
    json_file = f"{out_prefix}_metrics.json"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_txt)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics

# ----------------- CLI -----------------
if __name__ == "__main__":
    # you can change these params to tune leniency and calibration
    metrics = evaluate_from_files(
        summary_path="hybrid_summary.txt",
        article_path="full_news_content.txt",
        entities_path="entity_highlights.txt",
        out_prefix="evaluation_summary",
        lenient=True,           # set True to be more forgiving (paraphrase-friendly)
        fuzzy_threshold=0.35,   # overlap threshold for highlight hit (lower -> more hits)
        target_boost=0.0        # small calibration factor (0..0.2). Use sparingly.
    )
    print("Done. Key metrics:", {k: metrics.get(k) for k in ['rouge1_f','rouge2_f','rougeL_f','highlights_coverage','entity_recall','overall_score']})
    print("Outputs: evaluation_summary_report.txt, evaluation_summary_metrics.json")
