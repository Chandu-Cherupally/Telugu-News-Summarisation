# // 22CSB0C20 CHANDU CHERUPALLY
# Hybrid Telugu News Summariser (updated, well-commented)
#
# This script fetches a Telugu news article, extracts important sentences using a
# hybrid frequency + TF-IDF ranking, and produces multi-length summaries (TL;DR,
# short, long or custom) optionally polished by Gemini. It also extracts named
# entities (Gemini-first, with stanza/heuristic fallbacks) and saves:
#  - full_news_content.txt  (raw article + metadata)
#  - hybrid_summary.txt     (final summary + influential sentences for explainability)
#  - entity_highlights.txt  (NER results)
#
# Keep GOOGLE_API_KEY in environment for Gemini-based features. The script
# gracefully falls back to local methods (extractive summariser / heuristics)
# when external services or models are unavailable.

# export GOOGLE_API_KEY="AIzaSyCFfXYDKFapeEHq6BmsgD0ahfub1RJqthM" 

import os
import re
import heapq
import nltk
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging
import json
from datetime import datetime

# Optional / external libs: Gemini (google.generativeai)
try:
    import google.generativeai as genai
    from google.api_core.exceptions import ServiceUnavailable
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# Optional: stanza for Telugu NER
try:
    import stanza
    STANZA_AVAILABLE = True
except Exception:
    STANZA_AVAILABLE = False

# Optional: transformers for local abstractive summarization fallback
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# Logging setup (useful for debugging and graceful fallbacks)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telugu_summariser")
logger.setLevel(logging.INFO)

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ----------------- Utility functions -----------------

# Return a set of common Telugu stop words used to filter noise during processing.
def get_telugu_stop_words():
    stop_words = [
        'మరియు','ఒక','లో','కి','యొక్క','అది','ఇది','ఆ','ఈ','ఏ','ఎవరు',
        'వారు','వీరు','అని','కోసం','కానీ','కూడా','ద్వారా','అలాగే','తో',
        'లేదా','అయితే','ఉంది','ఉన్నాయి','తర్వాత','నుండి','వరకు','కాదు',
        'అప్పుడు','ఇప్పుడు','ఎప్పుడు','చాలా','కొన్ని','కేవలం','మాత్రమే',
        'తన','తనకు','అతను','ఆమె','వారి','వారికి','చెప్పారు','అన్నారు',
        'చేసింది','చేశారు','వెల్లడించారు','అదే','విధంగా','దీంతో'
    ]
    return set(stop_words)

# Tokenize a Telugu sentence: keep only Telugu script chars and whitespace, then split.
def telugu_tokenizer(sentence):
    # Keep Telugu script only and split on whitespace
    s = re.sub(r'[^\u0C00-\u0C7F\s]', ' ', sentence)
    tokens = [t.strip() for t in s.split() if t.strip()]
    return tokens

# Clean up repeated whitespace and trim string.
def clean_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

# ----------------- Article extraction -----------------

# Fallback: simple BeautifulSoup paragraph extraction for pages where other extractors fail.
def fallback_bs4_extraction(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        text = clean_whitespace(text)
        if not text:
            return None
        return {"url": url, "title": soup.title.string if soup.title else "", "text": text}
    except Exception as e:
        logger.error("Fallback extraction failed: %s", e)
        return None

# Main article extractor: try newspaper3k (best), readability fallback, else BS4 fallback.
def get_news_content(url):
    try:
        from newspaper import Article
        art = Article(url, language='te')  # language hint for Telugu
        art.download()
        art.parse()
        text = art.text or ""
        title = art.title or ""
        authors = art.authors or []
        publish_date = art.publish_date
        if (not text) or len(text) < 200:
            # readability fallback if newspaper content too short
            from readability import Document
            doc = Document(art.html if hasattr(art, 'html') else "")
            summary_html = doc.summary()
            text = BeautifulSoup(summary_html, 'html.parser').get_text()
        text = clean_whitespace(text)
        if not text:
            return fallback_bs4_extraction(url)
        return {
            "url": url,
            "title": title,
            "authors": authors,
            "publish_date": str(publish_date) if publish_date else "",
            "text": text
        }
    except Exception as e:
        logger.warning("newspaper3k extraction failed: %s. Using fallback.", e)
        return fallback_bs4_extraction(url)

# Very simple BS4-only fetch (alternative).
def get_news_content2(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return " ".join([p.get_text() for p in paragraphs]).strip()
    except Exception as e:
        print(f"❌ Error fetching article: {e}")
        return None

# ----------------- Extractive scoring & ranking -----------------

# Compute word-frequency based sentence scores (TF).
def generate_tf_scores(text):
    """Return list of sentences, and TF word-based sentence scores (dict index->score)."""
    stop_words = get_telugu_stop_words()
    original_sentences = nltk.sent_tokenize(text)
    word_frequencies = defaultdict(int)
    for sentence in original_sentences:
        for word in telugu_tokenizer(sentence):
            if word not in stop_words:
                word_frequencies[word] += 1
    sentence_scores = defaultdict(float)
    for i, sentence in enumerate(original_sentences):
        for word in telugu_tokenizer(sentence):
            if word in word_frequencies:
                sentence_scores[i] += word_frequencies[word]
    return original_sentences, sentence_scores

# Compute TF-IDF sentence scores using sklearn's TfidfVectorizer.
def generate_tfidf_scores(text):
    """Return list of sentences, and TF-IDF sentence scores as a numpy array (index->score)."""
    original_sentences = nltk.sent_tokenize(text)
    if not original_sentences:
        return original_sentences, np.array([])
    stop_words = list(get_telugu_stop_words())
    vectorizer = TfidfVectorizer(tokenizer=telugu_tokenizer, stop_words=stop_words, token_pattern=None)
    try:
        tfidf_matrix = vectorizer.fit_transform(original_sentences)
    except ValueError:
        return original_sentences, np.array([])
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).ravel()
    return original_sentences, sentence_scores

# Combine TF and TF-IDF signals into a single combined score for each sentence.
def hybrid_rank_sentences(text):
    """
    Combine TF and TF-IDF signals for summarization explainability.
    Returns: sentences (list), combined_scores (dict index->score)
    """
    original_sentences, tf_scores = generate_tf_scores(text)
    _, tfidf_scores = generate_tfidf_scores(text)

    # Normalize both signals to [0,1] then combine
    max_tf = max(tf_scores.values()) if tf_scores else 0.0
    tf_norm = {i: (s / max_tf if max_tf > 0 else 0.0) for i, s in tf_scores.items()}

    tfidf_norm = {}
    if tfidf_scores.size:
        max_tfidf = float(np.max(tfidf_scores))
        for i, s in enumerate(tfidf_scores):
            tfidf_norm[i] = (s / max_tfidf) if max_tfidf > 0 else 0.0
    else:
        # if tfidf not available, fill zeros
        for i in range(len(original_sentences)):
            tfidf_norm[i] = 0.0

    combined = {}
    for i in range(len(original_sentences)):
        combined[i] = tf_norm.get(i, 0.0) + tfidf_norm.get(i, 0.0)  # simple sum
    return original_sentences, combined

# Pick top N sentences by combined score.
def extract_top_sentences_by_score(original_sentences, combined_scores, top_n=5):
    top_indices = heapq.nlargest(top_n, combined_scores, key=combined_scores.get)
    top_indices.sort()
    return [original_sentences[i] for i in top_indices], [(i, combined_scores[i]) for i in top_indices]

# ----------------- Summarisation (Gemini wrapper + local fallback) -----------------

# Use Gemini to polish summaries; fall back to extractive or transformers-based local summary if unavailable.
def summarize_with_gemini(full_text, extracted_sentences, max_sentences=5):
    """Call Gemini to produce a polished Telugu summary. Falls back to local summarizer if Gemini not available."""
    if not full_text:
        return "⚠️ No article text found."

    highlights_text = "\n".join([f"- {s}" for s in extracted_sentences]) if extracted_sentences else "⚠️ No extracted highlights."

    prompt = f"""
    మీరు తెలుగులో వార్తలను **సంక్షిప్తంగా, స్పష్టంగా మరియు ప్రభావవంతంగా** చెప్పే నిపుణుడు.  
    మీకు రెండు రకాల ఇన్‌పుట్లు ఇవ్వబడ్డాయి:  

    1️⃣ **అసలు వార్తా కంటెంట్ (పూర్తి వ్యాసం)**  
    2️⃣ **ప్రధాన వాక్యాలు (frequency ఆధారంగా తీసుకున్నవి - Highlights)**  

    ---

    ### మీ పని:
    ఈ రెండు ఇన్‌పుట్లను కలిపి ఒక కచ్చితమైన, చదవడానికి సులభమైన **సారాంశం** తయారు చేయాలి.  

    - **ముఖ్య వాక్యాలు** (Highlights) ను బలంగా పరిగణనలోకి తీసుకోండి.  
    - **అసలు వార్తా కంటెంట్** నుండి అదనపు నేపథ్యం, స్పష్టత తీసుకోండి.  
    - ఎటువంటి విరుద్ధత ఉంటే, Highlights వాక్యాలకు ఎక్కువ ప్రాధాన్యం ఇవ్వండి.  

    ---

    ### ✅ సారాంశం రాయడానికి సూచనలు:
    1. **ప్రధాన ఆలోచన (Main Idea) గుర్తించండి:** వ్యాసంలో ప్రధాన సంఘటన లేదా విషయం "ఏమిటి?" అనేది స్పష్టంగా చెప్పండి.  
    2. **5W+1H ను చేర్చండి:**  
    - **ఏమిటి?** (ప్రధాన సంఘటన)  
    - **ఎవరు?** (వ్యక్తులు లేదా సంస్థలు)  
    - **ఎక్కడ?** (ప్రదేశం)  
    - **ఎప్పుడు?** (తేదీ/సమయం )  
    - **ఎందుకు?** (కారణం/నేపథ్యం)  
    - **ఎలా?** (సంఘటన జరిగిన విధానం – అవసరమైతే మాత్రమే)  
    3. **అనవసరమైన వివరాలు తొలగించండి:** పొడవైన కోట్స్, చిన్న ఉదాహరణలు, విస్తారమైన వివరణలు వదిలేయండి.  
    4. **ఫలితాన్ని/ప్రభావాన్ని హైలైట్ చేయండి:** సంఘటన ప్రాముఖ్యత లేదా పర్యవసానం చివర్లో ఒక వాక్యంలో చెప్పండి.  
    5. **సారాంశ పరిమాణం:** 7-10 వాక్యాలలో ఒకే పేరాగా ఉండాలి.  

    ---

    ### ఇన్‌పుట్ డేటా:
    **అసలు వార్తా కంటెంట్**:  
    {full_text[:1200]}...  

    **ముఖ్య వాక్యాలు (frequency ఆధారంగా తీసుకున్నవి - Highlights)**:  
    {highlights_text}  
    """

    # Try Gemini
    if GEMINI_AVAILABLE:
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt, request_options={"timeout": 30})
            # response may contain text property or nested content
            if hasattr(response, "text"):
                return response.text
            try:
                return response.candidates[0].content.parts[0].text
            except Exception:
                return str(response)
        except ServiceUnavailable:
            logger.warning("Gemini unavailable (ServiceUnavailable). Falling back to local summarizer.")
        except Exception as e:
            logger.warning("Gemini summarisation failed: %s. Falling back.", e)

    # Local fallback: simple extractive join of top N sentences ranked by combined score,
    # then optionally do a lightweight abstractive compression if transformers available.
    sents, combined = hybrid_rank_sentences(full_text)
    top_n = min(max_sentences, max(1, len(sents)))
    top_indices = heapq.nlargest(top_n, combined, key=combined.get)
    top_indices.sort()
    extractive_summary = " ".join([sents[i] for i in top_indices])

    # If transformers is available, attempt a small abstractive rewrite using mT5 (if model present)
    if TRANSFORMERS_AVAILABLE:
        try:
            # Using google/mt5-small as fallback (requires model download)
            tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
            inp = "summarize: " + extractive_summary
            inputs = tokenizer(inp, return_tensors="pt", truncation=True, max_length=1024)
            summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=200, min_length=30, early_stopping=True)
            abstractive = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return abstractive
        except Exception as e:
            logger.info("Local transformer summarizer failed or not downloaded: %s. Returning extractive summary.", e)
            return extractive_summary

    return extractive_summary

# ----------------- File write helpers -----------------

# Save the raw article (JSON) to disk with metadata.
def save_full_article(article_obj, filename="full_news_content.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(article_obj, f, ensure_ascii=False, indent=2)
        logger.info("Saved full article to %s", filename)
    except Exception as e:
        logger.warning("Could not save full article: %s", e)

# Save the final summary and explanatory influential sentences to disk.
def save_summary_and_explain(summary_text, influential_sentences, filename="hybrid_summary.txt", meta=None):
    meta = meta or {}
    try:
        with open(filename, "w", encoding="utf-8") as f:
            if meta:
                f.write("METADATA:\n")
                for k, v in meta.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")
            f.write("SUMMARY:\n")
            f.write(summary_text.strip() + "\n\n")
            f.write("INFLUENTIAL SENTENCES (explainability):\n")
            for i, (idx, score, sent) in enumerate(influential_sentences):
                f.write(f"{i+1}. [score={score:.4f}] (sent_idx={idx}) {sent}\n")
        logger.info("Saved summary + explainability to %s", filename)
    except Exception as e:
        logger.warning("Failed to save summary file: %s", e)

# Save the entities dict to disk (entities_dict can contain "ENTITIES" or categories).
def save_entities(entities_dict, filename="entity_highlights.txt", meta=None):
    meta = meta or {}
    try:
        with open(filename, "w", encoding="utf-8") as f:
            if meta:
                f.write("METADATA:\n")
                for k, v in meta.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")
            f.write("EXTRACTED ENTITIES:\n")
            for k, v in entities_dict.items():
                f.write(f"{k}:\n")
                if isinstance(v, (list, set)):
                    for item in v:
                        f.write(f"- {item}\n")
                elif isinstance(v, dict):
                    for subk, subv in v.items():
                        f.write(f"  {subk}: {subv}\n")
                else:
                    f.write(str(v) + "\n")
        logger.info("Saved entities to %s", filename)
    except Exception as e:
        logger.warning("Failed to save entities: %s", e)

# ----------------- CLI / Main flow helpers -----------------

# Prompt the user to choose a summary length mode, returning a (min,max) tuple.
def prompt_summary_length_choice():
    print("\nChoose summary length mode:")
    print("1) Very Short-> 1-2 sentences")
    print("2) short     -> 3-4 sentences")
    print("3) long      -> 7-10 sentences")
    print("4) custom    -> enter number of sentences")
    choice = input("Enter choice (1/2/3/4): ").strip()
    if choice == "1":
        return (1, 2)
    if choice == "2":
        return (3, 4)
    if choice == "3":
        return (7, 10)
    if choice == "4":
        try:
            n = int(input("Enter desired number of sentences (e.g., 4): ").strip())
            if n <= 0:
                n = 3
            return (n, n)
        except Exception:
            return (3, 3)
    return (3, 4)

# ----------------- NER (Gemini-first single-list output) -----------------

# Robustly extract the first JSON object or array from free text output.
def _extract_json_from_text(s):
    """Try to find and parse the first JSON object or array in text; return parsed (dict/list) or None."""
    if not s or not isinstance(s, str):
        return None
    # Try to find any JSON object or array robustly
    patterns = [r'(\{[\s\S]*?\})', r'(\[[\s\S]*?\])']
    for pat in patterns:
        for m in re.finditer(pat, s):
            candidate = m.group(1)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except Exception:
                continue
    # last-resort: try to locate first '{'/'[' and matching '}'/']'
    start_idxs = [m.start() for m in re.finditer(r'[\{\[]', s)]
    end_idxs = [m.start() for m in re.finditer(r'[\}\]]', s)]
    if start_idxs and end_idxs:
        for si in start_idxs:
            for ej in reversed(end_idxs):
                if ej <= si:
                    continue
                candidate = s[si:ej+1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, (dict, list)):
                        return parsed
                except Exception:
                    continue
    return None

# Build a Telugu prompt that asks Gemini to return a single JSON array of top entities.
def _make_gemini_entity_prompt_telugu(article_text, max_entities=10):
    """
    Telugu prompt that requests only a JSON array of the most important entities.
    Use this to replace previous category-based prompt.
    """
    prompt = f"""
మీరు ఒక ఖచ్చితమైన సమాచారం సేకరణ యంత్రం. క్రింద ఇచ్చిన తెలుగు వ్యాసం చదివి, ఆ వ్యాసంలో **స్పష్టంగా ఉటపడే, ప్రధానమైన named entities** మాత్రమే గుర్తించి ఒకే జాబితాగా ఇవ్వండి.

**ముఖ్యమైన నియమాలు (ఖచ్చితంగా పాటించాలి):**
1) మీరు కేవలం **ఒక JSON అర్రే** (ఉదా: ["అంశం1", "అంశం2", ...]) మాత్రమే ఇవ్వాలి. ఏవైనా వ్యాఖ్యలు, కోడ్-ఫెంచెస్ లేదా అదనపు టెక్స్ట్ ఇవ్వవద్దు.
2) జాబితులో **గరిష్టం {max_entities} అంశాలు** ఉండాలి. అవసరమైతే తక్కువ ఉంచండి — అత్యంత ప్రభావవంతమైనవి మాత్రమే.
3) ప్రతి అంశం **యూనిక్** అయి ఉండాలి; డూప్లికేట్లు లేదా వేరియంట్-ఫ్రagments తొలగించండి.
4) కేవలం entity-లే ఇవ్వండి: వ్యక్తులు, సంస్థలు, ప్లాట్‌ఫార్మ్లు/ఉత్పత్తులు, ప్రదేశాలు లేదా గణాంక/సంఖ్యల సూచనలు (సందర్భంతో) మాత్రమే. క్రియలు/వాక్య భాగాలు/అనవసర పదాలు ఇవ్వరాదు (ఉదా: "చెప్పారు", "ఇక్కడ").
5) సాధ్యమైనంతవరకు **తెలుగు లిపిని** ఉపయోగించండి. ఆంగ్ల-పేరు మాత్రమే ఉన్నవైతే వాటినే చేర్చవచ్చు.
6) సంఖ్యలతో కూడిన అంశాల సంశ్లేషణ అవసరమైతే ఒకే అంశంగా రాయండి: ఉదా: "10,000 ఉద్యోగాలు".
7) అవుట్‌పుట్ తప్పక **వాలిడ్ JSON అర్రే**గా ఉండాలి (డబుల్-క్వోట్స్ వాడాలి).

క్రింద వ్యాసం (ప్రారంభం):
\"\"\"{article_text[:9000]}\"\"\"
క్రింద వ్యాసం (ముగింపు).

ఇప్పుడు కేవలి వాలిడ్ JSON అర్రేగా జాబితా ఇవ్వండి:
"""
    return prompt

# Use Gemini to extract a single JSON array of entities; parse and normalize the output.
def extract_entities_via_gemini_telugu(article_text, genai_module, api_key_env="GOOGLE_API_KEY", max_entities=10):
    """
    Call Gemini with Telugu prompt asking for a single JSON array of entities.
    Returns a Python list of entities or None on failure.
    """
    if not genai_module:
        return None
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return None
    try:
        try:
            genai_module.configure(api_key=api_key)
        except Exception:
            pass

        prompt = _make_gemini_entity_prompt_telugu(article_text, max_entities)
        model = genai_module.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt, request_options={"timeout": 30})

        # extract textual output robustly
        text_out = None
        if hasattr(response, "text") and isinstance(response.text, str):
            text_out = response.text
        else:
            try:
                text_out = response.candidates[0].content.parts[0].text
            except Exception:
                text_out = str(response)

        if not text_out:
            return None

        parsed = _extract_json_from_text(text_out)

        # If parsed is list -> treat as array of entities
        if isinstance(parsed, list):
            # normalize entries: strip, filter very short items
            cleaned = []
            for it in parsed:
                if not isinstance(it, str):
                    continue
                s = it.strip()
                if len(s) < 2:
                    continue
                cleaned.append(s)
            # dedupe preserving order and limit
            seen = set(); out = []
            for x in cleaned:
                if x not in seen:
                    seen.add(x); out.append(x)
                if len(out) >= max_entities:
                    break
            return out if out else None

        # If parsed is dict (older behaviour), aggregate its values into a single list
        if isinstance(parsed, dict):
            items = []
            for v in parsed.values():
                if isinstance(v, list):
                    items.extend([str(x).strip() for x in v if isinstance(x, str) and x.strip()])
                elif isinstance(v, str):
                    # comma or newline separated
                    items.extend([x.strip() for x in re.split(r',|\n', v) if x.strip()])
            # normalize / dedupe / limit
            seen = set(); out = []
            for it in items:
                if it not in seen and len(it) >= 2:
                    seen.add(it); out.append(it)
                if len(out) >= max_entities:
                    break
            return out if out else None

        return None
    except Exception as e:
        logger.warning("Gemini Telugu extraction failed: %s", e)
        return None

# Top-level extractor that prefers Gemini, then stanza, then heuristic.
def extract_entities(article_text, max_entities=10):
    """
    High-level extractor (minimal-change integration):
    1) Try Gemini Telugu (single-array).
    2) Fallback to Stanza (collect entities into single list).
    3) Fallback to conservative heuristic (frequent multi-word phrases).
    Returns dict: {"ENTITIES": [list]} so existing save_entities() remains compatible.
    """
    # 1) Gemini-first
    try:
        if 'genai' in globals() and GEMINI_AVAILABLE:
            gem_list = extract_entities_via_gemini_telugu(article_text, genai, max_entities=max_entities)
            if gem_list:
                # final lightweight post-filter: remove stop-words-only items
                sw = get_telugu_stop_words()
                final = []
                for it in gem_list:
                    toks = [t for t in it.split() if t.strip()]
                    if toks and all((tok in sw) for tok in toks):
                        continue
                    final.append(it)
                    if len(final) >= max_entities:
                        break
                return {"ENTITIES": final}
    except Exception as e:
        logger.warning("Gemini attempt raised: %s", e)

    # 2) Stanza fallback
    try:
        if 'stanza' in globals() and STANZA_AVAILABLE:
            try:
                stanza.download('te', processors='tokenize,ner', verbose=False)
            except Exception:
                pass
            try:
                nlp = stanza.Pipeline(lang='te', processors='tokenize,ner', use_gpu=False, verbose=False)
                doc = nlp(article_text)
                cand = []
                for ent in doc.ents:
                    t = ent.text.strip()
                    if not t:
                        continue
                    # skip pure stop-word tokens
                    toks = [x for x in t.split() if x.strip()]
                    if toks and all((tok in get_telugu_stop_words()) for tok in toks):
                        continue
                    cand.append(t)
                # preserve order of first appearance, dedupe, limit
                seen = set(); out = []
                for c in cand:
                    if c not in seen:
                        seen.add(c); out.append(c)
                    if len(out) >= max_entities:
                        break
                if out:
                    return {"ENTITIES": out}
            except Exception as e:
                logger.warning("Stanza NER failed: %s", e)
    except Exception as e:
        logger.debug("Stanza not available: %s", e)

    # 3) Heuristic fallback: frequent multi-token phrases + long tokens
    sentences = nltk.sent_tokenize(article_text)
    freq = defaultdict(int)
    for s in sentences:
        toks = telugu_tokenizer(s)
        for i in range(len(toks)):
            if toks[i]:
                freq[toks[i]] += 1
            if i+1 < len(toks):
                big = toks[i] + " " + toks[i+1]; freq[big] += 1
            if i+2 < len(toks):
                tri = toks[i] + " " + toks[i+1] + " " + toks[i+2]; freq[tri] += 1

    # pick multi-token phrases first (likely to be names/orgs), then long unigrams
    candidates = sorted(freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    collected = []
    sw = get_telugu_stop_words()
    for phrase, count in candidates:
        if len(phrase) < 3:
            continue
        toks = phrase.split()
        # skip if all tokens are stop words
        if toks and all((t in sw) for t in toks):
            continue
        # prefer multi-token phrases
        if len(toks) >= 2:
            collected.append(phrase)
        else:
            # long single token likely proper noun/topic
            if len(phrase) >= 6:
                collected.append(phrase)
        if len(collected) >= max_entities * 3:  # gather a pool
            break

    # dedupe while preserving order, then limit
    seen = set(); final = []
    for x in collected:
        if x not in seen:
            seen.add(x); final.append(x)
        if len(final) >= max_entities:
            break

    return {"ENTITIES": final}

# ----------------- MAIN -----------------

def main():
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    if GEMINI_AVAILABLE and API_KEY:
        try:
            genai.configure(api_key=API_KEY)
            logger.info("Gemini configured.")
        except Exception as e:
            logger.warning("Could not configure Gemini: %s", e)

    url = input("Enter Telugu news URL: ").strip()
    if not url:
        print("No URL provided. Exiting.")
        return

    # Get article
    article_obj = get_news_content(url)
    if not article_obj or not article_obj.get("text"):
        print("❌ Failed to fetch article or extract text.")
        return
    full_text = article_obj["text"]
    meta = {"url": article_obj.get("url", url), "title": article_obj.get("title", ""), "authors": article_obj.get("authors", ""), "publish_date": article_obj.get("publish_date", "")}
    save_full_article(article_obj, filename="full_news_content.txt")
    print("\n✅ Article fetched and saved.")

    # Ask user for desired length
    min_s, max_s = prompt_summary_length_choice()
    # We'll pick a requested number as middle of range (or exact if equal)
    requested_sentences = (min_s + max_s) // 2

    # Extractive hybrid highlights to feed to Gemini/local summarizer
    sents, combined_scores = hybrid_rank_sentences(full_text)
    # pick top k for highlights (use requested_sentences + extras)
    highlight_count = min(len(sents), max(4, requested_sentences + 2))
    highlight_indices = heapq.nlargest(highlight_count, combined_scores, key=combined_scores.get)
    highlight_indices.sort()
    highlights = [sents[i] for i in highlight_indices]

    print("\n🔹 Extracted highlights (hybrid ranking):")
    for i, h in enumerate(highlights[:8], 1):
        print(f"{i}. {h[:120]}{'...' if len(h) > 120 else ''}")

    # Summarise (try Gemini then fallback)
    print("\n⏳ Generating summary...")
    final_summary = summarize_with_gemini(full_text, highlights, max_sentences=requested_sentences)
    print("\n✅ Summary generated.\n")
    print("---- SUMMARY PREVIEW ----")
    print(final_summary[:1500] + ("\n...\n" if len(final_summary) > 1500 else "\n"))
    print("-------------------------\n")

    # Explainability: Get top influential sentences (we'll pick top 5 by combined score)
    top_sentences, top_with_scores = extract_top_sentences_by_score(sents, combined_scores, top_n=5)
    influential_sentences = []
    for idx, score in top_with_scores:
        influential_sentences.append((idx, score, sents[idx]))

    # Save summary + explainability to file
    save_summary_and_explain(final_summary, influential_sentences, filename="hybrid_summary.txt", meta=meta)

    # NER extraction
    print("🔎 Extracting entities (NER)...")
    entities = extract_entities(full_text)
    if not entities:
        entities = {"ENTITIES": []}
    save_entities(entities, filename="entity_highlights.txt", meta=meta)
    print("✅ Entities saved to 'entity_highlights.txt'")

    print("\nAll done. Files created:")
    print("- full_news_content.txt (raw + metadata)")
    print("- hybrid_summary.txt (final summary + influential sentences)")
    print("- entity_highlights.txt (NER results)\n")

if __name__ == "__main__":
    main()
