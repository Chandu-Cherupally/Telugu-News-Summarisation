# 🌐 Hybrid Telugu News Summariser — README

**👨‍💻 Author:** 22CSB0C20 — Chandu Cherupally  
**🧠 Description:**  
A command-line tool that fetches **Telugu news articles**, extracts and summarises their content using **hybrid TF + TF-IDF ranking** and optional **Gemini-powered summarisation**, along with **entity highlighting** and **explainability** for key sentences.

---

## 🚀 Key Features
✅ **Multi-length summaries:** TL;DR (1–2 lines), short (3–4), long (7–10), and custom lengths.  
✅ **Hybrid summarisation:** Combines **term frequency** and **TF-IDF** scoring for robust extractive results.  
✅ **Explainability:** Lists the top influential sentences with scores to show why they were chosen.  
✅ **Entity extraction:** Uses a **Gemini Telugu prompt** for top named entities (people, places, orgs, etc.).  
✅ **Automatic fallback:** Uses **stanza** or **heuristic** NER if Gemini API fails.  
✅ **Clean outputs:** Saves structured data in readable `.txt` files.  

---


## 🧩 The Architecture Flow

**Extractive stage** (classical NLP)
Uses TF and TF-IDF scores to rank important sentences.
This part is not a sequence-to-sequence model — it’s purely statistical (bag-of-words style).
It identifies which sentences are likely most informative.

**Abstractive stage**(Gemini / Transformer)
Here’s where the Seq2Seq logic comes in.s
The model (like Gemini or optionally a Transformer like t5-base or flan-t5) takes the entire article text + top-ranked sentences as input sequence,
and generates a new summary as another sequence.
This generation step is context-aware, i.e., it considers the semantic relationships between words, sentences, and meaning (not just word frequency).

## 📁 Output Files
| File | Description |
|------|--------------|
| `full_news_content.txt` | Raw news content and metadata (URL, title, authors, publish_date, full text). |
| `hybrid_summary.txt` | The final hybrid summary with influential sentences and explainability data. |
| `entity_highlights.txt` | Extracted major entities (Telugu text, JSON format). |

---

## ⚙️ Quick Setup Guide

### 🧩 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate

📦 2. Install Dependencies
pip install requests beautifulsoup4 nltk scikit-learn newspaper3k readability-lxml

🔑 3. Set Gemini API Key (if using Gemini)
export GOOGLE_API_KEY="YOUR_KEY_HERE"

▶️ 4. Run the Script
python summariser.py (IN TERMINAL)
python app.py (In /ui_app directory - FOR UI)
🧩 How It Works (Step-by-Step)

📰 Article Fetching
get_news_content() — Tries newspaper3k, falls back to readability-lxml, then BeautifulSoup.

📊 Sentence Scoring
generate_tf_scores() + generate_tfidf_scores() compute scores;
hybrid_rank_sentences() combines and ranks them.

📝 Summarisation
summarize_with_gemini() — sends full article + highlights to Gemini for natural-language summary.
If Gemini unavailable → fallback to local transformer/extractive summary.

💬 Explainability
Stores the top influential sentences (with their scores and indices) in hybrid_summary.txt.

🔍 Entity Extraction
extract_entities() — uses Gemini Telugu prompt to extract top named entities (people, orgs, places).
If Gemini fails, it uses stanza Telugu model or a keyword heuristic.

⚙️ Configuration & Tuning
Parameter	        Description	Default
GOOGLE_API_KEY	     Enables Gemini summarisation + NER	None
max_entities	     Limits number of returned entities	10
summary_type	     TLDR / short / long / custom	custom
use_transformer	  Enable local abstractive summarisation	False

🪶 Tip: For cleaner Telugu entity results, filter non-Telugu words (Unicode range \u0C00–\u0C7F).

🔧 Future Enhancements

✅ Add caching for article and Gemini responses
✅ Integrate evaluation metrics (ROUGE, BLEU) for summaries
✅ Create FastAPI / Streamlit web interface
✅ Dockerize environment for portability
✅ Add small test suite for ranking & JSON parsing

🧩 Example Workflow (Simple)
Enter Telugu news URL: https://www.eenadu.net/telugu-news/andhra-pradesh/cm-chandrababu-speech-in-delhi/1701/125188538/
Choose summary type: TLDR
Summary generated successfully ✅
Entity highlights saved to entity_highlights.txt 🧠


🟢 Output ready!
hybrid_summary.txt → Your clean, concise Telugu summary
entity_highlights.txt → Important names, locations, organizations
