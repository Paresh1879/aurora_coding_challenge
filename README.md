# Member Q&A System

AI-powered question-answering system that understands natural language questions about member data.

**🚀 Live Demo:** https://aurora-coding-challenge-1.onrender.com

## Features

- **Smart Search**: Adapts strategy based on question type (counting, dates, preferences)
- **Cosine Similarity**: Better semantic understanding ("car service" ≠ "owns car")
- **Hybrid Search**: AI embeddings + keyword fallback
- **Natural Answers**: Conversational responses, no technical jargon

## How It Works

1. **Fetch Data**: Loads messages from API (cached for speed)
2. **Vector Search**: FAISS + OpenAI embeddings + cosine similarity
3. **Smart Agent**: Analyzes question → picks best search strategy → finds answers
4. **Natural Response**: GPT-4 generates conversational answers

**Why Cosine Similarity?** Measures meaning (angle between vectors), not just word matching. Industry standard for text similarity.

## ✅ Bonus 1: Alternative Approaches Considered

### 1. Hierarchical Embedding + Chunked Context Retrieval

* **Method:** Split messages into semantic chunks (e.g., by conversation thread), embed each chunk separately, and retrieve the top chunks relevant to the query.
* **Pros:**

  * Supports longer conversations beyond single-message embeddings.
  * Reduces irrelevant context being passed to the LLM.
* **Cons:**

  * Requires extra preprocessing and chunking logic.
* **Decision:** Useful for long threads, but adds implementation complexity.

### 2. Hybrid Keyword + Embedding Search

* **Method:** First filter messages using keywords or regex to narrow down candidates, then rank using vector similarity.
* **Pros:**

  * Reduces the number of messages sent to the embedding search, improving speed and cost.
  * Combines deterministic filtering with semantic reasoning.
* **Cons:**

  * Keywords must be carefully curated.
  * Some queries may be missed if keyword coverage is insufficient.
* **Decision:** Balances speed and accuracy, but maintenance of keywords can be cumbersome.

### 3. **Vector Database + Semantic Search (Chosen Approach)**

   * Embed all messages with OpenAI embeddings, store in a vector DB (FAISS/Chroma), and use cosine similarity to retrieve relevant context.
   * **Pros:** Scalable, semantic, flexible, cost-effective, and transparent reasoning.
   * **Decision:** Selected for implementation.

## 📊 Bonus 2: Dataset Insights & Anomalies

### Key Anomalies Identified

1. **Missing or Null User Information**

   * Some messages had missing `user_id` or `user_name`.
   * **Solution:** Filtered out incomplete messages during indexing.

2. **Inconsistent Date Formats**

   * Dates appeared as ISO timestamps, natural language, or full dates (e.g., “next week”, “November 7, 2025”).
   * **Solution:** Used `dateutil` parser and calculated relative dates for consistent temporal queries.

3. **Name Variations**

   * Members referred by first name, full name, or nickname.
   * **Solution:** Flexible name matching including partial and fuzzy matching.

4. **Duplicate or Near-Duplicate Messages**

   * Some messages repeated or were very similar.
   * **Solution:** Deduplicated messages using `message_id`.

5. **Incomplete or Empty Messages**

   * Very short or empty messages created noise in embeddings.
   * **Solution:** Filtered them out during preprocessing.

## Quick Start

```bash
# Setup
git clone <repo-url>
cd aurora_coding_challenge
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
uvicorn app.main:app --reload
```

## 🧪 Test It

```bash
# Test live deployment
curl https://aurora-coding-challenge-1.onrender.com/health

# Ask questions (live)
curl -X POST https://aurora-coding-challenge-1.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla going to London?"}'

### `POST /ask`
```json
{
  "question": "When is Layla going to London?"
}
```

Response:
```json
{
  "answer": "Layla is planning her trip to London in November 2025."
}
```

### `GET /health`
```json
{
  "status": "healthy",
  "messages_loaded": 3349
}
```

## 🛠️ Stack

- FastAPI, LangChain, OpenAI GPT-4
- FAISS (cosine similarity)
- Python 3.11+

---

Built for Aurora Coding Challenge
