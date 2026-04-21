#!/usr/bin/env python3
"""HotpotQA-style RAG benchmark: inference + citation-aware evaluation.

Dependencies: aiohttp, python-dotenv, tqdm (plus the stdlib).

Usage
-----
  # Evaluate a direct OpenAI-compatible LLM using the benchmark's own context
  python rag_luciole_bench.py --benchmark bench.jsonl --generate --llm-judge \\
      --output results.jsonl --report report.json

  # Index the benchmark chunks into an OpenRAG partition, then evaluate
  # OpenRAG end-to-end (retrieval + generation)
  python rag_luciole_bench.py --benchmark bench.jsonl --openrag-index \\
      --partition hotpot
  python rag_luciole_bench.py --benchmark bench.jsonl --generate --openrag-query \\
      --partition hotpot --llm-judge --output results.jsonl --report report.json

  # Evaluate pre-generated responses without re-running inference
  python rag_luciole_bench.py --benchmark bench.jsonl --responses responses.jsonl \\
      --llm-judge --output results.jsonl

Input format (JSONL, one chat-format SFT row per line):
  {"messages": [
     {"role": "system",    "content": "... Here are the retrieved documents : `<ctx>`"},
     {"role": "user",      "content": "<question>"},
     {"role": "assistant", "content": "<reasoning ... **Final Answer:** <ans>>"}],
   "supporting_facts_titles": ["<gold title>", ...],
   "chunks_total": <int>}

The context inside the system prompt must use the OpenRAG chunk format
(``* filename: <title>`` + ``[CHUNK_START] ... [CHUNK_END]``).
Responses use either ``[Title]`` brackets or ``##Cite "Title"##`` markers.

Environment (loaded from .env):
  LLM_API_URL, LLM_API_KEY, LLM_MODEL    # direct-LLM inference and judge
  AUTH_TOKEN, APP_URL, APP_PORT           # OpenRAG backend

Metrics: citation P/R/F1, distractor rate, token-F1, ROUGE-L, joint-F1,
plus a factual 1-5 judge score when --llm-judge is passed.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

LLM_API_URL = os.getenv(
    "LLM_API_URL",
    "https://chat.lucie.ovh.linagora.com/v1/chat/completions",
)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "Mistral-Small-3.1-24B-Instruct-2503")
OPENRAG_AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
OPENRAG_APP_URL = os.getenv("APP_URL", "127.0.0.1")
OPENRAG_APP_PORT = os.getenv("APP_PORT", "8080")
MAX_CONCURRENT = 10

# ── regex constants ─────────────────────────────────────────────────

_CITE_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_CITE_SOURCE_BRACKET_RE = re.compile(r'\[Source:\s*"([^"]+)"\s*\]')
_CITE_ANGLE_RE = re.compile(r'<<cite:\s*"([^"]+)"\s*>>')
_CITE_TAG_RE = re.compile(r'<cite>\s*"?([^"<]+?)"?\s*</cite>')
_CHUNK_TITLE_RE = re.compile(r"\*\s*filename:\s*(.+)")
_CHUNK_TEXT_RE = re.compile(
    r"\[CHUNK_START\]\n(.*?)\n\[CHUNK_END\]", re.DOTALL,
)

# ── inference: system prompts & context formatting ──────────────────

CONTEXT_SEPARATOR = "\n" + "-" * 10 + "\n\n"

_CITATION_INSTRUCTION = {
    "en": (
        'When quoting from the context, wrap the excerpt with `##begin_quote##` and `##end_quote##`, '
        'and attribute it with `##Cite "source title"##`.'
    ),
    "fr": (
        'Lorsque vous citez le contexte, encadrez l\'extrait avec `##begin_quote##` et `##end_quote##`, '
        'et attribuez-le avec `##Cite "titre de la source"##`.'
    ),
}

INFERENCE_SYSTEM_PROMPT = {
    "en": (
        "You are an AI conversational assistant specialized in **information retrieval and synthesis**.\n"
        "Your goal is to provide **precise, reliable, and well-structured answers** using **only the retrieved documents** (`Context`).\n"
        "Prioritize **clarity, accuracy, and completeness** in your responses.\n"
        "\n"
        "## Rules\n"
        "\n"
        "1. Use only the provided Context\n"
        "   * Base your answer **exclusively** on the information contained in the `Context`.\n"
        "   * **Never infer**, assume, or rely on any external knowledge.\n"
        "   * If the context is **insufficient**, **invite the user** to clarify their query or provide additional keywords.\n"
        "   * {citation_instruction}\n"
        "\n"
        "2. Language Consistency\n"
        "   * Always respond **in the same language** as the user's query.\n"
        "\n"
        "3. Structure and Readability\n"
        "   * Ensure responses are **concise yet complete**, avoiding omission of key details.\n"
        "\n"
        "Here are the retrieved documents : `{context}`"
    ),
    "fr": (
        "Vous êtes un assistant conversationnel IA spécialisé dans la **recherche et la synthèse d'informations**.\n"
        "Votre objectif est de fournir des **réponses précises, fiables et bien structurées** en utilisant **uniquement les documents récupérés** (`Contexte`).\n"
        "Privilégiez la **clarté, l'exactitude et l'exhaustivité** dans vos réponses.\n"
        "\n"
        "## Règles\n"
        "\n"
        "1. Utilisez uniquement le Contexte fourni\n"
        "   * Basez votre réponse **exclusivement** sur les informations contenues dans le `Contexte`.\n"
        "   * **N'inférez jamais**, ne supposez pas et ne vous appuyez pas sur des connaissances externes.\n"
        "   * Si le contexte est **insuffisant**, **invitez l'utilisateur** à préciser sa requête ou à fournir des mots-clés supplémentaires.\n"
        "   * {citation_instruction}\n"
        "\n"
        "2. Cohérence linguistique\n"
        "   * Répondez toujours **dans la même langue** que la requête de l'utilisateur.\n"
        "\n"
        "3. Structure et lisibilité\n"
        "   * Assurez-vous que les réponses sont **concises mais complètes**, en évitant d'omettre les détails clés.\n"
        "\n"
        "Voici les documents récupérés : `{context}`"
    ),
}

OPENRAG_QUERY_SYSTEM_PROMPT = {
    "en": (
        "You are an AI conversational assistant specialized in retrieval over an indexed knowledge base.\n"
        "Answer the user's question using only retrieved documents from OpenRAG.\n"
        "Cite supporting documents inline with bracket citations in the exact format [Title].\n"
        "Do not mention documents you did not use.\n"
        "If retrieval is insufficient, say so explicitly."
    ),
    "fr": (
        "Vous êtes un assistant conversationnel IA spécialisé dans la recherche sur une base de connaissances indexée.\n"
        "Répondez à la question de l'utilisateur en utilisant uniquement les documents récupérés par OpenRAG.\n"
        "Citez les documents de support inline avec des citations entre crochets au format exact [Title].\n"
        "Ne mentionnez pas de documents que vous n'avez pas utilisés.\n"
        "Si la récupération est insuffisante, dites-le explicitement."
    ),
}


def _is_openrag_format(context: str) -> bool:
    return "[CHUNK_START]" in context and "[CHUNK_END]" in context


def _reformat_openrag_chunks(raw_context: str) -> str:
    chunk_pattern = re.compile(
        r'((?:\[CONTEXT\].*?)?(?:\*\s*filename:.*?)\[CHUNK_START\].*?\[CHUNK_END\])',
        re.DOTALL,
    )
    chunks = chunk_pattern.findall(raw_context)
    if not chunks:
        return raw_context.strip()
    return CONTEXT_SEPARATOR.join(c.strip() for c in chunks)


def reformat_context_chunks(raw_context: str) -> str:
    """Reformat context chunks with a standard separator between them."""
    if _is_openrag_format(raw_context):
        return _reformat_openrag_chunks(raw_context)

    clean_context = raw_context.replace("-" * 10, "\n")
    pattern = r'\n?\[([^\]]+)\]\n'
    parts = re.split(pattern, clean_context)

    chunks = []
    if parts[0].strip():
        chunks.append(parts[0].strip())

    for i in range(1, len(parts), 2):
        title = parts[i]
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        chunk = f"[{title}]\n{content}" if content else f"[{title}]"
        chunks.append(chunk)

    if not chunks:
        return raw_context.strip()

    return CONTEXT_SEPARATOR.join(chunks)


def build_inference_messages(question: str, context: str, language: str) -> list[dict]:
    """Build chat messages for inference."""
    template = INFERENCE_SYSTEM_PROMPT.get(language, INFERENCE_SYSTEM_PROMPT["en"])
    citation_instruction = _CITATION_INSTRUCTION.get(language, _CITATION_INSTRUCTION["en"])
    reformatted = reformat_context_chunks(context)
    system_content = template.format(
        context=reformatted,
        citation_instruction=citation_instruction,
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]


def build_openrag_query_messages(question: str, language: str) -> list[dict]:
    return [
        {"role": "user", "content": question},
    ]

# ── refusal detection (multilingual) ────────────────────────────────

REFUSAL_PATTERNS = [
    # English
    r"(?:do not|don't|cannot|can't|unable to) (?:allow me to |)(?:answer|provide|respond)",
    r"(?:not enough|insufficient) information",
    # French
    r"ne (?:me )?permettent? pas",
    r"pas en mesure de répondre",
    r"ne (?:peux|puis) pas (?:fournir|répondre)",
]

_REFUSAL_RE = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]

# ── factual judge prompt ────────────────────────────────────────────

JUDGE_FACTUAL_SYSTEM_PROMPT = """\
You are an impartial factual evaluator. You will be given:
1. A **question**.
2. A **correct answer** (ground truth).
3. The **supporting facts** (the specific document titles that contain the evidence needed to answer the question).
4. A **context** (retrieved documents).
5. A **reasoning trace** produced by an AI assistant.

Your task is to rate the **factual correctness and faithfulness** of the reasoning trace on a scale from 1 to 5:

- **1**: The final answer is wrong AND the reasoning does not use the correct supporting facts at all.
- **2**: The final answer is wrong, but the reasoning references some of the correct supporting facts; OR the answer is partially right but the reasoning is based on wrong evidence.
- **3**: The final answer is approximately correct but imprecise, or the reasoning misses one of the key supporting facts, or the reasoning contains a factual error despite reaching the right answer.
- **4**: The final answer is correct and the reasoning uses most of the supporting facts properly, with only minor omissions or imprecisions.
- **5**: The final answer is correct, the reasoning correctly identifies and uses all the supporting facts, and the logical chain from evidence to answer is flawless.

You MUST reply with ONLY a JSON object in this exact format (no other text):
{"score": <int>, "justification": "<one sentence>"}
"""


# ── utility functions ───────────────────────────────────────────────


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison (lowercase, no articles/punctuation)."""
    answer = answer.lower()
    answer = re.sub(r"\b(a|an|the|le|la|les|l|un|une|des|du|de|d)\b", " ", answer)
    answer = re.sub(r"[^\w\s]", "", answer)
    return " ".join(answer.split()).strip()


def compute_token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between normalized prediction and reference."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 between normalized prediction and reference (LCS-based)."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def detect_refusal(response: str) -> bool:
    """Detect if a model response is a refusal to answer."""
    for pattern in _REFUSAL_RE:
        if pattern.search(response):
            return True
    return False


def extract_answer_from_reasoning(reasoning: str) -> str | None:
    """Extract the final answer from a reasoning trace (EN/FR)."""
    patterns = [
        # English
        r"\*\*(?:Final\s+)?Answer[:\*]*\**[:\s]*(.+?)(?:\n|$)",
        r"(?:Final\s+)?Answer[:\s]+(.+?)(?:\n|$)",
        r"[Tt]he (?:final )?answer is[:\s]+(.+?)(?:\.|$)",
        # French
        r"\*\*Réponse\s+finale\s*[:\*]*\**[:\s]*(.+?)(?:\n|$)",
        # Fallback: last stand-alone bold span
        r"\*\*(.+?)\*\*\s*(?:\.|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, reasoning, re.IGNORECASE | re.MULTILINE)
        if matches:
            answer = matches[-1].strip()
            answer = re.sub(r"\*+", "", answer).strip(".")
            if 0 < len(answer) <= 1000:
                return answer
    return None


def extract_cited_titles(reasoning: str) -> list[str]:
    """Extract chunk titles cited via any of the supported markers:

    - ``##Cite "title"##``          (hash-cite)
    - ``[Source: "title"]``         (bracket-source)
    - ``<<cite: "title">>``         (angle-cite)
    - ``<cite>"title"</cite>``      (xml-cite)
    """
    titles = re.findall(r'##Cite\s*"([^"]+)"\s*##', reasoning)
    titles += re.findall(r'##Cite\s*[«»]\s*([^«»]+?)\s*[«»]\s*##', reasoning)
    titles += _CITE_SOURCE_BRACKET_RE.findall(reasoning)
    titles += _CITE_ANGLE_RE.findall(reasoning)
    titles += _CITE_TAG_RE.findall(reasoning)
    if not titles:
        titles = re.findall(r'##Cite\s+([^#"«»]+?)\s*##', reasoning)
    seen: set[str] = set()
    unique: list[str] = []
    for t in titles:
        norm = t.strip().lower()
        if norm not in seen:
            seen.add(norm)
            unique.append(t.strip())
    return unique


def evaluate_chunk_citations(
    cited: list[str], expected: list[str],
) -> tuple[float | None, float | None]:
    """Compute precision and recall of cited chunks vs ground-truth supporting facts."""
    cited_norm = {t.strip().lower() for t in cited}
    expected_norm = {t.strip().lower() for t in expected}
    if not cited_norm and not expected_norm:
        return None, None
    correct = cited_norm & expected_norm
    precision = len(correct) / len(cited_norm) if cited_norm else None
    recall = len(correct) / len(expected_norm) if expected_norm else None
    return precision, recall


def _extract_json(text: str) -> dict:
    """Extract a JSON object from text that may contain markdown fences or extra prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise json.JSONDecodeError("No JSON object found", text, 0)


def load_jsonl_by_id(path: str | None) -> dict[str, dict]:
    """Load a JSONL file into ``{row["id"]: row}``. Missing file → empty dict."""
    out: dict[str, dict] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = row.get("id")
            if rid is not None:
                out[str(rid)] = row
    return out


def sha256_file(path: str) -> str:
    """Compute SHA-256 of a file (hex digest)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── LLM helpers ─────────────────────────────────────────────────────


async def post_chat_completion(
    session: aiohttp.ClientSession,
    *,
    api_url: str,
    api_key: str,
    payload: dict,
    max_retries: int = 3,
    timeout_seconds: int = 120,
) -> dict | None:
    """Post a chat completion request with retries and exponential backoff."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(
        total=timeout_seconds,
        connect=timeout_seconds,
        sock_connect=timeout_seconds,
        sock_read=timeout_seconds,
    )
    for attempt in range(max_retries):
        try:
            async with session.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    print(
                        f"HTTP attempt {attempt + 1}/{max_retries} failed with "
                        f"status {response.status}: {body[:500]}"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return None
                return await response.json()
        except Exception as exc:
            print(f"HTTP attempt {attempt + 1}/{max_retries} failed: {type(exc).__name__}: {exc}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    return None


def build_openrag_base_url(explicit_url: str | None = None) -> str:
    if explicit_url:
        return explicit_url.rstrip("/")
    host = OPENRAG_APP_URL.strip()
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return f"{host}:{OPENRAG_APP_PORT}".rstrip("/")


def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
    }


async def openrag_request_json(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    *,
    api_key: str,
    expected_statuses: tuple[int, ...] = (200,),
    json_payload: dict | None = None,
    data: aiohttp.FormData | None = None,
    timeout_seconds: int = 180,
) -> dict | list | str | None:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    try:
        async with session.request(
            method,
            url,
            headers=_auth_headers(api_key),
            json=json_payload,
            data=data,
            timeout=timeout,
        ) as response:
            body = await response.text()
            if response.status not in expected_statuses:
                raise RuntimeError(f"{method} {url} failed with {response.status}: {body[:500]}")
            ctype = response.headers.get("Content-Type", "")
            if "application/json" in ctype:
                return json.loads(body) if body else {}
            return body
    except Exception as exc:
        print(f"OpenRAG request failed: {exc}")
        return None


def _extract_titles_from_files_payload(payload: object) -> set[str]:
    titles: set[str] = set()
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("files", "data", "items", "documents"):
            value = payload.get(key)
            if isinstance(value, list):
                items = value
                break
        else:
            items = [payload]
    else:
        return titles

    for item in items:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = (
                item.get("filename")
                or item.get("file_name")
                or item.get("name")
                or item.get("title")
            )
        else:
            continue
        if not name:
            continue
        stem = Path(str(name)).stem.strip()
        if stem:
            titles.add(stem)
    return titles


async def ensure_openrag_partition(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    partition: str,
    api_key: str,
) -> None:
    """Create the partition if missing (409 means it already exists)."""
    result = await openrag_request_json(
        session,
        "POST",
        f"{base_url}/partition/{partition}",
        api_key=api_key,
        expected_statuses=(200, 201, 202, 204, 409),
    )
    if result is None:
        raise RuntimeError(f"Unable to create or verify OpenRAG partition '{partition}'")


async def add_openrag_file(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    partition: str,
    api_key: str,
    file_id: str,
    filename: str,
    content: str,
) -> str | None:
    form = aiohttp.FormData()
    form.add_field("file", content.encode("utf-8"), filename=filename, content_type="text/markdown")
    url = f"{base_url}/indexer/partition/{partition}/file/{file_id}"
    timeout = aiohttp.ClientTimeout(total=180)
    try:
        async with session.post(
            url,
            headers=_auth_headers(api_key),
            data=form,
            timeout=timeout,
        ) as response:
            body = await response.text()
            if response.status == 409:
                return "__SKIP__"
            if response.status not in (200, 201, 202):
                raise RuntimeError(f"POST {url} failed with {response.status}: {body[:500]}")
            ctype = response.headers.get("Content-Type", "")
            if "application/json" not in ctype:
                return None
            result = json.loads(body) if body else {}
            task_url = result.get("task_status_url")
            return str(task_url) if task_url else None
    except Exception as exc:
        print(f"OpenRAG upload failed: {exc}")
        return None


async def wait_for_openrag_task(
    session: aiohttp.ClientSession,
    *,
    task_url: str,
    api_key: str,
    poll_interval_seconds: float = 1.0,
    max_polls: int = 300,
) -> bool:
    for _ in range(max_polls):
        result = await openrag_request_json(
            session,
            "GET",
            task_url,
            api_key=api_key,
            expected_statuses=(200,),
            timeout_seconds=60,
        )
        if isinstance(result, dict):
            state = result.get("task_state")
            if state in {"SUCCESS", "COMPLETED"}:
                return True
            if state == "FAILED":
                return False
        await asyncio.sleep(poll_interval_seconds)
    return False


def build_factual_judge_user_prompt(
    question: str, correct_answer: str, supporting_facts: list[str],
    context: str, reasoning: str,
) -> str:
    sf_text = "\n".join(f"- {t}" for t in supporting_facts) if supporting_facts else "(none available)"
    return (
        f"**Question:**\n{question}\n\n"
        f"**Correct answer:**\n{correct_answer}\n\n"
        f"**Supporting facts (document titles):**\n{sf_text}\n\n"
        f"**Context:**\n{context}\n\n"
        f"**Reasoning trace:**\n{reasoning}"
    )


async def call_judge(
    session: aiohttp.ClientSession,
    system_prompt: str,
    user_content: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    *,
    api_key: str,
    api_url: str,
    model: str,
    max_retries: int = 3,
) -> dict | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": 512,
    }
    async with semaphore:
        result = await post_chat_completion(
            session,
            api_url=api_url,
            api_key=api_key,
            payload=payload,
            max_retries=max_retries,
            timeout_seconds=120,
        )
        if result is None:
            return None
        try:
            content = result["choices"][0]["message"]["content"].strip()
            parsed = _extract_json(content)
            score = int(parsed["score"])
            if score < 1 or score > 5:
                raise ValueError(f"Score {score} out of range 1-5")
            return {"score": score, "justification": parsed.get("justification", "")}
        except Exception as exc:
            print(f"  Judge response parsing failed: {exc}")
    return None


# ── inference ─────────────────────────────────────────────────────��─


def _get_context_string(row: dict) -> str:
    """Get context as a string, handling both augmented (str) and raw (dict) formats."""
    ctx = row.get("context", "")
    if isinstance(ctx, str):
        return ctx
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sentences = ctx.get("sentences", [])
        chunks = []
        for title, sents in zip(titles, sentences):
            text = " ".join(sents).strip() if isinstance(sents, list) else str(sents).strip()
            chunks.append(f"[{title}]\n{text}")
        return "\n\n".join(chunks)
    return str(ctx)


async def infer_single(
    session: aiohttp.ClientSession,
    row: dict,
    semaphore: asyncio.Semaphore,
    *,
    api_url: str,
    api_key: str,
    model: str,
    language: str,
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> str | None:
    """Run inference on a single row."""
    context = _get_context_string(row)
    messages = build_inference_messages(row["question"], context, language)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with semaphore:
        result = await post_chat_completion(
            session,
            api_url=api_url,
            api_key=api_key,
            payload=payload,
            max_retries=3,
            timeout_seconds=180,
        )
        if result is not None:
            try:
                return result["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                pass
    return None


async def infer_single_openrag(
    session: aiohttp.ClientSession,
    row: dict,
    semaphore: asyncio.Semaphore,
    *,
    api_url: str,
    api_key: str,
    model: str,
    language: str,
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> str | None:
    messages = build_openrag_query_messages(row["question"], language)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with semaphore:
        result = await post_chat_completion(
            session,
            api_url=api_url,
            api_key=api_key,
            payload=payload,
            max_retries=3,
            timeout_seconds=180,
        )
        if result is not None:
            try:
                content = result["choices"][0]["message"]["content"]
                extra = result.get("extra")
                titles = extract_openrag_source_titles(extra)
                if titles:
                    content = append_title_citations(content, titles)
                return content
            except (KeyError, IndexError):
                pass
    return None


async def run_inference(
    rows: list[dict],
    *,
    api_url: str,
    api_key: str,
    model: str,
    language: str,
    concurrency: int = 10,
    cache_writer: "Callable[[str, str], None] | None" = None,
) -> list[str | None]:
    """Run inference on all rows with progress tracking (preserves order).

    If ``cache_writer`` is provided, it is called synchronously with
    ``(row_id, response)`` as soon as each inference succeeds, so a
    crash mid-run does not lose completed responses.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: list[str | None] = [None] * len(rows)

    async with aiohttp.ClientSession() as session:
        async def _task(idx: int, row: dict) -> None:
            response = await infer_single(
                session, row, semaphore,
                api_url=api_url, api_key=api_key, model=model,
                language=language,
            )
            results[idx] = response
            if response is not None and cache_writer is not None:
                cache_writer(str(row["id"]), response)

        coros = [_task(i, row) for i, row in enumerate(rows)]
        with tqdm(total=len(coros), desc="Inference") as pbar:
            for coro in asyncio.as_completed(coros):
                await coro
                pbar.update(1)
    return results


async def run_inference_openrag(
    rows: list[dict],
    *,
    api_url: str,
    api_key: str,
    model: str,
    language: str,
    concurrency: int = 10,
    cache_writer: "Callable[[str, str], None] | None" = None,
) -> list[str | None]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[str | None] = [None] * len(rows)

    async with aiohttp.ClientSession() as session:
        async def _task(idx: int, row: dict) -> None:
            response = await infer_single_openrag(
                session, row, semaphore,
                api_url=api_url, api_key=api_key, model=model,
                language=language,
            )
            results[idx] = response
            if response is not None and cache_writer is not None:
                cache_writer(str(row["id"]), response)

        coros = [_task(i, row) for i, row in enumerate(rows)]
        with tqdm(total=len(coros), desc="OpenRAG inference") as pbar:
            for coro in asyncio.as_completed(coros):
                await coro
                pbar.update(1)
    return results


def extract_openrag_source_titles(extra: object) -> list[str]:
    titles: list[str] = []
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except json.JSONDecodeError:
            return titles
    if not isinstance(extra, dict):
        return titles

    seen: set[str] = set()
    for source in extra.get("sources", []):
        if not isinstance(source, dict):
            continue
        title = (
            source.get("file_id")
            or source.get("title")
            or source.get("filename")
            or source.get("file_name")
            or source.get("name")
            or source.get("original_filename")
        )
        if not title:
            raw_source = source.get("source") or source.get("file_url") or ""
            if raw_source:
                title = Path(str(raw_source)).stem
        if not title:
            continue
        title = unquote(str(title)).strip()
        if "." in title:
            title = Path(title).stem
        if not title or title in seen:
            continue
        seen.add(title)
        titles.append(title)
    return titles


def append_title_citations(content: str, titles: list[str]) -> str:
    citation_block = " ".join(f"[{title}]" for title in titles)
    if not citation_block:
        return content
    if content.rstrip().endswith(citation_block):
        return content
    return f"{content.rstrip()}\n\n{citation_block}"


def collect_partition_documents(rows: list[dict]) -> dict[str, str]:
    documents: dict[str, str] = {}
    for row in rows:
        for chunk in row.get("chunks", []):
            title = str(chunk.get("id", "")).strip()
            text = str(chunk.get("text", "")).strip()
            if not title or not text or title in documents:
                continue
            documents[title] = f"# {title}\n\n{text}\n"
    return documents


async def index_openrag_partition(
    rows: list[dict],
    *,
    base_url: str,
    partition: str,
    api_key: str,
    concurrency: int = 60,
) -> None:
    documents = collect_partition_documents(rows)
    if not documents:
        print("No documents found to index")
        return

    async with aiohttp.ClientSession() as session:
        await ensure_openrag_partition(
            session, base_url=base_url, partition=partition, api_key=api_key,
        )

        # HTTP 409 from add_openrag_file is our dedup signal — already-indexed
        # files are reported as skipped rather than re-uploaded.
        to_index = list(documents.items())
        semaphore = asyncio.Semaphore(concurrency)
        counters = {"indexed": 0, "failed": 0, "skipped": 0}

        async def _index_one(title: str, content: str) -> None:
            async with semaphore:
                task_url = await add_openrag_file(
                    session,
                    base_url=base_url,
                    partition=partition,
                    api_key=api_key,
                    file_id=title,
                    filename=f"{title}.md",
                    content=content,
                )
                if task_url == "__SKIP__":
                    counters["skipped"] += 1
                    return
                if not task_url:
                    counters["failed"] += 1
                    return
                ok = await wait_for_openrag_task(
                    session,
                    task_url=task_url,
                    api_key=api_key,
                    poll_interval_seconds=0.2,
                    max_polls=1500,
                )
                if ok:
                    counters["indexed"] += 1
                else:
                    counters["failed"] += 1

        tasks = [_index_one(t, c) for t, c in to_index]
        with tqdm(total=len(tasks), desc=f"Indexing {partition}") as pbar:
            for coro in asyncio.as_completed(tasks):
                await coro
                pbar.update(1)

        print(
            f"OpenRAG indexing finished for '{partition}': "
            f"{counters['indexed']} indexed, "
            f"{counters['skipped']} skipped, "
            f"{counters['failed']} failed"
        )


# ── augmented-format adapter ───────────��────────────────────────────


def _parse_chunks_from_context_string(context: str, gold_titles: list[str]) -> list[dict]:
    """Reconstruct a ``chunks`` list from the formatted context string
    produced by ``format_chunk_base`` / ``format_chunk_openrag``.
    """
    titles = [m.strip() for m in _CHUNK_TITLE_RE.findall(context)]
    texts = _CHUNK_TEXT_RE.findall(context)
    gold_set = {t.strip().lower() for t in gold_titles}
    chunks: list[dict] = []
    for title, text in zip(titles, texts):
        chunks.append({
            "id": title,
            "text": text.strip(),
            "is_gold": title.strip().lower() in gold_set,
            "source": "unknown",
        })
    return chunks


_CONTEXT_MARKER = "Here are the retrieved documents : `"


def adapt_chat_row(row: dict, idx: int) -> dict:
    """Convert a chat-format SFT row to the benchmark schema.

    Expected input (one JSONL line):
      {"messages": [
         {"role": "system",    "content": "... Here are the retrieved documents : `<ctx>`"},
         {"role": "user",      "content": "<question>"},
         {"role": "assistant", "content": "<reasoning ... **Final Answer:** <ans>>"}],
       "supporting_facts_titles": ["<gold title>", ...],
       "chunks_total": <int>}

    Chunk IDs are scoped by row ID (``{row_id}__{title}``) so they stay
    globally unique when indexed into a shared OpenRAG partition. The
    original title is preserved as ``chunks[i]["title"]`` so that inline
    SFT citations (``##Cite "Table"##``) still match during evaluation.
    """
    messages = row["messages"]
    if len(messages) < 3:
        raise ValueError(f"row {idx}: expected >=3 messages, got {len(messages)}")
    system = messages[0]["content"]
    question = messages[1]["content"]
    reasoning = messages[2]["content"]

    start = system.find(_CONTEXT_MARKER)
    if start == -1:
        raise ValueError(f"row {idx}: context marker not found in system message")
    context = system[start + len(_CONTEXT_MARKER):].rstrip()
    if context.endswith("`"):
        context = context[:-1].rstrip()

    row_id = str(row.get("id", idx))
    gold_titles = row.get("supporting_facts_titles", [])
    raw_chunks = _parse_chunks_from_context_string(context, gold_titles)
    chunks = [
        {
            "id": f"{row_id}__{c['id']}",
            "title": c["id"],
            "text": c["text"],
            "is_gold": c["is_gold"],
            "source": c["source"],
        }
        for c in raw_chunks
    ]
    answer = extract_answer_from_reasoning(reasoning) or ""

    return {
        "id": row_id,
        "question": question,
        "answer": answer,
        "reasoning_trace": reasoning,
        "response": reasoning,
        "chunks": chunks,
        "context": context,
        "supporting_facts_titles": list(gold_titles),
    }


# ── citation parsing ────────────────────────────────────────────────


def parse_citations(response: str, chunks: list[dict]) -> list[str]:
    """Extract cited chunks from *response*, returning their canonical IDs.

    A citation may reference a chunk either by its canonical ``id``
    (e.g. ``row5__Table`` — what OpenRAG returns) or by its original
    ``title`` (e.g. ``Table`` — what the SFT assistant cites). Both
    aliases are mapped back to the same canonical chunk id.

    Supports all SFT citation formats:
    - ``[Title]``                 (plain bracket)
    - ``##Cite "Title"##``        (hash-cite)
    - ``[Source: "Title"]``       (bracket-source)
    - ``<<cite: "Title">>``       (angle-cite)
    - ``<cite>"Title"</cite>``    (xml-cite)

    Duplicates are removed (first occurrence wins).
    """
    id_by_alias: dict[str, str] = {}
    for c in chunks:
        cid = c["id"]
        id_by_alias[cid.strip().lower()] = cid
        title = c.get("title")
        if title:
            id_by_alias[title.strip().lower()] = cid

    seen: set[str] = set()
    cited: list[str] = []

    def _collect(raw: str) -> None:
        canonical = id_by_alias.get(raw.strip().lower())
        if canonical is not None and canonical not in seen:
            seen.add(canonical)
            cited.append(canonical)

    # plain bracket style [Title]
    for m in _CITE_BRACKET_RE.finditer(response):
        _collect(m.group(1))

    # alternative markers (always run so mixed formats are handled)
    for raw_title in extract_cited_titles(response):
        _collect(raw_title)

    return cited


def extract_answer(response: str, chunks: list[dict] | None = None) -> str:
    """Extract the answer from *response*.

    Tries structured patterns first (``**Answer:**``, etc.). Otherwise
    returns the response with citation markers removed. Only brackets
    that match a known chunk id or title (case-insensitive) are
    stripped — other bracketed spans such as ``[Figure 1]`` or
    ``[note]`` are preserved, so token-F1 / ROUGE-L are not degraded
    by incidental brackets.
    """
    answer = extract_answer_from_reasoning(response)
    if answer:
        return answer

    if chunks:
        aliases: set[str] = set()
        for c in chunks:
            aliases.add(c["id"].strip().lower())
            if c.get("title"):
                aliases.add(c["title"].strip().lower())

        def _strip_known(match: re.Match) -> str:
            return "" if match.group(1).strip().lower() in aliases else match.group(0)

        cleaned = _CITE_BRACKET_RE.sub(_strip_known, response)
    else:
        cleaned = _CITE_BRACKET_RE.sub("", response)

    # Also strip alternative citation markers
    cleaned = re.sub(r'##Cite\s*"[^"]+"\s*##', "", cleaned)
    cleaned = re.sub(r'##Cite\s*[«»][^«»]+?[«»]\s*##', "", cleaned)
    cleaned = _CITE_SOURCE_BRACKET_RE.sub("", cleaned)
    cleaned = _CITE_ANGLE_RE.sub("", cleaned)
    cleaned = _CITE_TAG_RE.sub("", cleaned)
    return cleaned.strip()


# ── per-row evaluation ──────────────────────────────────────────────


async def evaluate_row(
    session: aiohttp.ClientSession,
    bench: dict,
    response_text: str,
    semaphore: asyncio.Semaphore,
    *,
    use_llm_judge: bool = True,
    api_key: str = "",
    api_url: str = "",
    model: str = "",
) -> dict:
    chunks = bench["chunks"]
    gold_ids = [c["id"] for c in chunks if c["is_gold"]]
    distractor_ids = {c["id"] for c in chunks if not c["is_gold"]}
    bm25_ids = {c["id"] for c in chunks if c.get("source") == "bm25_distractor"}
    is_unanswerable = bench.get("_is_unanswerable", bench.get("is_unanswerable", False))

    # -- citations --
    cited = parse_citations(response_text, chunks)
    precision, recall = evaluate_chunk_citations(cited, gold_ids)
    citation_f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        citation_f1 = 2 * precision * recall / (precision + recall)

    cited_distractor = [c for c in cited if c in distractor_ids]
    cited_bm25 = [c for c in cited if c in bm25_ids]
    distractor_rate = len(cited_distractor) / len(cited) if cited else 0.0
    bm25_rate = len(cited_bm25) / len(cited) if cited else 0.0

    # -- answer --
    reference = bench["answer"]
    predicted = extract_answer(response_text, chunks)
    refused = detect_refusal(response_text)

    token_f1 = compute_token_f1(predicted, reference)
    rouge_l = compute_rouge_l(predicted, reference)

    judge_score: int | None = None
    judge_justification: str | None = None
    answer_correct: bool | None = None

    if is_unanswerable:
        answer_correct = refused
    elif use_llm_judge:
        # Build context string from chunks for the factual judge
        context_str = bench.get("context", "")
        if not context_str and chunks:
            context_str = "\n\n".join(
                f"[{c['id']}]\n{c.get('text', '')}" for c in chunks
            )
        judge_result = await call_judge(
            session,
            JUDGE_FACTUAL_SYSTEM_PROMPT,
            build_factual_judge_user_prompt(
                bench["question"],
                reference,
                gold_ids,
                context_str,
                response_text,
            ),
            0.1,
            semaphore,
            api_key=api_key,
            api_url=api_url,
            model=model,
        )
        if judge_result is not None:
            judge_score = judge_result["score"]
            judge_justification = judge_result.get("justification", "")
            answer_correct = judge_score >= 5

    # -- joint --
    joint_f1 = None
    if citation_f1 is not None:
        joint_f1 = token_f1 * citation_f1

    return {
        "id": bench["id"],
        "question": bench["question"],
        "response": response_text,
        "reference_answer": reference,
        "predicted_answer": predicted,
        "cited_chunks": cited,
        "gold_chunks": gold_ids,
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": citation_f1,
        "distractor_citation_rate": distractor_rate,
        "bm25_citation_rate": bm25_rate,
        "cited_distractor_count": len(cited_distractor),
        "cited_bm25_count": len(cited_bm25),
        "token_f1": token_f1,
        "rouge_l": rouge_l,
        "answer_correct": answer_correct,
        "factual_judge_score": judge_score,
        "factual_judge_justification": judge_justification,
        "refused": refused,
        "is_unanswerable": is_unanswerable,
        "joint_f1": joint_f1,
        "level": bench.get("level", ""),
        "type": bench.get("type", ""),
    }


# ── aggregate report ────────────────────────────────────────────────


def _safe_avg(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _group_report(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    answerable = [r for r in rows if not r.get("is_unanswerable")]
    unanswerable = [r for r in rows if r.get("is_unanswerable")]

    # citation (answerable only -- unanswerable have no gold)
    cit_rows = [r for r in answerable if r.get("citation_f1") is not None]
    perfect = sum(
        1 for r in cit_rows
        if r["citation_precision"] == 1.0 and r["citation_recall"] == 1.0
    )
    no_cite = sum(1 for r in answerable if not r.get("cited_chunks"))

    # answer correctness (answerable only — judged by LLM)
    ans_judged = [r for r in answerable if r.get("answer_correct") is not None]
    ans_correct = sum(1 for r in ans_judged if r["answer_correct"])

    # factual judge score distribution (answerable only)
    score_5 = sum(1 for r in answerable if r.get("factual_judge_score") == 5)
    score_gte4 = sum(1 for r in answerable if (r.get("factual_judge_score") or 0) >= 4)

    report: dict = {
        "n": n,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "citation": {
            "precision": _safe_avg(cit_rows, "citation_precision"),
            "recall": _safe_avg(cit_rows, "citation_recall"),
            "f1": _safe_avg(cit_rows, "citation_f1"),
            "perfect": perfect,
            "no_citation_count": no_cite,
            "distractor_citation_rate": _safe_avg(answerable, "distractor_citation_rate"),
            "bm25_citation_rate": _safe_avg(answerable, "bm25_citation_rate"),
        },
        "answer": {
            "accuracy": ans_correct / len(ans_judged) if ans_judged else None,
            "correct": ans_correct,
            "judged": len(ans_judged),
            "token_f1": _safe_avg(answerable, "token_f1"),
            "rouge_l": _safe_avg(answerable, "rouge_l"),
            "factual_judge_avg": _safe_avg(answerable, "factual_judge_score"),
            "factual_judge_score_5": score_5,
            "factual_judge_score_gte4": score_gte4,
        },
        "joint_f1": _safe_avg(answerable, "joint_f1"),
    }

    if unanswerable:
        refused_correct = sum(1 for r in unanswerable if r.get("refused"))
        refused_wrong = sum(1 for r in answerable if r.get("refused"))
        report["refusal"] = {
            "total_unanswerable": len(unanswerable),
            "correctly_refused": refused_correct,
            "refusal_recall": refused_correct / len(unanswerable),
            "false_refusals": refused_wrong,
            "refusal_precision": (
                refused_correct / (refused_correct + refused_wrong)
                if (refused_correct + refused_wrong) > 0 else None
            ),
        }

    return report


def compute_report(results: list[dict]) -> dict:
    report = {"overall": _group_report(results)}

    # breakdown by level
    levels: dict[str, list[dict]] = {}
    for r in results:
        lvl = r.get("level") or "unknown"
        levels.setdefault(lvl, []).append(r)
    if len(levels) > 1:
        report["by_level"] = {k: _group_report(v) for k, v in sorted(levels.items())}

    return report


def print_report(report: dict) -> None:
    for section_name, section in report.items():
        if section_name == "by_level":
            for lvl, sub in section.items():
                _print_section(f"level={lvl}", sub)
        else:
            _print_section(section_name, section)


def _print_section(name: str, data: dict) -> None:
    n = data.get("n", 0)
    n_ans = data.get("n_answerable", n)
    n_unans = data.get("n_unanswerable", 0)
    cit = data.get("citation", {})
    ans = data.get("answer", {})
    ref = data.get("refusal", {})

    print(f"\n{'─'*50}")
    print(f"  {name}  (n={n}, answerable={n_ans}, unanswerable={n_unans})")
    print(f"{'─'*50}")

    if cit:
        p = cit.get("precision")
        r = cit.get("recall")
        f = cit.get("f1")
        print(f"  Citation P / R / F1:   {_fmt(p)} / {_fmt(r)} / {_fmt(f)}")
        print(f"  Perfect citation:      {cit.get('perfect', 0)}/{n_ans}")
        print(f"  No citation at all:    {cit.get('no_citation_count', 0)}/{n_ans}")
        print(f"  Distractor cite rate:  {_fmt(cit.get('distractor_citation_rate'))}")
        print(f"  BM25 distractor rate:  {_fmt(cit.get('bm25_citation_rate'))}")

    if ans:
        print(f"  Answer accuracy (=5): {_fmt(ans.get('accuracy'))}  ({ans.get('correct', 0)}/{ans.get('judged', 0)})")
        print(f"  Factual judge avg:     {_fmt(ans.get('factual_judge_avg'))}")
        print(f"  Factual judge =5:      {ans.get('factual_judge_score_5', 0)}/{n_ans}")
        print(f"  Factual judge >=4:     {ans.get('factual_judge_score_gte4', 0)}/{n_ans}")
        print(f"  Token F1:              {_fmt(ans.get('token_f1'))}")
        print(f"  ROUGE-L:              {_fmt(ans.get('rouge_l'))}")

    jf = data.get("joint_f1")
    if jf is not None:
        print(f"  Joint F1:              {_fmt(jf)}")

    if ref:
        print(f"  Refusal recall:        {_fmt(ref.get('refusal_recall'))}  ({ref.get('correctly_refused', 0)}/{ref.get('total_unanswerable', 0)})")
        print(f"  Refusal precision:     {_fmt(ref.get('refusal_precision'))}  (false refusals: {ref.get('false_refusals', 0)})")


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "  n/a"


# ── main ────────────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate citation quality on a RAG citation benchmark (standalone)",
    )
    parser.add_argument("--benchmark", required=True, help="Benchmark JSONL (with chunks)")
    parser.add_argument(
        "--responses", default=None,
        help="Responses JSONL (id + response). If omitted, uses inline response/reasoning_trace.",
    )
    parser.add_argument("--output", default=None, help="Per-row results JSONL")
    parser.add_argument("--report", default=None, help="Aggregate report JSON")
    parser.add_argument("--generate", action="store_true",
                        help="Run inference to generate responses before evaluating")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Enable LLM factual judge")
    parser.add_argument("--openrag-index", action="store_true",
                        help="Index benchmark chunks into an OpenRAG partition before evaluation")
    parser.add_argument("--openrag-query", action="store_true",
                        help="Generate answers by querying OpenRAG without sending benchmark context")
    parser.add_argument("--partition", default="hotpotqa_bench",
                        help="OpenRAG partition name (default: hotpotqa_bench)")
    parser.add_argument("--openrag-url", default=None,
                        help="OpenRAG base URL (default: http://$APP_URL:$APP_PORT)")
    parser.add_argument("--openrag-token", default=None,
                        help="OpenRAG auth token (default: AUTH_TOKEN env)")

    gen_group = parser.add_argument_group("generation", "LLM settings for --generate")
    gen_group.add_argument("--api-url", default=None,
                           help=f"API URL (default: LLM_API_URL env or {LLM_API_URL})")
    gen_group.add_argument("--api-key", default=None,
                           help="API key (default: LLM_API_KEY env)")
    gen_group.add_argument("--model", default=None,
                           help=f"Model name (default: LLM_MODEL env or {LLM_MODEL})")
    gen_group.add_argument("--language", default="en", choices=["en", "fr"],
                           help="Language for system prompt (default: en)")

    judge_group = parser.add_argument_group("judge", "LLM settings for --llm-judge (falls back to generation settings)")
    judge_group.add_argument("--judge-api-url", default=None,
                             help="API URL for judge (default: same as --api-url)")
    judge_group.add_argument("--judge-api-key", default=None,
                             help="API key for judge (default: same as --api-key)")
    judge_group.add_argument("--judge-model", default=None,
                             help="Model name for judge (default: same as --model)")

    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT)
    args = parser.parse_args()

    # Generation LLM config
    gen_api_url = args.api_url or LLM_API_URL
    gen_api_key = args.api_key or LLM_API_KEY
    gen_model = args.model or LLM_MODEL
    openrag_base_url = build_openrag_base_url(args.openrag_url)
    openrag_token = args.openrag_token or OPENRAG_AUTH_TOKEN

    # Judge LLM config (falls back to generation config)
    judge_api_url = args.judge_api_url or gen_api_url
    judge_api_key = args.judge_api_key or gen_api_key
    judge_model = args.judge_model or gen_model

    if args.generate and not args.openrag_query and not gen_api_key:
        raise SystemExit("ERROR: --generate requires an API key. Set LLM_API_KEY in .env or pass --api-key.")
    if args.llm_judge and not judge_api_key:
        raise SystemExit("ERROR: --llm-judge requires an API key. Set LLM_API_KEY in .env or pass --judge-api-key.")
    if (args.openrag_index or args.openrag_query) and not openrag_token:
        raise SystemExit("ERROR: OpenRAG mode requires AUTH_TOKEN or --openrag-token.")

    # load benchmark (chat-format SFT JSONL)
    bench: dict[str, dict] = {}
    with open(args.benchmark, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            adapted = adapt_chat_row(row, idx)
            bench[adapted["id"]] = adapted
    print(f"Loaded {len(bench)} rows from {args.benchmark}")

    ordered_ids = sorted(bench.keys())
    ordered_rows = [bench[rid] for rid in ordered_ids]

    if args.openrag_index:
        print(f"Indexing OpenRAG partition '{args.partition}' via {openrag_base_url}")
        await index_openrag_partition(
            ordered_rows,
            base_url=openrag_base_url,
            partition=args.partition,
            api_key=openrag_token,
        )

    # load or generate responses
    responses: dict[str, str] = {}
    if args.responses:
        with open(args.responses, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                responses[row["id"]] = row.get("response", "")
        mode = "separate file"
    elif args.generate:
        if args.openrag_query:
            model_name = f"openrag-{args.partition}"
            chat_url = f"{openrag_base_url}/v1/chat/completions"
            print(f"Running OpenRAG inference: {model_name} @ {chat_url}")
            traces = await run_inference_openrag(
                ordered_rows,
                api_url=chat_url,
                api_key=openrag_token,
                model=model_name,
                language=args.language,
                concurrency=args.concurrency,
            )
        else:
            print(f"Running inference: {gen_model} @ {gen_api_url}")
            traces = await run_inference(
                ordered_rows,
                api_url=gen_api_url, api_key=gen_api_key, model=gen_model,
                language=args.language, concurrency=args.concurrency,
            )
        failed = 0
        for rid, trace in zip(ordered_ids, traces):
            if trace is not None:
                responses[rid] = trace
            else:
                failed += 1
        print(f"Inference done: {len(responses)} ok, {failed} failed")
        mode = "generated"
    else:
        for rid, row in bench.items():
            if "response" in row:
                responses[rid] = row["response"]
        mode = "inline"

    matched = sorted(set(bench) & set(responses))
    missing = set(bench) - set(responses)
    extra = set(responses) - set(bench)
    print(f"Benchmark rows:  {len(bench)}")
    print(f"Responses ({mode}): {len(responses)}")
    print(f"Matched:         {len(matched)}")
    if missing:
        print(f"Missing:         {len(missing)} benchmark rows without response")
    if extra:
        print(f"Extra:           {len(extra)} responses without benchmark row")

    if not matched:
        print("ERROR: no matching IDs")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            evaluate_row(session, bench[rid], responses[rid], semaphore,
                         use_llm_judge=args.llm_judge,
                         api_key=judge_api_key, api_url=judge_api_url,
                         model=judge_model)
            for rid in matched
        ]
        with tqdm(total=len(tasks), desc="Evaluating") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

    results.sort(key=lambda r: r["id"])

    # per-row output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Per-row results -> {args.output}")

    # report
    report = compute_report(results)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report -> {args.report}")

    print_report(report)


if __name__ == "__main__":
    asyncio.run(main())
