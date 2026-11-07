"""
Swappable LLM backend.

Every agent talks to the model through ONE interface: `backend.generate(...)`.
Switch between Mock / Ollama (local) / Groq / Gemini by changing config.BACKEND
(or the LLM_BACKEND env var). Nothing else in the codebase needs to change.

Tiers:
  - "mock"   : no network, no deps. For smoke-testing the pipeline plumbing.
  - "ollama" : local CPU model (e.g. qwen2.5:1.5b). Slow but free & offline.
  - "groq"   : free hosted API. Set GROQ_MODEL to a NON-reasoning chat model.
  - "gemini" : Google free tier. Set GEMINI_MODEL.
Real numbers for the project come from the Kaggle notebook (kaggle/step1_kaggle.py,
Qwen2.5-7B-Instruct on a T4), not from these hosted backends.

Confidence: generate_with_confidence() returns (text, confidence in [0,1]).
For Week 1 the mock/ollama backends use a verbalized-confidence fallback;
Week 2 swaps in token-logprob confidence where the API supports it.
"""

from __future__ import annotations
import os
import re
import json
import collections
from typing import Optional, Tuple

try:
    import config
    _DEFAULT_BACKEND = config.BACKEND
except Exception:  # config import is optional
    _DEFAULT_BACKEND = os.environ.get("LLM_BACKEND", "mock")


# --------------------------------------------------------------------------- #
# Base interface
# --------------------------------------------------------------------------- #
class LLMBackend:
    name = "base"

    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 256, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def generate_with_confidence(self, prompt: str, system: Optional[str] = None,
                                 max_tokens: int = 256,
                                 temperature: float = 0.0) -> Tuple[str, float]:
        """Default: verbalized-confidence fallback. Append a confidence request,
        parse it back out. Backends that support logprobs override this."""
        sys2 = (system or "") + (
            "\nAfter your answer, on a new line output exactly: "
            "CONFIDENCE=<a number between 0 and 1>"
        )
        raw = self.generate(prompt, system=sys2, max_tokens=max_tokens,
                            temperature=temperature)
        conf = 0.5
        m = re.search(r"CONFIDENCE\s*=\s*([01]?\.?\d+)", raw)
        if m:
            try:
                conf = max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                pass
        text = re.sub(r"\n?CONFIDENCE\s*=.*$", "", raw).strip()
        return text, conf


# --------------------------------------------------------------------------- #
# Mock backend  (no network, no dependencies)  -- the smoke-test engine
# --------------------------------------------------------------------------- #
_STOP = set("""a an the of to in on at for and or but is are was were be been being
this that these those who what when where why how which whom with without from into
do does did done as by it its their his her them they we you i he she him me my our
about over under than then so such not no yes can could should would may might will
title content query doc id retrieved chunks question pool existing documents
play plays played game games team teams league character series include includes named""".split())


class MockBackend(LLMBackend):
    """Deterministic stand-in that *attempts* each task from the prompt text only.

    It does not cheat with a side channel — it parses the prompt the way a real
    model would see it. Accuracy is mediocre by design; the point is to prove the
    pipeline runs end-to-end and that EM/F1 compute correctly. Replace with Groq
    for real numbers.
    """
    name = "mock"

    @staticmethod
    def _content_words(text: str):
        toks = re.findall(r"[A-Za-z0-9']+", text.lower())
        return [t for t in toks if t not in _STOP and len(t) > 2]

    def _role(self, system: str) -> str:
        s = (system or "").lower()
        if "sufficient information" in s and "insufficient" in s:
            return "judge"
        if "retrieval query generation agent" in s:
            return "query"
        return "answer"

    @staticmethod
    def _content_spans(context: str):
        """Pull just the retrieved chunk texts (after 'Content:'), ignoring all
        prompt scaffolding so the heuristic reasons over evidence, not boilerplate."""
        spans = re.findall(r"Content:\s*(.+)", context)
        return spans if spans else [context]

    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 256, temperature: float = 0.0) -> str:
        role = self._role(system or "")
        question = ""
        qm = re.search(r"Question:\s*(.+)", prompt)
        if qm:
            question = qm.group(1).splitlines()[0]

        if role == "query":
            # Emit up to 8 salient keywords from the question (mimics gap-filling).
            kws = []
            for w in self._content_words(question):
                if w not in kws:
                    kws.append(w)
            return " ".join(kws[:8]) or "context"

        # Retrieved chunks may live in the system prompt (answer agent) or the
        # user prompt (judge/query agents); search both, content only.
        context = (system or "") + "\n" + prompt
        spans = self._content_spans(context)
        q_words = set(self._content_words(question))

        if role == "judge":
            # "Sufficient" if some retrieved chunk overlaps the question well.
            best = max((len(q_words & set(self._content_words(s))) for s in spans),
                       default=0)
            return "Sufficient Information" if best >= 2 else "Insufficient Information"

        # role == "answer": focus on the most question-relevant chunk, then
        # extract a salient entity/number from that chunk only.
        q_lower = question.lower()
        best_span, best_ov = spans[0], -1
        for s in spans:
            ov = len(q_words & set(self._content_words(s)))
            if ov > best_ov:
                best_ov, best_span = ov, s
        if "how many" in q_lower:
            nums = re.findall(r"\b\d{1,4}\b", best_span)
            if nums:
                return nums[0]
        phrases = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z.]+){0,2})\b",
                            best_span)
        counts = collections.Counter()
        for p in phrases:
            if p.lower() in q_lower or all(w.lower() in _STOP for w in p.split()):
                continue
            counts[p] += 1
        if counts:
            return counts.most_common(1)[0][0]
        nums = re.findall(r"\b\d{1,4}\b", best_span)
        return nums[0] if nums else "unknown"


# --------------------------------------------------------------------------- #
# Ollama backend (local, CPU)  -- offline fallback
# --------------------------------------------------------------------------- #
class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 256, temperature: float = 0.0) -> str:
        import requests
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        r = requests.post(f"{self.host}/api/generate", json=payload, timeout=600)
        r.raise_for_status()
        return r.json().get("response", "").strip()


# --------------------------------------------------------------------------- #
# Groq backend (free hosted, OpenAI-compatible)
# --------------------------------------------------------------------------- #
class GroqBackend(LLMBackend):
    name = "groq"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        from openai import OpenAI  # pip install openai
        # No default on purpose: Groq retires model ids (llama-3.1-8b-instant was
        # retired June 2026), and reasoning models (e.g. gpt-oss) spend small
        # max_tokens budgets on hidden reasoning and return EMPTY text.
        self.model = model or os.environ.get("GROQ_MODEL")
        if not self.model:
            raise RuntimeError("Set GROQ_MODEL to a current non-reasoning chat model "
                               "listed at console.groq.com/docs/models.")
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Set GROQ_API_KEY (get a free key at console.groq.com).")
        self.client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 256, temperature: float = 0.0) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model, messages=msgs,
            max_tokens=max_tokens, temperature=temperature)
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError(f"{self.model} returned empty text; pick a non-reasoning model.")
        return text

    def generate_with_confidence(self, prompt, system=None, max_tokens=256,
                                 temperature=0.0):
        """Token-logprob confidence: mean exp(logprob) of generated tokens.
        Groq exposes OpenAI-style logprobs. Falls back to verbalized on error."""
        try:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, max_tokens=max_tokens,
                temperature=temperature, logprobs=True)
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            import math
            lps = [t.logprob for t in choice.logprobs.content] if choice.logprobs else []
            conf = sum(math.exp(lp) for lp in lps) / len(lps) if lps else 0.5
            return text, float(max(0.0, min(1.0, conf)))
        except Exception:
            return super().generate_with_confidence(prompt, system, max_tokens, temperature)


# --------------------------------------------------------------------------- #
# Gemini backend (Google free tier)
# --------------------------------------------------------------------------- #
class GeminiBackend(LLMBackend):
    name = "gemini"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        import google.generativeai as genai  # pip install google-generativeai
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (aistudio.google.com/apikey).")
        genai.configure(api_key=key)
        self._genai = genai
        self.model = model or os.environ.get("GEMINI_MODEL")   # ids change; no stale default
        if not self.model:
            raise RuntimeError("Set GEMINI_MODEL to a current model id from aistudio.google.com.")

    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 256, temperature: float = 0.0) -> str:
        model = self._genai.GenerativeModel(
            self.model, system_instruction=system or None)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": temperature,
                               "max_output_tokens": max_tokens})
        return (resp.text or "").strip()


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_REGISTRY = {
    "mock": MockBackend,
    "ollama": OllamaBackend,
    "groq": GroqBackend,
    "gemini": GeminiBackend,
}


def get_backend(name: Optional[str] = None, **kwargs) -> LLMBackend:
    name = (name or _DEFAULT_BACKEND or "mock").lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backend '{name}'. Options: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


if __name__ == "__main__":
    b = get_backend("mock")
    print("backend:", b.name)
    demo_prompt = ("Question: Who directed Inception?\n"
                   "Query Pool:\n"
                   "Query 0: Who directed Inception?\n"
                   "Retrieved Chunks for Query 0:\n"
                   "(1) Title: Inception; Content: Inception is a 2010 film "
                   "directed by Christopher Nolan.")
    print(b.generate(demo_prompt,
                     system="Answer the question based on the given document."))
