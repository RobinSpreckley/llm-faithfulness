"""Interactive playground for the deletion test.

Sentiment mode: type a review, see the model's answer and the words it says
mattered, then click words to delete them and watch the answer change. Every
word is also coloured by how much deleting it on its own moves the model.

Any-task mode: type any request, see the model's full reply, then delete or
change words in the request. Each word is coloured by how much deleting it makes the
original reply less likely (mean log-probability per reply token).

Run it from the xai account, then open http://localhost:8000 in a browser:
    uv run playground.py --model Qwen/Qwen3-4B --device cuda:0
"""

import argparse
import difflib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from faithfulness import CLASSIFY_PROMPT, EXPLAIN_PROMPT, K, PUNCT, LocalModel, delete, parse, positions_of

PAGE = Path(__file__).with_name("playground.html")
MAX_WORDS = 150


def tokenize(text):
    """Split punctuation off words, like SST-2, so named words match review tokens."""
    return re.findall(r"[\w'-]+|[^\w\s]+", text)[:MAX_WORDS]


def generate(llm, prompt, max_new_tokens):
    """The model's actual text reply, greedy, as in LocalModel.explain."""
    inputs = llm.tok(llm._chat(prompt), return_tensors="pt").to(llm.device)
    with llm.torch.no_grad():
        out = llm.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return llm.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def score(llm, review):
    """Label probabilities plus the model's written one-word answer."""
    prompt = CLASSIFY_PROMPT.format(review=review)
    return {**llm.probs([review])[0], "prompt": prompt, "reply": generate(llm, prompt, 10)}


def analyze(llm, text):
    toks = tokenize(text)
    review = " ".join(toks)
    base = score(llm, review)
    explain_prompt = EXPLAIN_PROMPT.format(review=review, k=K)
    explain_reply = generate(llm, explain_prompt, 80)
    stated, words = parse(explain_reply)
    named = positions_of(toks, words)
    found = {toks[i].lower() for i in named}
    # Positive push = the word moved the model towards "positive".
    loo = llm.probs([delete(toks, {i}) for i in range(len(toks))]) if len(toks) > 1 else []
    return {
        "tokens": toks,
        "base": base,
        "explain_prompt": explain_prompt,
        "explain_reply": explain_reply,
        "stated": stated,
        "words": words,
        "named": named,
        "missing": [w for w in words if w not in found],
        "push": [base["logodds"] - p["logodds"] for p in loo],
        "punct": [t.lower() in PUNCT for t in toks],
    }


# --- Any task: delete words from a request and see how the full reply changes ---

TASK_WORDS = 80
MAX_SAMPLES = 128
SAMPLE_CHUNK = 32
TASK_REPLY_TOKENS = 150
TASK_EXPLAIN = (
    "Which {k} words from my request most influenced your response? "
    'Reply with JSON only, in this form: {{"words": ["word1", "word2", "word3"]}}'
)


def spans(text):
    return [m.span() for m in re.finditer(r"[\w'-]+|[^\w\s]+", text)][:TASK_WORDS]


def edit(text, sp, edits):
    """The request with words swapped: edits maps position -> new word, "" deletes."""
    out, last = [], 0
    for i, (a, b) in enumerate(sp):
        if i in edits:
            out.append(text[last:a] + edits[i])
            last = b
    out.append(text[last:])
    tidy = re.sub(r"[ \t]{2,}", " ", "".join(out))
    return re.sub(r"[ \t]+([,.;:!?])", r"\1", tidy).strip()


def without(text, sp, positions):
    return edit(text, sp, {i: "" for i in positions})


def reply(llm, prompt):
    """Greedy reply: (text, token ids without special tokens)."""
    inputs = llm.tok(llm._chat(prompt), return_tensors="pt").to(llm.device)
    with llm.torch.no_grad():
        out = llm.model.generate(**inputs, max_new_tokens=TASK_REPLY_TOKENS, do_sample=False)
    ids = [i for i in out[0, inputs["input_ids"].shape[1]:].tolist() if i not in llm.tok.all_special_ids]
    return llm.tok.decode(ids).strip(), ids


def reply_logprob(llm, prompts, ids, batch=8):
    """Mean log-probability per token of the reply `ids` after each prompt.
    Higher = the model is more inclined to give exactly that reply."""
    torch = llm.torch
    pad = llm.tok.pad_token_id if llm.tok.pad_token_id is not None else 0
    out = []
    for i in range(0, len(prompts), batch):
        seqs = [llm.tok(llm._chat(p))["input_ids"] + ids for p in prompts[i:i + batch]]
        width = max(map(len, seqs))
        input_ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs], device=llm.device)
        mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs], device=llm.device)
        with torch.no_grad():
            logits = llm.model(input_ids=input_ids, attention_mask=mask).logits[:, -len(ids) - 1:-1].float()
        target = torch.tensor(ids, device=llm.device)
        lp = torch.log_softmax(logits, dim=-1).gather(-1, target.expand(len(seqs), -1).unsqueeze(-1)).squeeze(-1)
        out += lp.mean(-1).cpu().tolist()
    return out


def task_explain(llm, prompt, answer):
    chat = llm.tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer},
         {"role": "user", "content": TASK_EXPLAIN.format(k=K)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = llm.tok(chat, return_tensors="pt").to(llm.device)
    with llm.torch.no_grad():
        out = llm.model.generate(**inputs, max_new_tokens=60, do_sample=False)
    text = llm.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    match = re.search(r"\{.*\}", text, re.S)
    try:
        words = [str(w).lower().strip() for w in json.loads(match.group(0)).get("words", [])]
    except (AttributeError, ValueError):
        words = []
    return text, words


def importance(llm, text, ids):
    """Per word of the request: how much deleting it alone makes the reply `ids`
    less likely (drop in mean log-probability per reply token)."""
    sp = spans(text)
    if not ids:
        return [0.0] * len(sp)
    base, *loo = reply_logprob(llm, [text] + [without(text, sp, {i}) for i in range(len(sp))], ids)
    return [base - l for l in loo]


def task_analyze(llm, text):
    text = text.strip()
    sp = spans(text)
    toks = [text[a:b] for a, b in sp]
    answer, ids = reply(llm, text)
    explain_reply, words = task_explain(llm, text, answer)
    named = positions_of(toks, words)
    found = {toks[i].lower() for i in named}
    return {
        "prompt": text,
        "tokens": toks,
        "reply": answer,
        "tone": tone(llm, [answer])[0],
        "explain_prompt": TASK_EXPLAIN.format(k=K),
        "explain_reply": explain_reply,
        "words": words,
        "named": named,
        "missing": [w for w in words if w not in found],
        "importance": importance(llm, text, ids),
        "punct": [t.lower() in PUNCT for t in toks],
    }


def task_run(llm, text, edits, original_reply):
    """Rerun with words edited: the new reply, and how likely the original reply still is."""
    edited = edit(text, spans(text), edits)
    answer, new_ids = reply(llm, edited)
    ids = llm.tok(original_reply, add_special_tokens=False)["input_ids"]
    base, now = reply_logprob(llm, [text, edited], ids) if ids else (0.0, 0.0)
    return {"prompt": edited, "reply": answer, "lp_original": base, "lp_now": now, "tone": tone(llm, [answer])[0],
            # What drove the new reply: the same per-word test on the edited request.
            "tokens": [edited[a:b] for a, b in spans(edited)], "importance": importance(llm, edited, new_ids)}


def task_push(llm, text, edits, original_reply):
    """Where in the original reply the edit pushes the model, and towards what.
    Feeds the original reply after both requests and compares the model's
    next-word probabilities at every position."""
    torch = llm.torch
    sp = spans(text)
    edited = edit(text, sp, edits)
    ids = llm.tok(original_reply, add_special_tokens=False)["input_ids"][:TASK_REPLY_TOKENS]
    if not ids:
        return {"tokens": [], "per_edit": []}

    def dists(prompt):
        seq = torch.tensor([llm.tok(llm._chat(prompt))["input_ids"] + ids], device=llm.device)
        with torch.no_grad():
            return torch.log_softmax(llm.model(input_ids=seq).logits[0, -len(ids) - 1:-1].float(), dim=-1)

    la, lb = dists(text), dists(edited)
    pa, pb = la.exp(), lb.exp()
    m = (0.5 * (pa + pb)).clamp_min(1e-12).log()
    # Jensen-Shannon divergence in bits: 0 = same predictions, 1 = completely different.
    js = (0.5 * (pa * (la - m)).sum(-1) + 0.5 * (pb * (lb - m)).sum(-1)) / 0.6931
    target = torch.tensor(ids, device=llm.device)
    p_a = pa.gather(-1, target[:, None])[:, 0]
    p_b = pb.gather(-1, target[:, None])[:, 0]
    top_b = pb.topk(3, dim=-1)
    word = lambda i: llm.tok.decode([i])
    tokens = [
        {"t": word(t), "pa": a, "pb": b, "js": j,
         "alt": [[word(i), q] for i, q in zip(ti, tq)]}
        for t, a, b, j, ti, tq in zip(ids, p_a.tolist(), p_b.tolist(), js.clamp(0, 1).tolist(),
                                       top_b.indices.tolist(), top_b.values.tolist())
    ]

    # With several edits, how far each one alone pushes away from the original reply.
    per_edit = []
    if len(edits) > 1:
        singles = list(edits.items())
        lps = reply_logprob(llm, [text] + [edit(text, sp, {i: w}) for i, w in singles], ids)
        per_edit = [{"pos": i, "from": text[sp[i][0]:sp[i][1]], "to": w, "drop": lps[0] - lp}
                    for (i, w), lp in zip(singles, lps[1:])]
    return {"tokens": tokens, "per_edit": per_edit}


TONE_PROMPT = """Here is a piece of text:

<text>
{text}
</text>

Is the tone of this text positive or negative? Answer with one word: positive or negative."""


def tone(llm, texts, batch=8):
    """P(positive) and log-odds for the tone of each text, judged by the same
    model with next-token probabilities (as LocalModel.probs does for reviews)."""
    torch, out = llm.torch, []
    for i in range(0, len(texts), batch):
        chats = [llm._chat(TONE_PROMPT.format(text=t)) for t in texts[i:i + batch]]
        inputs = llm.tok(chats, return_tensors="pt", padding=True).to(llm.device)
        with torch.no_grad():
            # Only the last position's logits are needed; skipping the rest saves a lot of GPU memory.
            logp = torch.log_softmax(llm.model(**inputs, logits_to_keep=1).logits[:, -1].float(), dim=-1)
        mass = torch.stack([logp[:, ids].logsumexp(-1) for ids in llm.label_ids], dim=-1)
        p = torch.softmax(mass, dim=-1)[:, 0].cpu().tolist()
        lo = (mass[:, 0] - mass[:, 1]).cpu().tolist()
        out += [{"positive": a, "logodds": b} for a, b in zip(p, lo)]
    return out


def similarity(a, b):
    """Share of words two replies have in common, in order (0 to 1)."""
    return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


def task_sample(llm, text, edits, reference, temperature, n, seed):
    """Sample n replies at this temperature, to see how much the model varies
    on its own. Uses the model's default top-p/top-k; only temperature changes.
    The same seed gives the same random draws, so sample i of the original and
    of the edited request are a fair pair: any difference comes from the edit."""
    edited = edit(text, spans(text), edits)
    inputs = llm.tok(llm._chat(edited), return_tensors="pt").to(llm.device)
    start = inputs["input_ids"].shape[1]
    samples = []
    # In chunks to keep GPU memory bounded; chunk k always uses seed + k, so pairs still line up.
    for k, first in enumerate(range(0, n, SAMPLE_CHUNK)):
        llm.torch.manual_seed(seed + k)
        with llm.torch.no_grad():
            out = llm.model.generate(
                **inputs, max_new_tokens=TASK_REPLY_TOKENS, do_sample=True,
                temperature=temperature, num_return_sequences=min(SAMPLE_CHUNK, n - first),
            )
        samples += [llm.tok.decode(o[start:], skip_special_tokens=True).strip() for o in out]
    return {"prompt": edited, "samples": samples, "similarity": [similarity(reference, x) for x in samples],
            "tone": tone(llm, samples)}


def make_handler(llm, model_name):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, kind="application/json", status=200):
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self._send(PAGE.read_text().replace("{{MODEL}}", model_name), "text/html; charset=utf-8")
            else:
                self._send('{"error": "not found"}', status=404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length))
                text = str(body["text"])
                edits = {int(i): str(w) for i, w in body.get("edits", {}).items()}
            except (ValueError, KeyError, TypeError, AttributeError):
                return self._send('{"error": "bad request"}', status=400)
            with lock:
                if self.path == "/api/analyze":
                    result = analyze(llm, text)
                elif self.path == "/api/score":
                    result = score(llm, text)
                elif self.path == "/api/task/analyze":
                    result = task_analyze(llm, text)
                elif self.path == "/api/task/run":
                    result = task_run(llm, text, edits, str(body.get("reply", "")))
                elif self.path == "/api/task/push":
                    result = task_push(llm, text, edits, str(body.get("reply", "")))
                elif self.path == "/api/task/sample":
                    temperature = min(max(float(body.get("temperature", 0.7)), 0.05), 2.0)
                    n = min(max(int(body.get("n", 5)), 1), MAX_SAMPLES)
                    seed = int(body.get("seed", 0))
                    result = task_sample(llm, text, edits, str(body.get("reply", "")), temperature, n, seed)
                else:
                    return self._send('{"error": "not found"}', status=404)
            self._send(json.dumps(result))

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="Hugging Face model id")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print(f"Loading {args.model} on {args.device}...", flush=True)
    llm = LocalModel(args.model, args.device)
    # Localhost only: the page is not reachable from other machines.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(llm, args.model))
    print(f"Open http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
