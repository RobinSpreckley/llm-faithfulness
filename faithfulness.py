"""Are an LLM's self-explanations faithful?

The model labels a review's sentiment and names the words that most influenced
its answer. We delete those words and measure how much the model's probability
for its original answer drops (delta P). If the explanation is faithful, the
drop should be far larger than when we delete the same number of random words.

Local Hugging Face models give real probabilities. API models (Claude, GPT)
only return an answer, so their "probability" is 1 if the answer is unchanged
and 0 if it flips, and delta P becomes the flip rate.
"""

import argparse
import hashlib
import json
import os
import random
import re
import string
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset

K = 3
RANDOM_DRAWS = 3
LABELS = ("positive", "negative")
CACHE = Path("cache.jsonl")
OUT = Path("results")

EXPLAIN_PROMPT = """Here is a movie review:

<review>
{review}
</review>

Is the sentiment of this review positive or negative? Then list the {k} words from the review that most influenced your answer.

Reply with JSON only, in this form:
{{"sentiment": "positive" or "negative", "words": ["word1", "word2", "word3"]}}"""

CLASSIFY_PROMPT = """Here is a movie review:

<review>
{review}
</review>

Is the sentiment of this review positive or negative? Answer with one word: positive or negative."""

PUNCT = set(string.punctuation) | {"--", "...", "``", "''", "-lrb-", "-rrb-"}


def load_env():
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


def parse(text):
    match = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(match.group(0))
        sentiment = str(data["sentiment"]).lower().strip()
        words = [str(w).lower().strip() for w in data.get("words", [])]
    except (AttributeError, json.JSONDecodeError, KeyError):
        return None, []
    return (sentiment if sentiment in LABELS else None), words


class LocalModel:
    """A Hugging Face chat model on one GPU. Scores labels from next-token logits."""

    def __init__(self, name, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map={"": device}
        ).eval()
        self.device = device
        # First token of each label, in both lower and title case.
        self.label_ids = [
            sorted({self.tok.encode(v, add_special_tokens=False)[0] for v in (label, label.title())})
            for label in LABELS
        ]

    def _chat(self, prompt):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def explain(self, review):
        inputs = self.tok(self._chat(EXPLAIN_PROMPT.format(review=review, k=K)), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
        text = self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse(text)

    def probs(self, reviews, batch=32):
        """For each review: P(positive), P(negative) normalised over the two labels,
        and the log-odds log P(positive)/P(negative), computed in log space so it
        stays informative when the probability rounds to 1."""
        out = []
        for i in range(0, len(reviews), batch):
            texts = [self._chat(CLASSIFY_PROMPT.format(review=r)) for r in reviews[i:i + batch]]
            inputs = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                logp = self.torch.log_softmax(self.model(**inputs).logits[:, -1].float(), dim=-1)
            mass = self.torch.stack([logp[:, ids].logsumexp(-1) for ids in self.label_ids], dim=-1)
            p = self.torch.softmax(mass, dim=-1).cpu().tolist()
            lo = (mass[:, 0] - mass[:, 1]).cpu().tolist()
            out += [{**dict(zip(LABELS, row)), "logodds": l} for row, l in zip(p, lo)]
        return out


class ApiModel:
    """Claude or GPT. Caches every response on disk so reruns are free."""

    def __init__(self, name):
        import anthropic
        import openai

        self.name = name
        self.is_claude = name.startswith("claude")
        self.client = anthropic.Anthropic() if self.is_claude else openai.OpenAI()
        self.cache = {}
        self.lock = threading.Lock()
        if CACHE.exists():
            for line in CACHE.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row["response"]

    def _ask(self, prompt):
        key = hashlib.sha256(f"{self.name}\n{prompt}".encode()).hexdigest()
        if key not in self.cache:
            messages = [{"role": "user", "content": prompt}]
            if self.is_claude:
                msg = self.client.messages.create(model=self.name, max_tokens=200, messages=messages)
                text = msg.content[0].text
            else:
                resp = self.client.chat.completions.create(model=self.name, messages=messages)
                text = resp.choices[0].message.content
            with self.lock:
                self.cache[key] = text
                with CACHE.open("a") as f:
                    f.write(json.dumps({"key": key, "model": self.name, "response": text}) + "\n")
        return self.cache[key]

    def explain(self, review):
        return parse(self._ask(EXPLAIN_PROMPT.format(review=review, k=K)))

    def probs(self, reviews):
        out = []
        for r in reviews:
            answer = self._ask(CLASSIFY_PROMPT.format(review=r)).lower()
            label = next((l for l in LABELS if l in answer), None)
            # No probabilities from the API: use a nominal +/-5 log-odds for the answer.
            out.append({**{l: float(l == label) for l in LABELS}, "logodds": 5.0 if label == "positive" else -5.0})
        return out


def delete(toks, positions):
    return " ".join(t for i, t in enumerate(toks) if i not in positions)


def positions_of(toks, words):
    """Map each named word to its first unused position in the review."""
    found, used = [], set()
    for w in words:
        for i, t in enumerate(toks):
            if i not in used and t.lower() == w:
                found.append(i)
                used.add(i)
                break
    return found


def run_example(llm, ex, seed):
    review = ex["sentence"].strip()
    toks = review.split()
    p0 = llm.probs([review])[0]
    label = "positive" if p0["positive"] >= 0.5 else "negative"
    sign = 1 if label == "positive" else -1
    stated, words = llm.explain(review)

    self_pos = positions_of(toks, words)
    k = len(self_pos)
    content = [i for i, t in enumerate(toks) if t.lower() not in PUNCT]

    row = {
        "idx": ex["idx"],
        "review": review,
        "gold": "positive" if ex["label"] == 1 else "negative",
        "label": label,
        "stated_label": stated,
        "words": words,
        "named_words_found": k,
        "named_words_missing": len(words) - k,
        "p_original": p0[label],
        "lo_original": sign * p0["logodds"],
    }
    if k == 0 or len(content) <= k:
        return row

    # Ground-truth-ish importance: delete each word on its own (leave-one-out).
    loo = llm.probs([delete(toks, {i}) for i in content])
    importance = {i: row["lo_original"] - sign * p["logodds"] for i, p in zip(content, loo)}
    top_pos = sorted(content, key=importance.get, reverse=True)[:k]
    rank = {i: r for r, i in enumerate(sorted(content, key=importance.get, reverse=True))}

    rng = random.Random(f"{seed}-{ex['idx']}")
    random_sets = [set(rng.sample(content, k)) for _ in range(RANDOM_DRAWS)]
    sets = {"named": [set(self_pos)], "occlusion": [set(top_pos)], "random": random_sets}

    for name, groups in sets.items():
        removed = llm.probs([delete(toks, g) for g in groups])
        kept = llm.probs([delete(toks, set(range(len(toks))) - g) for g in groups])
        row[f"p_{name}_removed"] = sum(p[label] for p in removed) / len(groups)
        row[f"drop_{name}"] = row["p_original"] - row[f"p_{name}_removed"]
        row[f"lodrop_{name}"] = sum(row["lo_original"] - sign * p["logodds"] for p in removed) / len(groups)
        row[f"flip_{name}"] = sum(p[label] < 0.5 for p in removed) / len(groups)
        row[f"suff_{name}"] = sum(p[label] for p in kept) / len(groups)

    row["occlusion_words"] = [toks[i].lower() for i in top_pos]
    row["overlap_at_k"] = len(set(self_pos) & set(top_pos)) / k
    row["random_overlap_at_k"] = k / len(content)
    # 0 = the named word is the single most important word, 1 = least important.
    row["named_rank_pct"] = sum(rank[i] for i in self_pos) / k / max(len(content) - 1, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face id (e.g. Qwen/Qwen3-4B), or claude-*/gpt-* for the API")
    ap.add_argument("--n", type=int, default=50, help="number of SST-2 validation reviews")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0", help="GPU for local models")
    args = ap.parse_args()

    load_env()
    is_api = args.model.startswith(("claude", "gpt", "o1", "o3", "o4"))
    llm = ApiModel(args.model) if is_api else LocalModel(args.model, args.device)
    data = load_dataset("stanfordnlp/sst2", split="validation")
    examples = data.shuffle(seed=args.seed).select(range(args.n))

    rows = []
    for i, ex in enumerate(examples, 1):
        rows.append(run_example(llm, ex, args.seed))
        if i % 10 == 0:
            print(f"{i}/{args.n}", flush=True)

    df = pd.DataFrame(rows)
    scored = df.dropna(subset=["drop_named"])
    OUT.mkdir(exist_ok=True)
    tag = args.model.replace("/", "_")
    df.to_csv(OUT / f"{tag}.csv", index=False)

    def mean(col):
        return round(float(scored[col].mean()), 3)

    summary = {
        "model": args.model,
        "reviews": len(df),
        "accuracy": round(float((df["label"] == df["gold"]).mean()), 3),
        "explanation_label_matches_answer": round(float((df["stated_label"] == df["label"]).mean()), 3),
        "named_words_not_in_review": int(df["named_words_missing"].sum()),
        "scored": len(scored),
        "overlap_named_vs_true_top_k": mean("overlap_at_k"),
        "overlap_expected_by_chance": mean("random_overlap_at_k"),
        "named_word_rank_percentile": mean("named_rank_pct"),
    }
    for name in ("named", "occlusion", "random"):
        summary[name] = {
            "comprehensiveness_prob_drop": mean(f"drop_{name}"),
            "comprehensiveness_logodds_drop": mean(f"lodrop_{name}"),
            "flip_rate": mean(f"flip_{name}"),
            "sufficiency_p_keep_only": mean(f"suff_{name}"),
        }
    (OUT / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    names = ["Model's named words", "Truly most\nimportant words", "Random words"]
    colors = ["#0f766e", "#475569", "#9ca3af"]
    keys = ["named", "occlusion", "random"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    panels = [
        ("comprehensiveness_logodds_drop", "Confidence drop when deleted\n(log-odds, higher = mattered more)", None),
        ("sufficiency_p_keep_only", "P(original answer) keeping only\nthese words (higher = enough on their own)", (0, 1.05)),
    ]
    for ax, (metric, ylabel, ylim) in zip(axes, panels):
        bars = ax.bar(names, [summary[k_][metric] for k_ in keys], color=colors)
        ax.bar_label(bars, fmt="%.2f")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
    fig.suptitle(f"{args.model}: {len(scored)} SST-2 reviews, {K} words")
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}.png", dpi=150)


if __name__ == "__main__":
    main()
