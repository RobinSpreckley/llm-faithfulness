# Are an LLM's explanations faithful?

When you ask an LLM *why* it gave an answer, does its explanation describe what actually drove the answer, or does it just sound plausible?

This project tests that with a **deletion test**:

1. The model labels a movie review as positive or negative, and names the 3 words that most influenced its answer.
2. Those words are deleted from the review, and the model is asked again.
3. The drop in the model's probability for its original answer is compared with deleting 3 **random** words from the same review.

If the explanation is faithful, deleting the named words should hurt the answer much more than deleting random ones.

## First result: Qwen3-4B on SST-2

All 872 reviews in the SST-2 validation set, with Qwen/Qwen3-4B run locally on one GPU.

| | Named words deleted | Random words deleted |
|---|---|---|
| Mean drop in probability of the original answer | **0.270** | 0.114 |
| Answer flips | **27.5%** | 11.9% |

<p align="center"><img src="https://raw.githubusercontent.com/RobinSpreckley/llm-faithfulness/main/results/Qwen_Qwen3-4B.png" alt="Probability drop: named versus random words" width="560"/></p>

- **On average, the explanations point at words that matter.** Deleting them hurts about 2.4× more than deleting random words.
- **But per review they are weak.** In 71% of reviews the drop was under 0.05, and in 60% the named words did no better than random ones.
- **Some explanations are made up.** The model named 69 words that are not in the review at all.
- Accuracy was 93.5%, and the label in the explanation matched the model's direct answer 98.4% of the time.

**Caveat:** the model is very confident (mean probability 0.996), so probabilities saturate near 1 and small effects are hidden. The next version fixes this by measuring log-odds.

## What's next

The current `faithfulness.py` adds the following. It is written but **not yet run**, so the results above come from the earlier version (commit `4e672ec`).

- **Log-odds** instead of probability, so saturation doesn't hide effects.
- **Occlusion baseline:** delete each word on its own to find which words *actually* matter most, then check how many of the model's 3 named words are in that top 3 (overlap@3) and where they rank.
- **Sufficiency:** keep *only* the named words and see whether the model still gives the same answer.

After that:

1. Response quality: compare the model's scores on 200 [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) responses with human ratings, then test whether its stated reasons survive perturbing the evidence.
2. Agent decisions: the same test applied to why an agent chose a tool or action.

## Run it

```bash
uv sync
uv run faithfulness.py --model Qwen/Qwen3-4B --n 50 --device cuda:0
```

- Any Hugging Face causal LM works for `--model`. Weights are loaded with `trust_remote_code=False`.
- `claude-*` and `gpt-*` models are supported through their APIs (keys in a `.env` file). APIs return only the answer, not probabilities, so for them the test measures flip rate only.
- Outputs go to `results/`: a CSV with every review, a summary JSON and a chart.

## Method notes

- The probability of each label is the model's next-token probability for "positive"/"Positive" versus "negative"/"Negative", after a one-word classification prompt. Qwen3's thinking mode is turned off.
- Explanations are generated greedily. Named words are matched to the review's tokens. Words not in the review are counted and skipped.
- Random baselines are averaged over 3 draws of the same number of words, excluding punctuation.
- Based on the comprehensiveness and sufficiency metrics from [ERASER (DeYoung et al., 2020)](https://arxiv.org/abs/1911.03429).

## Tools

Python · PyTorch · Hugging Face Transformers · Datasets · Pandas · Matplotlib · uv
