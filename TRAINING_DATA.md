# TRAINING_DATA.md — how every training question was produced

od5n trained on 22 slices of questions: `od3-r1` … `od3-r19` and `od3g-r20` … `od3g-r22`. None of these questions comes
from the challenge's test set. This file explains:

- where each question came from;
- how it was checked against the evaluation questions;
- how slices were drawn to look like the test;
- what is and is not shipped.

File references are to the code bundle (`thinkingcap/`). The rules that turn answers into pairs are in `RULES.md`.

## 1. Pools and their public sources

Builders are in `prepare_types.py`, pooling is in `build_pool.py`, and later top-ups are in `track.py:newtop`. The
"source" field of each row names the builder.

| source | Part | Public data and split | Filter / construction | How the answer or check is derived |
|---|---|---|---|---|
| `gsm8k_easy` | math | GSM8K, **train** split | 1–2 calculations | GSM8K's final number |
| `gsm8k_easy_test` | math | GSM8K, **test** split (see section 6) | 1–2 calculations | GSM8K's final number |
| `metamath_easy` | math | MetaMathQA, `GSM_Rephrased` rows | kept only if linked (word Jaccard ≥ 0.4) to an easy GSM8K question with the same answer; added 27 Oct | MetaMathQA's / GSM8K's answer |
| `gsm8k_matched` | math | the training part of a GSM8K corpus; plus a few SVAMP-style problems written by a reviewing model (origin `tgt-math-am1`) | matched to SVAMP's operation count × kind and length, profiled on the choosing-set SVAMP questions | the corpus answer; for written problems, the answer computed with the problem |
| `kodcode` | code | KodCode-Light-RL-10K; plus tasks written by a reviewing model (origins `tgt-code-*`) | one-function easy/medium tasks, task ≤ 45 tokens, solution ≤ 90; written tasks kept only if their reference solution passes every assert | asserts taken from the dataset's test functions |
| `ifeval_like` | instructions | argilla/ifeval-like-data; plus "built" / "built-long" rows | drawn to IFEval's instruction mix; built rows = a human-written request (no_robots, Dolly/OASST; WildChat possibly) + IFEval-style instruction sentences, coding requests removed | the IFEval checker on the row's instruction ids and kwargs |
| `toolace` | tools | ToolACE, single-turn | one-function / several-function calls; "no function fits" variants built from the nearest functions of *other* conversations, kept when a small model (Haiku-class) said twice that none fits (`near-verified`) | the gold call; no call for no-fit |
| `glaive_tools` | tools | glaive-function-calling-v2, cleaned (BFCL type names, every argument stated in the question) | hard no-fit variants by nearest-function rank (1–4, 4–15, 15–60), kept on the same two NO answers; plus questions written by a small model (`haiku`, `haiku-nofit`, `haiku-multiple`, `tgt-haiku-*`) and by a reviewing model (`tgt-argon`) | the gold call; no call for no-fit |
| `hermes_tools` | tools | Hermes function-calling | — | the gold call |

All math rows use the same wording as `svamp_row`: "The last line of your reply must start with 'Final answer:' followed
by the number alone."

## 2. Decontamination, as coded

`build_pool.py` (and `newtop`, which calls the same check):

- **Tokens:** lowercase, split into `[a-z0-9]+` tokens. Only non-assistant messages are used.
- **Rule:** a candidate is dropped if it shares **any** 8-token sequence with **any** evaluation question.
- **Exception:** 8-grams that appear in 3 or more evaluation questions are ignored, because they are boilerplate.

Slice selection (`track.py:od3prep`) then removes:

- the Dev-512 and Dev-SVAMP question ids;
- near-duplicates of Dev: a candidate with ≥ 30% of its 8-grams among Dev's rare 8-grams (those found in fewer than
  3 Dev questions);
- the `--r2-drop` default `source:metamath_sv+origin:xlam-irrelevance+id:nofit4-&source:glaive_tools&origin:`. This
  removes MetaMathQA's GSM_SV / FOBAR rows, xLAM no-fit rows, and one glaive no-fit batch;
- questions already used in the earlier r2 step-2/3 question files.

## 3. The original's answers, used for matching and for the brake

Every candidate was answered by the original before selection. The pool was answered ×10; the newtop sets were answered
10 times with seed 41. The exact count and seed per set are **not verified**. From these answers `od3prep` writes:

- `od3-base-stats.json`: per question, the part, the mean output tokens and the accuracy;
- `od3-base-think.json`: the median thinking length of the right answers;
- `od3-base-resp.json`: the median response length of the right answers (`--od3-resp`).

## 4. Drawing slices to look like the test

**od3prep** (`track.py:od3prep`) draws slices 1–19:

- Each question gets a bin from the original's solve rate on it: < 25%, 25–50%, 50–80%, 80–<100%, 100% (`R2_BINS`).
- Each part's draw follows the bin shares of the original's solve rate on the **full test**
  (`test_difficulty_data.py`).

  | Part | Test solve rate | <25% | 25–50% | 50–80% | 80–<100% | 100% |
  |---|---|---|---|---|---|---|
  | svamp | 0.887 | .047 | .023 | .112 | .117 | .701 |
  | mbpp | 0.524 | .357 | .074 | .159 | .119 | .291 |
  | ifeval | 0.587 | .266 | .115 | .163 | .116 | .340 |
  | simple | 0.822 | .105 | .050 | .087 | .092 | .665 |
  | multiple | 0.761 | .130 | .065 | .170 | .075 | .560 |
  | irrelevance | 0.834 | .117 | .025 | .079 | .054 | .725 |

- A bin that is short of candidates borrows from the nearest bins (`r2_select`). The seed is data_seed + k.
- **First build:** `od3prep --od-rounds 6 --data-seed 18`.
- **Later slices:** `--od3-more` calls with seeds 47–60, after newtop added more questions.
- A slice is written only if every part is full and its solve rate is within 3 points of the test's (5 points for
  irrelevance).
- Per-part sizes: slice 1 holds svamp 440, mbpp 280, ifeval 400, simple 160, multiple 120, irrelevance 160 (1,560). Every
  later slice holds svamp 220, mbpp 140, ifeval 200, simple 80, multiple 60, irrelevance 80 (780).

**od3branch** (`track.py:od3branch`) draws slices 20–22, with
`--ob-into od3g --ob-from 20 --ob-more 5 --od3-tops r2x-code-m1,r2x-math-m1 --ob-trim 0.05 --ob-trim-max 0.4 --data-seed 61`:

- od3 slices 2–19 are copied into od3g.
- New slices are drawn from candidates that are in neither slices 2–19 nor the r2 step files.
- A new slice must match on:
  - solve rate: within 3 points of the test;
  - against slices 2–19: mean tokens within 12%, mean words within 20%, median and p90 tokens within 15%.
- When a part's draw is too long or too short, up to 5% of its tail is trimmed per try, and at most 40% of its
  candidates can be dropped overall.
- Up to 150 tries per slice, with seed data_seed + 1000 × k + try.
- 5 slices were asked for and 3 matched: od3g-r20, r21 and r22.

## 5. Row format

Each line of `data/<set>-rK-questions.jsonl` is one question with these fields:

- `prompt`: the chat messages;
- `answer`: the gold answer, call, asserts or IFEval instruction ids/kwargs, depending on the task;
- `task`;
- `source`: the builder from section 1;
- `part`: one of the six parts;
- `pool_id`;
- `id`.

Some rows also carry `origin` (the sub-builder, as named in section 1).

The `id` prefix (for example `math2-svamp-…`, `mbpp-…`, `ifeval-…`, `bfcl_simple_python-…`) names the evaluation task a question was matched to, not where it came from; its origin is in `source`.

## 6. Counts per slice

Questions per source, over all 22 slices: 17,940 in total, of which 17,616 are distinct.

| Slice | gsm8k_matched | metamath_easy | gsm8k_easy | gsm8k_easy_test | kodcode | ifeval_like | toolace | glaive_tools | hermes_tools | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| od3-r1 | 235 | 157 | 43 | 5 | 280 | 400 | 193 | 247 | 0 | 1,560 |
| od3-r2 | 106 | 92 | 19 | 3 | 140 | 200 | 113 | 107 | 0 | 780 |
| od3-r3 | 110 | 86 | 23 | 1 | 140 | 200 | 105 | 115 | 0 | 780 |
| od3-r4 | 107 | 86 | 23 | 4 | 140 | 200 | 112 | 108 | 0 | 780 |
| od3-r5 | 113 | 91 | 14 | 2 | 140 | 200 | 101 | 118 | 1 | 780 |
| od3-r6 | 109 | 82 | 27 | 2 | 140 | 200 | 97 | 123 | 0 | 780 |
| od3-r7 | 109 | 84 | 25 | 2 | 140 | 200 | 92 | 128 | 0 | 780 |
| od3-r8 | 123 | 83 | 11 | 3 | 140 | 200 | 95 | 125 | 0 | 780 |
| od3-r9 | 108 | 79 | 30 | 3 | 140 | 200 | 88 | 131 | 1 | 780 |
| od3-r10 | 62 | 143 | 11 | 4 | 140 | 200 | 69 | 151 | 0 | 780 |
| od3-r11 | 69 | 143 | 6 | 2 | 140 | 200 | 62 | 158 | 0 | 780 |
| od3-r12 | 68 | 140 | 12 | 0 | 140 | 200 | 48 | 172 | 0 | 780 |
| od3-r13 | 81 | 121 | 16 | 2 | 140 | 200 | 35 | 185 | 0 | 780 |
| od3-r14 | 62 | 145 | 11 | 2 | 140 | 200 | 47 | 173 | 0 | 780 |
| od3-r15 | 76 | 129 | 13 | 2 | 140 | 200 | 42 | 178 | 0 | 780 |
| od3-r16 | 73 | 132 | 14 | 1 | 140 | 200 | 40 | 180 | 0 | 780 |
| od3-r17 | 21 | 190 | 8 | 1 | 140 | 200 | 36 | 184 | 0 | 780 |
| od3-r18 | 25 | 188 | 7 | 0 | 140 | 200 | 37 | 183 | 0 | 780 |
| od3-r19 | 30 | 186 | 4 | 0 | 140 | 200 | 34 | 186 | 0 | 780 |
| od3g-r20 | 107 | 106 | 6 | 1 | 140 | 200 | 56 | 164 | 0 | 780 |
| od3g-r21 | 92 | 118 | 9 | 1 | 140 | 200 | 53 | 167 | 0 | 780 |
| od3g-r22 | 87 | 125 | 8 | 0 | 140 | 200 | 48 | 172 | 0 | 780 |
| **All** | **1,973** | **2,706** | **340** | **41** | **3,220** | **4,600** | **1,603** | **3,455** | **2** | **17,940** |

Origins, over all slices:

- **ifeval_like:** built-long 4,159; built 255; plain argilla 186.
- **glaive_tools:** haiku-nofit 383; haiku-multiple 310; haiku 260; nearest-hard 276; near-verified-glaive 267;
  near-hard 232; tgt-haiku-b1 110; tgt-haiku 14; tgt-argon 17; haiku-multiple-removed 2.
- **toolace:** near-verified 579; far 40; near 9.
- **gsm8k_matched:** tgt-math-am1 164.
- **kodcode:** tgt-code-am1 102; ar2 64; ar1 29; ar0 7.

Things to know about these counts:

- **Repeated questions.** 324 questions in od3g-r20 / r21 / r22 (102 / 111 / 111) are also in slice 1. od3branch's
  "already used" set covers slices 2–19 and not slice 1. In round 1 the model did not answer slice 1; the original's
  stored answers were used instead.
- **GSM8K test-split questions.** 41 questions come from the **test split of GSM8K** (`gsm8k_easy_test`), 5 of them in
  slice 1. GSM8K is not one of the challenge's benchmarks, and the 8-gram check against the evaluation questions applied
  to them as to every row. They are flagged because they are a test split of another benchmark.
- The math mix changes over the slices: `metamath_easy` grows from slice 10 and dominates slices 17–19.
  `gsm8k_matched` returns in the od3g slices.

## 7. What is shipped, and what is not

- **Shipped** (`training_data/`, per `REPRODUCE.md` section 2): the slice question files and the original's
  statistics files (`od3-base-stats.json`, `od3-base-think.json`, `od3-base-resp.json`). The bundle holds the code that
  builds every pool and slice.
- **Not shipped:**
  - the per-round answers (8 per question) and the pair files of od5n's rounds;
  - the original's stored answers to slice 1 (`runs/bon/rp-od3-r1-stored/rollouts.jsonl`, 149 MB). Without them,
    round 1 must regenerate them from the original. Sampling makes that rebuild differ.

## 8. Not verified

- The exact number of the original's answers per question and their seeds, per set (section 3).
- Whether WildChat loaded when the "built" / "built-long" IFEval rows were made, and which human-request source each
  of those rows came from.
- Whether `od3-base-resp.json` covers the newtop questions.
