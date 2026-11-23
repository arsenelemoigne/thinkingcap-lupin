# RULES.md — every rule the final run applied, as coded

This file lists the rules of the submitted run, od5n, as the code ran them. It covers each flag of the two od5n command
lines in `REPRODUCE.md` section 3, and each default those lines rely on. File references are to the code bundle
(`thinkingcap/`). If a rule is not pinned down by the code or the data, it says **not verified**.

Words used throughout:

- **the original**: Qwen3-0.6B with no adapter.
- **part**: one of the six task types: `svamp` (math), `mbpp` (code), `ifeval` (instructions), `bfcl_simple_python` (one
  function), `bfcl_multiple` (several functions), `bfcl_irrelevance` (no function fits).
- **slice**: the fixed set of training questions used in one round.
- **thinking length**: the number of tokens between the start of the answer and `</think>`. **Response length** is the
  output tokens minus the thinking length.

## 0. The two command lines

The first line ran rounds 1–18 (`--od-rounds 18 --od-min-rounds 18`). The second line is the same, except for
`--od-slices od3g --od-rounds 22 --od-min-rounds 22`. Rerunning it skips the finished rounds 1–18 and continues from
round 18's adapter for rounds 19–22 (`track.py:odpo_fixed`: a round already in the run's history whose adapter exists is
skipped).

- `--od-fixed-ref` routes `odpo` to `odpo_fixed` (`track.py:odpo`).
- `--rp-base Qwen/Qwen3-0.6B` sets the base model.
- One LoRA adapter is continued over all rounds. The DPO reference is always the original: the base with the adapter
  turned off (`a.rp_ref_base = True`).
- `--od-slices od3g` (line 2 only) reads slices from `data/od3g-rK-questions.jsonl`. Without it the default is `od3`.
  od3g slices 2–19 are copies of od3's. Slices 20–22 are new (see `TRAINING_DATA.md` section 4).
- `--od-min-rounds 18` / `22` aborts the run if fewer slices exist.
- `--note consult37-B` is a label only.

## 1. Answer

Each round, the current model answers its slice (`track.py:ptrepair`, `rp_own_data` path).

- **Who answers.** In round 1 nobody answers. The run reuses the original's stored answers to slice 1
  (`rp_rollouts_from = "od3-r1-stored"`), so slice 1 has 1,560 questions. In round k ≥ 2, the model merged at the end
  of round k−1 answers (`models/od5n-r{k-1}`).
- **Questions.** In round k ≥ 2 the question file is `data/{od3|od3g}-r{k}-questions.jsonl`: 780 questions, matching
  `--rp-questions` (svamp 220, mbpp 140, ifeval 200, simple 80, multiple 60, irrelevance 80). Round 1's slice holds
  svamp 440, mbpp 280, ifeval 400, simple 160, multiple 120, irrelevance 160.
- **How many answers.** `--rp-samples 8`: 8 answers per question. The call is
  `probe_curve.py --samples 8 --seed 53 --chunk 300 --generate-only --dtype bfloat16`. Seed 53 is the default
  `rp_gen_seed`. `--rp-seed 18` seeds the question draw only when no question file is given, so it has no effect for
  rounds ≥ 2. Its effect on round 1's stored answers is **not verified**.
- **Sampling** (`probe_curve.py`, `SAMPLING` from the challenge harness `utils/eval/recipe.py`):
  - temperature 0.6, top_p 0.95, top_k 20;
  - thinking on (`enable_thinking=True`);
  - at most 4,096 new tokens, or 8,192 for code.
- **Grading.**
  - `truncated` means the last token is not EOS. A truncated answer is wrong.
  - Math and multiple choice use the harness's `is_task_correct`.
  - Code runs the sanitized solution plus the asserts in a subprocess, with a 10 s timeout.
  - Tool calls use the harness's `grade_call` (the BFCL checker).
  - Instructions use the IFEval checker.
- **Refill.** Normally ptrepair answers more questions when a part is short of pairs. That step is skipped when a
  question file is set, so a short part simply gets fewer pairs.

## 2. Measure, then brake

These rules run in round k ≥ 2 only (`--od-caps` is set and k > 0, `track.py:odpo_fixed`).

**Step 1: answer and measure.** A first `ptrepair` pass (`rp_pairs_only`) produces the answers. The second pass reuses
them. `od3_monitor` (`track.py:od3_monitor`) then compares them with the original's statistics in
`data/od3-base-stats.json` (per question: part, mean output tokens, accuracy):

- **saved** for a part = 100 × (1 − Σ mean output tokens of this round's answers ÷ Σ the original's mean tokens), over
  the part's questions. Tokens are **total output tokens**, thinking plus response.
- **accuracy change** for a part = 100 × the mean, over questions, of (this round's share right − the original's share
  right). `correct` alone counts here; this measure does not also require a finished answer.
- The overall figures (a macro mean over math, code, instructions and tools, with tools weighted 400 / 200 / 240) are
  logged but not used by the brake.

**Step 2: brake.** A part is **held** this round if either condition is true:

- its saved % ≥ its cap. `--od-caps`: svamp 61, mbpp 60, ifeval 60, simple 45, multiple 53, irrelevance 72;
- its accuracy change ≤ `--od-acc-hold` −2.0 points.

`--od-unlatch` re-decides the hold every round from that round's numbers. Without it, a part held once would stay held.
`od_smooth` keeps its default of 1, so the brake reads one slice, not an average.

A held part gets `--no-shorten-parts <parts>` added to the pair-builder arguments. It also gets `--right-hold <parts>`,
except `bfcl_irrelevance`. On the od5 path `--right-hold` has **no effect**: it is read only in the non-od5 branch of
`repair_pairs.py`. Pairs are then rebuilt from the same answers (`rp_rebuild_pairs`).

**Worked example.** In round k, the incoming model's IFEval answers save 61% of the original's tokens at −0.5 points.
61 ≥ 60, so IFEval is held: it gets no shorten pairs this round. Next round it is measured again. If it saves 58%, it is
no longer held.

**Stop rules, all disabled:**

- `--od-stop-save 999`: the "target reached" stop needs saved ≥ 999%.
- `--od-conv-gain -999`: the "saving stopped growing" stop needs (saved − last round's saved) < −999.
- `--od-dev-rounds 99`: no Dev scoring happens inside the run. The Dev scores in `REPRODUCE.md` section 5 were run
  separately.

## 3. Build pairs

`repair_pairs.py` runs with `--kinds mix --mix-preset od5 --od5-v2 --seed 0` and the `--rp-pair-args` of the line
(lines 340–497 build candidates, lines 700–754 apply the quota). Every rule below is per question, against the
original's own record on that question.

### 3.1 Definitions

- **finished**: thinking closed and not truncated. **right**: correct and finished. **wrong**: finished and not correct.
  Unfinished answers are never in the wrong pool, but they can be "looping" (3.3).
- **size**: thinking length for parts in `--think-only` (all six parts are listed), so size is always thinking length.
- **k1**: the number of right answers before the floor. **n**: the number of answers (8). **p1** = k1 / n.
  **p0**: the original's accuracy on this question (`--base-stats data/od3-base-stats.json`).
- **bt**: the original's median thinking length over its right answers on this question
  (`--base-think data/od3-base-think.json`). If the question is missing there, bt is the median thinking of this round's
  right answers.
- **b_resp**: the original's median response length over its right answers (`--base-resp data/od3-base-resp.json`).
  Whether this file covers every slice question is **not verified**; a missing question gives 0.

### 3.2 Per-part floor on chosen answers

`--floor-by-part` takes ABS:RATIO pairs: svamp 30:0.45, mbpp 36:0.38, ifeval 14:0.25, simple 25:0.44, multiple 22:0.38,
irrelevance 10:0.20.

A right answer whose thinking is below max(ABS, RATIO × bt) is removed from the right answers and can never be chosen.

**Example.** On a math question where bt = 300 tokens, the floor is max(30, 0.45 × 300) = 135. A right answer that
thinks 120 tokens is dropped; one that thinks 140 stays.

### 3.3 Candidates by kind

Two more definitions:

- **looping**: not finished, or (with `--prefer-reread`, which loads the Qwen3-0.6B tokenizer to decode) any run of 8
  words that appears 4 or more times.
- **clean**: the right answers (after the floor) that are not looping. Clean answers are sorted by thinking length.

The kinds:

- **loop**. Applies when the question has a clean answer and a looping non-right answer. Chosen = the shortest clean
  answer. Rejected = the looping non-right answer with the most output tokens.
- **format**. The rejected answer is the shortest wrong answer with a bad format:
  - the response is empty;
  - or, for math, the answer has no final number (the `_nofinal` flag on the answer record; where it is set is **not
    verified**);
  - or, for one-function and several-function tool calls, the text after `</think>` has no "(".

  Chosen = the clean answer whose thinking length is closest to the rejected one's. The pair is kept only if that ratio
  is between 0.85 and 1.25.
- **count** (`right_count`). Applies to IFEval questions with an "at least" rule (`--count-mode atleast`,
  `repair_pairs.py:at_least`):
  - number_words, number_sentences or keywords:frequency with relation "at least";
  - letter_frequency with let_relation "at least";
  - any number_placeholders or number_highlighted_sections rule.

  Rejected = the shortest wrong answer. Chosen = among clean answers with:
  - response ≥ `--count-resp-min` 1.10 × the rejected response + 15;
  - thinking ≤ 1.2 × the rejected thinking;
  - response ≤ `--count-resp-cap` 1.75 × the rejected response + `--count-resp-pad` 60,

  the one with the least thinking. These questions never get a shorten pair.
- **shorten** ("think less, stay right"). A question can be shortened (`--od5-v2`) when:
  - if p0 = 1: p1 ≥ 0.875 (at most one wrong in 8);
  - if 0.5 ≤ p0 < 1: p1 ≥ max(0.5, p0 − 0.125);
  - if p0 < 0.5: p1 ≥ max(0.25, p0).

  It also needs at least 2 clean answers and must not be an "at least" IFEval question.

  - long = the clean answer with the longest thinking; short = the shortest of the others.
  - IFEval only (`--od5-v2` restricts the response rules to IFEval): the short answer's response must be
    ≥ max(`--resp-median` 0.95 × the median clean response, 0.92 × b_resp) and ≤ `--resp-max` 1.05 × long's response.
  - The pair is kept if long's thinking ÷ short's thinking ≥ `--shorten-ratio` 1.30. For ifeval and irrelevance the
    ratio is 1.25 (`--shorten-ratio-by-part`).
  - `--shorten-second ifeval+bfcl_irrelevance`: with 3 or more clean answers, a second pair (short over the longest
    remaining answer) is added if it also meets the ratio.
  - `--shorten-min-right 0.85` applies only to a question with no record of the original (p0 missing). Every slice
    question has one (slices are drawn from questions with stored answers of the original), so in practice this flag is
    not used. `--count-rules` is read only outside the od5 branch, so it is **inert here**.

  **Example.** On a math question with p0 = 1.0, 8 answers have 7 right and clean, thinking 190, 210, …, 420 tokens.
  p1 = 0.875 ≥ 0.875, so the question can be shortened. long = 420, short = 190, and 420 / 190 = 2.2 ≥ 1.30, so the pair
  (190 over 420) is a candidate.
- **recovery** (the question was damaged). A question counts as damaged when 1 ≤ k1 < n and either:
  - p0 ≥ 0.875 and p1 ≤ p0 − 0.25;
  - or p0 < 0.875 and p1 ≤ p0 − 0.125.

  Wrong answers considered are those with thinking < 0.85 × bt, shortest first. For a wrong answer with thinking tw,
  a chosen clean answer needs thinking in [1.08 × tw + 10, min(1.05 × bt, 1.45 × tw + 50)].

  The target is min(0.85 × bt, 1.25 × the median clean thinking). The best pair is picked by, in order:
  1. wrong answers that thought less than 0.65 × bt first;
  2. chosen answers that match the re-read pattern first (`REREAD`, `repair_pairs.py` lines 36–43);
  3. the smallest |chosen thinking − target|.

  One recovery candidate per question. Recovery is never blocked by the brake.

  **Example.** bt = 400 and the wrong answer thought 180 (< 0.65 × 400 = 260, so it ranks first). The chosen answer must
  think between 204 and min(420, 311) = 311 tokens. The target is min(340, 1.25 × median clean).
- **hold** ("be right at about the same length"). Applies to questions that are not damaged and have 1 ≤ k1 < n.
  - **Not held part:** a compact pair, with chosen thinking ÷ rejected thinking between 0.50 and 1.00. For IFEval the
    chosen response must also be ≤ `--right-resp-cap` 1.15 × the rejected response + 15. The pair is picked by: re-read
    match first, then shortest chosen thinking, then |ratio − 0.75|.
  - **When no compact pair exists, or the part is held**, a length-neutral pair: ratio 0.90–1.12, nearest 1.0, re-read
    match breaking ties.

  In other words, a held part still gets hold pairs, and they never favour the shorter answer much.

### 3.4 Quota per part

Round 1 uses `--od-r1-quota`: svamp 80, mbpp 56, ifeval 88, simple 36, multiple 28, irrelevance 32, for 320 pairs.
Rounds ≥ 2 use `--rp-quota`: svamp 52, mbpp 48, ifeval 64, simple 24, multiple 22, irrelevance 30, for 240 pairs.

For each part with quota N:

1. **Count, loop and format first.** Take count pairs up to round(0.10 N) (IFEval only), then loop pairs up to
   round(0.12 N), then format pairs up to round(0.10 N). n_rem = N − the pairs taken.
2. **dmg** = the mean of (p1 − p0) over the part's questions on this slice.
3. Candidate counts:
   - c_sh = the number of shorten candidates, or 0 if the part is held;
   - c_rec = the number of recovery candidates, doubled if dmg < −0.01;
   - c_hold = the number of hold candidates.
4. **Hold share.** n_hold = min(c_hold, round(0.20 × n_rem)) if c_sh + c_rec > 0, otherwise n_hold = n_rem.
   n_act = n_rem − n_hold.
5. **Shorten share.** n_sh = round(n_act × c_sh / (c_sh + c_rec)). When c_sh > 0 and dmg > −0.015, recovery is capped:
   n_sh = min(c_sh, n_act − min(recovery candidates, round(0.20 × n_act))).
6. **Fill in order:**
   - shorten up to n_sh;
   - then recovery up to n_act − shorten taken;
   - then hold up to n_rem − shorten − recovery;
   - then more recovery for any gap.

   Shortening never fills a gap.
7. A shortfall is recorded. Nothing is filled from elsewhere.

Rules for `take()`, which picks pairs from a candidate list:

- It shuffles the list with a generator seeded at 0.
- It skips a pair already used.
- It skips a question that already has 3 pairs, or 2 pairs of a non-"guard" kind. Every od5 kind counts as non-guard,
  so in practice a question gives at most 2 pairs per round.
- Count and hold pairs are written with kind "right".

**Worked example** (a math part in a round ≥ 2, N = 52, not held):

- 3 loop and 2 format pairs → n_rem = 47.
- Say c_sh = 120, recovery = 10, c_hold = 30, dmg = −0.005.
- n_hold = min(30, round(9.4)) = 9 → n_act = 38.
- Recovery cap = min(10, round(7.6)) = 8 → n_sh = min(120, 38 − 8) = 30.
- Result: 30 shorten, 8 recovery, 9 hold, plus the 5 above = 52.

**Output order.** Within a part, kinds are interleaved by share. Parts are interleaved by share. `dpo.py --keep-order`
trains in that order.

## 4. Train

`dpo.py` (called from `track.py:ptrepair`). The first round creates a fresh adapter with `sft.py --init-only --seed 0`:
LoRA r 16, alpha 32, dropout 0, on q/k/v/o/gate/up/down_proj. Each later round continues the previous round's adapter.

The call:

```
dpo.py --mix all --keep-order --nll 0.0 --beta 0.1 --batch 16 --lr 1.4e-5 --lr-schedule cosine --warmup 2 --lr-min 1.4e-5
       --max-steps 15 (round 1: 20) --epochs 2 --save-every 5 --held-out 0 --max-len 8704 --fp16-autocast
       --dtype bfloat16 --seed 0 --ref-base
```

- **Loss:** −log σ(β × [(log π(chosen) − log ref(chosen)) − (log π(rejected) − log ref(rejected))]), with β 0.1 and no
  NLL term. The reference log-probabilities are computed once, with the adapter disabled.
- **Length filter:** a pair whose prompt + longer side exceeds 8,704 tokens is dropped.
- **Steps:** min(ceil(rows × 2 / 16), max_steps).
  - Rounds ≥ 2: 240 pairs give 30 steps, capped at 15, so each pair is seen once (15 × 16 = 240).
  - Round 1: 320 pairs give 40, capped at 20, so again once each.
  - A short round trains fewer steps, and the last saved step is used.
- **Learning rate:**
  - warm-up for 2 steps: 0.7e-5, then 1.4e-5;
  - then "cosine" from 1.4e-5 to `--lr-min` 1.4e-5. Because the end value equals the start value, the rate stays
    constant at 1.4e-5.
- **Optimiser:** AdamW, weight decay 0, gradient clip 1.0.
- **Precision (`--fp16-autocast`):**
  - the base is loaded in float32;
  - the forward pass runs under float16 autocast with a gradient scaler;
  - LoRA weights stay in float32.

  This default's origin is explained in `IDEA.md` section 6.
- **Tripwire.** `dpo.py` has a held-out tripwire. With `--held-out 0` its baseline and threshold are 0, so it never
  fires.
- The adapter is saved every 5 steps. The round's adapter is `adapters/od5n-r{k}-s0/step-15` (step-20 in round 1).

## 5. Merge

`merge.py --model Qwen/Qwen3-0.6B --adapters <round adapter>:1 --out models/od5n-r{k}`:

- the base is loaded in float32;
- the LoRA is applied with weight 1, then `merge_and_unload`;
- the result is saved in bfloat16 with the tokenizer.

This merged model answers round k+1. At the end of the line, `od5n-best` (`--od-best-name`) is the last round's merged
model, scored on Dev with `--rp-dev-vs rpE80`. The submitted checkpoint is round 22.

## 6. Not verified

- The number of stored answers per question and the sampling seed for the original's round-1 answers. `od3prep`'s text
  says ×8, while the newtop sets were answered 10 times with seed 41.
- Whether `data/od3-base-resp.json` covers every slice question. A missing question gives b_resp = 0, which weakens the
  IFEval response floor to 0.95 × median.
- Whether decoding for `--prefer-reread` changes any choice beyond the looping test and the re-read tie-breaks.
- Where the math `_nofinal` flag is set.
