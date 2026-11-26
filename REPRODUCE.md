# Reproducing the submitted checkpoint (od5n, round 22)

The submitted model is the original Qwen3-0.6B plus one LoRA adapter (rank 16), trained by 22 rounds of online DPO.
Its full test (2,759 questions, 8 answers each, official `evaluate.py` and `compare.py`): 58.6% tokens saved
[57.6, 59.7] at +0.07 accuracy points [-0.69, +0.86].

Everything below uses the code in the git bundle: one commit holding the final state of the code, with the code
files exactly as they ran. I did not write a separate training script for the submission. Each statement was checked
against the code or the experiment log; what could not be checked is listed at the end.

## 1. Environment

- One Google Colab L4 GPU (24 GB) per run. Python 3.13 and vLLM 0.23.0 (read from the runtime's log on 22 Nov).
- The official harness, `BottleCapAI/ThinkingCap-Test`, cloned from `main` when the runtime starts (not pinned). `main`
  has been at commit `965a54c` since 6 Nov; the last three commits change only the README. Its `requirements.txt` pins
  transformers 4.57.1, peft 0.17.1 and accelerate 1.10.1.
- The notebook (`full_eval_colab.ipynb`) mounts Google Drive, clones the harness, installs its requirements, copies this
  repository's `*.py` and `plan*.txt` files next to `evaluate.py`, then runs `python track.py setup` (which runs the
  harness's `utils/train/prepare_data.py` and `utils/eval/benchmarks.py`, and installs `langdetect`).
- The queue is a text file with one stage of `track.py` per line: `python track.py plan --plan-file <file>`. My own
  queue files that ran it are not in the bundle (they hold many unrelated lines; the plan*.txt files that are in it are older queues); to re-run, put the two lines of section 3
  in a text file and pass it with `--plan-file`.
  The runner keeps `runs/`, `data/`, `adapters/` and `models/` on Drive (folder `thinkingcap-mvtf`).

## 2. Training data

The training questions never come from the four evaluation benchmarks. They come from public training sets (GSM8K,
MetaMath, KodCode, ToolACE, Glaive and Hermes function-calling sets, and instruction-following requests built by
`prepare_types.py`), plus sets of questions written by larger language models in the style of the tasks (Claude Haiku
5.5 and Gemini: tool-call, "no function fits" and some instruction questions from slice 10 on, medium code and
SVAMP-style math for slices 20 to 22). Only the questions come from those models. Every answer used in training is Qwen3-0.6B's own. The question-writing scripts (`gen_*.py`, `nofit_verify.py`) read their chat endpoint and key from the environment variables `LLM_API_URL` and `LLM_API_KEY`; the files they wrote are in the bundle, so they need not be re-run.
Candidate questions sharing rare 8-word sequences with the evaluation prompts were dropped (`build_pool.py`).

The questions are cut into slices of 780 (math 220, code 140, instructions 200, one function 80, several
functions 60, no function fits 80). Each slice is matched to the full test per task, on how often the original model
solves the questions (`test_difficulty_data.py` holds the original's solve-rate bins on the test). So the test's
difficulty profile was used to choose training questions; the test questions themselves were not. The slices are meant to be disjoint, but 324 questions of slices 20 to 22 repeat slice 1: when slices 20 to 22 were
drawn, the check for already-used questions covered slices 2 to 19 and missed slice 1 (details in `TRAINING_DATA.md`).

- Slices 1 to 19 (`data/od3-rK-questions.jsonl`): stage `od3prep`, run in several calls between 13 and 16 Nov
  (slices 1 to 6 first, with `--data-seed 18`; slices 17 to 19 last, with `--data-seed 58`). The exact calls
  are recorded in the experiment log sent with the submission (13 to 16 Nov). Each call added top-up question sets built by stage `newtop`.
- Slices 20 to 22 (`data/od3g-rK-questions.jsonl`): stage `od3branch --ob-into od3g --ob-from 20 --ob-more 5
  --od3-tops r2x-code-m1,r2x-math-m1 --ob-trim 0.05 --ob-trim-max 0.4 --data-seed 61` (19 Nov). It asked for five
  slices; three passed its checks.
- The original model's statistics per question: `data/od3-base-stats.json` (solve rate, mean tokens),
  `od3-base-think.json` (median thinking of its right answers) and `od3-base-resp.json` (median response length).
- Round 1 trains on the original's stored answers to slice 1 (`runs/bon/rp-od3-r1-stored/rollouts.jsonl`).

The slice files and the original model's statistics are shipped with the submission (`training_data/`), so the data
step can be skipped. The original's stored answers to slice 1, used by round 1 (149 MB), are not shipped.

How each training question was built, source by source, with per-slice counts: `TRAINING_DATA.md`. Every rule the run applied: `RULES.md`.

## 3. The training run

Two queue lines, as run on 20 and 21 Nov. The first trains rounds 1 to 18. The second is the same line asked for 22
rounds on the extended slice set; it skips the 18 finished rounds.

```
odpo --name od5n --od-fixed-ref --rp-base Qwen/Qwen3-0.6B --od-rounds 18 --od-steps 15 --od-r1-steps 20 --od-r1-quota svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32 --rp-quota svamp=52,mbpp=48,ifeval=64,bfcl_simple_python=24,bfcl_multiple=22,bfcl_irrelevance=30 --rp-questions svamp=220,mbpp=140,ifeval=200,bfcl_simple_python=80,bfcl_multiple=60,bfcl_irrelevance=80 --rp-own-data --rp-kinds mix --rp-samples 8 --rp-seed 18 --rp-pair-args=--think-only,svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance,--prefer-reread,--count-rules,--shorten-ratio,1.30,--shorten-ratio-by-part,ifeval=1.25+bfcl_irrelevance=1.25,--shorten-second,ifeval+bfcl_irrelevance,--shorten-min-right,0.85,--mix-preset,od5,--od5-v2,--base-stats,data/od3-base-stats.json,--base-resp,data/od3-base-resp.json,--base-think,data/od3-base-think.json,--floor-by-part,svamp=30:0.45+mbpp=36:0.38+ifeval=14:0.25+bfcl_simple_python=25:0.44+bfcl_multiple=22:0.38+bfcl_irrelevance=10:0.20,--count-mode,atleast,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--resp-median,0.95,--resp-max,1.05,--right-resp-cap,1.15 --rp-lr 1.4e-5 --rp-warmup 2 --rp-beta 0.1 --rp-dpo-seed 0 --rp-dev-vs rpE80 --od-dev-rounds 99 --data-seed 18 --od-caps svamp=61,mbpp=60,ifeval=60,bfcl_simple_python=45,bfcl_multiple=53,bfcl_irrelevance=72 --od-acc-hold -2.0 --od-unlatch --od-stop-save 999 --od-conv-gain -999 --od-min-rounds 18 --od-best-name od5n-best --note consult37-B

odpo --name od5n --od-slices od3g --od-fixed-ref --rp-base Qwen/Qwen3-0.6B --od-rounds 22 --od-steps 15 --od-r1-steps 20 --od-r1-quota svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32 --rp-quota svamp=52,mbpp=48,ifeval=64,bfcl_simple_python=24,bfcl_multiple=22,bfcl_irrelevance=30 --rp-questions svamp=220,mbpp=140,ifeval=200,bfcl_simple_python=80,bfcl_multiple=60,bfcl_irrelevance=80 --rp-own-data --rp-kinds mix --rp-samples 8 --rp-seed 18 --rp-pair-args=--think-only,svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance,--prefer-reread,--count-rules,--shorten-ratio,1.30,--shorten-ratio-by-part,ifeval=1.25+bfcl_irrelevance=1.25,--shorten-second,ifeval+bfcl_irrelevance,--shorten-min-right,0.85,--mix-preset,od5,--od5-v2,--base-stats,data/od3-base-stats.json,--base-resp,data/od3-base-resp.json,--base-think,data/od3-base-think.json,--floor-by-part,svamp=30:0.45+mbpp=36:0.38+ifeval=14:0.25+bfcl_simple_python=25:0.44+bfcl_multiple=22:0.38+bfcl_irrelevance=10:0.20,--count-mode,atleast,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--resp-median,0.95,--resp-max,1.05,--right-resp-cap,1.15 --rp-lr 1.4e-5 --rp-warmup 2 --rp-beta 0.1 --rp-dpo-seed 0 --rp-dev-vs rpE80 --od-dev-rounds 99 --data-seed 18 --od-caps svamp=61,mbpp=60,ifeval=60,bfcl_simple_python=45,bfcl_multiple=53,bfcl_irrelevance=72 --od-acc-hold -2.0 --od-unlatch --od-stop-save 999 --od-conv-gain -999 --od-min-rounds 22 --od-best-name od5n-best --note consult37-B
```

What one round does (`track.py`: `odpo_fixed`, which calls `ptrepair`):

1. **Answer.** The current model (the original with the adapter merged in) answers the round's 780 new questions
   8 times each (`probe_curve.py`, vLLM, the harness's own sampling settings, seed 53, the evaluation's token caps).
2. **Brake** (`--od-caps`, `--od-acc-hold`). These answers are compared with the original's on the same questions. A
   task whose saving has reached its cap (math 61%, code 60%, instructions 60%, one function 45%, several functions
   53%, no function fits 72%), or whose accuracy is 2 points or more below the original's, gets no "shorter" pairs
   this round. The decision is taken again every round (`--od-unlatch`).
3. **Pairs** (`repair_pairs.py`, `--rp-quota`: 240 pairs per round, 320 in round 1). Pairs come only from the model's
   own answers: a shorter right answer preferred to a longer right one (thinking at least 1.30 times shorter, 1.25
   for instructions and "no function fits"), a right answer preferred to a wrong one, and recovery pairs on
   questions the model now solves less often than the original did.
4. **Floors** (`--floor-by-part`). A right answer is never preferred if its thinking is below a token count or below
   a share of the original's median right thinking on that question: it must reach the larger of the two (for
   example math: 30 tokens and 45%).
5. **Train** (`dpo.py`). DPO with beta 0.1, 16 pairs per step, 15 steps (20 in round 1), learning rate 1.4e-5
   constant after 2 warm-up steps, no NLL term. The same adapter continues from round to round. The DPO reference is
   always the original model, with the adapter switched off (`--ref-base`). The adapter is LoRA rank 16, alpha 32, on
   the attention and MLP projections (`sft.py --init-only`). During the update the base weights are in float32, with
   float16 autocast and a loss scaler (the default of this training path in `track.py`; od5n's line does not switch
   it off; `IDEA.md`, section 6, says where the setting comes from).
6. **Merge** (`merge.py`). The adapter is merged into the original (summed in fp32, saved in bf16) as
   `models/od5n-rK`, which answers the next round.

The submitted checkpoint is `adapters/od5n-r22-s0/step-15` (the adapter) and `models/od5n-r22` (the merged model).

## 4. Seeds and what is not deterministic

`--rp-dpo-seed 0` fixes the adapter's initialisation and the pair order. `--data-seed` and the `od3prep` /
`od3branch` seeds fix the slices. `--rp-seed 18` plays no role here, because the questions come from the slice files.
Answers are sampled (vLLM, seed 53), and each round trains on the previous round's sampled answers, so a rerun will
not reproduce the weights bit for bit. I did not rerun od5n with another seed. The closest check is an earlier recipe
of the same family, run twice with different seeds: 54.8% / -0.78 and 52.1% / -0.58 on the full test.

## 5. Time, evaluation and selection

- Training: line 1 started on 20 Nov at about 23:40 and round 18 ended at about 10:00 the next day; rounds 19 to 22
  ended at about 14:00, after a Dev scoring of about 40 minutes. That is about 13 to 14 hours on one L4, or roughly
  35 minutes per round (log timestamps, accurate to about 15 minutes).
- Full test: `fullmodel --nothink-models models/od5n-r22 --fm-no-screen`, which runs
  `python evaluate.py --questions questions --samples 8 --out runs/mvtf/full-od5n-r22 --model models/od5n-r22`, then
  `python compare.py runs/mvtf/full-base runs/mvtf/full-od5n-r22`. A full test of a trained model takes about one to two hours
  on an L4. The original was scored the same way (`runs/mvtf/full-base`). The numbers above were measured on the
  merged model, not through `--adapter`.
- Selection: rounds were first scored on Dev-512, a set of 512 held-out training-style questions (8 answers each) that
  no run trained on. For od5n, rounds 13, 18, 20 and 22 were scored on Dev. The automatic rule (Dev at 60.5% saved and
  +1.75 or better) selected none of them. I then chose to run the full test on round 22 by hand, because it had the
  best Dev accuracy at over 58% saved. Round 22 is the only od5n round with a full test. Across all my runs, many
  models had a full test before this one, and some settings (the per-task floors and caps) were adjusted after seeing
  earlier full tests.

## 6. Hardware over the whole project

Two Colab L4 runtimes, used in parallel for most of the project (19 Oct to 23 Nov 2026); the very first runs used a T4.
I did not keep a total of GPU hours, so the total is unknown.

## 7. Not verified

- A complete rerun from an empty machine. The runner expects its folder layout on Google Drive, and on Colab the
  training stage stops if Drive is not mounted. The first queue line also ends with a Dev-512 scoring, which needs the
  Dev files (not in this package).
- That the answers through `evaluate.py --adapter` match the merged model's exactly. They should agree up to bf16
  rounding, but the full test was only run on the merged model.
