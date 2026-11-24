# IDEA.md — Arsène le Moigne, ThinkingCap-Test

**Result (best checkpoint, in the archive root):** Qwen3-0.6B writes **58.6% fewer tokens** [57.6, 59.7] at **+0.07 points** of accuracy [−0.69, +0.86], on the full evaluation (2,759 questions × 8 samples, `evaluate.py` and `compare.py`, settings unchanged).

| | Math | Code | Instructions | One function | Several functions | No function fits |
|---|---|---|---|---|---|---|
| Tokens saved | 61% | 56% | 58% | 55% | 57% | 69% |
| Accuracy change (points) | −1.5 | −0.1 | +1.8 | **−4.3** | −1.7 | +8.5 |

The average hides a real cost on one-function tool calls. I say where it comes from in section 4.

The best checkpoint is in the archive root as a LoRA adapter; the same model, merged into full weights (the form I measured), is at https://huggingface.co/arsenelemoigne/qwen3-0.6b-lupin. A longer write-up with every chart is at https://arsenelemoigne.eu/posts/thinkingcap/. How to re-run the final model is in `REPRODUCE.md`.

## 1. The idea

Among its 8 answers to a question, the original model usually writes at least one that is right and short. It already knows the short way; it does not use it reliably. So I do not teach it a new style. I train it to **prefer its own shorter right answers**, and I stop each task before its accuracy breaks.

The method is **online DPO against a fixed reference**:

1. **Fresh questions.** Each round uses 780 new training questions (220 math, 140 code, 200 instructions, 220 tool calls); one exception, found while writing this up: 324 questions of rounds 20 to 22 had already been used in round 1. They come from other public datasets, never from the evaluation benchmarks.
2. **Answer.** The current model answers each question 8 times.
3. **Pair.** Pairs are built from those answers and from the original model's stored answers to the same questions:
   - *shorter beats longer*: a right answer clearly shorter than the original's right answers is preferred to a longer right one;
   - *right beats wrong*, always;
   - *recovery*: if the current model now fails a question the original solved, its wrong answer is paired against the original's right one.
4. **Minimum length.** A short answer is never preferred if it falls below a per-task floor (a share of the original's thinking on that question, and a number of tokens). Without it, a task keeps shortening until its accuracy collapses and does not come back.
5. **Brake.** After each round, each task is checked on that round's questions. If it has saved more than its cap, or lost 2 points, it gets no "shorter" pairs in the next round, only right-over-wrong and recovery pairs. The check is repeated every round, so a task can resume.
6. **Train.** 15 steps of 16 pairs (20 steps in round 1), one LoRA adapter of rank 16 that continues from round to round, constant learning rate 1.4e-5, β = 0.1. **The DPO reference is always the original model**, never the previous round. This is what stopped the slow drift of my first online runs.
7. Repeat for 22 rounds, in one run from the original model. No repair stage, no merging of separate models. The submitted model is the original plus that one adapter.

Per-task settings of the final run (floor = share of the original's thinking · minimum tokens):

| | Math | Code | Instructions | One function | Several functions | No function fits |
|---|---|---|---|---|---|---|
| Pairs per round (all kinds) | 52 | 48 | 64 | 24 | 22 | 30 |
| Floor | 45% · 30 t | 38% · 36 t | 25% · 14 t | 44% · 25 t | 38% · 22 t | 20% · 10 t |
| Saving cap | 61% | 60% | 60% | 45% | 53% | 72% |

**Hardware.** Google Colab, L4 GPUs (24 GB), one or two at a time; a T4 for the first probes. One round takes 30 to 40 minutes on one L4, the 22 rounds of the final run took about 13 hours, and a full evaluation takes about 1 to 2 hours.

Every rule of the final run, as coded, is in `RULES.md`; how each training question was built is in `TRAINING_DATA.md`.

## 2. Explore, then exploit

This was my first machine-learning project, so I split it in two phases: first explore many directions cheaply, then put the compute into the one that worked.

**Explore.** Numbers marked *(screen)* are from small screens, not the full evaluation.

| Direction | Result | Why I moved on |
|---|---|---|
| System prompt "think briefly" | 22% saved at −2.7 | Changes the average length, not the model's judgement of when it is done. Code and tool calls mostly ignore it. |
| Activation steering (a "brief vs verbose" direction) | up to 18% at no cost *(screen)*; stronger pushes cost 5 to 8 points | Shortens, but does not teach when to stop. |
| Pruning weights | 10% removed: no change; 20% or more: it thinks *longer* | Removing weights does not remove habits. |
| Fine-tuning on its shortest right answers | 12% at −0.7 *(screen)* | Imitation shows a short answer but not what to prefer. |
| The same answers used as preference pairs | 40–48% *(screen)* | The contrast that pointed me to preference training. |
| Cutting each answer where it is already decided, then a repair stage | 57.4% at −3.5 | Needs an oracle to find the cuts and a second stage to repair accuracy; math paid most. Didn't seem to generalize. |
| Reinforcement learning (GRPO with a length penalty, DRPO, a ShorterBetter-style reward, a larger model as coach) | barely shortened | Not proof it cannot work: I tuned these far less than DPO, and the length term was tiny next to right-or-wrong. |
| One adapter per task, merged | 53.6% at −3.3 | The adapters interfered once combined. |
| Offline DPO (pairs built once, before training) | 54.1% at −0.93 after a six-stage chain; 51.7% at −0.86 as a clean three-step procedure | Pairs go stale: the model stops writing the long answers it is told to avoid, so every further gain needed a new stage. |
| Online DPO, no brake | 60.5% at −4.8 (math −9) | Ran away from itself. The floor, the brake and the fixed reference exist to stop this. |
| Online DPO with floor, brake and fixed reference | first recipe 54.8% at −0.78 (second seed 52.1% at −0.58); tuned per-task limits 59.5% at −0.80; final limits **58.6% at +0.07** | Kept. |

**Why I dropped reinforcement learning.** Reinforcement learning is said to work poorly on models this small, and my own runs agreed: the length term was hard to balance against the right-or-wrong reward, and I could not find hyperparameters that shortened the model without breaking it. Getting them right would have taken many more runs than I could afford.

**Why online DPO.** Offline DPO was the first method that clearly worked: preference pairs taught the model what to prefer, not just a style to imitate. But each gain needed a new stage, with pairs rebuilt from the model's newest answers, then a repair, five to six stages in all. That chain was in effect online DPO done by hand, one stage at a time. It suggested the same result could come from one careful procedure: rebuild the pairs every round from the current model, and add the controls the chain had needed (a minimum length per task, a brake, and the original model as a fixed reference so the model cannot drift).

**Exploit.** From then on, every run was online DPO from the original model, and I only tuned its per-task limits. The final model is one such run.

## 3. How the algorithm trades length for accuracy

You asked for several checkpoints. All are full evaluations; the extra ones are under `other-checkpoints/`.

| Checkpoint | Tokens saved | Accuracy change [95%] | Note |
|---|---|---|---|
| Earlier tuned run, continued, round 20 | 57.4% | +0.14 [−0.63, +0.88] | |
| **Final recipe, round 22 (root)** | **58.6%** | **+0.07 [−0.69, +0.86]** | one run from the original |
| Sister variant of the final recipe, round 15 | 59.3% | −1.72 [−2.49, −0.96] | differs only in per-task limits |
| Earlier tuned run, round 18 | 59.5% | −0.80 [−1.53, −0.02] | math −3.6 |
| Earlier tuned run, continued with looser caps, round 19 | 60.6% | −1.00 [−1.76, −0.28] | the only one past 60% |
| Looser variant, rounds 19 and 21 (not sent) | 61.7%, 62.7% | −1.68, −1.69 | tool calls lose 6 to 11 points |
| For reference: the original with thinking switched off | 94% | −22.3 | math falls from 89% to 21% |

Past about 59%, accuracy starts to go, first on math and on tool calls.

## 4. What the shorter model actually does

Measured on the final model. The "settle point" tests use 1,230 training-type questions it never trained on; the rest use its full-evaluation answers.

- **Overthinking.** The original keeps writing after its answer is decided: 28% of its math thinking, and 69–94% on the other tasks, comes after the point where the answer no longer changes.
- **Two kinds of saving.** On code, instructions and tool calls, the model mostly stopped re-checking. On math, 62% of the cut came *before* the answer was settled: at the same 224-token thinking budget it solves 60% of problems against 30% for the original, and the original forced to stop as early drops to 54% (the trained model: 87%).
- **Dropped, not compressed.** About 98% of the saving is whole steps removed. The steps it kept are written the same way (17.1 → 16.2 tokens each).
- **It carries over.** On four benchmarks not used in training (ARC-Challenge, CommonsenseQA, MMLU, HumanEval; 8 samples) it writes 48–70% fewer tokens, and no accuracy change is outside its 95% interval (−2.0, −1.3, +2.4, −0.5 points).
- **But shorter is not smarter.** The average holds because losses and gains cancel. About 10 answers in 100 swap between right and wrong (83 questions' worth beyond what sampling noise explains). Math problems solved 8 times out of 8 fall from 70% to 65%, while "solved at least once" stays at 96.8%: the model did not lose the ability, its answers vary more.
- **The one-function loss is dropped arguments.** Calls with a missing argument rise from 2.3 to 5.4 per 100 answers, and answers with no usable call from 2.0 to 3.7. A wrong function name stays rare (0.8 to 1.4).
- **A negative result.** I hoped that, after training, a long answer would flag a wrong one. It does not: among answers to the same question, length no longer predicts which is wrong.

## 5. Limits, and what the evaluation set influenced

I want to be plain about where the evaluation set touched this work.

- **Training questions are matched to the evaluation set, not drawn from it.** They come from other public datasets (GSM8K and MetaMathQA for math, KodCode for code, no_robots / Dolly / OpenAssistant requests with IFEval-style verifiable constraints, ToolACE / Glaive / Hermes for tool calls). Each round's set is drawn to match the evaluation set's task mix, length and difficulty (how often the original model solves a question). Candidates sharing an 8-word sequence with an evaluation question are dropped. When the public pools ran out of hard tool and code questions, I generated new ones with a large language model and kept only verified ones.
- **The per-task floors and caps came from earlier full evaluations.** I set them from the collapse points I had seen: for each task, the best saving any earlier model reached while losing one or two points.
- **Many models were scored on the evaluation set.** Over the project about 45 models got a full evaluation, and in the first weeks I also screened on a 720-question subset of it. From 11 November, choices were made on my own 512 held-out training-type questions ("Dev"). For the final run, four rounds were scored on Dev and only round 22 got a full evaluation; my automatic Dev gate had not selected it, and I chose to test it anyway. The headline number still carries a selection effect from the whole project.
- **The four other benchmarks were never used to tune training, but they did help me choose.** Several earlier models were measured on them along the way. At the end I compared four finished candidates on them, and kept the one with no significant loss there. So they are a check on the method, not a fully blind test of the final pick.
- **Other test splits.** 41 training questions come from GSM8K's test split. GSM8K is not one of the four evaluation benchmarks, but I mention it.
- **One run.** The final recipe was run once (seeds in `REPRODUCE.md`). The first recipe gave 54.8% and 52.1% on two seeds. The sister variant shows the result is sensitive to the per-task limits.
- **The limits are per task.** The rules are general, but their values are set per benchmark task. I have not shown how to set them for a task with no earlier measurements.
- **My in-training readings were optimistic.** Accuracy read on each round's training questions, and on Dev, came out 3 to 4 points higher than on the full evaluation.

## 6. How I worked

I am a lawyer in training, not an ML engineer, and this was my first machine-learning project. I did it alone, with the AI tools described below.

**Other candidates.** To know where the bar was, I read the other candidates' public write-ups and code, and set myself the goal of going past it. I understood their methods, and I re-ran one published recipe as a reference point; it is not part of the final method, though its code paths are still in this repository. My method is not built on theirs: online rounds, the fixed reference, the per-task floors and the brake came out of my own failures (section 2). Two things I did take from that reading: one public dataset, MetaMathQA (GSM8K problems rewritten in wordings closer to SVAMP's), which I added as a math source after seeing it in another candidate's data mix; and one numerical setting, the base weights kept in float32 with float16 autocast during the update, which I first saw in another candidate's public training script. A third idea from that reading, an extra loss term that keeps the preferred answer likely (an "NLL anchor"), is still in the code but switched off in the final run.

**AI tools.** I worked with a coding agent (Claude Code): I chose the directions, set the constraints and made the decisions; the agent wrote and ran most of the code and drafted the analyses. I also consulted other language models as reviewers: mainly Gemini, another model early on, and once a second Claude model. They reviewed plans (made by me and Claude), and Gemini proposed setting variations from the data of my earlier runs, including the per-task limits of the final run, which is one of two variations it proposed. Language models (Claude Haiku and Gemini) also wrote part of the training questions for tool calls, code and instructions; each was kept only after automatic checks (`TRAINING_DATA.md`). I found that as its context grew, Claude became less reliable, and a reviewer starting from a clean context helped me decide on clean data.

**What is attached.** `AGENT_TRACES.txt` is an edited selection of my conversation with the coding agent, with the commands it ran. `EXPERIMENT_LOG.md` is the dated record of the runs, numbers and decisions, including the dead ends. Both leave out unrelated work, credentials, off-topic exchanges, and my notes and comparison runs on other candidates' public submissions. Some early sessions were deleted and cannot be recovered. The code bundle is one commit of the final code: I squashed the history for the same reasons.
