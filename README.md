# thinkingcap-lupin

Code for **qwen3-0.6b-lupin**: Qwen3-0.6B trained with online DPO to think less at the same accuracy, for BottleCap AI's open ThinkingCap-Test challenge.

**Result** on the challenge's official evaluation (2,759 questions, 8 answers each): **58.6% fewer tokens** [57.6, 59.7] at **+0.07 points** of accuracy [−0.69, +0.86].

- Write-up with every chart: https://arsenelemoigne.eu/posts/thinkingcap/
- Model (merged weights + LoRA adapter): https://huggingface.co/arsenelemoigne/qwen3-0.6b-lupin

## What is here

**The final recipe**
- `track.py`: the runner. Each line of a plan file is one stage (`python track.py plan --plan-file <file>`); the final model is two `odpo` lines (see `REPRODUCE.md`).
- `repair_pairs.py`: builds each round's preference pairs (shorter right over longer right, right over wrong, recovery) with the per-task floors.
- `dpo.py`: DPO training of the LoRA adapter against the original model as fixed reference.
- `sft.py`: creates the empty LoRA adapter (`--init-only`).
- `merge.py`: merges the adapter into the base weights after each round.
- `probe_curve.py`, `typed.py`: generation of the model's answers with vLLM, and answer checking.
- `full_eval_colab.ipynb`: the Colab notebook that ran the queue.

**Data building**
- `prepare_types.py`, `build_pool.py`, `select_pool.py`, `pool_check.py`, `subset_questions.py`, `subset_check.py`: the training-question pools and the per-round slices matched to the test.
- `contamination.py`, `audit_contamination.py`: the 8-word overlap check against the evaluation questions.
- `test_difficulty_data.py`, `test_difficulty.json`: the original model's solve-rate profile on the test, used to match slices.
- `gen_*.py`, `nofit_verify.py`, `verify_removed_nofit.py` and the `.jsonl` files: the questions written by language models (tool calls, "no function fits", code, math, instructions) and their checks.

**Analysis (article figures)**
- `budget_curves.py`, `budget_ladder.py`, `prune_traces.py`, `check_vllm.py`, `surprisal.py`, `eff_allocate.py`: forced early stops, equal-budget curves, information per token.
- `churn_metrics.py`, `settle_text.py`, `trim_condense.py`, `content_density.py`, `onefn_split.py`, `lenflag.py`, `wrong_lengths.py`: question-by-question churn, thinking after the answer, trimming against condensing, the one-function errors, length as an error signal.
- `transfer_score.py`, `dev_score.py`, `article_final_stats.py`, `article_final_examples.py`: generalization scores, Dev scores, the article's statistics and examples.

**Documents**
- `IDEA.md`: the method, what I tried, the analysis, the limits.
- `REPRODUCE.md`: how the final model was trained and evaluated, step by step.
- `RULES.md`: every rule the final run applied, with exact values and where each lives in the code.
- `TRAINING_DATA.md`: how each training question was built.
- `EXPERIMENT_LOG.md`: dated record of the runs and decisions (edited copy).

`track.py` also contains the stages of the directions I explored before (see `IDEA.md`, section 2); their helper scripts are not in this repository.

The evaluation itself (`evaluate.py`, `compare.py`, `utils/`) comes from the official harness, `BottleCapAI/ThinkingCap-Test`, which the notebook clones next to this code. The notebook still points to the private development repository where the code ran, so read `REPRODUCE.md` before re-running.
