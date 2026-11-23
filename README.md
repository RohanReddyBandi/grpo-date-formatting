# GRPO Date Formatting

CS 173 project: fine-tuning a tiny language model with GRPO (Group Relative Policy Optimization) to convert dates written in any style into `MM-DD-YYYY`.

Example: `July 23, 1983` → `07-23-1983`

## Task

Each prompt asks the model to convert one date:

```
Convert the following date to MM-DD-YYYY format. Answer with only the date. Date: <date>
```

Dates come in many styles: full and abbreviated month names, ordinals (`23rd of May, 1973`), weekday prefixes, numeric US-order dates (`01/23/2036`), year-first dates (`2053-09-23`), day-month-year (`08-Feb-2075`), dates inside a sentence, and random all-caps or lowercase versions. Years are always four digits and numeric slash dates are always month/day order, so every prompt has exactly one correct answer.

### Reward function

Graded on the first line of the model's output:

- Exact match: **1.0**
- Valid `MM-DD-YYYY` format but wrong: **0.2** for format, plus **0.2** for each correct month, day, and year (max 0.8)
- Anything else: **0**

## Files

| File | What it is |
|---|---|
| `generate_dates.py` | Generates the dataset. Answers are computed from the `datetime` object, never parsed back from text. |
| `dates_dataset.csv` | 1,000 prompts (1900–2099, 10 styles, random casing, at least 20 leap days). Fixed seed. |
| `grpo_train.py` | Minimal GRPO trainer written from scratch in PyTorch + transformers (no TRL). |
| `test_grpo.py` | Unit tests for the reward function and per-group advantage normalization. |
| `training_log.csv` | Average reward at every training step. |
| `reward_curve.png` | Plot of average reward vs. step. |
| `test_set.csv` | The last 100 rows, held out from training for Phase 3 evaluation. |

## Setup

- **Model:** Qwen/Qwen2.5-0.5B-Instruct
- **Hardware:** one NVIDIA A40 on Purdue RCAC's Scholar cluster
- **Hyperparameters** (the defaults the LLM recommended): batch size 8 prompts, G = 4 generations per prompt, learning rate 1e-5, KL coefficient 0.02, 2,000 steps, temperature 0.7, top-p 0.9, max 64 new tokens

## How the code was built and audited

1. **Generated** the GRPO code with gpt-oss:120b in Purdue GenAI Studio.
2. **Reviewed it by hand** and sent back fixes over a few rounds:
   - Output folder was written to before it existed (crash)
   - Empty model output crashed the reward function
   - Advantages were normalized across the whole batch instead of per prompt group (the core of GRPO)
   - No chat template for the instruct model
   - Policy loaded in bf16, so small updates could round away
   - Simple KL estimate swapped for the standard non-negative GRPO estimator
   - A fix accidentally removed `import random`
   - Setting the pad token to `<|im_end|>` broke the masks, since that token appears inside the chat template
   - Padding after the end token was being counted as generated output
3. **Audited** the final script with a different model (qwen3.8:27b), which reviewed the GRPO logic and wrote `test_grpo.py`. It confirmed the core logic was correct and flagged design limitations (no PPO clipping, policy loss scales with length, bf16 reference model).
4. **All 10 unit tests passed** on Scholar.
5. **Ran a 5-step smoke test** before the full run. It caught two environment problems: an old `jinja2` version, and transformers 5.x returning a dict from `apply_chat_template` (fixed with `return_dict=False`).

## Running it

```
pip install torch transformers pandas matplotlib tqdm pytest
pytest test_grpo.py -v
python grpo_train.py --steps 5 --save_dir smoke_test   # quick check
python grpo_train.py                                     # full run
```

Model weights aren't in this repo because they're too large for GitHub.
