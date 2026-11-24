#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal GRPO (Group‑Relative Policy Optimization) trainer for the
date‑generation task using Qwen2.5‑0.5B‑Instruct.

Key fixes in this version
-------------------------
1️⃣  Added missing ``import random``.
2️⃣  Did **not** overwrite ``tokenizer.pad_token`` – the model already defines a proper pad token.
3️⃣  Built attention‑mask and generated‑token mask from the *actual* lengths of each prompt
    and each **real** generated sequence (EOS‑aware), not from ``!= pad_token_id``.
4️⃣  When a generation is padded, the real length is taken as the index of the first
    ``eos_token_id`` in the generated part (plus one). If no EOS is present, the
    first pad token is used; if there is no pad either, the full length is kept.
5️⃣  All previously‑listed improvements (per‑prompt advantage normalisation,
    chat‑template prompting, float32 policy model, GRPO KL estimator, no ``labels=``
    in the forward pass) remain unchanged.
"""

import os
import re
import csv
import argparse
import random                     # <-- required for random.sample
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    set_seed,
)

# ==============================================================
# Hyper‑parameters (can be overridden from the CLI)
# ==============================================================
SEED = 42                     # reproducibility
BATCH_SIZE = 8                # prompts per optimisation step
G = 4                         # generations per prompt
MAX_NEW_TOKENS = 64           # max tokens generated per sample
LR = 1e-5                     # AdamW learning rate
KL_COEF = 0.02                # KL‑penalty weight
NUM_TRAIN_STEPS = 2000        # total optimisation steps
GRAD_ACCUM_STEPS = 1          # gradient accumulation steps
SAVE_DIR = Path("./grpo_finetuned")   # where artefacts are stored

# -------------------------------------------------------------
# Paths (filled after SAVE_DIR is created)
# -------------------------------------------------------------
LOG_CSV = SAVE_DIR / "training_log.csv"
PLOT_PNG = SAVE_DIR / "reward_curve.png"
TEST_SET = SAVE_DIR / "test_set.csv"

# ==============================================================
# Helper utilities (unit‑testable)
# ==============================================================
DATE_PATTERN = re.compile(r"^(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])-(\d{4})$")


def compute_reward(generation: str, answer: str) -> float:
    """
    Reward based on the *first line* of the model output.

    * Exact match               → 1.0
    * Correct ``MM‑DD‑YYYY`` format but not exact → 0.2 (format)
      + 0.2 for each correct component (month, day, year)
    * Empty or non‑matching    → 0.0
    """
    first_line = generation.strip().splitlines()[0].strip() if generation.strip() else ""

    if not first_line:
        return 0.0

    if first_line == answer:
        return 1.0

    m = DATE_PATTERN.fullmatch(first_line)
    if not m:
        return 0.0

    reward = 0.2                     # format is correct
    month_ans, day_ans, year_ans = answer.split("-")
    month_pred, day_pred, year_pred = m.groups()

    if month_pred == month_ans:
        reward += 0.2
    if day_pred == day_ans:
        reward += 0.2
    if year_pred == year_ans:
        reward += 0.2

    return reward


def normalize_advantages(rewards: torch.Tensor, g: int) -> torch.Tensor:
    """
    Per‑prompt (group) normalisation.

    Input shape: (B * g, )
    Returns a flattened tensor of the same shape where each row (prompt)
    has been normalised by its own mean/std.
    """
    eps = 1e-8
    B = rewards.shape[0] // g
    rewards = rewards.view(B, g)                         # (B, g)
    mean = rewards.mean(dim=1, keepdim=True)            # (B, 1)
    std = rewards.std(dim=1, unbiased=False, keepdim=True)  # (B, 1)
    adv = (rewards - mean) / (std + eps)                # (B, g)
    return adv.view(-1)                                 # flatten


# ==============================================================
# Main training routine
# ==============================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    # -----------------------------------------------------------------
    # 1️⃣  Prepare data & output directory
    # -----------------------------------------------------------------
    os.makedirs(SAVE_DIR, exist_ok=True)          # create before writing test set
    df = pd.read_csv("dates_dataset.csv", dtype=str).fillna("")
    test_df = df.tail(100).reset_index(drop=True)
    train_df = df.iloc[:-100].reset_index(drop=True)

    test_df.to_csv(TEST_SET, index=False)         # now safe

    prompts = train_df["prompt"].tolist()
    answers = train_df["answer"].tolist()
    assert len(prompts) == len(answers)

    # -----------------------------------------------------------------
    # 2️⃣  Tokenizer & models
    # -----------------------------------------------------------------
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Policy model – **float32** (preserves tiny updates)
    policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    policy.train()

    # Frozen reference model – keep bfloat16 for speed
    ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)

    # -----------------------------------------------------------------
    # 3️⃣  Generation configuration (shared)
    # -----------------------------------------------------------------
    gen_cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # -----------------------------------------------------------------
    # 4️⃣  Training loop
    # -----------------------------------------------------------------
    reward_history = []

    # CSV header
    with open(LOG_CSV, "w", newline="") as f_log:
        csv_writer = csv.writer(f_log)
        csv_writer.writerow(["step", "avg_reward"])

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    for step in trange(1, NUM_TRAIN_STEPS + 1, desc="GRPO steps"):
        # ----------------- sample a batch of prompts -----------------
        batch_idx = random.sample(range(len(prompts)), BATCH_SIZE)
        batch_prompts = [prompts[i] for i in batch_idx]
        batch_answers = [answers[i] for i in batch_idx]

        # ----------------- generate G completions per prompt ---------
        all_prompt_ids = []        # tokenised prompt (chat‑template)
        all_full_ids = []          # prompt + completion (padded)
        prompt_lens = []           # true length of each prompt
        full_lens = []             # **real** length of each full sequence (EOS‑aware)
        all_rewards = []           # scalar reward per generation

        for prompt, answer in zip(batch_prompts, batch_answers):
            # Chat‑template input (tensor shape (1, L_prompt))
            prompt_tensor = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt", return_dict=False,
            ).to(device)                     # (1, L_prompt)

            # Generate G different completions (sampling)
            gen_out = policy.generate(
                input_ids=prompt_tensor,
                generation_config=gen_cfg,
                num_return_sequences=G,
                do_sample=True,
                top_k=0,
            )  # (G, L_prompt + L_gen)

            L_prompt = prompt_tensor.shape[-1]

            for gen_seq in gen_out:
                # ---------- Determine real length of the generated part ----------
                generated_part = gen_seq[L_prompt:]                 # tensor of shape (L_gen,)
                # Find first EOS token in the generated part
                eos_positions = (generated_part == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    # Include the EOS token itself
                    gen_real_len = eos_positions[0].item() + 1
                else:
                    # No EOS → fall back to first pad token, else full length
                    pad_positions = (generated_part == pad_id).nonzero(as_tuple=True)[0]
                    if len(pad_positions) > 0:
                        gen_real_len = pad_positions[0].item()
                    else:
                        gen_real_len = generated_part.shape[0]   # whole generated part

                full_real_len = L_prompt + gen_real_len

                # ---------- Reward ----------
                gen_text = tokenizer.decode(
                    gen_seq[L_prompt:L_prompt + gen_real_len], skip_special_tokens=True
                )
                reward = compute_reward(gen_text, answer)
                all_rewards.append(reward)

                # ---------- Store tensors & lengths ----------
                all_prompt_ids.append(prompt_tensor.squeeze(0))   # (L_prompt,)
                all_full_ids.append(gen_seq.squeeze(0))          # (padded full seq,)
                prompt_lens.append(L_prompt)
                full_lens.append(full_real_len)

        # -----------------------------------------------------------------
        # Pad sequences (using model’s native pad token)
        # -----------------------------------------------------------------
        padded_prompt_ids = torch.nn.utils.rnn.pad_sequence(
            all_prompt_ids, batch_first=True, padding_value=pad_id
        )  # (N, max_prompt_len)
        padded_full_ids = torch.nn.utils.rnn.pad_sequence(
            all_full_ids, batch_first=True, padding_value=pad_id
        )  # (N, max_full_len)

        # -----------------------------------------------------------------
        # 5️⃣  Build masks from the stored **real** lengths
        # -----------------------------------------------------------------
        max_full_len = padded_full_ids.size(1)

        # attention mask: 1 for real tokens (up to real length), 0 for padding
        seq_lengths_tensor = torch.tensor(full_lens, device=device).unsqueeze(1)  # (N,1)
        attention_mask = (torch.arange(max_full_len, device=device).unsqueeze(0) < seq_lengths_tensor).long()

        # generated‑token mask: positions after the prompt but before the real end
        prompt_lengths_tensor = torch.tensor(prompt_lens, device=device).unsqueeze(1)  # (N,1)
        gen_token_mask = (
            (torch.arange(max_full_len, device=device).unsqueeze(0) >= prompt_lengths_tensor)
            & (torch.arange(max_full_len, device=device).unsqueeze(0) < seq_lengths_tensor)
        )  # (N, max_full_len) bool

        # -----------------------------------------------------------------
        # 6️⃣  Forward passes (policy & reference) – no labels
        # -----------------------------------------------------------------
        policy_out = policy(
            input_ids=padded_full_ids,
            attention_mask=attention_mask,
        )
        with torch.no_grad():
            ref_out = ref(
                input_ids=padded_full_ids,
                attention_mask=attention_mask,
            )

        # Log‑probabilities
        pol_logp = torch.log_softmax(policy_out.logits, dim=-1)   # (N, S, V)
        ref_logp = torch.log_softmax(ref_out.logits, dim=-1)

        # ---------------------------------------------------------------
        # Align logits with the *next* token (shift by one)
        # ---------------------------------------------------------------
        shifted_pol_logp = pol_logp[:, :-1, :].gather(
            2, padded_full_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)          # (N, S-1)
        shifted_ref_logp = ref_logp[:, :-1, :].gather(
            2, padded_full_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)          # (N, S-1)

        # Shifted mask (drop first token because we predict token t from t‑1)
        shifted_gen_mask = gen_token_mask[:, 1:]               # (N, S-1)

        # ---------------------------------------------------------------
        # Sequence log‑probability under the policy (sum over generated tokens)
        # ---------------------------------------------------------------
        pol_seq_logprob = (shifted_pol_logp * shifted_gen_mask).sum(dim=1)   # (N,)

        # ---------------------------------------------------------------
        # 7️⃣  Advantages (per‑prompt normalisation)
        # ---------------------------------------------------------------
        rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, device=device)
        advantages = normalize_advantages(rewards_tensor, G)   # (N,)

        # Policy‑gradient loss
        pg_loss = -(advantages * pol_seq_logprob).mean()

        # ---------------------------------------------------------------
        # 8️⃣  GRPO KL estimator (per generated token)
        # ---------------------------------------------------------------
        # estimator: exp(ref - pol) - (ref - pol) - 1
        kl_est = torch.exp(shifted_ref_logp - shifted_pol_logp) \
                 - (shifted_ref_logp - shifted_pol_logp) - 1.0
        kl = (kl_est * shifted_gen_mask).sum() / shifted_gen_mask.sum()

        loss = pg_loss + KL_COEF * kl

        # ---------------------------------------------------------------
        # 9️⃣  Optimisation step
        # ---------------------------------------------------------------
        loss.backward()
        if step % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        # ---------------------------------------------------------------
        # 🔟  Logging
        # ---------------------------------------------------------------
        avg_reward = rewards_tensor.mean().item()
        reward_history.append(avg_reward)

        # CSV log (append)
        with open(LOG_CSV, "a", newline="") as f_log:
            csv_writer = csv.writer(f_log)
            csv_writer.writerow([step, f"{avg_reward:.5f}"])

        if step % 10 == 0 or step == NUM_TRAIN_STEPS:
            tqdm.write(
                f"Step {step:4d} | Avg reward: {avg_reward:.4f} | loss: {loss.item():.4f}"
            )

    # -----------------------------------------------------------------
    # 11️⃣  Save artefacts
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(reward_history) + 1), reward_history, label="Avg reward")
    plt.xlabel("Training step")
    plt.ylabel("Average reward")
    plt.title("GRPO – Reward vs. Step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PNG)

    policy.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("\n=== Training complete ===")
    print(f"Model checkpoint : {SAVE_DIR}")
    print(f"Reward curve     : {PLOT_PNG}")
    print(f"Training log     : {LOG_CSV}")
    print(f"Test set saved   : {TEST_SET}")


# ==============================================================
# CLI entry point
# ==============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimal GRPO trainer for date‑format generation (Qwen2.5‑0.5B‑Instruct)"
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--g", type=int, default=G, help="Generations per prompt")
    parser.add_argument("--steps", type=int, default=NUM_TRAIN_STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--kl_coef", type=float, default=KL_COEF)
    parser.add_argument("--save_dir", type=str, default=str(SAVE_DIR))
    args = parser.parse_args()

    # Apply possible overrides
    BATCH_SIZE = args.batch_size
    G = args.g
    NUM_TRAIN_STEPS = args.steps
    LR = args.lr
    KL_COEF = args.kl_coef
    SAVE_DIR = Path(args.save_dir)

    # Refresh dependent paths after a possible new SAVE_DIR
    LOG_CSV = SAVE_DIR / "training_log.csv"
    PLOT_PNG = SAVE_DIR / "reward_curve.png"
    TEST_SET = SAVE_DIR / "test_set.csv"

    main(args)