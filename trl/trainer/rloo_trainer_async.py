# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from typing import Callable, Optional, Union

import ray
from ray.util.queue import Queue
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback

from ..trainer.utils import (
    OnlineTrainerState,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    prepare_deepspeed,
    print_rich_table,
    selective_log_softmax,
    truncate_response,
)
from .rloo_config import RLOOConfig


if is_wandb_available():
    import wandb

INVALID_LOGPROB = 1.0

class RLOOTrainerAsync(Trainer):
    _tag_names = ["trl", "rloo"]

    def __init__(
        self,
        config: RLOOConfig,
        processing_class,
        policy: nn.Module,
        ref_policy: nn.Module,
        replay_buffer: Queue,
        data_collator: Optional[DataCollatorWithPadding] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
        async_replay_buffer: bool = True,
    ) -> None:
        if ref_policy is policy:
            raise ValueError(
                "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
                "same as `policy`, you must mass a copy of it, or `None` if you use peft."
            )

        self.args = config
        args = config
        self.processing_class = processing_class
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.async_replay_buffer = async_replay_buffer

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        self.policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.ref_policy = ref_policy
        self.data_collator = data_collator
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = exact_div(
            args.local_batch_size, args.rloo_k, "`local_batch_size` must be a multiple of rloo_k"
        )  # RLOO logic: needed because RLOO repeats the same prompt args.rloo_k times

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, ref_policy]:
            if isinstance(module, nn.Module):
                disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = self.processing_class.eos_token_id
        self.model = policy
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )

        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if self.is_deepspeed_enabled:
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            self.model = prepare_deepspeed(
                self.model, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            self.deepspeed = self.model
        else:
            self.ref_policy = self.accelerator.prepare(self.ref_policy)
            self.model = self.accelerator.prepare(self.model)

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_policy
        processing_class = self.processing_class
        device = accelerator.device

        def repeat_generator():
            if not self.async_replay_buffer:
                iter = self.replay_buffer.run()
            while True:
                curr_batch = []
                for _ in range(args.batch_size):
                    if self.async_replay_buffer:
                        item = self.replay_buffer.get(block=True)
                    else:
                        item = next(iter)
                    curr_batch.append(item)
                yield curr_batch

        iter_dataloader = iter(repeat_generator())

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = (args.num_total_batches * args.num_mini_batches) // 2
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                # Process each conversation trajectory
                all_input_ids = []
                all_response_masks = []
                all_rewards = []
                
                for traj in data:
                    turns = traj if isinstance(traj, list) else [traj]
                    if not turns:
                        continue
                        
                    # Get the reward for the entire conversation TODO: deal with multiple rewards
                    conversation_reward = turns[-1]["reward"]
                    
                    # Build the full conversation sequence in token space
                    full_sequence_tokens = []
                    response_mask = []  # 1 for response tokens, 0 for prompt tokens
                    
                    for turn in turns:
                        # Tokenize prompt and response separately
                        prompt_tokens = processing_class.encode(turn["prompt"])
                        response_tokens = processing_class.encode(turn["response"])
                        
                        # Add prompt tokens to sequence with mask=0 (not a response)
                        full_sequence_tokens.extend(prompt_tokens)
                        response_mask.extend([0] * len(prompt_tokens))
                        
                        # Add response tokens to sequence with mask=1 (is a response)
                        full_sequence_tokens.extend(response_tokens)
                        response_mask.extend([1] * len(response_tokens))
                    
                    all_input_ids.append(full_sequence_tokens)
                    all_response_masks.append(response_mask)
                    all_rewards.append(conversation_reward)
                
                # Pad sequences to the same length
                max_length = max(len(seq) for seq in all_input_ids)
                padded_input_ids = []
                padded_masks = []
                
                for i, (seq, mask) in enumerate(zip(all_input_ids, all_response_masks)):
                    # Pad with pad_token_id
                    padded_seq = seq + [processing_class.pad_token_id] * (max_length - len(seq))
                    padded_mask = mask + [0] * (max_length - len(mask))  # Pad masks with 0
                    
                    padded_input_ids.append(padded_seq)
                    padded_masks.append(padded_mask)
                
                # Convert to tensors
                input_ids = torch.tensor(padded_input_ids, device=device)
                response_masks = torch.tensor(padded_masks, dtype=torch.bool, device=device)
                rewards = torch.tensor(all_rewards, device=device)
                
                # Forward pass through both models
                with torch.no_grad():
                    outputs = forward(model, input_ids, processing_class.pad_token_id)
                    ref_outputs = forward(ref_policy, input_ids, processing_class.pad_token_id)
                
                logits = outputs.logits[:, :-1]  # Shift to align with next tokens
                ref_logits = ref_outputs.logits[:, :-1]
                
                # Prepare target tokens (shifted right for next-token prediction)
                target_ids = input_ids[:, 1:]
                response_masks_shifted = response_masks[:, 1:]  # Shift mask to match
                
                # Create a tensor for logprobs of the right shape
                batch_size = input_ids.shape[0]
                seq_length = target_ids.shape[1]
                
                # Calculate logprobs only for response tokens
                logprobs = torch.full((batch_size, seq_length), INVALID_LOGPROB, device=device, dtype=torch.float)
                ref_logprobs = torch.full((batch_size, seq_length), INVALID_LOGPROB, device=device, dtype=torch.float)
                
                # We need to efficiently compute logprobs for all tokens
                for i in range(batch_size):
                    # Get logprobs for all positions where we have a response token
                    mask = response_masks_shifted[i]
                    if not mask.any():
                        continue
                    
                    # Get the logits and target ids for this sequence where responses are
                    seq_logits = logits[i][mask]
                    seq_ref_logits = ref_logits[i][mask]
                    seq_targets = target_ids[i][mask]
                    
                    # Compute log probabilities for response tokens
                    seq_logprobs = selective_log_softmax(seq_logits, seq_targets)
                    seq_ref_logprobs = selective_log_softmax(seq_ref_logits, seq_targets)
                    
                    # Place logprobs in the right positions
                    logprobs[i][mask] = seq_logprobs.flatten()
                    ref_logprobs[i][mask] = seq_ref_logprobs.flatten()
                
                # Compute KL divergence on response tokens
                kl = logprobs - ref_logprobs
                
                # Normalize rewards if needed
                if args.normalize_reward:
                    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                    rewards = torch.clamp(rewards, -args.reward_clip_range, args.reward_clip_range)
                    
                # Compute rewards with KL penalty
                if args.token_level_kl:
                    # Token-level KL penalty applied to response tokens only
                    kl_reward = -args.kl_coef * kl
                    
                    # Get the last response token for each conversation
                    last_indices = []
                    for i in range(batch_size):
                        # Find the last position where response_mask is True
                        last_true = response_masks_shifted[i].nonzero()
                        last_idx = last_true[-1] if last_true.numel() > 0 else 0
                        last_indices.append(last_idx.item())
                    
                    # Create tensor for final rewards
                    last_reward = torch.zeros_like(kl)
                    for i in range(batch_size):
                        last_reward[i, last_indices[i]] = rewards[i]
                    
                    # Combine KL reward and final reward
                    non_score_reward = torch.sum(kl_reward * response_masks_shifted.float(), dim=1)
                    reward = last_reward + kl_reward
                    rlhf_reward = torch.sum(reward * response_masks_shifted.float(), dim=1)
                else:
                    # Sequence-level KL penalty
                    sequence_kl = torch.sum(kl * response_masks_shifted.float(), dim=1)
                    non_score_reward = -args.kl_coef * sequence_kl
                    rlhf_reward = non_score_reward + rewards
                
                # vectorized RLOO advantages implementation
                rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
                baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1)
                advantages = rlhf_reward - baseline
                advantages = advantages.flatten()

                # Normalize advantages
                if args.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            # Get batch data
                            mb_advantage = advantages[micro_batch_inds]
                            mb_input_ids = input_ids[micro_batch_inds]
                            mb_response_masks = response_masks_shifted[micro_batch_inds]  # Get shifted response masks
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_target_ids = target_ids[micro_batch_inds]

                            # Forward pass
                            output = forward(model, mb_input_ids, processing_class.pad_token_id)
                            logits = output.logits[:, :-1]  # Shift to align with targets
                            logits /= args.temperature + 1e-7

                            # Compute new logprobs only for response tokens
                            new_logprobs_full = torch.full_like(mb_logprobs, INVALID_LOGPROB)
                            
                            # Calculate logprobs only where response masks are True
                            for i in range(mb_input_ids.shape[0]):
                                mask = mb_response_masks[i]
                                if not mask.any():
                                    continue
                                    
                                seq_logits = logits[i][mask]
                                seq_targets = mb_target_ids[i][mask]
                                
                                # Compute log probabilities for response tokens
                                seq_logprobs = selective_log_softmax(seq_logits, seq_targets)
                                
                                # Place logprobs in the right positions
                                new_logprobs_full[i][mask] = seq_logprobs.flatten()
                            
                            # Only use response tokens for ratio calculation
                            # Sum logprobs across sequence dimension but only for response tokens
                            mb_logprobs_sum = torch.sum(mb_logprobs * mb_response_masks.float(), dim=1)
                            new_logprobs_sum = torch.sum(new_logprobs_full * mb_response_masks.float(), dim=1)
                            
                            # Compute log ratio
                            logprobs_diff = new_logprobs_sum - mb_logprobs_sum
                            ratio = torch.exp(logprobs_diff)
                            
                            # Compute token-wise ratios for metrics
                            token_ratio = (new_logprobs_full - mb_logprobs).exp()
                            
                            # PPO clipped loss
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = pg_loss_max.mean()

                            # Final loss
                            loss = pg_loss

                            # Optimization step
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()

                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                
                                # Entropy calculation (only for response tokens)
                                entropy = torch.zeros_like(new_logprobs_full)
                                for i in range(mb_input_ids.shape[0]):
                                    mask = mb_response_masks[i]
                                    if not mask.any():
                                        continue
                                        
                                    seq_logits = logits[i][mask]
                                    prob_dist = torch.nn.functional.softmax(seq_logits, dim=-1)
                                    seq_entropy = torch.logsumexp(seq_logits, dim=-1) - torch.sum(prob_dist * seq_logits, dim=-1)
                                    entropy[i][mask] = seq_entropy
                                
                                # Average entropy only over response tokens
                                mean_entropy = torch.sum(entropy * mb_response_masks.float()) / (mb_response_masks.sum() + 1e-8)
                                
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = mean_entropy
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = token_ratio[mb_response_masks].mean()
                        
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1

                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, logits, new_logprobs_full, logprobs_diff, ratio, pg_losses,
                        pg_losses2, pg_loss, loss, pg_clipfrac, entropy, approxkl,
                        mb_advantage, mb_input_ids, mb_response_masks, mb_logprobs, mb_target_ids,
                        token_ratio
                    )
                    # fmt: on
                    torch.cuda.empty_cache()

            # TODO: update weights of vllm
                    
            # Compute metrics
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(rewards.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (input_ids == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.local_dataloader_batch_size
                self.log(metrics)
            del kl, mean_kl, mean_entropy, rewards

            self.lr_scheduler.step()
            self.state.global_step += 1
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            torch.cuda.empty_cache()
            gc.collect()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
