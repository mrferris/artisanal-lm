import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch.nn.utils.rnn import pad_sequence

from lm.model.components.attention import Rope
from lm.model.components.linear import Embedding, Linear, RMSNorm
from lm.model.components.transformer import Transformer
from lm.training.optimization.adamw import AdamW
from lm.training.reinforcement.dpo import calculate_simpo_loss
from lm.training.reinforcement.grpo import calculate_grpo_loss
from lm.training.reinforcement.log_probs import calculate_model_log_probs
from lm.training.utils.checkpointing import load_checkpoint
from lm.training.utils.gradient_clipping import clip_gradients


class TransformerLM(nn.Module):
    """
    Implements an autoregressive transformer language model.
    Architecture is most similar to Llama1/Llama2:
    - Pre-norm via RMSNorm
    - RoPE for position encodings
    - SwiGLU activations in FFN
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ):
        """
        d_model: Embedding dimension of model, aka width
        num_heads: number of heads per attention instance
        num_layers: number of transformer layers
        rope: shared between all transformer layers
        d_ff: width of the feedforward networks
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = Rope(theta=rope_theta, d_k=d_model // num_heads, max_seq_len=context_length, device=device)
        self.device = device
        self.dtype = dtype

        self.embedding_layer = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)

        self.transformer_layers = []
        self.transformer_layers = nn.ModuleList(
            [Transformer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=self.rope, device=device, dtype=dtype) for _ in range(num_layers)],
        )
        self.output_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.register_buffer(
            "position_ids",
            torch.arange(context_length, device=device),
            persistent=False,
        )

        self.output_embedding = Linear(d_model, vocab_size, device, dtype)

    def forward(
        self,
        input: Int[torch.Tensor, "batch_size sequence_length"],
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        """
        Args:
            kv_cache: Controls KV caching behavior.
                None — no caching, return logits only.
                []   — initial encode: no prior cache, return (logits, kv_cache).
                [(K,V), ...] — continue: use prior cache, return (logits, new_kv_cache).
        """
        output = self.embedding_layer(input)

        batch, seq_len, _ = output.shape
        use_cache = kv_cache is not None

        # When using KV cache, token positions start after the cached prefix
        cache_len = kv_cache[0][0].shape[-2] if kv_cache else 0
        # All examples in a training batch share positions. Keeping this 1-D
        # lets RoPE broadcast its cached tables rather than copying them across
        # every batch item and attention head.
        token_positions = self.position_ids[cache_len : cache_len + seq_len]

        new_kv_cache = [] if use_cache else None

        for i, layer in enumerate(self.transformer_layers):
            # () = initial encode for this layer, (K,V) = continue with prior
            layer_cache = kv_cache[i] if kv_cache else (() if use_cache else None)
            result = layer(output, token_positions, kv_cache=layer_cache)

            if use_cache:
                output, layer_kv = result
                new_kv_cache.append(layer_kv)
            else:
                output = result

        output = self.output_norm(output)
        output = self.output_embedding(output)

        if use_cache:
            return output, new_kv_cache
        return output

    def encode_kv(self, prefix_tokens: Int[torch.Tensor, "batch_size seq_len"]):
        """Compute KV cache for a prefix sequence.

        Args:
            prefix_tokens: Token tensor of shape [batch, seq_len].

        Returns:
            (logits, kv_cache) where logits has shape [batch, seq_len, vocab_size]
            and kv_cache is a list of per-layer (K, V) tuples.
        """
        with torch.no_grad():
            logits, kv = self.forward(prefix_tokens, kv_cache=[])
        return logits, kv

    def forward_with_kv(
        self,
        suffix_tokens: Int[torch.Tensor, "batch_size seq_len"],
        kv_cache,
    ) -> Float[torch.Tensor, "batch_size seq_len vocab_size"]:
        """Forward suffix tokens using a pre-computed prefix KV cache.

        Automatically expands the cache's batch dimension to match
        suffix_tokens if needed (e.g. cache is batch=1, suffix is batch=N).

        Args:
            suffix_tokens: Tokens to forward, shape [batch, seq_len].
            kv_cache: Cache returned by encode_kv().

        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size].
        """
        batch_size = suffix_tokens.shape[0]
        cache_batch = kv_cache[0][0].shape[0]
        if batch_size != cache_batch and cache_batch == 1:
            kv_cache = [
                (k.expand(batch_size, -1, -1, -1), v.expand(batch_size, -1, -1, -1))
                for k, v in kv_cache
            ]
        with torch.no_grad():
            logits, _ = self.forward(suffix_tokens, kv_cache=kv_cache)
        return logits

    def forward_incremental(self, token, kv_cache):
        """Forward a single token and return updated KV cache.

        Args:
            token: Token tensor of shape [batch, 1].
            kv_cache: Running KV cache from encode_kv() or a prior
                      forward_incremental() call.

        Returns:
            (logits, updated_kv_cache) where logits has shape
            [batch, 1, vocab_size] and updated_kv_cache contains
            the full sequence K/V (prior cache + new token).
        """
        with torch.no_grad():
            logits, updated_kv = self.forward(token, kv_cache=kv_cache)
        return logits, updated_kv

    def param_count(self) -> tuple[int, int]:
        """
        Get the param count of the model.
        Returns:
            A 2-element tuple containing:
            - param count including embedding params
            - param count not including embedding params
        """
        non_embedding_parameters = 0
        for name, param in self.named_parameters():
            if "embedding" not in name:
                non_embedding_parameters += param.numel()
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return [num_parameters, non_embedding_parameters]

    def load_checkpoint(self, checkpoint: str):
        """
        Load a model for inference with no optimizer state.
        """
        load_checkpoint(checkpoint, self, None)


class TrainableModel:
    """
    A model wrapped in a bunch of training utilities to enable continuous learning.
    """

    def __init__(self, model: nn.Module):
        self.model = model

        self.optimizer = AdamW(
            model.parameters(),
            lr=5e-5,
            betas=[0.9, 0.95],
            eps=1e-4,
            weight_decay=0.01,
        )

        self.grpo_optimizer = AdamW(
            model.parameters(),
            lr=5e-5,
            betas=[0.9, 0.95],
            eps=1e-4,
            weight_decay=0.01,
        )

    def do_simpo_step(
        self,
        prompt: list[int],
        positive: list[int],
        negative: list[int],
    ) -> tuple[float, float]:
        """
        Executes SimPO training loop on a single (prompt, positive, negative) triple.
        Aligns the model to respond more like the positive response example,
        and less like the negative.
        Args:
            prompt: List of token IDs for the prompt
            positive: List of token IDs for the positive
            negative: List of token IDs for the negative
        Returns:
            Two floats in a tuple, representing the change in likelihood of the
            positive and negative responses respectively.
        """
        # calculate_log_probs expects the same number of prompts in the batch dimension as responses.
        # We will stack positive on top of negative, for outputs and output lengths.
        prompt_tensor = torch.tensor(prompt, dtype=torch.int).expand(2, -1).to(self.model.device)
        prompt_length_tensor = torch.Tensor([len(prompt), len(prompt)]).to(self.model.device)

        # Responses can vary in length, pad with zeros.
        response_tensor = pad_sequence(
            [
                torch.tensor(positive, dtype=int),
                torch.tensor(negative, dtype=int),
            ],
            batch_first=True,
            padding_value=0,
        ).to(self.model.device)
        response_length_tensor = torch.tensor(
            [len(positive), len(negative)],
            dtype=int,
        ).to(self.model.device)

        per_token_log_probs, _ = calculate_model_log_probs(
            self.model,
            prompt_token_sequence=prompt_tensor,
            prompt_lengths=prompt_length_tensor,
            output_token_sequence=response_tensor,
            output_length=response_length_tensor,
        )
        log_probs = per_token_log_probs.sum(dim=-1)

        loss = calculate_simpo_loss(
            policy_positive_log_prob=log_probs[0],
            policy_negative_log_prob=log_probs[1],
            positive_length=prompt_length_tensor[0],
            negative_length=prompt_length_tensor[1],
        )

        # Take the step!
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Measure the step's effects.
        after_per_token_log_probs, _ = calculate_model_log_probs(
            self.model,
            prompt_token_sequence=prompt_tensor,
            prompt_lengths=prompt_length_tensor,
            output_token_sequence=response_tensor,
            output_length=response_length_tensor,
        )
        after_log_probs = after_per_token_log_probs.sum(dim=-1)
        before_probs = torch.exp(log_probs) * 100
        after_probs = torch.exp(after_log_probs) * 100
        print(tuple((after_probs - before_probs).tolist()))

        return tuple((after_probs - before_probs).tolist())

    def do_grpo_step(
        self,
        prompt: list[int],
        responses: list[list[int]],
        rewards: list[float],
        target_kl: float,
        max_steps: int = 100,
        clip_epsilon: float = 0.2,
        gradient_limit: float = 1.0,
    ) -> dict:
        """
        Executes GRPO training loop on a group of (prompt, response, reward) tuples,
        continuing until the response-prediction KL from the policy at the start
        of this update reaches target_kl. This is a stopping threshold, not a
        hard upper bound: the final optimizer step can overshoot it.

        Args:
            prompt: List of token IDs for the prompt.
            responses: List of response token ID sequences.
            rewards: List of reward scores for each response.
            target_kl: Target KL divergence to reach before stopping.
            max_steps: Safety cap on number of optimization steps.
            clip_epsilon: PPO-style clip range for importance ratios.
            gradient_limit: Maximum L2 norm for gradient clipping.

        Returns:
            Dict with losses, kl_values, steps_taken, and final_kl.
        """
        assert len(rewards) == len(responses)

        device = self.model.device
        group_size = len(rewards)
        prompt_len = len(prompt)

        # Make tensors of the inputs
        prompt_lengths = torch.tensor([prompt_len] * group_size, dtype=torch.long, device=device)
        prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device).expand(group_size, -1)
        response_lengths = torch.tensor([len(r) for r in responses], dtype=torch.long, device=device)
        response_tensor = pad_sequence(
            [torch.tensor(r, dtype=torch.long) for r in responses],
            batch_first=True,
            padding_value=0,
        ).to(device)

        # Build full sequences for KL computation
        full_sequences = torch.cat((prompt_tensor, response_tensor), dim=1)

        # Calculate advantages: mean-subtracted rewards
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
        advantages = rewards_tensor - rewards_tensor.mean()

        # Calculate original log probs (frozen reference) and reference logits for KL
        with torch.no_grad():
            generation_policy_log_probs, response_mask = calculate_model_log_probs(
                self.model,
                prompt_tensor,
                prompt_lengths,
                response_tensor,
                response_lengths,
            )
            # Logit j predicts token j+1. Reuse the policy loss's mask so KL
            # includes the first response prediction and excludes prompt,
            # padding, and the prediction after the final response token.
            before_logits = self.model(full_sequences)[:, :-1]
            before_log_probs_full = torch.log_softmax(before_logits, dim=-1)
            before_probs_full = torch.exp(before_log_probs_full)

        losses = []
        kl_values = []
        step = 0
        while step < max_steps:
            # Calculate per-token log probs for current model
            log_probs, response_mask = calculate_model_log_probs(
                self.model,
                prompt_tensor,
                prompt_lengths,
                response_tensor,
                response_lengths,
            )

            # Calculate GRPO loss (scalar)
            loss = calculate_grpo_loss(
                log_probs,
                generation_policy_log_probs,
                response_mask=response_mask,
                advantages=advantages,
                clip_epsilon=clip_epsilon,
            )

            # Reset gradients
            self.grpo_optimizer.zero_grad()

            # Backprop GRPO loss
            loss.backward()

            # Clip gradients
            clip_gradients(list(self.model.parameters()), max_l2_norm=gradient_limit)

            # Step optimizer
            self.grpo_optimizer.step()

            losses.append(loss.item())
            step += 1

            # Compute KL(old || current) over response-predicting positions.
            with torch.no_grad():
                after_logits = self.model(full_sequences)[:, :-1]
                after_log_probs_full = torch.log_softmax(after_logits, dim=-1)
                kl_per_position = (
                    before_probs_full * (before_log_probs_full - after_log_probs_full)
                ).sum(dim=-1)
                masked_kl = kl_per_position * response_mask
                kl = masked_kl.sum() / response_mask.sum().clamp(min=1)
                kl = kl.item()

            kl_values.append(kl)
            if kl >= target_kl:
                break

        return {
            "losses": losses,
            "kl_values": kl_values,
            "steps_taken": step,
            "final_kl": kl_values[-1] if kl_values else 0.0,
        }
