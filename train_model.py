import argparse
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy
import torch
import torch.nn as nn
from tqdm import tqdm

# wandb and torch.utils.tensorboard are imported lazily inside TrainingLogger so an
# embedded run (YouGPT) with both disabled doesn't require them to be installed.
from lm.model.model import TransformerLM
from lm.performance.reference.model import BasicsTransformerLM as ReferenceTransformerLM
from lm.performance.utils import estimate_mfu, synchronize_accelerator
from lm.tokenization.bpe import Tokenizer
from lm.training.loss.cross_entropy import cross_entropy, cross_entropy_masked
from lm.training.optimization.adamw import AdamW
from lm.training.utils.checkpointing import load_checkpoint, save_checkpoint
from lm.training.utils.data_batching import (
    ConversationBatchLoader,
    ResponseBatchLoader,
    load_batch,
)
from lm.training.utils.gradient_clipping import clip_gradients
from lm.training.utils.scheduler import learning_rate_scheduler


@dataclass
class TrainingConfig:
    """
    Defines a pre-training run.
    Defaults defined via argparse at instantiation of TrainingConfig.
    """

    # Model configs
    batch_size: int
    context_length: int
    d_model: int
    vocab_size: int
    num_heads: int
    num_layers: int
    d_ff: int
    rope_theta: int
    device: str
    dtype: torch.dtype

    # AdamW configs
    betas: tuple[float]
    eps: float
    weight_decay: float

    # Training run configs
    min_learning_rate: float
    learning_rate: float
    training_steps: int
    warmup_steps: int
    lr_schedule: str
    gradient_limit: float
    metrics_interval: int
    preload_batches: bool

    # How often to do certain things
    checkpoint_interval: int
    validation_interval: int
    mfu_interval: int

    # Data paths
    training_data_path: str
    validation_data_path: str
    vocab_path: str | None
    merges_path: str | None
    checkpoint_resume_path: str | None

    # Compile the model, only works on cuda
    compile: bool
    # Whether we should use a reference transformer implementation to sanity check ours.
    train_reference: bool

    # Logging configs
    disable_wandb: bool
    disable_tensorboard: bool
    run_name: str

    # Mixed precision training
    use_mixed_precision: bool = False

    # Optional source-aware profiling of this exact training loop.
    profile_dir: str | None = None
    profile_trigger_file: str | None = None
    profile_wait_steps: int = 5
    profile_warmup_steps: int = 5
    profile_active_steps: int = 20
    seed: int | None = None

    # Which batch loader to use: "conversation" (conversation-aligned, padded) or
    # "plain" (uniform random fixed-length windows over the token stream).
    loader: str = "conversation"
    # Optional second stage: switch a plain pretraining run to conversation-
    # aligned batches with loss on <|Me|> content plus conversation structure.
    finetune_start_step: int | None = None
    finetune_loader: str = "conversation"
    reaction_loss_weight: int = 1
    emoji_loss_weight: int = 1
    response_prefix_tokens: int = 0
    response_prefix_weight: int = 1


def fine_tune_token_loss_weights(config: TrainingConfig) -> dict[int, int]:
    """Build optional emphasis weights for rare Me reaction/emoji targets."""
    weights = {
        token_id: config.reaction_loss_weight
        for token_id in range(4, 10)
        if config.reaction_loss_weight != 1
    }
    if config.emoji_loss_weight == 1 or not config.vocab_path:
        return weights

    with open(config.vocab_path) as vocab_file:
        vocab = json.load(vocab_file)
    # IDs 0:10 are core/reaction tokens. Emoji specials follow them until
    # the first one-byte base token, matching MikeGPT's tokenizer builder.
    for token_id in range(10, config.vocab_size):
        encoded = vocab.get(str(token_id))
        if encoded is None or len(bytes.fromhex(encoded)) == 1:
            break
        weights[token_id] = config.emoji_loss_weight
    return weights


def train(config: TrainingConfig, step_callback=None):
    """
    Run a pre-training loop.

    step_callback: optional callable(step:int, step_state:dict) invoked after each
    step's metrics are logged. Lets an embedder (e.g. the YouGPT app) stream live
    loss without depending on wandb/tensorboard.
    """
    if config.seed is not None:
        numpy.random.seed(config.seed)
        torch.manual_seed(config.seed)

    if config.train_reference:
        model = ReferenceTransformerLM(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
            device=config.device,
            dtype=config.dtype,
        )

    else:
        model = TransformerLM(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
            device=config.device,
            dtype=config.dtype,
        )
        param_count = model.param_count()[1]
        print(f"Non-embedding param count: {param_count:,}")

    if config.compile:
        model = torch.compile(model)

    if config.device == "cuda":
        torch.set_float32_matmul_precision("high")

    model.to(config.device)

    optimizer = AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps,
    )

    LoaderClass = BatchLoader if config.loader == "plain" else ConversationBatchLoader

    loader_kwargs = {
        "file_path": config.training_data_path,
        "batch_size": config.batch_size,
        "context_length": config.context_length,
        "device": config.device,
    }
    if LoaderClass is BatchLoader and config.preload_batches:
        loader_kwargs["num_batches"] = config.training_steps
    training_data_loader = LoaderClass(**loader_kwargs)
    finetune_data_loader = None
    if config.finetune_start_step is not None:
        if config.loader != "plain":
            raise ValueError("--finetune-start-step requires --loader plain")
        if not 1 <= config.finetune_start_step <= config.training_steps + 1:
            raise ValueError(
                "--finetune-start-step must be between 1 and training_steps + 1"
            )
        fine_tune_weights = fine_tune_token_loss_weights(config)
        FineTuneLoader = (
            ResponseBatchLoader
            if config.finetune_loader == "response"
            else ConversationBatchLoader
        )
        finetune_data_loader = FineTuneLoader(
            file_path=config.training_data_path,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=config.device,
            seed=config.seed,
            token_loss_weights=fine_tune_weights,
            **(
                {
                    "response_prefix_tokens": config.response_prefix_tokens,
                    "response_prefix_weight": config.response_prefix_weight,
                }
                if FineTuneLoader is ResponseBatchLoader
                else {}
            ),
        )

    # Validation is optional — an embedded run may only have training data.
    if config.validation_data_path:
        validation_batch_loader = LoaderClass(
            file_path=config.validation_data_path,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=config.device,
        )
        masked_validation_batch_loader = (
            ResponseBatchLoader(
                file_path=config.validation_data_path,
                batch_size=config.batch_size,
                context_length=config.context_length,
                device=config.device,
                seed=None if config.seed is None else config.seed + 1,
                token_loss_weights=fine_tune_weights,
            )
            if finetune_data_loader is not None
            else None
        )
    else:
        validation_batch_loader = None
        masked_validation_batch_loader = None

    # Logged and used for MFU calculations.
    param_count = model.param_count()[1]

    logger = TrainingLogger(config=config, param_count=param_count)
    checkpoint_meta = vocab_fingerprint(config.vocab_path, config.vocab_size)
    checkpoint_meta.update(
        {
            "d_model": config.d_model,
            "d_ff": config.d_ff,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "context_length": config.context_length,
            "rope_theta": config.rope_theta,
        }
    )
    checkpointer = Checkpointer(meta=checkpoint_meta)
    if config.checkpoint_resume_path:
        checkpointer.load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=config.checkpoint_resume_path,
        )

    if config.vocab_path is not None and config.merges_path is not None:
        tokenizer = Tokenizer.from_files(
            vocab_filepath=config.vocab_path,
            merges_filepath=config.merges_path,
        )
    else:
        tokenizer = None

    t0 = time.perf_counter()
    # Last finite loss is retained for display if a non-finite batch is skipped.
    last_good_loss = None
    recovered_steps = 0
    clipped_steps = 0
    metrics_samples = 0

    device_type = config.device.split(":")[0]
    if config.use_mixed_precision:
        # MPS GradScaler is not usable in the project's PyTorch 2.6 build: its
        # unscale path attempts an unsupported float64 MPS operation. BF16 has
        # FP32's exponent range, so it does not need loss scaling. CUDA keeps
        # the conventional FP16 + device-aware GradScaler path.
        amp_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
        scaler = torch.amp.GradScaler("cuda") if device_type == "cuda" else None
        print(
            f"Mixed precision enabled: {device_type} autocast {amp_dtype}, "
            f"loss scaling {'enabled' if scaler is not None else 'not required'}"
        )
    else:
        amp_dtype = None
        scaler = None

    def amp_context():
        if not config.use_mixed_precision:
            return nullcontext()
        return torch.amp.autocast(device_type=device_type, dtype=amp_dtype)

    range_context = torch.profiler.record_function if config.profile_dir else lambda _: nullcontext()
    profiler = None
    profiler_steps_remaining = 0
    profile_start_step = None
    if config.profile_dir:
        os.makedirs(config.profile_dir, exist_ok=True)

        def save_profile(prof):
            end_step = (
                profile_start_step + config.profile_active_steps - 1
                if profile_start_step is not None
                else prof.step_num
            )
            trace_path = os.path.join(
                config.profile_dir,
                f"training-steps-{profile_start_step or 1}-{end_step}.json",
            )
            summary_path = trace_path.removesuffix(".json") + "-summary.txt"
            prof.export_chrome_trace(trace_path)
            with open(summary_path, "w") as summary_file:
                summary_file.write(
                    prof.key_averages(group_by_stack_n=1).table(
                        sort_by="self_cpu_time_total",
                        row_limit=100,
                    )
                )
            print(f"PROFILE trace {trace_path}", flush=True)

        def start_profiler(wait, warmup, active):
            active_profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(
                    wait=wait,
                    warmup=warmup,
                    active=active,
                    repeat=1,
                ),
                on_trace_ready=save_profile,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            active_profiler.start()
            return active_profiler

        # CLI profiling without a trigger retains the original start-of-run
        # behavior. MikeGPT supplies a trigger file and starts captures live.
        if config.profile_trigger_file is None:
            profile_start_step = 1 + config.profile_wait_steps + config.profile_warmup_steps
            profiler_steps_remaining = (
                config.profile_wait_steps
                + config.profile_warmup_steps
                + config.profile_active_steps
            )
            profiler = start_profiler(
                config.profile_wait_steps,
                config.profile_warmup_steps,
                config.profile_active_steps,
            )
            print(
                "Source profiling enabled for the exact training loop: "
                f"{config.profile_dir}",
                flush=True,
            )

    # When embedded (a step_callback is driving progress), suppress the tqdm bar so its
    # carriage-return output doesn't interleave with the machine-readable progress lines.
    timing_window_started = time.perf_counter()
    for step in tqdm(range(1, config.training_steps + 1), disable=step_callback is not None):
        collect_metrics = (
            step == 1
            or step % config.metrics_interval == 0
            or step % 100 == 0
            or step % config.validation_interval == 0
            or step % config.checkpoint_interval == 0
            or step % config.mfu_interval == 0
        )
        if (
            profiler is None
            and config.profile_trigger_file
            and os.path.exists(config.profile_trigger_file)
        ):
            try:
                os.unlink(config.profile_trigger_file)
            except FileNotFoundError:
                pass
            profile_start_step = step
            profiler_steps_remaining = config.profile_active_steps
            profiler = start_profiler(0, 0, config.profile_active_steps)
            print(
                f"PROFILE capturing steps {step}-"
                f"{step + config.profile_active_steps - 1}",
                flush=True,
            )

        # Put the model in training mode.
        model.train()

        # Determine the currrent learning rate.
        if config.lr_schedule == "constant":
            if config.warmup_steps > 0 and step < config.warmup_steps:
                lr = config.learning_rate * step / config.warmup_steps
            else:
                lr = config.learning_rate
        else:
            lr = learning_rate_scheduler(
                current_step=step,
                max_rate=config.learning_rate,
                min_rate=config.min_learning_rate,
                cosine_annealing_iterations=config.training_steps,
                warmup_iterations=config.warmup_steps,
            )
        optimizer.set_learning_rate(lr)

        fine_tuning = (
            finetune_data_loader is not None
            and step >= config.finetune_start_step
        )

        # Get a batch of data using the active phase's loader.
        with range_context("train/data_loader"):
            active_loader = (
                finetune_data_loader if fine_tuning else training_data_loader
            )
            loaded_batch = active_loader.load_batch()
            train, label = loaded_batch[:2]
            loss_mask = loaded_batch[2] if len(loaded_batch) == 3 else None

        # Print de-tokenized first sequence of the batch
        if tokenizer is not None:
            first_sequence_ids = train[0].tolist()
            decoded_text = tokenizer.decode(first_sequence_ids)
            tqdm.write(f"Step {step} sample: {decoded_text}")

        with range_context("train/forward"):
            with amp_context():
                output = model(train)
        with range_context("train/loss"):
            with amp_context():
                loss = (
                    cross_entropy_masked(output, label, train, loss_mask)
                    if fine_tuning or config.loader == "conversation"
                    else cross_entropy(output, label)
                )

        # Backpropogate and calculate gradients.
        optimizer.zero_grad(set_to_none=True)

        with range_context("train/backward"):
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # Unscale gradients before clipping if using mixed precision
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)

        # Clip the gradients to some max total l2 norm.
        with range_context("train/gradient_clipping"):
            gradient_norm, was_clipped = clip_gradients(
                model.parameters(),
                config.gradient_limit,
                synchronize=collect_metrics,
            )
        if collect_metrics:
            metrics_samples += 1
            if was_clipped is True:
                clipped_steps += 1

        loss_val = loss.item() if collect_metrics else None
        # gradient_norm is already reduced to a synchronized Python scalar by
        # clipping, so it doubles as the non-finite-gradient check. Avoid
        # scanning every parameter from Python (one MPS synchronization per
        # tensor), scanning weights again, and copying the entire model into a
        # rollback snapshot on every healthy step.
        step_healthy = (
            not collect_metrics
            or (math.isfinite(loss_val) and math.isfinite(gradient_norm))
        )
        if step_healthy:
            with range_context("train/optimizer"):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            if loss_val is not None:
                last_good_loss = loss_val
        else:
            optimizer.zero_grad(set_to_none=True)
            recovered_steps += 1
            if last_good_loss is not None:
                loss_val = last_good_loss
            print(f"[stability] step {step}: non-finite loss/grad — skipped update "
                  f"(total skipped: {recovered_steps})", flush=True)

        step_state = None
        if collect_metrics:
            step_state = {
                "gradient_norm": gradient_norm,
                "gradient_clipped": int(was_clipped),
                "gradient_clip_rate": clipped_steps / metrics_samples,
                "phase": "finetune" if fine_tuning else "pretrain",
            }
        if step % config.mfu_interval == 0:
            synchronize_accelerator(config.device)
            t1 = time.perf_counter()
            dt = t1 - t0
            t0 = t1

            token_rate = (config.batch_size * config.context_length * config.mfu_interval) / dt
            print(f"Token rate: {token_rate}/s")

            mfu = estimate_mfu(num_params=param_count, batch_size=config.batch_size, model=model, dt=dt / config.mfu_interval)
            print(f"MFU: {mfu}")
            step_state["mfu"] = mfu

        if step % 100 == 0:
            # Measure the actual training steps, excluding periodic validation
            # and checkpoint I/O. Synchronize only at the reporting boundary so
            # queued MPS optimizer work is included without taxing every step.
            synchronize_accelerator(config.device)
            now = time.perf_counter()
            step_state["avg_step_time_ms"] = (
                (now - timing_window_started) * 1000.0 / 100
            )

        if step % config.checkpoint_interval == 0:
            with range_context("train/checkpoint"):
                checkpointer.save_checkpoint(model, optimizer, step, config.run_name)
        if validation_batch_loader is not None and step % config.validation_interval == 0:
            with range_context("train/validation"):
                validation_loss = calculate_validation_loss(
                    model=model,
                    loader=validation_batch_loader,
                    amp_context=amp_context,
                )
            step_state["val_loss"] = validation_loss.item()
            if masked_validation_batch_loader is not None:
                with range_context("train/masked_validation"):
                    masked_validation_loss = calculate_validation_loss(
                        model=model,
                        loader=masked_validation_batch_loader,
                        amp_context=amp_context,
                        masked=True,
                    )
                step_state["masked_val_loss"] = masked_validation_loss.item()

        # loss_val was computed above (and, for a skipped step, set to the last good loss).
        if collect_metrics:
            step_state["loss"] = loss_val
            try:
                # perplexity is display-only; a transient loss spike (huge/inf loss)
                # must never crash the run via math.exp overflow.
                step_state["perplexity"] = math.exp(loss_val)
            except (OverflowError, ValueError):
                step_state["perplexity"] = float("inf")
            logger.log(step_state=step_state, step=step)

            if step_callback is not None:
                step_callback(step, step_state)

        if step % 100 == 0:
            timing_window_started = time.perf_counter()

        if profiler is not None:
            profiler.step()
            profiler_steps_remaining -= 1
            if profiler_steps_remaining == 0:
                profiler.stop()
                profiler = None

    if profiler is not None:
        profiler.stop()

    return


class TrainingLogger:
    def __init__(self, config, param_count):
        self.run_name = config.run_name
        self.disable_wandb = config.disable_wandb
        self.disable_tensorboard = config.disable_tensorboard

        current_time = datetime.now().strftime("%-m-%-d-%y_%H:%M")
        config_dict = asdict(config)

        config_dict["non_embedding_params"] = param_count
        if not self.disable_wandb:
            import wandb

            self.wandb_handler = wandb.init(
                name=f"{self.run_name}-{current_time}",
                entity="michael-ferris-1928-michael-ferris",
                project="Artisinal-LLM",
                config=config_dict,
            )

        if not self.disable_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard_writer = SummaryWriter(f"runs/{self.run_name}-{current_time}")

    def log(self, step_state, step):
        if not self.disable_wandb:
            self.wandb_handler.log(step_state, step=step)
        if not self.disable_tensorboard:
            for key, value in step_state.items():
                self.tensorboard_writer.add_scalar(key, value, step)


def vocab_fingerprint(vocab_path, vocab_size):
    meta = {"vocab_size": vocab_size}
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            meta["vocab_sha256"] = hashlib.sha256(f.read()).hexdigest()
    return meta


def validate_encoded_vocab(training_data_path, vocab_path):
    """Fail before training when encoded tokens and tokenizer do not match."""
    if not vocab_path:
        return
    fingerprint_path = os.path.join(
        os.path.dirname(os.path.abspath(training_data_path)),
        "vocab_sha.txt",
    )
    if not os.path.exists(fingerprint_path):
        return
    with open(fingerprint_path) as fingerprint_file:
        expected = fingerprint_file.read().strip()
    with open(vocab_path, "rb") as vocab_file:
        actual = hashlib.sha256(vocab_file.read()).hexdigest()
    if actual != expected:
        raise ValueError(
            "Tokenizer mismatch: training data was encoded with vocab "
            f"{expected[:12]}..., but --vocab-path is {actual[:12]}.... "
            "Use the vocab_sha.txt-matched tokenizer or rebuild the encoded data."
        )


class Checkpointer:
    def __init__(self, meta=None):
        self.start_time = datetime.now().strftime("%-m-%-d-%y_%H:%M")
        self.meta = meta

    def save_checkpoint(
        self,
        model,
        optimizer,
        iteration,
        run_name,
    ):
        os.makedirs(os.path.join("checkpoints", f"{run_name}-{self.start_time}"), exist_ok=True)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=iteration,
            out=os.path.join("checkpoints", f"{run_name}-{self.start_time}", f"checkpoint_step_{iteration}"),
            meta=self.meta,
        )

    def load_checkpoint(self, model, optimizer, checkpoint_path):
        load_checkpoint(src=checkpoint_path, model=model, optimizer=optimizer)


class BatchLoader:
    def __init__(
        self,
        file_path: str,
        batch_size: int,
        context_length: int,
        device: torch.device,
        num_batches: int | None = None,
    ):
        self.file = numpy.load(file_path, mmap_mode="r")
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.preloaded_batches = None
        self.batch_index = 0
        if num_batches:
            max_index = len(self.file) - context_length - 1
            starts = numpy.random.randint(
                0, max_index + 1, size=(num_batches, batch_size)
            )
            offsets = numpy.arange(context_length + 1)
            windows = numpy.asarray(
                self.file[starts[:, :, None] + offsets[None, None, :]]
            )
            self.preloaded_batches = torch.from_numpy(windows).to(
                device=device, dtype=torch.long
            )

    def load_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.preloaded_batches is not None:
            batch = self.preloaded_batches[self.batch_index]
            self.batch_index += 1
            return batch[:, :-1], batch[:, 1:]
        return load_batch(self.file, batch_size=self.batch_size, context_length=self.context_length, device=self.device)


def calculate_validation_loss(
    model: nn.Module,
    loader: BatchLoader,
    amp_context=nullcontext,
    masked: bool = False,
) -> float:
    model.eval()
    with torch.no_grad():
        # One random batch is far too noisy to select checkpoints. Average enough
        # batches to cover 16 * batch_size contexts at every validation interval.
        losses = []
        for _ in range(16):
            loaded_batch = loader.load_batch()
            validation_data, validation_label = loaded_batch[:2]
            validation_loss_mask = (
                loaded_batch[2] if len(loaded_batch) == 3 else None
            )
            with amp_context():
                validation_output = model(validation_data)
                losses.append(
                    cross_entropy_masked(
                        validation_output,
                        validation_label,
                        validation_data,
                        validation_loss_mask,
                    )
                    if masked
                    else cross_entropy(validation_output, validation_label)
                )
        return torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser(description="Train LLM")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of batches per training step")
    parser.add_argument("--context-length", type=int, default=256, help="length of model's context length")
    parser.add_argument("--d-model", type=int, default=512, help="Dimension of model's embeddings")
    parser.add_argument("--vocab-size", type=int, default=32_000, help="Number of tokens in the model's vocab")
    parser.add_argument("--num-heads", type=int, default=2, help="Heads per attention mechanism in the model")
    parser.add_argument("--num-layers", type=int, default=16, help="Number of transformer layers in the model")
    parser.add_argument("--d-ff", type=int, default=1344, help="Dimension of the feedforward networks in the model")
    parser.add_argument("--rope-theta", type=int, default=10000, help="Constant used in RoPE rotation calculations")
    parser.add_argument("--min-learning-rate", type=float, default=3e-5, help="Slowest learning rate")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Nominal learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay rate for AdamW optimization")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 constant for AdamW Optimization")
    parser.add_argument("--beta2", type=float, default=0.95, help="Beta2 constant for AdamW Optimization")
    parser.add_argument("--epsilon", type=float, default=1e-5, help="Epsilon cosntant for AdamW Optimization")
    parser.add_argument("--training-steps", type=int, default=10_000, help="Number of training iterations to run")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Steps before specified learning rate reached")
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine", help="Learning-rate schedule after warmup")
    parser.add_argument("--gradient-limit", type=float, default=1.0, help="L2 norm above which gradients will be clipped")
    parser.add_argument("--metrics-interval", type=int, default=1, help="Synchronize and report scalar metrics every n steps")
    parser.add_argument("--preload-batches", action="store_true", help="Stage all plain-loader batches on the accelerator before training")
    parser.add_argument("--training-data-path", type=str, required=True, help="Path to training data (.npy)")
    parser.add_argument("--validation-data-path", type=str, required=False, help="Path to validation data (.npy)")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="Save checkpoint every n training steps")
    parser.add_argument("--validation-interval", type=int, default=100, help="Calculate validation loss every n training steps")
    parser.add_argument("--mfu-interval", type=int, default=100, help="Interval at which to calculate MFU")
    parser.add_argument("--device", type=str, default="mps", help="Device on which to train model")
    parser.add_argument("--dtype", type=torch.dtype, default=torch.float32, help="Data type for model weights")
    parser.add_argument("--mixed-precision", dest="use_mixed_precision", action="store_true", help="Use mixed precision (BF16 on MPS/CPU, FP16 with loss scaling on CUDA)")
    parser.add_argument("--profile-dir", type=str, default=None, help="Write a source-aware PyTorch trace of this exact training loop to this directory")
    parser.add_argument("--profile-trigger-file", type=str, default=None, help="Start a live profile when this file appears; usable repeatedly")
    parser.add_argument("--profile-wait-steps", type=int, default=5, help="Unrecorded profiler steps before warmup")
    parser.add_argument("--profile-warmup-steps", type=int, default=5, help="Profiler warmup steps")
    parser.add_argument("--profile-active-steps", type=int, default=20, help="Training steps captured in each profiler trace")
    parser.add_argument("--seed", type=int, default=None, help="Seed model initialization and batch sampling for paired experiments")
    parser.add_argument("--compile", dest="compile", action="store_true", help="Compile the model before training")
    parser.add_argument("--train-reference", dest="train_reference", action="store_true", help="Train reference model instead")
    parser.add_argument("--run-name", type=str, help="Name of the training run as it will appear in WandB and Tensorboard")
    parser.add_argument("--disable-wandb", dest="disable_wandb", action="store_true", help="Turn off W&B logging")
    parser.add_argument("--disable-tensorboard", dest="disable_tensorboard", action="store_true", help="Turn off Tensorboard logging")
    parser.add_argument("--vocab-path", type=str, default=None, help="Path to .json vocab file for example training sequences")
    parser.add_argument("--merges-path", type=str, default=None, help="Path to .pkl merge file for example training sequences")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path of checkpoint from which to resume training")
    parser.add_argument("--progress-stdout", dest="progress_stdout", action="store_true", help="Print a machine-readable 'PROGRESS step N total loss L' line each step for embedders to parse")
    parser.add_argument("--loader", type=str, default="conversation", choices=["conversation", "plain"], help="Batch loader: 'conversation' (conversation-aligned, padded) or 'plain' (uniform random fixed-length windows)")
    parser.add_argument("--finetune-start-step", type=int, default=None, help="At this step, switch a plain-loader run to conversation batches and selective Me-content plus structural loss")
    parser.add_argument("--finetune-loader", choices=("conversation", "response"), default="conversation", help="Fine-tuning sampler: arbitrary conversation chunks or response-anchored windows")
    parser.add_argument("--reaction-loss-weight", type=int, default=1, help="Fine-tuning loss weight for Me reaction tokens (IDs 4-9)")
    parser.add_argument("--emoji-loss-weight", type=int, default=1, help="Fine-tuning loss weight for Me emoji special tokens")
    parser.add_argument("--response-prefix-tokens", type=int, default=0, help="Number of initial content tokens after each Me marker to emphasize with ResponseBatchLoader")
    parser.add_argument("--response-prefix-weight", type=int, default=1, help="Loss multiplier for emphasized response-prefix tokens")
    parser.set_defaults(
        train_reference=False,
        compile=False,
        use_mixed_precision=False,
        disable_wandb=False,
        disable_tensorboard=False,
    )

    args = parser.parse_args()
    validate_encoded_vocab(args.training_data_path, args.vocab_path)

    config = TrainingConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        d_model=args.d_model,
        vocab_size=args.vocab_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        min_learning_rate=args.min_learning_rate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=[args.beta1, args.beta2],
        eps=args.epsilon,
        training_steps=args.training_steps,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        gradient_limit=args.gradient_limit,
        metrics_interval=args.metrics_interval,
        preload_batches=args.preload_batches,
        checkpoint_interval=args.checkpoint_interval,
        validation_interval=args.validation_interval,
        mfu_interval=args.mfu_interval,
        training_data_path=args.training_data_path,
        validation_data_path=args.validation_data_path,
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        checkpoint_resume_path=args.resume_from_checkpoint,
        device=args.device,
        dtype=args.dtype,
        use_mixed_precision=args.use_mixed_precision,
        profile_dir=args.profile_dir,
        profile_trigger_file=args.profile_trigger_file,
        profile_wait_steps=args.profile_wait_steps,
        profile_warmup_steps=args.profile_warmup_steps,
        profile_active_steps=args.profile_active_steps,
        seed=args.seed,
        compile=args.compile,
        train_reference=args.train_reference,
        run_name=args.run_name,
        disable_wandb=args.disable_wandb,
        disable_tensorboard=args.disable_tensorboard,
        loader=args.loader,
        finetune_start_step=args.finetune_start_step,
        finetune_loader=args.finetune_loader,
        reaction_loss_weight=args.reaction_loss_weight,
        emoji_loss_weight=args.emoji_loss_weight,
        response_prefix_tokens=args.response_prefix_tokens,
        response_prefix_weight=args.response_prefix_weight,
    )

    print(f"Training with config: {config}")

    step_callback = None
    if args.progress_stdout:
        total_steps = config.training_steps

        def step_callback(step, step_state):
            loss = step_state.get("loss")
            msg = f"PROGRESS step {step} {total_steps} loss {loss:.6f}"
            if "val_loss" in step_state:
                msg += f" val {step_state['val_loss']:.6f}"
            msg += (
                f" grad {step_state['gradient_norm']:.6f}"
                f" clipped {step_state['gradient_clipped']}"
                f" clip_rate {step_state['gradient_clip_rate']:.6f}"
            )
            if "avg_step_time_ms" in step_state:
                msg += f" avg_step_ms {step_state['avg_step_time_ms']:.3f}"
            if "masked_val_loss" in step_state:
                msg += f" masked_val {step_state['masked_val_loss']:.6f}"
            msg += f" phase {step_state.get('phase', 'pretrain')}"
            print(msg, flush=True)

    train(config=config, step_callback=step_callback)


if __name__ == "__main__":
    main()
