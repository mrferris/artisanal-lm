import argparse
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb
from lm.model.model import TransformerLM
from lm.performance.reference.model import BasicsTransformerLM as ReferenceTransformerLM
from lm.performance.utils import estimate_mfu, synchronize_accelerator
from lm.tokenization.bpe import Tokenizer
from lm.training.loss.cross_entropy import cross_entropy
from lm.training.optimization.adamw import AdamW
from lm.training.utils.checkpointing import save_checkpoint
from lm.training.utils.data_batching import load_batch
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
    gradient_limit: float

    # How often to do certain things
    checkpoint_interval: int
    validation_interval: int
    mfu_interval: int

    # Data paths
    training_data_path: str
    validation_data_path: str
    vocab_path: str | None
    merges_path: str | None

    # Compile the model, only works on cuda
    compile: bool
    # Whether we should use a reference transformer implementation to sanity check ours.
    train_reference: bool

    # Logging configs
    disable_wandb: bool
    disable_tensorboard: bool
    run_name: str


def train(config: TrainingConfig):
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

    training_data_loader = BatchLoader(
        file_path=config.training_data_path,
        batch_size=config.batch_size,
        context_length=config.context_length,
        device=config.device,
    )

    validation_batch_loader = BatchLoader(
        file_path=config.validation_data_path,
        batch_size=config.batch_size,
        context_length=config.context_length,
        device=config.device,
    )

    # Logged and used for MFU calculations.
    param_count = model.param_count()[1]

    logger = TrainingLogger(config=config, param_count=param_count)
    checkpointer = Checkpointer()

    if config.vocab_path is not None and config.merges_path is not None:
        tokenizer = Tokenizer.from_files(
            vocab_filepath=config.vocab_path,
            merges_filepath=config.merges_path,
        )
    else:
        tokenizer = None

    t0 = time.time()
    for step in tqdm(range(1, config.training_steps + 1)):
        # Put the model in training mode.
        model.train()

        # Determine the currrent learning rate.
        lr = learning_rate_scheduler(
            current_step=step,
            max_rate=config.learning_rate,
            min_rate=config.min_learning_rate,
            cosine_annealing_iterations=config.training_steps,
            warmup_iterations=config.warmup_steps,
        )
        optimizer.set_learning_rate(lr)

        # Get a batch of data using the data loader
        train, label = training_data_loader.load_batch()

        # Print de-tokenized first sequence of the batch
        if tokenizer is not None:
            first_sequence_ids = train[0].tolist()
            decoded_text = tokenizer.decode(first_sequence_ids)
            tqdm.write(f"Step {step} sample: {decoded_text}")

        output = model(train)
        loss = cross_entropy(output, label)

        # Backpropogate and calculate gradients.
        optimizer.zero_grad()
        loss.backward()

        # Clip the gradients to some max total l2 norm.
        clip_gradients(model.parameters(), config.gradient_limit)

        # Optimize the gradients.
        optimizer.step()

        step_state = {}
        if step % config.mfu_interval == 0:
            synchronize_accelerator(config.device)
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            token_rate = (config.batch_size * config.context_length * config.mfu_interval) / dt
            print(f"Token rate: {token_rate}/s")

            mfu = estimate_mfu(num_params=param_count, batch_size=config.batch_size, model=model, dt=dt / config.mfu_interval)
            print(f"MFU: {mfu}")
            step_state["mfu"] = mfu

        if step % config.checkpoint_interval == 0:
            checkpointer.save_checkpoint(model, optimizer, step)
        if step % config.validation_interval == 0:
            validation_loss = calculate_validation_loss(
                model=model,
                loader=validation_batch_loader,
            )
            step_state["val_loss"] = validation_loss.item()

        step_state["loss"] = loss.item()
        step_state["perplexity"] = math.exp(loss.item())
        logger.log(step_state=step_state, step=step)

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
            self.wandb_handler = wandb.init(
                name=f"{self.run_name}-{current_time}",
                entity="michael-ferris-1928-michael-ferris",
                project="Artisinal-LLM",
                config=config_dict,
            )

        if not self.disable_tensorboard:
            self.tensorboard_writer = SummaryWriter(f"runs/{self.run_name}-{current_time}")

    def log(self, step_state, step):
        if not self.disable_wandb:
            self.wandb_handler.log(step_state, step=step)
        if not self.disable_tensorboard:
            for key, value in step_state.items():
                self.tensorboard_writer.add_scalar(key, value, step)


class Checkpointer:
    def __init__(self):
        self.start_time = datetime.now().strftime("%-m-%-d-%y_%H:%M")
        os.makedirs(os.path.join("checkpoints", self.start_time), exist_ok=True)

    def save_checkpoint(
        self,
        model,
        optimizer,
        iteration,
    ):
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=iteration,
            out=os.path.join("checkpoints", self.start_time, f"checkpoint_step_{iteration}"),
        )


class BatchLoader:
    def __init__(
        self,
        file_path: str,
        batch_size: int,
        context_length: int,
        device: torch.device,
    ):
        self.file = numpy.memmap(file_path, dtype=numpy.uint16, mode="r")
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def load_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return load_batch(self.file, batch_size=self.batch_size, context_length=self.context_length, device=self.device)


def calculate_validation_loss(model: nn.Module, loader: BatchLoader) -> float:
    model.eval()
    with torch.no_grad():
        validation_data, validation_label = loader.load_batch()
        validation_output = model(validation_data)

        validation_loss = cross_entropy(validation_output, validation_label)

        return validation_loss


def main():
    parser = argparse.ArgumentParser(description="Train LLM")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of batches per training step")
    parser.add_argument("--context-length", type=int, default=256, help="length of model's context length")
    parser.add_argument("--d-model", type=int, default=512, help="Dimension of model's embeddings")
    parser.add_argument("--vocab-size", type=int, default=32_000, help="Number of tokens in the model's vocab")
    parser.add_argument("--num-heads", type=int, default=16, help="Heads per attention mechanism in the model")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers in the model")
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
    parser.add_argument("--gradient-limit", type=int, default=1.0, help="L2 gabove which will be clipped")
    parser.add_argument("--training-data-path", type=str, required=True, help="Path to training data (.npy)")
    parser.add_argument("--validation-data-path", type=str, required=False, help="Path to validation data (.npy)")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="Save checkpoint every n training steps")
    parser.add_argument("--validation-interval", type=int, default=100, help="Calculate validation loss every n training steps")
    parser.add_argument("--mfu-interval", type=int, default=100, help="Interval at which to calculate MFU")
    parser.add_argument("--device", type=str, default="mps", help="Device on which to train model")
    parser.add_argument("--dtype", type=torch.dtype, default=torch.float32, help="Data type for model weights")
    parser.add_argument("--compile", dest="compile", action="store_true", help="Compile the model before training")
    parser.add_argument("--train-reference", dest="train_reference", action="store_true", help="Train reference model instead")
    parser.add_argument("--run-name", type=str, help="Name of the training run as it will appear in WandB and Tensorboard")
    parser.add_argument("--disable-wandb", dest="disable_wandb", action="store_true", help="Turn off W&B logging")
    parser.add_argument("--disable-tensorboard", dest="disable_tensorboard", action="store_true", help="Turn off Tensorboard logging")
    parser.add_argument("--vocab-path", type=str, default=None, help="Path to .json vocab file for example training sequences")
    parser.add_argument("--merges-path", type=str, default=None, help="Path to .pkl merge file for example training sequences")
    parser.set_defaults(
        train_reference=False,
        compile=False,
        disable_wandb=False,
        disable_tensorboard=False,
    )

    args = parser.parse_args()

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
        gradient_limit=args.gradient_limit,
        checkpoint_interval=args.checkpoint_interval,
        validation_interval=args.validation_interval,
        mfu_interval=args.mfu_interval,
        training_data_path=args.training_data_path,
        validation_data_path=args.validation_data_path,
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        device=args.device,
        dtype=args.dtype,
        compile=args.compile,
        train_reference=args.train_reference,
        run_name=args.run_name,
        disable_wandb=args.disable_wandb,
        disable_tensorboard=args.disable_tensorboard,
    )

    print(f"Training with config: {config}")
    train(config=config)


if __name__ == "__main__":
    main()
