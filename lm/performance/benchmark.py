import argparse
import statistics
import timeit
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from reference.model import BasicsTransformerLM as ReferenceTransformerLM

from lm.model.components.attention import MultiHeadSelfAttention, Rope
from lm.model.components.transformer import TransformerLM
from lm.performance.utils import synchronize_accelerator
from lm.training.loss.cross_entropy import cross_entropy
from lm.training.optimization.adamw import AdamW


@dataclass
class BenchmarkConfig:
    """
    Configures benchmarking and the model parameters to be benchmarked.
    """

    warmup_steps: int
    benchmark_steps: int
    forward_only: bool

    batch_size: int
    d_model: int
    vocab_size: int
    context_length: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: int
    device: torch.device
    compile: bool
    reference: bool
    optimization: bool
    autocast: bool
    attention_only: bool


def benchmark(config: BenchmarkConfig):
    """
    Runs warmup steps and then times model forward (and optionally backward) passes
    on random data.
    """
    begin_memory_profiling(config)
    print("=======================================================")
    print("Running benchmarks: ")
    print(f"warmup_steps={config.warmup_steps}, benchmark_steps={config.benchmark_steps}, forward_only={config.forward_only}")
    print(f"attention_only={config.attention_only}, optimization={config.optimization},             ")
    print("Model hyperparameters:")
    print(
        f"  d_model={config.d_model}, vocab_size={config.vocab_size}, context_length={config.context_length}\n"
        f"  num_layers={config.num_layers}, num_heads={config.num_heads}, d_ff={config.d_ff}, rope_theta={config.rope_theta}\n",
    )
    print(f"device={config.device}, compile={config.compile}, batch_size={config.batch_size}")

    if config.reference:
        model = ReferenceTransformerLM(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
        )
    else:
        if config.attention_only:
            model = MultiHeadSelfAttention(
                d_model=config.d_model,
                num_heads=1,
                rope=Rope(
                    theta=config.rope_theta,
                    d_k=config.d_model,
                    max_seq_len=config.context_length,
                ),
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
            )
            param_count = model.param_count()[1]
            print(f"Non-embedding param count: {param_count:,}")

    if config.device == "cuda":
        torch.set_float32_matmul_precision("high")

    if config.compile and config.device == "cuda":
        model = torch.compile(model)

    if config.optimization:
        optimizer = AdamW(
            params=model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
            betas=[0.9, 0.95],
            eps=1e-5,
        )
    else:
        optimizer = None

    if config.forward_only:
        model.eval()
    else:
        model.train()

    if config.attention_only:
        input = torch.randn(size=(config.context_length, config.d_model))
        desired_output = torch.arange(config.context_length)
    else:
        input = torch.randint(low=0, high=config.vocab_size - 1, size=(config.batch_size, config.context_length))
        desired_output = torch.randint(low=0, high=config.vocab_size - 1, size=(config.batch_size, config.context_length))

    model.to(config.device)

    input = input.to(config.device)
    desired_output = desired_output.to(config.device)

    # Warmup steps
    for _ in range(config.warmup_steps):
        model_step(
            model,
            input,
            desired_output,
            config.forward_only,
            optimizer,
            config.device,
        )

    # Benchmarking
    times = timeit.repeat(
        lambda: model_step(
            model,
            input,
            desired_output,
            config.forward_only,
            optimizer,
            config.device,
        ),
        number=1,
        repeat=config.benchmark_steps,
    )

    end_memory_profiling(config)

    mean = statistics.mean(times)
    stdev = statistics.stdev(times)

    print(f"Times: {times}")
    print(f"Mean: {mean} seconds")
    print(f"Std dev: {stdev} seconds")
    if config.device == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        print(f"Peak allocated memory: {peak_allocated / 1024**2:.2f} MiB")
        print(f"Peak reserved memory:  {peak_reserved / 1024**2:.2f} MiB")

    print("=======================================================")


def model_step(model, input, desired_output, forward_only, optimizer, device):
    with optionally_autocast(config.autocast):
        if forward_only:
            with torch.no_grad():
                output = model(input)
        else:
            output = model(input)
            loss = cross_entropy(output, desired_output)
            loss.backward()
            if optimizer is not None:
                optimizer.step()

        synchronize_accelerator(device)


def begin_memory_profiling(config):
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history(
            stacks="all",
            max_entries=1000000,
        )


def end_memory_profiling(config):
    if config.device == "cuda":
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)


def optionally_autocast(autocast: bool):
    return torch.autocast("cuda", dtype=torch.float16) if autocast else nullcontext()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLM")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of batches per training step")
    parser.add_argument("--context-length", type=int, default=256, help="length of model's context length")
    parser.add_argument("--d-model", type=int, default=512, help="Dimension of model's embeddings")
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Number of tokens in the model's vocab")
    parser.add_argument("--num-heads", type=int, default=16, help="Heads per attention mechanism in the model")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers in the model")
    parser.add_argument("--d-ff", type=int, default=1344, help="Dimension of the feedforward networks in the model")
    parser.add_argument("--rope-theta", type=int, default=10_000, help="Constant used in RoPE rotation calculations")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Steps before beginning benchamrk measurements")
    parser.add_argument("--benchmark-steps", type=int, default=10, help="Number of steps to benchmark")
    parser.add_argument("--forward-only", dest="forward_only", action="store_true", help="Benchmark forward passes only")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run benchmarks")
    parser.add_argument("--no-compile", dest="compile", action="store_false", help="Disable torch.compile")
    parser.add_argument("--reference", dest="reference", action="store_true", help="Run reference model instead of artisinal")
    parser.add_argument("--optimization", dest="optimization", action="store_true", help="Benchmark optimization step")
    parser.add_argument("--autocast", dest="autocast", action="store_true", help="Autocast model to FP16")
    parser.add_argument("--attention-only", dest="attention_only", action="store_true", help="Benchmark attention mechanism only")
    parser.set_defaults(compile=True, forward_only=False, reference=False, optimization=False, autocast=False, attention_only=False)
    args = parser.parse_args()

    config = BenchmarkConfig(
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        forward_only=args.forward_only,
        batch_size=args.batch_size,
        d_model=args.d_model,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        compile=args.compile,
        reference=args.reference,
        optimization=args.optimization,
        autocast=args.autocast,
        attention_only=args.attention_only,
    )

    benchmark(config=config)
