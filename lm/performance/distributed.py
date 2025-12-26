import os
import timeit

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

timer = timeit.default_timer


def config_rank(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(rank)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def measure_all_reduce(rank, world_size, bytes, num_warmups, num_iterations):
    config_rank(rank, world_size)
    print(f"Finished configuring rank {rank}")
    data_size = int(bytes / 4)

    tensor = torch.randn((data_size,), device=f"cuda:{rank}")
    # tensor = torch.randn((data_size,))
    times = []

    print(f"Allocated data of Size: {tensor.numel() * tensor.element_size()} bytes")

    for step in range(num_warmups):
        print(f"Running rank {rank} warmup step {step}")
        dist.all_reduce(tensor=tensor, async_op=False)
    print(f"Rank {rank} finished warmup")

    dist.barrier()
    torch.cuda.synchronize()

    for step in range(num_iterations):
        start = timer()
        dist.all_reduce(tensor=tensor, async_op=False)
        torch.cuda.synchronize()
        elapsed = timer() - start
        times.append(elapsed)
        print(f"Rank {rank} finished step {step} in {elapsed}s")

    times = torch.tensor(times, dtype=torch.float64, device=f"cuda:{rank}")
    # times = torch.tensor(times, dtype=torch.float64)
    gather_result = [torch.zeros_like(times) for _ in range(world_size)]

    dist.all_gather(gather_result, times)

    if rank == 0:
        durations = torch.stack(gather_result).cpu()

        rank_table = pd.DataFrame(
            {
                "Rank": range(world_size),
                "Mean (ms)": durations.mean(dim=1).mul(1_000).tolist(),
                "Std (ms)": durations.std(dim=1).mul(1_000).tolist(),
                "Min (ms)": durations.min(dim=1).values.mul(1_000).tolist(),
                "Max (ms)": durations.max(dim=1).values.mul(1_000).tolist(),
            },
        )
        print(
            rank_table.to_string(
                index=False,
                float_format=lambda value: f"{value:.3f}",
            ),
        )
        results = pd.DataFrame(
            [
                {
                    "rank": rank,
                    "iteration": iteration,
                    "duration_ms": durations[rank, iteration].item() * 1000,
                    "message_bytes": tensor.numel() * tensor.element_size(),
                    "world_size": world_size,
                }
                for rank in range(world_size)
                for iteration in range(num_iterations)
            ],
        )

        results.to_csv(
            "all_reduce.csv",
            mode="a",
            header=not os.path.exists("all_reduce.csv"),
            index=False,
        )

    dist.destroy_process_group()

    return


def run_benchmark(bytes: int, world_size: int, num_warmups: int, num_iterations: int):
    print(f"Running all-reduce benchmark on {bytes} bytes with world size {world_size}")
    mp.spawn(fn=measure_all_reduce, args=(world_size, bytes, num_warmups, num_iterations), nprocs=world_size, join=True)


if __name__ == "__main__":
    world_size = 2
    num_warmups = 5
    num_iterations = 10
    run_benchmark(1_000_000, world_size, num_warmups, num_iterations)
    run_benchmark(10_000_000, world_size, num_warmups, num_iterations)
    run_benchmark(100_000_000, world_size, num_warmups, num_iterations)
    run_benchmark(1_000_000_000, world_size, num_warmups, num_iterations)
