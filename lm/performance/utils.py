import torch


def synchronize_accelerator(device: str):
    """
    For properly measuring execution time,
    call `synchronize` on the device on which the model is running.
    """
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def estimate_mfu(num_params: int, batch_size: int, model: torch.nn.Module, dt: float) -> float:
    """
    Calculates Model Flops Utilization
    Flops per token: 6*N + 12*L*H*Q*T
    From PaLM: https://arxiv.org/pdf/2204.02311
    """
    num_layers = model.num_layers
    num_heads = model.num_heads
    head_dim = model.d_model // num_heads
    seq_len = model.context_length
    flops_actual = (num_params * 6) + (12 * num_layers * num_heads * head_dim * seq_len)
    flops_actual *= batch_size
    flops_actual = flops_actual / dt

    expected_h100_flops = {torch.bfloat16: 1979e12, torch.float32: 67e12}

    expected_m3_max_flops = {torch.float32: 14.2e12, torch.float16: 28.4e12}

    expected_flops = {"cuda": expected_h100_flops, "mps": expected_m3_max_flops}

    device = model.device
    dtype = model.dtype

    flops_expected = expected_flops[device][dtype]

    mfu = (flops_actual / flops_expected) * 100.0 * 100.0

    print(f"Flops expected: {flops_expected:,}")
    print(f"Flops actual:   {flops_actual:,}")
    print(f"MFU:            {mfu}")

    return mfu
