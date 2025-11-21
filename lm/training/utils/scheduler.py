import math


def learning_rate_scheduler(current_step: int, max_rate: float, min_rate: float, warmup_iterations: int, cosine_annealing_iterations: int) -> float:
    """
    Implements cosine-annealed learning rate schedule.
    """
    if current_step < warmup_iterations:
        return (current_step / warmup_iterations) * max_rate

    if current_step > cosine_annealing_iterations:
        return min_rate

    proportion_of_annealing_complete = (current_step - warmup_iterations) / (cosine_annealing_iterations - warmup_iterations)

    cos_annealing_term = 1 + math.cos(proportion_of_annealing_complete * math.pi)

    return min_rate + (cos_annealing_term * (max_rate - min_rate) / 2)
