"""Structured prefix prompt builder for RIS-LLM.

Constructs a 4-component prompt that integrates:
1. DataDesc: dataset domain and statistical characteristics
2. TaskInst: prediction objective, horizon, evaluation focus
3. PatchInfo: mean, variance, extrema per patch
4. PatchLabels: semantic annotations from TSS module

"""

import torch


def build_structured_prompt(
    data_desc: str,
    task_inst: str,
    patch_stats: torch.Tensor,
    patch_labels: list,
    seq_len: int,
    pred_len: int,
) -> list:
    """Build structured prefix prompts for a batch.

    Args:
        data_desc: textual dataset description
        task_inst: task instruction text
        patch_stats: [B, N, 4*D] per-patch (mean, var, min, max)
        patch_labels: list of lists, [B, N] or empty strings
        seq_len: input sequence length
        pred_len: prediction horizon

    Returns:
        prompts: list of B prompt strings
    """
    B = patch_stats.shape[0]
    N = patch_stats.shape[1]
    D = patch_stats.shape[2] // 4  # each has 4 stats

    prompts = []
    for b in range(B):
        parts = []

        # 1. Data Description
        parts.append(f"<|DataDesc|> {data_desc}")

        # 2. Task Instruction
        parts.append(
            f"<|TaskInst|> Predict the next {pred_len} steps "
            f"given the previous {seq_len} steps of electricity price data. "
            f"Focus on capturing volatility patterns and structural trends."
        )

        # 3. Patch Information
        patch_info_parts = ["<|PatchInfo|>"]
        for n in range(N):
            stats = patch_stats[b, n]  # [4*D]
            stat_strs = []
            for d in range(D):
                mean_val = stats[0 * D + d].item()
                var_val = stats[1 * D + d].item()
                min_val = stats[2 * D + d].item()
                max_val = stats[3 * D + d].item()
                stat_strs.append(
                    f"d{d}:mean={mean_val:.3f},var={var_val:.3f},"
                    f"min={min_val:.3f},max={max_val:.3f}"
                )
            patch_info_parts.append(f"Patch{n}: {'; '.join(stat_strs)}")
        parts.append(' '.join(patch_info_parts))

        # 4. Patch Labels (semantic annotations from TSS)
        label_parts = ["<|PatchLabels|>"]
        if patch_labels and len(patch_labels) > b:
            for n, label in enumerate(patch_labels[b]):
                if label:
                    label_parts.append(f"Patch{n}: {label}")
                else:
                    label_parts.append(f"Patch{n}: Normal")
        else:
            label_parts.append("No specific labels")
        parts.append(' '.join(label_parts))

        prompt = ' '.join(parts)
        prompts.append(prompt)

    return prompts


def load_data_description(dataset_name: str) -> str:
    """Load domain-specific data description.

    Args:
        dataset_name: name of the dataset (e.g., 'EPF', 'ETTh1')

    Returns:
        data_desc: textual description string
    """
    descriptions = {
        'EPF': (
            "The Spanish electricity price dataset contains hourly day-ahead and "
            "real-time electricity prices for the Spanish electricity market. "
            "It includes generation features (fossil, renewable, other), "
            "consumption (total electricity demand), and weather features "
            "(temperature, humidity, pressure, wind speed, wind direction, "
            "precipitation, cloud cover). Prices exhibit high volatility due to "
            "renewable integration and demand variability."
        ),
        'ETTh1': (
            "The ETT (Electricity Transformer Temperature) dataset records "
            "hourly electricity transformer temperature and load data. "
            "It includes measurements of oil temperature and various load levels."
        ),
        'ETTh2': (
            "The ETT (Electricity Transformer Temperature) dataset records "
            "hourly electricity transformer temperature and load data."
        ),
        'ETTm1': (
            "The ETT minute-level dataset records electricity transformer "
            "temperature and load data at 15-minute intervals."
        ),
        'ETTm2': (
            "The ETT minute-level dataset records electricity transformer "
            "temperature and load data at 15-minute intervals."
        ),
        'ECL': (
            "The ECL (Electricity Consuming Load) dataset records hourly "
            "electricity consumption data."
        ),
        'Weather': (
            "The Weather dataset contains hourly weather measurements "
            "including temperature, humidity, wind speed, and pressure."
        ),
        'Traffic': (
            "The Traffic dataset records hourly road traffic occupancy rates."
        ),
    }
    return descriptions.get(
        dataset_name,
        f"The {dataset_name} time series dataset. "
        f"Forecast the next steps given historical observations."
    )
