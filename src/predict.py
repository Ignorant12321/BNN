from __future__ import annotations

import numpy as np


def mc_predict(model, loader, device, mc_samples: int = 50) -> dict[str, np.ndarray]:
    import torch

    model.train()
    all_means = []
    all_log_vars = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            target = batch.pop("target")
            sample_means = []
            sample_vars = []
            for _ in range(mc_samples):
                mean, log_var = model(batch)
                sample_means.append(mean.detach().cpu().numpy())
                sample_vars.append(torch.exp(log_var).detach().cpu().numpy())
            all_means.append(np.stack(sample_means, axis=0))
            all_log_vars.append(np.stack(sample_vars, axis=0))
            targets.append(target.detach().cpu().numpy())

    mean_samples = np.concatenate(all_means, axis=1)
    aleatoric_vars = np.concatenate(all_log_vars, axis=1)
    targets_np = np.concatenate(targets, axis=0)
    mean = mean_samples.mean(axis=0)
    total_var = mean_samples.var(axis=0) + aleatoric_vars.mean(axis=0)
    return {
        "samples": mean_samples,
        "target": targets_np,
        "mean": mean,
        "std": np.sqrt(np.maximum(total_var, 1e-8)),
    }


def interval_from_samples(samples: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - level
    return np.quantile(samples, alpha / 2, axis=0), np.quantile(samples, 1 - alpha / 2, axis=0)
