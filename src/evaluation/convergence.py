from collections.abc import Iterable, Mapping

from src.utils.config import ConvergenceConfig


def summarize_convergence(
    rows: Iterable[Mapping[str, str]], config: ConvergenceConfig | None
) -> dict[str, str | int | float | None]:
    """Summarize the first sustained validation HardSum crossing from epoch logs."""
    if config is None:
        return {"status": "not_configured"}
    result = {
        "status": "not_reached",
        "metric": "val_hard_sum_accuracy",
        "hard_sum_threshold": config.hard_sum_threshold,
        "consecutive_epochs": config.consecutive_epochs,
        "streak_start_epoch": None,
        "epoch": None,
        "steps": None,
        "elapsed_seconds": None,
    }
    streak = 0
    for row in rows:
        if float(row["val_hard_sum_accuracy"]) >= config.hard_sum_threshold:
            streak += 1
        else:
            streak = 0
        if streak == config.consecutive_epochs:
            epoch = int(row["epoch"])
            return {
                **result,
                "status": "reached",
                "streak_start_epoch": epoch - streak + 1,
                "epoch": epoch,
                "steps": int(row["steps"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
            }
    return result
