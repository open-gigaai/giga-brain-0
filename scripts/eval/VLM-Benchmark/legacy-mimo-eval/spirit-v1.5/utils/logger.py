# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# Released by Spirit AI Team.
# ==============================================================================

import json
from pathlib import Path

import wandb


class Logger:
    def __init__(self, config, rank: int):
        self.rank = rank
        self.run = None

        if rank == 0:
            self.run = wandb.init(
                project=config.wandb_project,
                mode=config.wandb_mode,
                config=self._config_to_dict(config),
            )
            print(f"WandB run: {self.run.name}")

    def log(self, metrics: dict, step: int):
        if self.rank == 0 and self.run is not None:
            self.run.log(metrics, step=step)

    def print(self, msg: str):
        if self.rank == 0:
            print(msg)

    def save_config(self, config, path: str):
        if self.rank == 0:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            config_dict = self._config_to_dict(config)
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)

            print(f"Saved config to {path}")

    def finish(self):
        if self.rank == 0 and self.run is not None:
            self.run.finish()

    @staticmethod
    def _config_to_dict(config) -> dict:
        from dataclasses import asdict
        return asdict(config)
    