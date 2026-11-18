from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint delegating to runner.run_training."""
    from runner import run_training

    run_training(cfg)


if __name__ == "__main__":
    main()
