from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="offline")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for one offline measurement of one checkpoint."""
    from common.checkpoint import load_checkpoint
    from config import OfflineConfig
    from offline import (
        ExtraMeasurement,
        OfflineSetup,
        PairMeasurement,
        SignalMeasurement,
    )

    raw = cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True))
    raw["sigmas"] = tuple(raw["sigmas"])
    raw["step_sigmas"] = tuple(raw["step_sigmas"])
    raw["pair_sigmas"] = tuple(raw["pair_sigmas"])
    offline_cfg = OfflineConfig(**raw)
    ckpt = load_checkpoint(Path(offline_cfg.checkpoint))
    setup = OfflineSetup(offline_cfg, ckpt)
    measurement = {
        "signals": SignalMeasurement,
        "extra": ExtraMeasurement,
        "pairs": PairMeasurement,
    }[offline_cfg.measurement](setup)
    result = measurement.run()
    result["config"] = raw
    out = Path(offline_cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"offline | {offline_cfg.measurement} | {out}")


if __name__ == "__main__":
    main()
