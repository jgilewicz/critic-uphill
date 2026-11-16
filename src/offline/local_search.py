import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

from analysis.runs import load_spec
from common.checkpoint import BUFFER, load_nets
from offline.critic_degeneracy import (
    PERTURB,
    build_heads,
    check_reproduction,
    checkpoint_key,
    checkpoints,
    episodes_matrix,
)


def episode_seeds(source: dict[str, Any]) -> list[int]:
    # the source's truth seeds, then its validation seeds
    off, val = source["offline"], source["validation"]
    truth = [off["eval_seed"] + e for e in range(off["truth_episodes"])]
    fresh = [val["seed_base"] + e for e in range(val["episodes"])]
    return truth + fresh


def agreement(
    returns: np.ndarray,
    seeds: list[int],
    cell: dict[str, Any],
    vcell: dict[str, Any],
) -> dict[str, float]:
    # rerun vs the source: validation seeds ran on CPU there too (exact),
    # truth seeds on GPU (TF32 actions, chaotic dynamics: only correlated)
    n_truth = len(cell["actor"]["truth"])
    stored = np.array(
        [cell["actor"]["truth"]] + [r["truth"] for r in cell["perturbations"]]
    )
    col = {s: i for i, s in enumerate(seeds)}
    fresh = [col[s] for s in vcell["seeds"]]
    diffs = [np.abs(returns[0, fresh] - np.array(vcell["actor"]))]
    for c in vcell["candidates"]:
        diffs.append(np.abs(returns[c["row"] + 1, fresh] - c["returns"]))
    means = returns[:, :n_truth].mean(axis=1)
    return {
        "validation_max_abs_diff": float(np.max(np.concatenate(diffs))),
        "truth_mean_corr": float(np.corrcoef(means, stored.mean(1))[0, 1]),
    }


def episodes_checkpoint(
    source: dict[str, Any],
    path: Path,
    cell: dict[str, Any],
    vcell: dict[str, Any],
) -> dict[str, Any]:
    off = source["offline"]
    nets, meta = load_nets(path)
    key = jax.random.fold_in(checkpoint_key(off, meta), PERTURB)
    heads, levels = build_heads(
        nets.actor, cell["sigmas"], off["perturbations"], key
    )
    checked = (path / BUFFER).is_file()
    if checked:
        check_reproduction(off, path, heads, cell)
    seeds = episode_seeds(source)
    t0 = time.time()
    returns, lengths = episodes_matrix(
        meta["env"], nets.embedding, heads, seeds, off["horizon"]
    )
    rows = [
        {"level": int(lv), "returns": r.tolist(), "lengths": n.tolist()}
        for lv, r, n in zip(levels, returns, lengths, strict=True)
    ]
    return {
        "meta": meta,
        "sigmas": cell["sigmas"],
        "seeds": seeds,
        "reproduction_checked": checked,
        "agreement": agreement(returns, seeds, cell, vcell),
        "actor": rows[0],
        "perturbations": rows[1:],
        "seconds": time.time() - t0,
    }


def episodes(spec: dict[str, Any], root: Path, env: str) -> None:
    source = load_spec((spec["dir"] / spec["source"]).resolve())
    offline_dir = source["dir"] / source["analysis"]["offline_cache"] / env
    out = spec["dir"] / spec["analysis"]["episodes_cache"] / env
    out.mkdir(parents=True, exist_ok=True)
    for seed in source["seeds"]:
        for path in checkpoints(root, env, seed):
            name = f"s{seed}_{path.name}.json"
            dest = out / name
            if dest.is_file():
                print(f"skip {dest}: exists")
                continue
            cell_file = offline_dir / name
            vcell_file = offline_dir / f"validate_{name}"
            for f in (cell_file, vcell_file):
                if not f.is_file():
                    raise FileNotFoundError(
                        f"{f} missing. run critic_degeneracy evaluate and "
                        "validate for the source experiment first"
                    )
            result = episodes_checkpoint(
                source,
                path,
                json.loads(cell_file.read_text()),
                json.loads(vcell_file.read_text()),
            )
            dest.write_text(json.dumps(result))
            print(f"-> {dest} {result['agreement']} {result['seconds']:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="local-search-v1 offline data")
    parser.add_argument("command", choices=["episodes"])
    parser.add_argument("toml", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("env")
    args = parser.parse_args()
    episodes(load_spec(args.toml), args.checkpoint_root, args.env)


if __name__ == "__main__":
    main()
