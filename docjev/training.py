"""Run preparation → frozen reference → FSDP RLCD → fixed-cohort evaluation."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from .io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "source", "image-root", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a fresh output directory; individual stage commands support resume")
    if not args.model.is_dir() or not args.source.is_file() or not args.image_root.is_dir():
        parser.error("Model, source and image root must exist")
    config = json.loads(args.config.read_text())
    if config["epochs"] != 1:
        parser.error("The validated pipeline currently supports exactly one epoch")
    import torch

    world = config["world_size"]
    if torch.cuda.device_count() != world:
        parser.error(f"Expose exactly {world} GPUs via CUDA_VISIBLE_DEVICES")
    args.out.mkdir(parents=True)
    runtime_config = args.out / "config.json"
    shutil.copyfile(args.config, runtime_config)
    data, reference, train, after = [
        args.out / name for name in ("data", "reference", "train", "after")
    ]
    commands = [
        [
            sys.executable,
            "-m",
            "docjev.rlcd.prepare_infinity",
            "--source",
            str(args.source),
            "--image-root",
            str(args.image_root),
            "--model",
            str(args.model),
            "--config",
            str(runtime_config),
            "--out",
            str(data),
        ],
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world}",
            "-m",
            "docjev.rlcd.parallel_score",
            "--model",
            str(args.model),
            "--data",
            str(data),
            "--config",
            str(runtime_config),
            "--out",
            str(reference),
            "--phase",
            "reference",
        ],
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world}",
            "-m",
            "docjev.rlcd.train",
            "--model",
            str(args.model),
            "--data",
            str(data),
            "--references",
            str(reference),
            "--config",
            str(runtime_config),
            "--out",
            str(train),
        ],
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world}",
            "-m",
            "docjev.rlcd.parallel_score",
            "--model",
            str(train / "checkpoint-final"),
            "--data",
            str(data),
            "--config",
            str(runtime_config),
            "--out",
            str(after),
            "--phase",
            "after",
            "--references",
            str(reference),
        ],
    ]
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "2")
    for index, command in enumerate(commands):
        print(f"Stage {index + 1}/{len(commands)}", flush=True)
        with (args.out / f"stage-{index + 1}.log").open("w") as log:
            subprocess.run(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
            )
    before = json.loads((reference / "metrics.json").read_text())["metrics"]["dev"]
    final = json.loads((after / "metrics.json").read_text())["metrics"]["dev"]
    write_json(
        args.out / "comparison.json",
        {
            "before": before,
            "after": final,
            "accuracy_delta_percentage_points": 100
            * (final["original_order"]["accuracy"] - before["original_order"]["accuracy"]),
            "checkpoint": "train/checkpoint-final",
            "validation_used_for_optimizer": False,
        },
    )
    (args.out / "_SUCCESS").write_text("All four stages completed successfully\n")
    print(json.dumps({"status": "complete", "output": str(args.out)}))


if __name__ == "__main__":
    main()
