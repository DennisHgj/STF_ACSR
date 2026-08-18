import os
from glob import glob
import torch


def average_checkpoints(last):
    avg = None
    for path in last:
        states = torch.load(path, map_location=lambda storage, loc: storage)[
            "state_dict"
        ]
        states = {k[6:]: v for k, v in states.items() if k.startswith("model.")}
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            if avg[k].is_floating_point():
                avg[k] /= len(last)
            else:
                avg[k] //= len(last)
    return avg


def ensemble(args, count=3):
    experiment_dir = os.path.join(args.exp_dir, args.exp_name)
    candidates = sorted(
        glob(os.path.join(experiment_dir, "epoch=*.ckpt")),
        key=os.path.getmtime,
    )
    last = candidates[-int(count):]
    if not last:
        raise FileNotFoundError(
            f"No epoch checkpoints were found in {experiment_dir}."
        )
    model_path = os.path.join(
        experiment_dir, f"model_avg_{len(last)}.pth"
    )
    torch.save(average_checkpoints(last), model_path)
    return model_path
