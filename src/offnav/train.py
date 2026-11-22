"""Offline training of BC and IQL on one data condition (fixed dataset, no environment interaction).

The only simulator use during a training run is the optional development evaluation enabled in
pilot configs, which is logged separately and never feeds data back into training.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .bc import BC
from .data import load_task_bundle
from .iql import IQL
from .networks import Normalizer, count_parameters
from .splits import indices_digest, load_manifest

ALGOS = {"bc": BC, "iql": IQL}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_determinism(seed: int, device: torch.device) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def run_paths(cfg: dict, rid: str) -> dict:
    return {
        "results": C.results_root(cfg) / "runs" / rid,
        "artifacts": C.artifacts_root(cfg) / "runs" / rid,
        "manifests": C.results_root(cfg) / "manifests",
    }


def fit_normalizer(obs: np.ndarray, std_floor: float) -> Normalizer:
    mean = obs.astype(np.float64).mean(axis=0)
    std = np.maximum(obs.astype(np.float64).std(axis=0), std_floor)
    return Normalizer(torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32))


def run_fingerprint(cfg: dict, maze: str, condition: str, method: str, seed_index: int,
                    manifest_digest: str, dataset_digest: str, version: int = C.FINGERPRINT_VERSION,
                    training_code_sha256: str | None = None) -> tuple[str, dict]:
    """Everything that determines the trained weights. A mismatch forbids resuming a checkpoint.

    version 1 (all runs made before 2026-11-22): configuration, seeds, manifest and dataset only.
    version 2: additionally the digest of the training code (config.TRAINING_CODE_FILES), so a
    change to the training code also invalidates checkpoints and completed runs. Version 1 content
    is unchanged, so legacy fingerprints can still be recomputed exactly.
    """
    content = {
        "maze": maze, "condition": condition, "method": method, "seed_index": seed_index,
        "seeds": C.seeds_for(cfg, seed_index), "manifest_sha256": manifest_digest,
        "dataset_sha256": dataset_digest, "training": cfg["training"],
        "method_cfg": cfg[method], "task": cfg["task"],
    }
    if version >= 2:
        content["fingerprint_version"] = version
        content["training_code_sha256"] = training_code_sha256 or C.training_code_digest()
    return C.fingerprint(content), content


class LegacyRunError(RuntimeError):
    """A completed version-1 run exists where the current code would produce a version-2 run."""


class EagerStepper:
    """Sample a minibatch uniformly with replacement from the condition's transitions and update."""

    def __init__(self, algo, data: dict, batch_size: int, gen: torch.Generator):
        self.algo, self.data, self.batch_size, self.gen = algo, data, batch_size, gen
        self.n = len(next(iter(data.values())))
        self.device = next(iter(data.values())).device

    def __call__(self) -> dict:
        idx = torch.randint(0, self.n, (self.batch_size,), device=self.device, generator=self.gen)
        return self.algo.update({k: v[idx] for k, v in self.data.items()})


class GraphStepper(EagerStepper):
    """Same computation as EagerStepper, with algo.update captured once into a CUDA graph.

    On Windows/WDDM a small IQL update is dominated by kernel-launch overhead; replaying a captured
    graph removes it (measured ~7x faster on the study machine). Index sampling stays outside the
    graph and uses the same generator calls as the eager path. Warm-up iterations needed before
    capture would change parameters and optimiser moments, so all of those tensors are snapshotted
    first and restored in place afterwards (the graph keeps pointers to these exact tensors).
    The LR scheduler runs outside the graph and writes into tensor learning rates.
    """

    def __init__(self, algo, data: dict, batch_size: int, gen: torch.Generator, warmup: int = 3):
        super().__init__(algo, data, batch_size, gen)
        opts = [o for o in algo.optimizers().values() if isinstance(o, torch.optim.Optimizer)]
        for opt in opts:
            for group in opt.param_groups:
                group["capturable"] = True
                if not torch.is_tensor(group["lr"]):
                    group["lr"] = torch.tensor(float(group["lr"]), device=self.device)
        self.static = {k: v[:batch_size].clone() for k, v in data.items()}
        _ensure_optimizer_state(opts)
        snapshot = _state_tensors(algo, opts)
        saved = [t.detach().clone() for t in snapshot]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                algo.update(self.static)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = algo.update(self.static)
        with torch.no_grad():
            for t, s in zip(_state_tensors(algo, opts), saved):
                t.copy_(s)
        torch.cuda.synchronize()

    def __call__(self) -> dict:
        idx = torch.randint(0, self.n, (self.batch_size,), device=self.device, generator=self.gen)
        for k, v in self.data.items():
            torch.index_select(v, 0, idx, out=self.static[k])
        self.graph.replay()
        return self.out


def _ensure_optimizer_state(opts) -> None:
    """Create Adam state without changing parameters: one step with zero gradients, then reset step=0.

    With zero gradients and zero moments, Adam's update is exactly zero. Existing (resumed) state is
    left untouched.
    """
    for opt in opts:
        params = [p for g in opt.param_groups for p in g["params"]]
        if all(p in opt.state and len(opt.state[p]) > 0 for p in params):
            continue
        for p in params:
            p.grad = torch.zeros_like(p)
        opt.step()
        for p in params:
            opt.state[p]["step"].zero_()


def _state_tensors(algo, opts) -> list[torch.Tensor]:
    ts = []
    for m in algo.modules().values():
        ts += list(m.parameters()) + list(m.buffers())
    for opt in opts:
        for st in opt.state.values():
            ts += [v for v in st.values() if torch.is_tensor(v)]
    return ts


def build_algo(method: str, cfg: dict, normalizer: Normalizer, init_seed: int, device: torch.device):
    torch.manual_seed(init_seed)
    hyper = dict(cfg["training"]["network"])
    hyper.update(cfg[method])
    return ALGOS[method](obs_dim=6, act_dim=2, cfg=hyper, normalizer=normalizer,
                         total_steps=int(cfg["training"]["steps"]), device=device)


def train_run(cfg: dict, maze: str, condition: str, method: str, seed_index: int,
              device_name: str | None = None, dev_eval_fn=None, force: bool = False,
              stop_after: int | None = None, accept_legacy: bool = False) -> dict:
    """Train one run to cfg['training']['steps'] updates, resuming a matching checkpoint if present.

    stop_after: optionally stop (with a checkpoint) once this many total updates are done; a later
    call resumes. Used to test that interrupted and uninterrupted runs give identical weights.
    accept_legacy: a completed version-1 run whose legacy fingerprint matches the current
    configuration, manifest and dataset is accepted as complete instead of raising LegacyRunError.
    """
    rid = C.run_id(maze, condition, method, seed_index)
    paths = run_paths(cfg, rid)
    paths["results"].mkdir(parents=True, exist_ok=True)
    paths["artifacts"].mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name or cfg["training"].get("device", "auto"))
    seeds = C.seeds_for(cfg, seed_index)

    bundle = load_task_bundle(maze)
    task = bundle["task"]
    idx = load_manifest(paths["manifests"] / f"{C.manifest_id(maze, condition, seed_index)}.npz")
    mdigest = indices_digest(idx)
    fp, fp_content = run_fingerprint(cfg, maze, condition, method, seed_index, mdigest, bundle["dataset_sha256"])
    legacy_fp, _ = run_fingerprint(cfg, maze, condition, method, seed_index, mdigest, bundle["dataset_sha256"], version=1)

    final_path = paths["artifacts"] / "final.pt"
    ckpt_path = paths["artifacts"] / "checkpoint.pt"
    if final_path.exists() and not force:
        saved = torch.load(final_path, map_location="cpu", weights_only=False)
        if saved["fingerprint"] == fp:
            return {"run_id": rid, "status": "already_complete", "fingerprint": fp}
        if saved["fingerprint"] == legacy_fp:
            if accept_legacy:
                return {"run_id": rid, "status": "legacy_complete", "fingerprint": legacy_fp}
            raise LegacyRunError(
                f"{rid}: completed legacy run (fingerprint version 1, no training-code digest). It matches the "
                "current configuration, manifest and dataset but cannot be checked against the current code. "
                "Pass --accept-legacy to reuse it, or train into empty OFFNAV_RESULTS/OFFNAV_ARTIFACTS folders.")
        raise RuntimeError(f"{final_path} has fingerprint {saved['fingerprint']} != {fp}; the configuration, data, "
                           "manifest or training code changed. Train into a fresh results location, or use "
                           "`offnav job --rerun-stale`, which sets the old run aside instead of overwriting it.")

    if not np.all(task.goal_consistent[idx]):
        raise RuntimeError("manifest contains goal-switch transitions")

    set_determinism(seeds["init"], device)
    tcfg = cfg["training"]
    total_steps = int(tcfg["steps"])
    batch_size = int(tcfg["batch_size"])
    normalizer = fit_normalizer(task.obs[idx], float(tcfg["obs_std_floor"]))
    algo = build_algo(method, cfg, normalizer, seeds["init"], device)

    data = {
        "obs": torch.as_tensor(task.obs[idx], device=device),
        "action": torch.as_tensor(task.action[idx], device=device),
        "next_obs": torch.as_tensor(task.next_obs[idx], device=device),
        "success": torch.as_tensor(task.success[idx], device=device),
        "mask": torch.as_tensor(task.mask[idx], device=device),
    }
    n_train = len(idx)
    gen = torch.Generator(device=device)
    gen.manual_seed(seeds["batch"])

    start_step, log_rows, dev_rows, elapsed_before = 0, [], [], 0.0
    if ckpt_path.exists() and not force:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck["fingerprint"] != fp:
            kind = "legacy (version-1) " if ck["fingerprint"] == legacy_fp else ""
            raise RuntimeError(f"stale {kind}checkpoint {ckpt_path} (fingerprint {ck['fingerprint']} != {fp})")
        _load_state(algo, ck, device)
        gen.set_state(ck["generator_state"].cpu())
        start_step, log_rows, dev_rows, elapsed_before = ck["step"], ck["log_rows"], ck["dev_rows"], ck["elapsed"]
        print(f"[{rid}] resuming from step {start_step}", flush=True)

    log_every = int(tcfg["log_every"])
    ckpt_every = int(tcfg["checkpoint_every"])
    dev_at = set(int(s) for s in cfg.get("dev_eval", {}).get("at_steps", [])) if dev_eval_fn else set()
    use_graph = device.type == "cuda" and bool(tcfg.get("cuda_graph", True))
    stepper = (GraphStepper if use_graph else EagerStepper)(algo, data, batch_size, gen)
    acc: dict[str, torch.Tensor] = {}
    t0 = time.perf_counter()
    dev_seconds = 0.0
    for step in range(start_step, total_steps):
        info = stepper()
        algo.actor_sched.step()
        for k, v in info.items():
            acc[k] = acc[k] + v if k in acc else v.clone()
        done = step + 1
        if done % log_every == 0:
            row = {"step": done}
            row.update({k: float(v) / log_every for k, v in acc.items()})
            row["actor_lr"] = float(algo.actor_opt.param_groups[0]["lr"])  # value, not the live tensor
            log_rows.append(row)
            acc = {}
        if done in dev_at:
            td = time.perf_counter()
            dev_rows.append({"step": done, **dev_eval_fn(algo)})
            dev_seconds += time.perf_counter() - td
        if (done % ckpt_every == 0 or done == stop_after) and done < total_steps:
            _save_ckpt(ckpt_path, algo, gen, done, fp, log_rows, dev_rows,
                       elapsed_before + time.perf_counter() - t0 - dev_seconds)
        if stop_after is not None and done == stop_after and done < total_steps:
            return {"run_id": rid, "status": "stopped", "step": done, "fingerprint": fp}
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = elapsed_before + time.perf_counter() - t0 - dev_seconds

    _save_ckpt(ckpt_path, algo, gen, total_steps, fp, log_rows, dev_rows, train_seconds)
    torch.save({"fingerprint": fp, "step": total_steps, "actor": algo.actor.state_dict(),
                "norm": algo.norm.state_dict(), "method": method,
                "network": cfg["training"]["network"]}, final_path)

    import pandas as pd
    pd.DataFrame(log_rows).to_csv(paths["results"] / "train_log.csv", index=False)
    if dev_rows:
        pd.DataFrame(dev_rows).to_csv(paths["results"] / "dev_eval.csv", index=False)
    resumed_or_new = "resumed" if start_step > 0 else "fresh"
    params = {name: count_parameters(m) for name, m in algo.modules().items() if name != "norm"}
    info = {
        "run_id": rid, "status": "trained", "start": resumed_or_new,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "config_path": cfg.get("_config_path"), "mode": cfg["mode"],
        "fingerprint": fp, "fingerprint_content": fp_content,
        "fingerprint_version": C.FINGERPRINT_VERSION, "training_code_sha256": fp_content["training_code_sha256"],
        "maze": maze, "condition": condition, "method": method, "seed_index": seed_index, "seeds": seeds,
        "dataset_id": bundle["dataset_id"], "dataset_sha256": bundle["dataset_sha256"],
        "manifest_sha256": mdigest, "n_train_transitions": n_train,
        "normalizer_mean": algo.norm.mean.cpu().tolist(), "normalizer_std": algo.norm.std.cpu().tolist(),
        "parameter_counts": params,
        "device": str(device), "train_seconds": train_seconds,
        "updates_per_second": total_steps / train_seconds if train_seconds > 0 else None,
        "dev_eval_seconds": dev_seconds,
        "code_revision": C.code_revision(), "environment": C.environment_info(),
    }
    C.write_json(paths["results"] / "run_info.json", info)
    return info


def _save_ckpt(path: Path, algo, gen, step, fp, log_rows, dev_rows, elapsed) -> None:
    state = {name: m.state_dict() for name, m in algo.modules().items()}
    state.update({name: o.state_dict() for name, o in algo.optimizers().items()})
    state.update({"step": step, "fingerprint": fp, "generator_state": gen.get_state(),
                  "torch_cpu_rng": torch.get_rng_state(), "log_rows": log_rows, "dev_rows": dev_rows,
                  "elapsed": elapsed})
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_state(algo, ck: dict, device) -> None:
    for name, m in algo.modules().items():
        m.load_state_dict(ck[name])
    for name, o in algo.optimizers().items():
        o.load_state_dict(ck[name])
    torch.set_rng_state(ck["torch_cpu_rng"].cpu())


def load_policy(cfg: dict, rid: str):
    """Load the final actor and normaliser of a run (CPU) and its fingerprint."""
    from .networks import GaussianActor
    path = run_paths(cfg, rid)["artifacts"] / "final.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    net = saved["network"]
    actor = GaussianActor(6, 2, tuple(net["hidden"]), net["log_std_min"], net["log_std_max"])
    actor.load_state_dict(saved["actor"])
    norm = Normalizer(torch.zeros(6), torch.ones(6))
    norm.load_state_dict(saved["norm"])
    return actor.eval(), norm.eval(), saved["fingerprint"]
