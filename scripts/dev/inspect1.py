import os, json, numpy as np, minari
for did in ["D4RL/pointmaze/umaze-v2", "D4RL/pointmaze/medium-v2"]:
    ds = minari.load_dataset(did)
    print("=====", did)
    print("total_episodes", ds.total_episodes, "total_steps", ds.total_steps)
    md = ds.storage.metadata
    print({k: (v if len(str(v)) < 300 else str(v)[:300]) for k, v in md.items() if k not in ("observation_space","action_space")})
    print("env_spec", ds.env_spec)
    print("eval_env_spec", ds.spec.eval_env_spec if hasattr(ds.spec,'eval_env_spec') else None)
    eps = list(ds.iterate_episodes(episode_indices=[0,1,2]))
    for ep in eps:
        print("ep", ep.id, "T", len(ep.actions), "obs lens", {k: v.shape for k, v in ep.observations.items()})
        print("  infos keys", {k: np.asarray(v).shape for k, v in ep.infos.items()})
        print("  rewards nonzero idx", np.flatnonzero(ep.rewards), "term", np.flatnonzero(ep.terminations), "trunc", np.flatnonzero(ep.truncations))
        o = ep.observations
        print("  dg first/2nd/last", o["desired_goal"][0], o["desired_goal"][1], o["desired_goal"][-1])
        print("  obs first", o["observation"][0], "last", o["observation"][-1])
        inf = ep.infos
        if "goal" in inf: print("  info goal first/2nd/last", inf["goal"][0], inf["goal"][1], inf["goal"][-1])
        if "success" in inf: print("  info success idx", np.flatnonzero(inf["success"]))
