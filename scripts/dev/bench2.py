import time, sys, torch
from offnav.bc import BC
from offnav.iql import IQL
from offnav.networks import Normalizer
cfg = {"hidden": [256,256], "log_std_min": -5.0, "log_std_max": 2.0, "actor_lr": 3e-4, "critic_lr": 3e-4, "value_lr": 3e-4,
       "discount": 0.99, "expectile": 0.9, "beta": 10.0, "max_adv_weight": 100.0, "polyak": 0.005, "reward_shift": -1.0}
mode = sys.argv[1]
if mode == "cpu1":
    torch.set_num_threads(1)
    d = torch.device("cpu")
    N = 900_000
    data = {"obs": torch.randn(N, 6), "action": torch.rand(N, 2)*2-1, "next_obs": torch.randn(N, 6),
            "success": (torch.rand(N) < 0.01).float(), "mask": torch.ones(N)}
    for name, cls in [("bc", BC), ("iql", IQL)]:
        algo = cls(6, 2, cfg, Normalizer(torch.zeros(6), torch.ones(6)), 10000, d)
        g = torch.Generator(); g.manual_seed(0)
        def step():
            idx = torch.randint(0, N, (256,), generator=g)
            return algo.update({k: v[idx] for k, v in data.items()})
        for _ in range(20): step()
        t = time.perf_counter(); n = 200
        for _ in range(n): step()
        print("cpu1", name, f"{n/(time.perf_counter()-t):.0f} updates/s", flush=True)
elif mode == "graph":
    d = torch.device("cuda")
    N = 900_000
    data = {"obs": torch.randn(N, 6, device=d), "action": torch.rand(N, 2, device=d)*2-1, "next_obs": torch.randn(N, 6, device=d),
            "success": (torch.rand(N, device=d) < 0.01).float(), "mask": torch.ones(N, device=d)}
    for name, cls in [("bc", BC), ("iql", IQL)]:
        algo = cls(6, 2, cfg, Normalizer(torch.zeros(6), torch.ones(6)), 10000, d)
        # make optimizers capturable with tensor lr
        for opt in [o for o in algo.optimizers().values() if isinstance(o, torch.optim.Optimizer)]:
            for gr in opt.param_groups:
                gr["capturable"] = True
                gr["lr"] = torch.tensor(gr["lr"], device=d)
        static = {k: v[:256].clone() for k, v in data.items()}
        sched = algo.actor_sched
        algo.actor_sched = type("NoSched", (), {"step": lambda self: None})()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5): algo.update(static)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = algo.update(static)
        g = torch.Generator(device=d); g.manual_seed(0)
        def step():
            idx = torch.randint(0, N, (256,), device=d, generator=g)
            for k, v in data.items(): torch.index_select(v, 0, idx, out=static[k])
            graph.replay(); sched.step()
        for _ in range(20): step()
        torch.cuda.synchronize(); t = time.perf_counter(); n = 2000
        for _ in range(n): step()
        torch.cuda.synchronize()
        print("graph", name, f"{n/(time.perf_counter()-t):.0f} updates/s", "lr", float(algo.actor_opt.param_groups[0]["lr"]), flush=True)
