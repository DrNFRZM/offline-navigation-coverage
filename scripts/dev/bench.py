import time, torch, numpy as np
from offnav.bc import BC
from offnav.iql import IQL
from offnav.networks import Normalizer
cfg = {"hidden": [256,256], "log_std_min": -5.0, "log_std_max": 2.0, "actor_lr": 3e-4, "critic_lr": 3e-4, "value_lr": 3e-4,
       "discount": 0.99, "expectile": 0.9, "beta": 10.0, "max_adv_weight": 100.0, "polyak": 0.005, "reward_shift": -1.0}
for dev in ["cuda", "cpu"]:
    d = torch.device(dev)
    N = 900_000
    data = {"obs": torch.randn(N, 6, device=d), "action": torch.rand(N, 2, device=d)*2-1, "next_obs": torch.randn(N, 6, device=d),
            "success": (torch.rand(N, device=d) < 0.01).float(), "mask": torch.ones(N, device=d)}
    for name, cls in [("bc", BC), ("iql", IQL)]:
        torch.manual_seed(0)
        algo = cls(6, 2, cfg, Normalizer(torch.zeros(6), torch.ones(6)), 10000, d)
        g = torch.Generator(device=d); g.manual_seed(0)
        def step():
            idx = torch.randint(0, N, (256,), device=d, generator=g)
            return algo.update({k: v[idx] for k, v in data.items()})
        for _ in range(50): step()
        if dev == "cuda": torch.cuda.synchronize()
        n = 1000 if dev == "cuda" else 300
        t = time.perf_counter()
        for _ in range(n): step()
        if dev == "cuda": torch.cuda.synchronize()
        print(dev, name, f"{n/(time.perf_counter()-t):.0f} updates/s", "threads", torch.get_num_threads())
