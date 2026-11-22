"""Maze geometry and the goal-conditioned task constructed from the recorded experience.

Geometry follows gymnasium_robotics.envs.maze (maze_v4.py): cell (row, col) has centre
    x = (col + 0.5) * s - n_cols * s / 2,   y = n_rows * s / 2 - (row + 0.5) * s
with maze_size_scaling s = 1 for PointMaze. Row 0 is the top (largest y).

Task (see docs/data.md): an episodic goal-reaching task. The input to every network is
(x, y, vx, vy, gx, gy). A transition reaches its goal when the successor position is within
GOAL_RADIUS of the goal, which is exactly the environment's sparse reward. Reaching the goal
terminates the episode (bootstrap mask 0); all other transitions bootstrap (mask 1),
including segment ends that the dataset marks as truncations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

GOAL_RADIUS = 0.45  # gymnasium_robotics MazeEnv.compute_reward, sparse variant

ENV_IDS = {"umaze": "PointMaze_UMaze-v3", "medium": "PointMaze_Medium-v3"}
DATASET_IDS = {"umaze": "D4RL/pointmaze/umaze-v2", "medium": "D4RL/pointmaze/medium-v2"}

# Copied from gymnasium_robotics/envs/maze/maps.py (Apache-2.0) so that geometry tests do not
# need the simulator; tests check that these match the installed package.
U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]
MEDIUM_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]
MAPS = {"umaze": U_MAZE, "medium": MEDIUM_MAZE}

Cell = tuple[int, int]


class MazeGeometry:
    """Free-cell graph of a maze map with metric conversions and shortest-path statistics."""

    def __init__(self, maze_map, scaling: float = 1.0):
        # 'g', 'r', 'c' markers denote free cells; 1 denotes a wall.
        self.grid = np.array([[1 if v == 1 else 0 for v in row] for row in maze_map], dtype=np.int64)
        self.n_rows, self.n_cols = self.grid.shape
        self.scaling = float(scaling)
        self.x_center = self.n_cols * self.scaling / 2.0
        self.y_center = self.n_rows * self.scaling / 2.0
        self.free_cells: list[Cell] = [(r, c) for r in range(self.n_rows) for c in range(self.n_cols)
                                       if self.grid[r, c] == 0]
        self.index = {cell: i for i, cell in enumerate(self.free_cells)}
        self.n_free = len(self.free_cells)
        self._all_pairs()

    @classmethod
    def for_maze(cls, maze: str) -> "MazeGeometry":
        return cls(MAPS[maze])

    # ------------------------------------------------------------------ coordinates
    def cell_to_xy(self, cell: Cell) -> np.ndarray:
        r, c = cell
        return np.array([(c + 0.5) * self.scaling - self.x_center,
                         self.y_center - (r + 0.5) * self.scaling])

    def xy_to_rowcol(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=np.float64)
        col = np.floor((xy[..., 0] + self.x_center) / self.scaling).astype(np.int64)
        row = np.floor((self.y_center - xy[..., 1]) / self.scaling).astype(np.int64)
        return row, col

    def cell_index(self, xy: np.ndarray) -> np.ndarray:
        """Index into free_cells for each position, -1 if the position lies in a wall cell or outside."""
        row, col = self.xy_to_rowcol(xy)
        out = np.full(row.shape, -1, dtype=np.int64)
        inside = (row >= 0) & (row < self.n_rows) & (col >= 0) & (col < self.n_cols)
        lut = np.full((self.n_rows, self.n_cols), -1, dtype=np.int64)
        for cell, i in self.index.items():
            lut[cell] = i
        out[inside] = lut[row[inside], col[inside]]
        return out

    def in_cell(self, xy: np.ndarray, cell: Cell) -> np.ndarray:
        row, col = self.xy_to_rowcol(xy)
        return (row == cell[0]) & (col == cell[1])

    def wall_cells(self) -> list[Cell]:
        return [(r, c) for r in range(self.n_rows) for c in range(self.n_cols) if self.grid[r, c] == 1]

    def extent(self) -> tuple[float, float, float, float]:
        return (-self.x_center, self.x_center, -self.y_center, self.y_center)

    # ------------------------------------------------------------------ graph
    def neighbors(self, cell: Cell) -> list[Cell]:
        r, c = cell
        out = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.n_rows and 0 <= cc < self.n_cols and self.grid[rr, cc] == 0:
                out.append((rr, cc))
        return out

    def _all_pairs(self) -> None:
        """BFS from every free cell: distances (in cells) and numbers of shortest paths."""
        n = self.n_free
        self.dist = np.full((n, n), -1, dtype=np.int64)
        self.npaths = np.zeros((n, n), dtype=np.int64)
        for s, src in enumerate(self.free_cells):
            self.dist[s, s] = 0
            self.npaths[s, s] = 1
            q = deque([src])
            while q:
                u = q.popleft()
                iu = self.index[u]
                for v in self.neighbors(u):
                    iv = self.index[v]
                    if self.dist[s, iv] < 0:
                        self.dist[s, iv] = self.dist[s, iu] + 1
                        q.append(v)
                    if self.dist[s, iv] == self.dist[s, iu] + 1:
                        self.npaths[s, iv] += self.npaths[s, iu]

    def betweenness(self) -> dict[Cell, float]:
        """Shortest-path betweenness over ordered pairs of distinct free cells (intermediate nodes only)."""
        n = self.n_free
        bc = np.zeros(n)
        for s in range(n):
            for t in range(n):
                if s == t:
                    continue
                for v in range(n):
                    if v in (s, t):
                        continue
                    if self.dist[s, v] + self.dist[v, t] == self.dist[s, t]:
                        bc[v] += self.npaths[s, v] * self.npaths[v, t] / self.npaths[s, t]
        return {cell: float(bc[i]) for i, cell in enumerate(self.free_cells)}

    def shortest_path(self, start: Cell, goal: Cell) -> list[Cell]:
        """One deterministic shortest cell path (neighbour order: up, down, left, right)."""
        s, g = self.index[start], self.index[goal]
        path = [start]
        cur = start
        while cur != goal:
            for nb in self.neighbors(cur):
                if self.dist[self.index[nb], g] == self.dist[self.index[cur], g] - 1:
                    cur = nb
                    break
            path.append(cur)
        assert len(path) == self.dist[s, g] + 1
        return path

    def route_category(self, start: Cell, goal: Cell, region: Cell) -> str:
        """Geometric route category of a start->goal pair relative to the restricted cell.

        through : every shortest cell path visits the region as an intermediate cell
        partial : some but not all shortest paths visit it
        avoid   : no shortest path visits it
        endpoint: the start or goal cell is the region cell
        """
        if region in (start, goal):
            return "endpoint"
        s, g, r = self.index[start], self.index[goal], self.index[region]
        if self.dist[s, r] + self.dist[r, g] != self.dist[s, g]:
            return "avoid"
        if self.npaths[s, r] * self.npaths[r, g] == self.npaths[s, g]:
            return "through"
        return "partial"


def select_region(geom: MazeGeometry, pool_cell_counts: np.ndarray) -> dict:
    """Pre-registered rule: the free cell with maximal shortest-path betweenness.

    Ties are broken by larger occupancy in the eligible training pool (never by policy outcomes).
    """
    bc = geom.betweenness()
    best = max(geom.free_cells, key=lambda c: (round(bc[c], 9), int(pool_cell_counts[geom.index[c]])))
    ranked = sorted(geom.free_cells, key=lambda c: -bc[c])
    return {
        "cell": list(best),
        "center_xy": geom.cell_to_xy(best).tolist(),
        "betweenness": bc[best],
        "betweenness_ranking": [{"cell": list(c), "betweenness": bc[c]} for c in ranked[:6]],
        "is_articulation_point": is_articulation_point(geom, best),
    }


def is_articulation_point(geom: MazeGeometry, cell: Cell) -> bool:
    """True if removing the cell disconnects the free-cell graph."""
    others = [c for c in geom.free_cells if c != cell]
    seen = {others[0]}
    q = deque([others[0]])
    while q:
        u = q.popleft()
        for v in geom.neighbors(u):
            if v != cell and v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) < len(others)


# ----------------------------------------------------------------------------- task construction

def sparse_reward(achieved_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    """Environment sparse reward: 1 if the achieved position is within GOAL_RADIUS of the goal."""
    d = np.linalg.norm(np.asarray(achieved_xy, dtype=np.float64) - np.asarray(goal_xy, dtype=np.float64), axis=-1)
    return (d <= GOAL_RADIUS).astype(np.float32)


@dataclass
class TaskArrays:
    """Goal-conditioned transitions (all recorded transitions; eligibility is a separate mask)."""

    obs: np.ndarray        # (N, 6) x, y, vx, vy, gx, gy at time t, goal = transition goal
    action: np.ndarray     # (N, 2) recorded action
    next_obs: np.ndarray   # (N, 6) successor state with the SAME transition goal
    success: np.ndarray    # (N,) 1 if successor is within GOAL_RADIUS of the transition goal
    mask: np.ndarray       # (N,) bootstrap mask = 1 - success
    goal: np.ndarray       # (N, 2) transition goal
    goal_consistent: np.ndarray  # (N,) stored goal at t equals stored goal at t+1


def build_task(flat: dict) -> TaskArrays:
    """Construct goal-conditioned transitions from the flattened recording.

    The transition goal is the desired_goal stored in the *successor* observation. In
    gymnasium_robotics PointMazeEnv.step the successor observation and the reward are both computed
    against the current goal *before* update_goal() may resample it, so this is exactly the goal
    the stored reward refers to. When the goal was resampled on the previous step, the stored
    observation at t still carries the old goal while the reward and successor use the new one;
    such "goal-switch" transitions are flagged (goal_consistent = False) and excluded from every
    training condition, so that the conditioning goal, the behaviour's goal, the reward and the
    Bellman target always agree.
    """
    goal = flat["next_desired_goal"].astype(np.float64)
    consistent = np.all(flat["desired_goal"] == flat["next_desired_goal"], axis=1)
    success = sparse_reward(flat["next_achieved_goal"], goal)
    obs = np.concatenate([flat["observation"], goal], axis=1).astype(np.float32)
    next_obs = np.concatenate([flat["next_observation"], goal], axis=1).astype(np.float32)
    return TaskArrays(
        obs=obs,
        action=flat["action"].astype(np.float32),
        next_obs=next_obs,
        success=success.astype(np.float32),
        mask=(1.0 - success).astype(np.float32),
        goal=goal.astype(np.float32),
        goal_consistent=consistent,
    )
