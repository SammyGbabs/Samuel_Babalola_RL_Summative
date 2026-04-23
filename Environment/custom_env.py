"""
Environment/custom_env.py
=========================
Custom Gymnasium environment: 20x20 residential grid-world.

Design
------
    * Observation : Dict { 'position' (row,col), 'goal' (row,col),
                           'local_view' (k x k window around the agent) }
    * Action      : Discrete(4) — 0:Up, 1:Down, 2:Left, 3:Right
    * Dynamics    : Obstacles are HARD WALLS (movement into them is cancelled)
    * Task        : Reach a per-episode random goal cell (resampled each reset)
    * Reward      : +1.0 on reaching the goal, -0.01 per step
    * Termination : agent lands on goal cell
    * Truncation  : max_steps reached (default 200)

Usage
-----
    >>> from Environment.custom_env import ResidentialGridEnv
    >>> env = ResidentialGridEnv()
    >>> obs, info = env.reset(seed=0)
    >>> obs, r, terminated, truncated, info = env.step(action=0)

Or via the Gymnasium registry (after importing this module once):
    >>> import gymnasium as gym
    >>> from Environment import custom_env   # registers the env id
    >>> env = gym.make("ResidentialGrid-v0")

The map (rooms, doorways, obstacles, agent start) lives in
`Environment/rendering.py` so that visualisation and RL dynamics cannot
drift apart — there is exactly one source of truth.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Relative import keeps the package self-contained when placed under
# Environment/ and imported via `from Environment.custom_env import ...`.
# We fall back to an absolute import when the file is run as a script.
try:
    from .rendering import (
        GRID_SIZE,
        AGENT_START,
        build_room_grid,
        get_doorways,
        get_obstacles,
        render_environment,
        render_rgb_array,
    )
except ImportError:  # running as a plain script, not as part of a package
    from rendering import (  # type: ignore[no-redef]
        GRID_SIZE,
        AGENT_START,
        build_room_grid,
        get_doorways,
        get_obstacles,
        render_environment,
        render_rgb_array,
    )


# ---------------------------------------------------------------------------
# Action encoding — (drow, dcol). Row grows downward (north-up display).
# ---------------------------------------------------------------------------
ACTION_DELTAS: dict[int, tuple[int, int]] = {
    0: (-1,  0),   # Up
    1: ( 1,  0),   # Down
    2: ( 0, -1),   # Left
    3: ( 0,  1),   # Right
}
ACTION_NAMES = ["Up", "Down", "Left", "Right"]


class ResidentialGridEnv(gym.Env):
    """Gymnasium environment for the 20x20 residential grid-world."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Cell-code semantics used inside the observation's `local_view` array:
    #   -1 : out-of-bounds (padding)
    #    0 : free floor
    #    1 : obstacle (hard wall)
    #    2 : doorway
    #    3 : goal
    #    4 : agent
    CODE_FREE, CODE_OBSTACLE, CODE_DOOR, CODE_GOAL, CODE_AGENT = 0, 1, 2, 3, 4

    def __init__(
        self,
        view_size: int = 5,
        max_steps: int = 200,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        assert view_size % 2 == 1 and view_size >= 3, \
            "view_size must be an odd integer >= 3"

        self.view_size = view_size
        self.view_radius = view_size // 2
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.render_mode = render_mode

        # --- Static map (built once, never mutated) --------------------
        self._room_grid = build_room_grid()
        self._doorways  = set(get_doorways())
        self._obstacles = {o["pos"] for o in get_obstacles()}

        # walkable[r, c] is True iff the cell is NOT an obstacle.
        self._walkable = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
        for (r, c) in self._obstacles:
            self._walkable[r, c] = False

        # Base semantic grid (obstacles + doorways); goal/agent overlaid per step.
        self._base_codes = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        for (r, c) in self._obstacles:
            self._base_codes[r, c] = self.CODE_OBSTACLE
        for (r, c) in self._doorways:
            self._base_codes[r, c] = self.CODE_DOOR

        # --- Spaces ----------------------------------------------------
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "position":   spaces.Box(0, GRID_SIZE - 1, shape=(2,), dtype=np.int32),
            "goal":       spaces.Box(0, GRID_SIZE - 1, shape=(2,), dtype=np.int32),
            "local_view": spaces.Box(-1, 4, shape=(view_size, view_size),
                                     dtype=np.int8),
        })

        # --- Dynamic episode state ------------------------------------
        self._agent_pos: tuple[int, int] = AGENT_START
        self._goal_pos:  tuple[int, int] = AGENT_START
        self._steps: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Start a new episode. Samples a new random goal each time."""
        super().reset(seed=seed)
        self._agent_pos = AGENT_START

        # Sample goal uniformly among walkable cells, excluding the start.
        walkable = np.argwhere(self._walkable)
        mask = ~((walkable[:, 0] == AGENT_START[0]) &
                 (walkable[:, 1] == AGENT_START[1]))
        candidates = walkable[mask]
        idx = self.np_random.integers(0, len(candidates))
        self._goal_pos = (int(candidates[idx, 0]), int(candidates[idx, 1]))

        self._steps = 0
        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply `action`. Obstacles and grid borders block movement."""
        assert self.action_space.contains(action), f"Invalid action: {action}"

        dr, dc = ACTION_DELTAS[int(action)]
        r, c = self._agent_pos
        nr, nc = r + dr, c + dc
        if (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE
                and self._walkable[nr, nc]):
            self._agent_pos = (nr, nc)

        self._steps += 1
        terminated = self._agent_pos == self._goal_pos
        truncated = (not terminated) and (self._steps >= self.max_steps)
        reward = self.goal_reward if terminated else self.step_penalty

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        """
        Render the current state.

        * render_mode="rgb_array": returns an (H, W, 3) uint8 image.
        * render_mode="human":     writes a 300-DPI PNG via the matplotlib
                                   renderer and returns the file path.
        """
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            return render_rgb_array(self._agent_pos, self._goal_pos)
        if self.render_mode == "human":
            return render_environment(
                agent_pos=self._agent_pos,
                goal_pos=self._goal_pos,
                save_path="custom_env_render.png",
                dpi=300,
            )
        raise ValueError(f"Unknown render_mode: {self.render_mode}")

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> dict[str, np.ndarray]:
        """Build the Dict observation: position + goal + local egocentric view."""
        codes = self._base_codes.copy()
        gr, gc = self._goal_pos
        codes[gr, gc] = self.CODE_GOAL
        ar, ac = self._agent_pos
        codes[ar, ac] = self.CODE_AGENT   # agent overwrites underlying cell

        # Pad with -1 so the agent can always extract a (k x k) centred view.
        vr = self.view_radius
        padded = np.full(
            (GRID_SIZE + 2 * vr, GRID_SIZE + 2 * vr),
            fill_value=-1, dtype=np.int8,
        )
        padded[vr:vr + GRID_SIZE, vr:vr + GRID_SIZE] = codes
        # (ar, ac) in `codes` corresponds to (ar+vr, ac+vr) in `padded`;
        # the top-left of the view in padded coords is therefore (ar, ac).
        local_view = padded[ar:ar + self.view_size,
                            ac:ac + self.view_size].copy()

        return {
            "position":   np.array(self._agent_pos, dtype=np.int32),
            "goal":       np.array(self._goal_pos,  dtype=np.int32),
            "local_view": local_view,
        }

    def _get_info(self) -> dict[str, Any]:
        ar, ac = self._agent_pos
        gr, gc = self._goal_pos
        return {
            "steps": self._steps,
            "manhattan_to_goal": abs(ar - gr) + abs(ac - gc),
            "agent_pos": self._agent_pos,
            "goal_pos":  self._goal_pos,
        }


# ---------------------------------------------------------------------------
# Register as a Gymnasium id so training scripts can use gym.make(...)
# ---------------------------------------------------------------------------
# Registration is idempotent-safe — try/except guards against double-registering
# when this module is re-imported (common in notebooks).
try:
    gym.register(
        id="ResidentialGrid-v0",
        entry_point=f"{__name__}:ResidentialGridEnv",
        max_episode_steps=200,
    )
except gym.error.Error:
    # Already registered — fine.
    pass


# ---------------------------------------------------------------------------
# Smoke tests: run this file directly to verify everything works.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    print("[1/3] Gymnasium env_checker ...")
    env = ResidentialGridEnv(view_size=5, max_steps=200)
    check_env(env, skip_render_check=True)
    print("      ✓ passed")

    print("[2/3] Random-policy rollout (3 episodes) ...")
    rng = np.random.default_rng(0)
    for ep in range(3):
        obs, info = env.reset(seed=ep)
        total_reward, done, term = 0.0, False, False
        while not done:
            a = int(rng.integers(0, 4))
            obs, r, term, trunc, info = env.step(a)
            total_reward += r
            done = term or trunc
        print(f"      ep={ep}  goal={info['goal_pos']}  "
              f"steps={info['steps']}  reached={term}  return={total_reward:+.2f}")

    print("[3/3] Tabular Q-learning sanity check (fixed walkable goal) ...")
    train_env = ResidentialGridEnv()
    fixed_goal = (0, 0)  # Corner of living room; guaranteed walkable.
    original_reset = train_env.reset

    def fixed_reset(*, seed=None, options=None):
        obs, info = original_reset(seed=seed, options=options)
        train_env._goal_pos = fixed_goal
        return train_env._get_obs(), train_env._get_info()

    train_env.reset = fixed_reset  # type: ignore[method-assign]

    Q = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float64)
    alpha, gamma, eps = 0.2, 0.95, 0.2
    returns: list[float] = []
    for ep in range(400):
        obs, _ = train_env.reset(seed=ep)
        rc = tuple(obs["position"])
        done, G = False, 0.0
        while not done:
            a = int(rng.integers(0, 4)) if rng.random() < eps \
                else int(np.argmax(Q[rc[0], rc[1]]))
            obs, r, term, trunc, _ = train_env.step(a)
            nxt = tuple(obs["position"])
            td = r + (0.0 if term else gamma * np.max(Q[nxt[0], nxt[1]]))
            Q[rc[0], rc[1], a] += alpha * (td - Q[rc[0], rc[1], a])
            rc = nxt
            G += r
            done = term or trunc
        returns.append(G)

    early, late = float(np.mean(returns[:50])), float(np.mean(returns[-50:]))
    print(f"      avg return first 50 eps:  {early:+.3f}")
    print(f"      avg return last  50 eps:  {late:+.3f}")
    assert late > early, "Q-learning did not improve — env may be broken"
    print("      ✓ agent's return improved — env is learnable")

    print("\nAll checks passed. From your training scripts, use either:")
    print('    env = ResidentialGridEnv()')
    print('    env = gymnasium.make("ResidentialGrid-v0")')
