"""Compares Stanley and Pure Pursuit automatically (no window).

Gas/brake is handled by a simple controller that holds a fixed target speed.
Both methods drive the same tracks (seeds) so the comparison is fair.

Usage:
    uv run python compare_lateral_control.py
    uv run python compare_lateral_control.py --speeds 40 60 --seeds 10
"""

import argparse

import gymnasium as gym
import numpy as np
from env_wrapper import CarRacingEnvWrapper
from lateral_control import LateralControl


def drive(env, mode, seed, target_speed, max_steps):
    lateral_control = LateralControl(mode=mode)
    _, info = env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(max_steps):
        steering = lateral_control.control(info["trajectory"], info["speed"])
        speed = info["speed"]
        gas = 0.3 if speed < target_speed else 0.0
        brake = 0.3 if speed > target_speed + 10 else 0.0
        _, reward, done, truncated, info = env.step([steering, gas, brake])
        total_reward += reward
        if done or truncated:
            break
    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speeds", type=float, nargs="+", default=[40, 60, 80])
    parser.add_argument("--seeds", type=int, default=5, help="number of tracks")
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()

    env = CarRacingEnvWrapper(
        gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False)
    )
    seeds = list(range(args.seeds))
    modes = ["stanley", "pure_pursuit"]

    print(f"Tracks (seeds): {seeds}\n")
    print(f"{'Speed':>6} | {'stanley':>10} | {'pure_pursuit':>12}")
    print("-" * 36)
    for speed in args.speeds:
        means = []
        for mode in modes:
            rewards = [drive(env, mode, s, speed, args.max_steps) for s in seeds]
            means.append(np.mean(rewards))
        print(f"{speed:>6.0f} | {means[0]:>10.1f} | {means[1]:>12.1f}")

    env.close()


if __name__ == "__main__":
    main()
