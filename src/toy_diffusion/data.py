from __future__ import annotations

import math

import numpy as np


def _sample_moons(n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    n1 = n // 2
    n2 = n - n1

    theta1 = rng.uniform(0.0, math.pi, size=n1)
    theta2 = rng.uniform(0.0, math.pi, size=n2)

    moon1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
    moon2 = np.stack([1.0 - np.cos(theta2), 0.5 - np.sin(theta2)], axis=1)

    x = np.concatenate([moon1, moon2], axis=0)
    x += rng.normal(scale=noise, size=x.shape)
    x *= 1.4
    return x.astype(np.float32)


def _sample_spiral(n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    t = np.sqrt(rng.uniform(0.0, 1.0, size=n)) * 4.0 * math.pi
    r = 0.25 * t
    x = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    x += rng.normal(scale=noise, size=x.shape)
    x /= 3.5
    return x.astype(np.float32)


def _sample_circle(n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    r = 1.5 + rng.normal(scale=noise, size=n)
    x = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x /= 2.0
    return x.astype(np.float32)


def sample_dataset(name: str, n: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    name = name.lower()
    if name == "moons":
        return _sample_moons(n, noise, rng)
    if name == "spiral":
        return _sample_spiral(n, noise, rng)
    if name == "circle":
        return _sample_circle(n, noise, rng)
    raise ValueError(f"unknown toy dataset: {name}")

