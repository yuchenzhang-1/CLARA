from __future__ import annotations

import argparse
import random
import hashlib
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional



@dataclass(frozen=True)
class Segment:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SampledPair:
    global_seg: Segment
    local_seg: Segment




@dataclass
class SegmentSamplingConfig:
    mode: str               
    num_pairs_per_video:int
    l_ratio:float
    g_ratio: float
    min_l: int
    min_g: int
    max_l: int
    max_g: int
    base_seed: int




def compute_l_g_lengths(
    T: int,
    l_ratio: float,
    g_ratio: float,
    min_l: int,
    min_g: int,
    max_l: int,
    max_g: int,
) -> Tuple[int, int]:

    if T <= 0:
        raise ValueError(f"T must be >= 1, got {T}")

    if T == 1:
        return 1, 1
    if T == 2:
        return 1, 2
    if T == 3:
        return 1, 2
    if T == 4:
        return 2, 3

    l_len = int(round(T * l_ratio))
    g_len = int(round(T * g_ratio))

    l_len = max(l_len, min_l)
    g_len = max(g_len, min_g)
    l_len = min(l_len, max_l)
    g_len = min(g_len, max_g)

    l_len = min(l_len, T)
    g_len = min(g_len, T)

    if g_len <= l_len:
        g_len = min(T, l_len + 1)

    l_len = max(1, l_len)
    g_len = max(1, g_len)
    return l_len, g_len


def sample_segment(T: int, seg_len: int, rng: random.Random) -> Segment:
    seg_len = min(seg_len, T)
    if seg_len == T:
        return Segment(0, T)
    start = rng.randrange(0, T - seg_len + 1)
    return Segment(start, start + seg_len)


def sample_one_pair(
    T: int,
    mode: str,
    l_ratio: float,
    g_ratio: float,
    min_l: int,
    min_g: int,
    max_l: int,
    max_g: int,
    rng: random.Random,
) -> SampledPair:
    if mode not in ("within", "independent"):
        raise ValueError(f"mode must be 'within' or 'independent', got {mode}")

    l_len, g_len = compute_l_g_lengths(
        T=T,
        l_ratio=l_ratio,
        g_ratio=g_ratio,
        min_l=min_l,
        min_g=min_g,
        max_l=max_l,
        max_g=max_g,
    )

    if T == 1:
        seg = Segment(0, 1)
        return SampledPair(global_seg=seg, local_seg=seg)

    g_seg = sample_segment(T, g_len, rng)

    if mode == "independent":
        l_seg = sample_segment(T, l_len, rng)
        return SampledPair(global_seg=g_seg, local_seg=l_seg)

    # within
    if g_seg.length <= l_len:
        l_seg = Segment(g_seg.start, g_seg.end)
        return SampledPair(global_seg=g_seg, local_seg=l_seg)

    start_in_g = rng.randrange(0, g_seg.length - l_len + 1)
    l_start = g_seg.start + start_in_g
    l_seg = Segment(l_start, l_start + l_len)
    return SampledPair(global_seg=g_seg, local_seg=l_seg)


def sample_pairs_for_video(
    T: int,
    num_pairs_per_video: int,
    mode: str,
    l_ratio: float,
    g_ratio: float,
    min_l: int,
    min_g: int,
    max_l: int,
    max_g: int,
    rng: random.Random,
) -> List[SampledPair]:
    if num_pairs_per_video <= 0:
        raise ValueError(f"num_pairs_per_video must be >=1, got {num_pairs_per_video}")

    pairs: List[SampledPair] = []
    for _ in range(num_pairs_per_video):
        pairs.append(sample_one_pair(
            T=T,
            mode=mode,
            l_ratio=l_ratio,
            g_ratio=g_ratio,
            min_l=min_l,
            min_g=min_g,
            max_l=max_l,
            max_g=max_g,
            rng=rng,
        ))
    return pairs



def stable_hash_str(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def build_video_rng(
    base_seed: int,
    epoch: int,
    rank: int,
    video_id: str,
    epoch_mult: int = 1000003,
    rank_mult: int = 10007,
) -> random.Random:
    seed = int(base_seed) + epoch_mult * int(epoch) + rank_mult * int(rank) + stable_hash_str(video_id)
    return random.Random(seed)

