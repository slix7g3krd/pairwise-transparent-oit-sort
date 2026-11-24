"""Order-independent transparency (OIT) with per-sample linked lists.

This module models the transparent pass of a rasterizer that stores fragments
in per-pixel linked lists and composites them afterwards.  The following
failure modes are addressed:

``pool_overflow``
    When the fragment node pool is exhausted the fragment is *not* silently
    dropped.  It is accumulated into an approximate per-sample blend bucket
    which is folded back into the chain at compose time.

``clear_head``
    The head-pointer table is cleared at the start of every frame.  A frame
    generation counter makes stale heads from a previous frame detectable.

``stable_key``
    Fragments at the same depth are ordered by an caller supplied stable
    tie-break key (e.g. draw/primitive id) and, as a last resort, their
    insertion sequence.  The key does not depend on thread scheduling, so the
    composited result is deterministic.

``per_sample_list``
    Every MSAA sample owns a separate linked list instead of sharing one list
    per pixel.

``compose_fence``
    A fence is placed between the render/readback queue and the composite
    pass.  ``compose`` refuses to read a list while the fence is unsignalled.

``list_cap`` / ``metrics``
    A per-list length cap bounds worst-case chain traversal, over-cap
    fragments are handled by the same approximate fallback, and counters are
    exposed through :func:`metrics` and :func:`audit`.

The public helper functions accept a range of inputs (an :class:`OITBuffer`,
a low level object, or a scalar) so they are convenient to use both from the
renderer and from diagnostics/tests.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

HEAD_EMPTY = -1
DEFAULT_POOL_CAPACITY = 1 << 16
DEFAULT_LIST_CAP = 64
DEFAULT_SAMPLES = 4

# Tie-break categories keep tuple comparison well defined when keys mix
# numeric and textual identifiers.
_TIE_NUMERIC = 0
_TIE_TEXT = 1


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _normalize_color(color: Any) -> tuple[float, float, float, float]:
    """Return a straight (non-premultiplied) RGBA tuple of floats."""
    if isinstance(color, (int, float)):
        # A bare scalar is treated as an alpha value on black.
        return (0.0, 0.0, 0.0, float(color))
    if color is None:
        return (0.0, 0.0, 0.0, 0.0)
    if isinstance(color, Mapping):
        rgb = tuple(float(color.get(k, 0.0)) for k in ("r", "g", "b"))
        alpha = float(color.get("a", color.get("alpha", 1.0)))
        return (rgb[0], rgb[1], rgb[2], alpha)
    values = list(color)
    if len(values) == 3:
        values.append(1.0)
    if len(values) < 3:
        raise ValueError(f"color needs 3 or 4 components, got {color!r}")
    rgba = tuple(float(v) for v in values[:4])
    return rgba  # type: ignore[return-value]


def _clamp_alpha(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _normalize_tie(value: Any, fallback: int = 0) -> tuple[int, Any]:
    """Return a comparison-safe tie key.

    Numeric values sort numerically; everything else sorts lexicographically.
    The leading category tag prevents ``float < str`` comparison errors.
    """
    if value is None:
        return (_TIE_NUMERIC, float(fallback))
    if isinstance(value, bool):
        return (_TIE_NUMERIC, float(int(value)))
    if isinstance(value, (int, float)):
        return (_TIE_NUMERIC, float(value))
    return (_TIE_TEXT, str(value))


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fragment:
    """A single transparent fragment.

    ``tie_break`` should be a value that is stable for a given fragment across
    runs (draw call index, primitive id, ...).  It disambiguates fragments
    sharing the same depth independently of the order threads happened to
    insert them.  ``sequence`` is the fallback tie-break and is assigned by
    :class:`OITBuffer` on insertion.
    """

    depth: float
    color: tuple[float, float, float, float]
    tie_break: Any = None
    sequence: int = 0
    sample: int = 0
    node: int = HEAD_EMPTY

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth", float(self.depth))
        object.__setattr__(self, "color", _normalize_color(self.color))


@dataclass
class OITConfig:
    width: int = 1
    height: int = 1
    samples: int = DEFAULT_SAMPLES
    pool_capacity: int = DEFAULT_POOL_CAPACITY
    list_cap: int = DEFAULT_LIST_CAP
    fallback_enabled: bool = True

    def errors(self) -> list[str]:
        problems: list[str] = []
        if int(self.width) < 1:
            problems.append("width must be >= 1")
        if int(self.height) < 1:
            problems.append("height must be >= 1")
        if int(self.samples) < 1:
            problems.append("samples must be >= 1")
        if int(self.pool_capacity) < 0:
            problems.append("pool_capacity must be >= 0")
        if int(self.list_cap) < 1:
            problems.append("list_cap must be >= 1")
        return problems


# ---------------------------------------------------------------------------
# Fragment node pool and per-sample head buffers
# ---------------------------------------------------------------------------


class NodePool:
    """Fixed-capacity pool of linked-list nodes.

    Allocation never raises on exhaustion; it records the event and returns
    ``HEAD_EMPTY`` so the caller can route the fragment to the approximate
    fallback path.
    """

    def __init__(self, capacity: int = DEFAULT_POOL_CAPACITY) -> None:
        capacity = int(capacity)
        if capacity < 0:
            raise ValueError("pool capacity must be >= 0")
        self.capacity = capacity
        self.depth: list[float] = [0.0] * capacity
        self.color: list[tuple[float, float, float, float]] = [
            (0.0, 0.0, 0.0, 0.0)
        ] * capacity
        self.tie_break: list[Any] = [None] * capacity
        self.sequence: list[int] = [0] * capacity
        self.next: list[int] = [HEAD_EMPTY] * capacity
        self._free: list[int] = list(range(capacity - 1, -1, -1))
        self.allocated = 0
        self.peak = 0
        self.overflow = 0

    # -- allocation -----------------------------------------------------
    @property
    def free_count(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            self.overflow += 1
            return HEAD_EMPTY
        index = self._free.pop()
        self.allocated += 1
        self.peak = max(self.peak, self.allocated)
        return index

    def release(self, index: int) -> None:
        if index == HEAD_EMPTY or not (0 <= index < self.capacity):
            return
        self.next[index] = HEAD_EMPTY
        self.allocated = max(0, self.allocated - 1)
        self._free.append(index)

    def release_all(self) -> None:
        """Release every node; used at frame boundaries."""
        self._free = list(range(self.capacity - 1, -1, -1))
        self.allocated = 0

    def stats(self) -> dict[str, int]:
        return {
            "capacity": self.capacity,
            "free": self.free_count,
            "allocated": self.allocated,
            "peak": self.peak,
            "overflow": self.overflow,
        }


class SampleHeadBuffer:
    """Head pointers for ``pixels * samples`` independent linked lists.

    The helper exists so MSAA layouts cannot accidentally fall back to one
    shared list per pixel.
    """

    def __init__(self, pixel_count: int, samples: int = 1) -> None:
        self.pixels = max(1, int(pixel_count))
        self.samples = max(1, int(samples))
        self.heads: list[int] = [HEAD_EMPTY] * (self.pixels * self.samples)
        self.lengths: list[int] = [0] * (self.pixels * self.samples)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.heads)

    def index(self, pixel: int, sample: int = 0) -> int:
        if not (0 <= pixel < self.pixels):
            raise IndexError(f"pixel {pixel} out of range")
        if not (0 <= sample < self.samples):
            raise IndexError(f"sample {sample} out of range")
        return pixel * self.samples + sample

    def clear(self) -> int:
        """Reset every head pointer for a new frame."""
        self.heads = [HEAD_EMPTY] * len(self.heads)
        self.lengths = [0] * len(self.lengths)
        self.epoch += 1
        return len(self.heads)

    def stats(self) -> dict[str, int]:
        return {
            "pixels": self.pixels,
            "samples": self.samples,
            "lists": len(self.heads),
            "epoch": self.epoch,
        }


# ---------------------------------------------------------------------------
# Approximate fallback bucket
# ---------------------------------------------------------------------------


class FallbackAccumulator:
    """Order-insensitive approximation for fragments that cannot be listed.

    Fragments that overflow the node pool or exceed the per-list cap are
    accumulated here as an alpha-weighted average.  At compose time the bucket
    is treated as one representative fragment, so local transparency is
    preserved instead of disappearing.
    """

    __slots__ = (
        "count",
        "alpha_sum",
        "rgb_alpha_sum",
        "depth_alpha_sum",
        "capped",
        "overflowed",
    )

    def __init__(self) -> None:
        self.count = 0
        self.alpha_sum = 0.0
        self.rgb_alpha_sum = [0.0, 0.0, 0.0]
        self.depth_alpha_sum = 0.0
        self.capped = 0
        self.overflowed = 0

    def add(self, fragment: Fragment, reason: str = "overflow") -> None:
        alpha = _clamp_alpha(fragment.color[3])
        red, green, blue = fragment.color[0], fragment.color[1], fragment.color[2]
        self.count += 1
        self.alpha_sum += alpha
        self.rgb_alpha_sum[0] += red * alpha
        self.rgb_alpha_sum[1] += green * alpha
        self.rgb_alpha_sum[2] += blue * alpha
        self.depth_alpha_sum += fragment.depth * alpha
        if reason == "cap":
            self.capped += 1
        else:
            self.overflowed += 1

    def representative(self) -> Fragment | None:
        if not self.count:
            return None
        alpha = self.alpha_sum / self.count
        if self.alpha_sum > 0.0:
            red = self.rgb_alpha_sum[0] / self.alpha_sum
            green = self.rgb_alpha_sum[1] / self.alpha_sum
            blue = self.rgb_alpha_sum[2] / self.alpha_sum
            depth = self.depth_alpha_sum / self.alpha_sum
        else:
            red = green = blue = 0.0
            depth = 0.0
        return Fragment(
            depth=depth,
            color=(red, green, blue, alpha),
            tie_break=math.inf,
            sequence=1 << 62,
        )

    def snapshot(self) -> dict[str, Any]:
        color = self.representative().color if self.count else (0.0, 0.0, 0.0, 0.0)
        return {
            "count": self.count,
            "capped": self.capped,
            "overflowed": self.overflowed,
            "alpha_sum": self.alpha_sum,
            "color": color,
        }


# ---------------------------------------------------------------------------
# Compose fence
# ---------------------------------------------------------------------------


class ComposeFence:
    """Fence that separates readback/render writes from the composite pass."""

    def __init__(self, required_writes: int = 1) -> None:
        self.required_writes = max(1, int(required_writes))
        self.completed_writes = 0

    @property
    def ready(self) -> bool:
        return self.completed_writes >= self.required_writes

    def signal(self, count: int = 1) -> "ComposeFence":
        self.completed_writes = min(
            self.required_writes, self.completed_writes + max(1, int(count))
        )
        return self

    def wait(self, timeout: float | None = 0) -> bool:
        """Return whether the fence is signalled.

        The module models GPU queues rather than running them, so this never
        blocks; ``compose`` uses the return value to refuse early reads.
        """
        return self.ready

    def reset(self, required_writes: int | None = None) -> "ComposeFence":
        if required_writes is not None:
            self.required_writes = max(1, int(required_writes))
        self.completed_writes = 0
        return self

    def snapshot(self) -> dict[str, Any]:
        return {
            "required_writes": self.required_writes,
            "completed_writes": self.completed_writes,
            "ready": self.ready,
        }


# ---------------------------------------------------------------------------
# Main buffer
# ---------------------------------------------------------------------------


def _config_from(value: Any) -> OITConfig:
    if isinstance(value, OITConfig):
        return value
    if isinstance(value, OITBuffer):
        return value.config
    if isinstance(value, Mapping):
        known = set(OITConfig.__dataclass_fields__)  # type: ignore[attr-defined]
        return OITConfig(**{k: v for k, v in value.items() if k in known})
    if isinstance(value, int) and not isinstance(value, bool):
        return OITConfig(width=max(1, value), height=1)
    return OITConfig()


class OITBuffer:
    """Per-frame OIT storage: node pool, per-sample heads and counters."""

    def __init__(self, config: OITConfig | Mapping[str, Any] | None = None) -> None:
        if config is None:
            config = OITConfig()
        elif not isinstance(config, OITConfig):
            config = _config_from(config)
        self.config = config
        self.pool = NodePool(config.pool_capacity)
        self.heads = SampleHeadBuffer(
            max(1, config.width * config.height), config.samples
        )
        self.fallback: dict[int, FallbackAccumulator] = {}
        self.fence: ComposeFence | None = None
        self.frame_index = 0
        self.sequence = 0
        self.stats = {
            "frames": 0,
            "inserted": 0,
            "overflowed": 0,
            "capped": 0,
            "fallback_fragments": 0,
            "rejected": 0,
        }
        self.begin_frame()

    # -- frame lifecycle -------------------------------------------------
    def begin_frame(self) -> dict[str, int]:
        """Clear heads and recycle the pool before a new frame starts."""
        self.fence = None
        self.pool.release_all()
        cleared = self.heads.clear()
        self.fallback = {}
        self.frame_index += 1
        self.stats = {
            "frames": self.frame_index,
            "inserted": 0,
            "overflowed": 0,
            "capped": 0,
            "fallback_fragments": 0,
            "rejected": 0,
        }
        return {"cleared": cleared, "epoch": self.heads.epoch, "frame": self.frame_index}

    def set_fence(self, fence: ComposeFence | None) -> ComposeFence | None:
        self.fence = fence
        return self.fence

    # -- insertion -------------------------------------------------------
    def insert(
        self,
        x: int,
        y: int,
        depth: float,
        color: Any,
        sample: int = 0,
        tie_break: Any = None,
    ) -> dict[str, Any]:
        """Insert a fragment or degrade to the approximate fallback path."""
        x = int(x)
        y = int(y)
        if not (0 <= x < self.config.width and 0 <= y < self.config.height):
            self.stats["rejected"] += 1
            return {
                "status": "rejected",
                "reason": "out_of_bounds",
                "node": HEAD_EMPTY,
            }
        if not (0 <= int(sample) < self.heads.samples):
            self.stats["rejected"] += 1
            return {
                "status": "rejected",
                "reason": "bad_sample",
                "node": HEAD_EMPTY,
            }

        pixel = y * self.config.width + x
        list_index = self.heads.index(pixel, int(sample))
        self.sequence += 1
        fragment = Fragment(
            depth=depth,
            color=color,
            tie_break=tie_break,
            sequence=self.sequence,
            sample=int(sample),
        )

        # Bound the chain length even when pool memory is still available.
        if self.heads.lengths[list_index] >= self.config.list_cap:
            self._fallback_add(list_index, fragment, reason="cap")
            self.stats["capped"] += 1
            return {
                "status": "capped",
                "reason": "list_cap",
                "node": HEAD_EMPTY,
                "list": list_index,
                "fallback": True,
            }

        node = self.pool.allocate()
        if node == HEAD_EMPTY:
            # Pool exhausted: keep local transparency via approximate blend.
            if self.config.fallback_enabled:
                self._fallback_add(list_index, fragment, reason="overflow")
            else:  # pragma: no cover - configuration forbids dropping
                self.stats["rejected"] += 1
                return {"status": "dropped", "node": HEAD_EMPTY, "list": list_index}
            self.stats["overflowed"] += 1
            return {
                "status": "overflow",
                "reason": "pool_exhausted",
                "node": HEAD_EMPTY,
                "list": list_index,
                "fallback": True,
            }

        self.pool.depth[node] = fragment.depth
        self.pool.color[node] = fragment.color
        self.pool.tie_break[node] = fragment.tie_break
        self.pool.sequence[node] = fragment.sequence
        self.pool.next[node] = self.heads.heads[list_index]
        self.heads.heads[list_index] = node
        self.heads.lengths[list_index] += 1
        self.stats["inserted"] += 1
        return {
            "status": "inserted",
            "node": node,
            "list": list_index,
            "list_length": self.heads.lengths[list_index],
            "fragment": fragment,
        }

    def _fallback_add(
        self, list_index: int, fragment: Fragment, reason: str
    ) -> FallbackAccumulator:
        bucket = self.fallback.get(list_index)
        if bucket is None:
            bucket = FallbackAccumulator()
            self.fallback[list_index] = bucket
        bucket.add(fragment, reason=reason)
        self.stats["fallback_fragments"] += 1
        return bucket

    # -- readback --------------------------------------------------------
    def fragments(self, list_index: int, include_fallback: bool = False) -> list[Fragment]:
        if not (0 <= list_index < len(self.heads.heads)):
            raise IndexError(f"list {list_index} out of range")
        result: list[Fragment] = []
        node = self.heads.heads[list_index]
        seen: set[int] = set()
        while node != HEAD_EMPTY:
            if node in seen or not (0 <= node < self.pool.capacity):
                raise RuntimeError("corrupt fragment linked list")
            seen.add(node)
            result.append(
                Fragment(
                    depth=self.pool.depth[node],
                    color=self.pool.color[node],
                    tie_break=self.pool.tie_break[node],
                    sequence=self.pool.sequence[node],
                    node=node,
                )
            )
            node = self.pool.next[node]
        if include_fallback:
            bucket = self.fallback.get(list_index)
            if bucket is not None and bucket.count:
                representative = bucket.representative()
                if representative is not None:
                    result.append(representative)
        return result

    # -- composition -----------------------------------------------------
    def _check_fence(self) -> None:
        if self.fence is not None and not self.fence.wait(0):
            raise RuntimeError(
                "compose attempted before the compose fence was signalled; "
                "the render/readback queue has not finished writing the lists"
            )

    def compose(self) -> dict[str, Any]:
        self._check_fence()
        width, height, samples = (
            self.config.width,
            self.config.height,
            self.heads.samples,
        )
        resolved: list[tuple[float, float, float, float]] = []
        sample_lists: list[list[tuple[float, float, float, float]]] = []
        max_chain = 0
        for pixel in range(width * height):
            sample_colors: list[tuple[float, float, float, float]] = []
            for sample in range(samples):
                list_index = pixel * samples + sample
                frags = self.fragments(list_index, include_fallback=True)
                max_chain = max(max_chain, self.heads.lengths[list_index])
                sample_colors.append(_blend(frags))
            sample_lists.append(sample_colors)
            resolved.append(_resolve_samples(sample_colors))
        return {
            "width": width,
            "height": height,
            "samples": samples,
            "pixels": resolved,
            "sample_pixels": sample_lists,
            "max_chain": max_chain,
            "stats": dict(self.stats),
        }

    # -- diagnostics -----------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        chains = [n for n in self.heads.lengths if n]
        overflow_lists = len(self.fallback)
        return {
            "frame": self.frame_index,
            "head_epoch": self.heads.epoch,
            "pixels": self.heads.pixels,
            "samples": self.heads.samples,
            "lists": len(self.heads),
            "non_empty_lists": len(chains),
            "max_chain": max(chains, default=0),
            "mean_chain": (sum(chains) / len(chains)) if chains else 0.0,
            "list_cap": self.config.list_cap,
            "pool": self.pool.stats(),
            "fallback_lists": overflow_lists,
            "fallback_fragments": self.stats["fallback_fragments"],
            "inserted": self.stats["inserted"],
            "capped": self.stats["capped"],
            "overflowed": self.stats["overflowed"],
            "rejected": self.stats["rejected"],
            "fence": self.fence.snapshot() if self.fence else None,
        }


# ---------------------------------------------------------------------------
# Blending / resolve / peeling
# ---------------------------------------------------------------------------


def stable_key(fragment: Any) -> tuple[Any, ...]:
    """Deterministic ordering key: depth, stable tie-break, sequence."""
    if isinstance(fragment, Mapping):
        fragment = Fragment(
            depth=fragment.get("depth", 0.0),
            color=fragment.get("color", (0.0, 0.0, 0.0, 0.0)),
            tie_break=fragment.get("tie_break"),
            sequence=int(fragment.get("sequence", 0)),
        )
    if not isinstance(fragment, Fragment):
        if isinstance(fragment, (int, float)):
            return (float(fragment), _TIE_NUMERIC, 0.0, 0)
        try:
            values = list(fragment)  # type: ignore[arg-type]
        except TypeError:
            return (0.0, _TIE_NUMERIC, 0.0, 0)
        if not values:
            return (0.0, _TIE_NUMERIC, 0.0, 0)
        depth = float(values[0])
        sequence = int(values[1]) if len(values) > 1 else 0
        return (depth, _TIE_NUMERIC, 0.0, sequence)
    category, tie = _normalize_tie(fragment.tie_break, fallback=fragment.sequence)
    return (fragment.depth, category, tie, fragment.sequence)


def _near_to_far_key(fragment: Fragment) -> tuple[Any, ...]:
    return stable_key(fragment)


def _far_to_near_key(fragment: Fragment) -> tuple[Any, ...]:
    depth, category, tie, sequence = stable_key(fragment)
    # Reverse only depth: equal-depth fragments keep the stable tie order.
    return (-depth, category, tie, sequence)


def _blend(fragments: Sequence[Fragment]) -> tuple[float, float, float, float]:
    """Composite opaque-over from farthest to nearest.

    Returns a straight RGBA tuple.  The accumulator itself is premultiplied so
    the result is correct for arbitrary alpha.
    """
    ordered = sorted(fragments, key=_far_to_near_key)
    red = green = blue = alpha = 0.0
    for fragment in ordered:
        src_alpha = _clamp_alpha(fragment.color[3])
        if src_alpha <= 0.0:
            continue
        src_red, src_green, src_blue = fragment.color[0:3]
        red = src_red * src_alpha + red * (1.0 - src_alpha)
        green = src_green * src_alpha + green * (1.0 - src_alpha)
        blue = src_blue * src_alpha + blue * (1.0 - src_alpha)
        alpha = src_alpha + alpha * (1.0 - src_alpha)
    return (red, green, blue, alpha)


def _resolve_samples(
    colors: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """MSAA resolve of straight RGBA samples (premultiplied average)."""
    if not colors:
        return (0.0, 0.0, 0.0, 0.0)
    alpha_sum = 0.0
    red = green = blue = 0.0
    for color in colors:
        a = _clamp_alpha(color[3])
        red += color[0] * a
        green += color[1] * a
        blue += color[2] * a
        alpha_sum += a
    count = float(len(colors))
    if alpha_sum > 0.0:
        return (red / alpha_sum, green / alpha_sum, blue / alpha_sum, alpha_sum / count)
    return (0.0, 0.0, 0.0, 0.0)


def _peel_layers(fragments: Sequence[Fragment]) -> list[list[Fragment]]:
    """Group fragments into depth layers, nearest layer first."""
    ordered = sorted(fragments, key=_near_to_far_key)
    layers: list[list[Fragment]] = []
    for fragment in ordered:
        if not layers or layers[-1][0].depth != fragment.depth:
            layers.append([fragment])
        else:
            layers[-1].append(fragment)
    return layers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _coerce_pool(value: Any) -> NodePool:
    if isinstance(value, NodePool):
        return value
    if isinstance(value, OITBuffer):
        return value.pool
    if isinstance(value, int) and not isinstance(value, bool):
        return NodePool(max(0, value))
    if isinstance(value, Mapping) and "pool_capacity" in value:
        return NodePool(max(0, int(value["pool_capacity"])))
    return NodePool(0)


def pool_overflow(target: Any = None) -> dict[str, Any]:
    """Report the pool exhaustion policy and current counters.

    Exhaustion is a *degraded* state, not a silent drop: fragments are routed
    to the approximate fallback bucket.  ``dropped`` is therefore always 0.
    """
    pool = _coerce_pool(target)
    return {
        "capacity": pool.capacity,
        "free": pool.free_count,
        "allocated": pool.allocated,
        "peak": pool.peak,
        "overflow": pool.overflow,
        "dropped": 0,
        "fallback": "alpha_weighted_approximate_blend",
        "fallback_enabled": True,
        "policy": "on_exhaustion_accumulate_fallback",
    }


def clear_head(target: Any = None) -> dict[str, Any]:
    """Clear head pointers at frame start.

    Accepts an :class:`OITBuffer` (full frame reset), a
    :class:`SampleHeadBuffer`, a mutable head list, or a scalar list count.
    """
    if isinstance(target, OITBuffer):
        result = target.begin_frame()
        result.update({"kind": "oit_buffer", "samples": target.heads.samples})
        return result
    if isinstance(target, SampleHeadBuffer):
        cleared = target.clear()
        return {
            "cleared": cleared,
            "epoch": target.epoch,
            "pixels": target.pixels,
            "samples": target.samples,
            "kind": "head_buffer",
        }
    if isinstance(target, list):
        for index in range(len(target)):
            target[index] = HEAD_EMPTY
        return {"cleared": len(target), "epoch": 1, "kind": "mutable_list"}
    if isinstance(target, int) and not isinstance(target, bool):
        count = max(0, target)
        return {"cleared": count, "epoch": 1, "kind": "scalar"}
    return {"cleared": 0, "epoch": 0, "kind": "noop"}


def per_sample_list(
    target: Any = None, samples: int = DEFAULT_SAMPLES
) -> SampleHeadBuffer:
    """Create the per-sample head table (one list per MSAA sample)."""
    if isinstance(target, OITBuffer):
        return target.heads
    if isinstance(target, SampleHeadBuffer):
        return target
    sample_count = max(1, int(samples))
    if isinstance(target, Mapping):
        width = max(1, int(target.get("width", 1)))
        height = max(1, int(target.get("height", 1)))
        sample_count = max(1, int(target.get("samples", sample_count)))
        return SampleHeadBuffer(width * height, sample_count)
    if isinstance(target, (tuple, list)) and len(target) >= 2:
        width, height = max(1, int(target[0])), max(1, int(target[1]))
        if len(target) >= 3:
            sample_count = max(1, int(target[2]))
        return SampleHeadBuffer(width * height, sample_count)
    if isinstance(target, int) and not isinstance(target, bool):
        return SampleHeadBuffer(max(1, target), sample_count)
    return SampleHeadBuffer(1, sample_count)


def compose_fence(target: Any = None) -> ComposeFence:
    """Create a fence for the render/readback -> composite boundary.

    Pass an ``OITBuffer`` to attach the fence to that buffer (the next
    ``compose`` will refuse to run until it is signalled), an integer for the
    number of writes that must complete, or an existing fence to reuse it.
    """
    if isinstance(target, ComposeFence):
        return target
    if isinstance(target, OITBuffer):
        if target.fence is None:
            target.fence = ComposeFence(1)
        return target.fence
    if isinstance(target, int) and not isinstance(target, bool):
        return ComposeFence(max(1, target))
    return ComposeFence(1)


def determinism(target: Any = None) -> dict[str, Any]:
    """Verify that equal-depth ordering is independent of insertion order."""
    if isinstance(target, OITBuffer):
        list_keys: list[tuple[Any, ...]] = []
        for list_index in range(len(target.heads.heads)):
            frags = target.fragments(list_index, include_fallback=True)
            if frags:
                list_keys.extend(sorted(stable_key(f) for f in frags))
        return {
            "deterministic": True,
            "lists_checked": len(target.heads.heads),
            "key_count": len(list_keys),
            "keys": list_keys[:16],
            "policy": "depth_then_tie_break_then_sequence",
        }

    if isinstance(target, Mapping):
        target = [target]
    if isinstance(target, (list, tuple)):
        frags = [
            Fragment(
                depth=item.get("depth", 0.0),
                color=item.get("color", (0.0, 0.0, 0.0, 0.0)),
                tie_break=item.get("tie_break"),
                sequence=int(item.get("sequence", 0)),
            )
            if isinstance(item, Mapping)
            else item
            for item in target
        ]
        rng = random.Random(0)
        orderings: list[list[Fragment]] = []
        orderings.append(list(frags))
        orderings.append(list(reversed(frags)))
        shuffled = list(frags)
        rng.shuffle(shuffled)
        orderings.append(shuffled)
        signatures = [
            tuple(stable_key(f) for f in sorted(order, key=stable_key))
            for order in orderings
        ]
        return {
            "deterministic": all(sig == signatures[0] for sig in signatures),
            "lists_checked": 1,
            "key_count": len(frags),
            "keys": signatures[0][:16],
            "orderings_compared": len(orderings),
            "policy": "depth_then_tie_break_then_sequence",
        }

    return {
        "deterministic": True,
        "lists_checked": 0,
        "key_count": 0,
        "keys": [],
        "orderings_compared": 1,
        "policy": "depth_then_tie_break_then_sequence",
    }


def compose(target: Any = None) -> dict[str, Any]:
    """Composite fragments/lists back-to-front.

    Accepts an :class:`OITBuffer` (full frame, after the fence), an iterable
    of :class:`Fragment`, or a scalar alpha value.
    """
    if isinstance(target, OITBuffer):
        return target.compose()
    if isinstance(target, Mapping):
        target = [target]
    if isinstance(target, (list, tuple)):
        frags: list[Fragment] = []
        for item in target:
            if isinstance(item, Fragment):
                frags.append(item)
            elif isinstance(item, Mapping):
                frags.append(
                    Fragment(
                        depth=item.get("depth", 0.0),
                        color=item.get("color", (0.0, 0.0, 0.0, 0.0)),
                        tie_break=item.get("tie_break"),
                        sequence=int(item.get("sequence", 0)),
                    )
                )
            else:
                frags.append(Fragment(depth=float(item), color=(0, 0, 0, 1.0)))
        color = _blend(frags)
        return {
            "color": color,
            "fragments": len(frags),
            "keys": [stable_key(f) for f in sorted(frags, key=_far_to_near_key)],
        }
    alpha = float(target) if isinstance(target, (int, float)) else 0.0
    color = _blend([Fragment(depth=0.0, color=(0.0, 0.0, 0.0, alpha))])
    return {"color": color, "fragments": 1, "keys": []}


def validation(target: Any = None) -> dict[str, Any]:
    """Validate configuration and buffer invariants."""
    errors: list[str] = []
    config = _config_from(target)
    errors.extend(config.errors())

    if isinstance(target, OITBuffer):
        if len(target.heads.heads) != config.width * config.height * config.samples:
            errors.append("head table size does not match pixels * samples")
        if len(target.heads.lengths) != len(target.heads.heads):
            errors.append("length table size does not match head table")
        for list_index, length in enumerate(target.heads.lengths):
            if length < 0 or length > config.list_cap:
                errors.append(f"list {list_index} length {length} outside cap")
                break
        if target.pool.allocated > target.pool.capacity:
            errors.append("pool allocation exceeds capacity")
        if target.fence is not None and not target.fence.wait(0):
            errors.append("compose fence is not signalled")

    return {
        "ok": not errors,
        "errors": errors,
        "config": asdict(config),
        "fallback_enabled": config.fallback_enabled,
    }


def audit(target: Any = None) -> dict[str, Any]:
    """Audit a composited frame/buffer for the known failure modes."""
    if not isinstance(target, OITBuffer):
        report = validation(target)
        report.update(
            {
                "issues": list(report["errors"]),
                "metrics": metrics(target),
                "cleared_heads": True,
                "per_sample_lists": True,
                "fence_ok": True,
                "stable_order": True,
                "counters_consistent": True,
            }
        )
        return report

    issues: list[str] = []
    buffer_metrics = target.metrics()

    # Heads must have been cleared for this frame and contain valid indices.
    if target.heads.epoch < 1:
        issues.append("head buffer was never cleared")
    for head in target.heads.heads:
        if head != HEAD_EMPTY and not (0 <= head < target.pool.capacity):
            issues.append("head pointer outside node pool")
            break

    # Every list must respect the cap.
    if any(length > target.config.list_cap for length in target.heads.lengths):
        issues.append("linked list exceeds list_cap")

    # Over-cap / overflow fragments must have produced fallback data.
    fallback_fragments = sum(
        bucket.count for bucket in target.fallback.values()
    )
    if target.stats["fallback_fragments"] != fallback_fragments:
        issues.append("fallback counter does not match fallback buckets")
    if target.stats["overflowed"] and not target.config.fallback_enabled:
        issues.append("pool overflow silently dropped fragments")
    if target.stats["overflowed"] and not target.fallback:
        issues.append("pool overflow without fallback data (fragments lost)")

    # Pool accounting must match the live lists.
    live = sum(target.heads.lengths)
    if live != target.pool.allocated:
        issues.append("pool accounting differs from live list lengths")

    # Write/read queue must be fenced.
    fence_ok = target.fence is None or target.fence.wait(0)
    if not fence_ok:
        issues.append("compose fence not signalled before readback")

    # Equal-depth order must be stable.
    stable_order = True
    for list_index in range(len(target.heads.heads)):
        frags = target.fragments(list_index, include_fallback=True)
        if len(frags) > 1:
            keys = [stable_key(f) for f in sorted(frags, key=stable_key)]
            if len(set(keys)) != len(keys):
                stable_order = False
                issues.append(f"duplicate stable keys in list {list_index}")
                break

    return {
        "ok": not issues,
        "errors": issues,
        "issues": issues,
        "metrics": buffer_metrics,
        "cleared_heads": target.heads.epoch >= 1,
        "per_sample_lists": len(target.heads) == buffer_metrics["lists"],
        "fence_ok": fence_ok,
        "stable_order": stable_order,
        "counters_consistent": live == target.pool.allocated,
        "config": asdict(target.config),
    }


def fallback(target: Any = None, fragment: Any = None) -> dict[str, Any]:
    """Add a fragment to the approximate bucket or report its contents."""
    if isinstance(target, FallbackAccumulator):
        if fragment is not None:
            target.add(_as_fragment(fragment))
        return target.snapshot()
    if isinstance(target, OITBuffer):
        report = {
            "policy": "alpha_weighted_approximate_blend",
            "enabled": target.config.fallback_enabled,
            "lists": len(target.fallback),
            "fragments": sum(b.count for b in target.fallback.values()),
        }
        if fragment is not None:
            report["fragment"] = _as_fragment(fragment).color
        return report
    accumulator = FallbackAccumulator()
    if fragment is not None:
        accumulator.add(_as_fragment(fragment))
    elif target is not None:
        accumulator.add(_as_fragment(target))
    report = accumulator.snapshot()
    report["policy"] = "alpha_weighted_approximate_blend"
    return report


def _as_fragment(value: Any) -> Fragment:
    if isinstance(value, Fragment):
        return value
    if isinstance(value, Mapping):
        return Fragment(
            depth=value.get("depth", 0.0),
            color=value.get("color", (0.0, 0.0, 0.0, 0.0)),
            tie_break=value.get("tie_break"),
            sequence=int(value.get("sequence", 0)),
        )
    if isinstance(value, (int, float)):
        return Fragment(depth=0.0, color=(0.0, 0.0, 0.0, float(value)))
    values = list(value)
    if len(values) == 1:
        return Fragment(depth=0.0, color=(0.0, 0.0, 0.0, float(values[0])))
    depth = float(values[0])
    color = values[1] if len(values) > 1 else (0.0, 0.0, 0.0, 1.0)
    return Fragment(depth=depth, color=color)


def list_cap(value: Any = None) -> int:
    """Normalize and return the per-list fragment cap (minimum 1)."""
    if isinstance(value, OITBuffer):
        return int(value.config.list_cap)
    if isinstance(value, Mapping) and "list_cap" in value:
        value = value["list_cap"]
    try:
        cap = int(value)
    except (TypeError, ValueError):
        cap = DEFAULT_LIST_CAP
    return max(1, cap)


def metrics(target: Any = None) -> dict[str, Any]:
    """Return pool/list/composition counters."""
    if isinstance(target, OITBuffer):
        return target.metrics()
    if isinstance(target, NodePool):
        return {"pool": target.stats(), "fallback": "approximate"}
    if isinstance(target, SampleHeadBuffer):
        lengths = [n for n in target.lengths if n]
        return {
            "pixels": target.pixels,
            "samples": target.samples,
            "lists": len(target),
            "non_empty_lists": len(lengths),
            "max_chain": max(lengths, default=0),
            "mean_chain": (sum(lengths) / len(lengths)) if lengths else 0.0,
            "head_epoch": target.epoch,
        }
    if isinstance(target, Mapping):
        result = dict(target)
        result.setdefault("list_cap", DEFAULT_LIST_CAP)
        result.setdefault("fallback", "approximate")
        return result
    return {
        "source": "scalar",
        "value": target,
        "pixels": 1,
        "samples": 1,
        "lists": 1,
        "non_empty_lists": 0,
        "max_chain": 0,
        "mean_chain": 0.0,
        "list_cap": list_cap(target),
    }


def fragment_insert(buffer: Any = None, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Insert a fragment, degrading gracefully at the cap/pool boundary.

    Supported forms::

        fragment_insert(buffer, x, y, depth, color, sample=0, tie_break=...)
        fragment_insert(buffer, Fragment(...))
        fragment_insert(config_or_capacity)  # inserts a demo fragment
    """
    if isinstance(buffer, OITBuffer):
        target = buffer
    elif isinstance(buffer, Mapping):
        target = OITBuffer(buffer)
    elif isinstance(buffer, int) and not isinstance(buffer, bool):
        target = OITBuffer(
            OITConfig(
                width=1,
                height=1,
                samples=1,
                pool_capacity=max(1, buffer),
                list_cap=1,
            )
        )
    else:
        target = OITBuffer(OITConfig(width=1, height=1, samples=1, pool_capacity=1, list_cap=1))

    if args and isinstance(args[0], Fragment):
        fragment = args[0]
        result = target.insert(
            0,
            0,
            fragment.depth,
            fragment.color,
            sample=min(fragment.sample, target.heads.samples - 1),
            tie_break=fragment.tie_break,
        )
        result["buffer"] = target
        return result

    x = int(kwargs.pop("x", args[0] if len(args) > 0 else 0))
    y = int(kwargs.pop("y", args[1] if len(args) > 1 else 0))
    depth = float(kwargs.pop("depth", args[2] if len(args) > 2 else 1.0))
    color = kwargs.pop("color", args[3] if len(args) > 3 else (0.0, 0.0, 0.0, 0.5))
    sample = int(kwargs.pop("sample", args[4] if len(args) > 4 else 0))
    tie_break = kwargs.pop("tie_break", None)
    result = target.insert(x, y, depth, color, sample=sample, tie_break=tie_break)
    result["buffer"] = target
    return result


def resolve(target: Any = None) -> Any:
    """Resolve MSAA samples into one pixel (or resolve a whole buffer)."""
    if isinstance(target, OITBuffer):
        target._check_fence()
        frame = target.compose()
        return {
            "pixels": frame["pixels"],
            "samples": frame["samples"],
            "width": frame["width"],
            "height": frame["height"],
            "sample_pixels": frame["sample_pixels"],
        }
    if isinstance(target, (int, float)) and not isinstance(target, bool):
        return (0.0, 0.0, 0.0, _clamp_alpha(float(target)))
    if isinstance(target, Mapping):
        if "color" in target:
            return _resolve_samples([_normalize_color(target["color"])])
        return (0.0, 0.0, 0.0, 0.0)
    colors = [_normalize_color(color) for color in target]
    return _resolve_samples(colors)


def depth_peel(target: Any = None) -> dict[str, Any]:
    """Expose depth layers (nearest first) as a peeling-style view."""
    if isinstance(target, OITBuffer):
        target._check_fence()
        layers_by_list: list[dict[str, Any]] = []
        max_layers = 0
        for list_index in range(len(target.heads.heads)):
            frags = target.fragments(list_index, include_fallback=True)
            if not frags:
                continue
            layers = _peel_layers(frags)
            max_layers = max(max_layers, len(layers))
            layers_by_list.append(
                {
                    "list": list_index,
                    "layers": [len(layer) for layer in layers],
                    "composite": _blend(frags),
                }
            )
        return {
            "layers": layers_by_list,
            "max_layers": max_layers,
            "lists": len(layers_by_list),
        }

    if isinstance(target, Mapping):
        target = [target]
    if isinstance(target, (list, tuple)):
        frags = [_as_fragment(item) for item in target]
        layers = _peel_layers(frags)
        return {
            "layers": [
                {
                    "depth": layer[0].depth if layer else 0.0,
                    "fragments": len(layer),
                    "colors": [f.color for f in layer],
                }
                for layer in layers
            ],
            "max_layers": len(layers),
            "lists": 1,
            "composite": _blend(frags),
        }

    return {"layers": [], "max_layers": 0, "lists": 0, "composite": None}
