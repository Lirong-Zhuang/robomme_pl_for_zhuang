"""Point-guided OpenCV segmentation for RoboMME's fixed front camera."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import cv2
import numpy as np

from .types import CameraFrame
from .types import SegmentationResult


_RGB_COLORS = {
    "red": (220, 45, 45),
    "green": (45, 180, 70),
    "blue": (45, 90, 220),
    "yellow": (235, 210, 40),
    "orange": (235, 130, 35),
    "purple": (145, 65, 190),
    "cyan": (40, 190, 200),
    "magenta": (210, 55, 175),
    "white": (235, 235, 235),
    "black": (25, 25, 25),
}
_COLOR_ALIASES = {
    "红": "red",
    "红色": "red",
    "绿": "green",
    "绿色": "green",
    "蓝": "blue",
    "蓝色": "blue",
    "黄": "yellow",
    "黄色": "yellow",
    "橙": "orange",
    "橙色": "orange",
    "紫": "purple",
    "紫色": "purple",
    "青": "cyan",
    "青色": "cyan",
    "白": "white",
    "白色": "white",
    "黑": "black",
    "黑色": "black",
}


@dataclass(frozen=True)
class SegmentationProfile:
    min_area: int = 50
    max_area_fraction: float = 0.35
    search_radius_px: int = 80
    depth_tolerance_m: float = 0.03
    color_delta_e: float = 58.0
    morphology_px: int = 5


DEFAULT_PROFILES = {
    "compact_colored": SegmentationProfile(
        min_area=50,
        max_area_fraction=0.08,
        search_radius_px=70,
        depth_tolerance_m=0.025,
        color_delta_e=55,
        morphology_px=3,
    ),
    "peg_colored": SegmentationProfile(
        min_area=50,
        max_area_fraction=0.12,
        search_radius_px=100,
        depth_tolerance_m=0.035,
        color_delta_e=58,
        morphology_px=3,
    ),
    "container": SegmentationProfile(
        min_area=100,
        max_area_fraction=0.35,
        search_radius_px=110,
        depth_tolerance_m=0.05,
        morphology_px=7,
    ),
    "button": SegmentationProfile(
        min_area=50,
        max_area_fraction=0.08,
        search_radius_px=65,
        depth_tolerance_m=0.018,
        morphology_px=3,
    ),
    "target": SegmentationProfile(
        min_area=80,
        max_area_fraction=0.20,
        search_radius_px=90,
        depth_tolerance_m=0.025,
        morphology_px=5,
    ),
    "box": SegmentationProfile(
        min_area=100,
        max_area_fraction=0.30,
        search_radius_px=110,
        depth_tolerance_m=0.05,
        morphology_px=7,
    ),
    "grid_targets": SegmentationProfile(
        min_area=50,
        max_area_fraction=0.08,
        search_radius_px=0,
        morphology_px=5,
    ),
}


def _morphology(mask: np.ndarray, size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8 > 0


def _components(mask: np.ndarray, profile: SegmentationProfile) -> list[dict[str, Any]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    image_area = mask.shape[0] * mask.shape[1]
    result = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < profile.min_area:
            continue
        if area > profile.max_area_fraction * image_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        result.append(
            {
                "mask": labels == label,
                "area": area,
                "bbox_xywh": (x, y, width, height),
                "center_xy": tuple(float(item) for item in centroids[label]),
            }
        )
    return result


def _distance_to_mask(mask: np.ndarray, point_xy: tuple[float, float]) -> float:
    x = int(np.clip(round(point_xy[0]), 0, mask.shape[1] - 1))
    y = int(np.clip(round(point_xy[1]), 0, mask.shape[0] - 1))
    if mask[y, x]:
        return 0.0
    inverse = (~mask).astype(np.uint8)
    distance = cv2.distanceTransform(inverse, cv2.DIST_L2, 3)
    return float(distance[y, x])


@dataclass
class OpenCVSegmenter:
    """Segment target masks without predicting any metric object geometry."""

    profiles: dict[str, SegmentationProfile] = field(
        default_factory=lambda: dict(DEFAULT_PROFILES)
    )

    def _profile(self, name: str) -> SegmentationProfile:
        try:
            return self.profiles[name]
        except KeyError as error:
            raise ValueError(f"unknown OpenCV segmentation profile: {name!r}") from error

    def segment(
        self,
        frame: CameraFrame,
        *,
        point_xy: tuple[float, float],
        color_hint: str | None,
        profile_name: str,
    ) -> SegmentationResult:
        profile = self._profile(profile_name)
        x, y = point_xy
        height, width = frame.rgb.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"point_xy={point_xy} is outside the image")
        if color_hint:
            return self._segment_colored(
                frame, point_xy, color_hint, profile_name, profile
            )
        if profile_name in {"compact_colored", "peg_colored"}:
            raise ValueError(f"profile {profile_name!r} requires color_hint")
        return self._segment_depth(frame, point_xy, profile_name, profile)

    def _segment_colored(
        self,
        frame: CameraFrame,
        point_xy: tuple[float, float],
        color_hint: str,
        profile_name: str,
        profile: SegmentationProfile,
    ) -> SegmentationResult:
        normalized_color = _COLOR_ALIASES.get(color_hint.strip().lower(), color_hint.strip().lower())
        if normalized_color not in _RGB_COLORS:
            raise ValueError(
                f"unsupported color_hint {color_hint!r}; expected one of "
                f"{sorted(_RGB_COLORS)}"
            )
        lab = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        prototype = np.asarray(_RGB_COLORS[normalized_color], dtype=np.uint8).reshape(1, 1, 3)
        prototype_lab = cv2.cvtColor(prototype, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
        delta = np.linalg.norm(lab - prototype_lab, axis=2)
        raw_mask = delta <= profile.color_delta_e
        raw_mask &= np.isfinite(frame.depth_m) & (frame.depth_m > 0)
        candidates = _components(
            _morphology(raw_mask, profile.morphology_px), profile
        )
        if not candidates:
            raise ValueError(
                f"OpenCV found no {normalized_color!r} component for {profile_name}"
            )
        for candidate in candidates:
            distance = _distance_to_mask(candidate["mask"], point_xy)
            mean_delta = float(np.mean(delta[candidate["mask"]]))
            candidate["distance_px"] = distance
            candidate["mean_color_delta"] = mean_delta
            candidate["score"] = (
                distance / max(profile.search_radius_px, 1)
                + 0.35 * mean_delta / profile.color_delta_e
            )
        best = min(candidates, key=lambda item: item["score"])
        if best["distance_px"] > profile.search_radius_px:
            raise ValueError(
                f"nearest color component is {best['distance_px']:.1f}px from VLM point"
            )
        confidence = float(np.clip(1.0 - best["score"] / 1.5, 0.05, 1.0))
        return SegmentationResult(
            mask=best["mask"],
            confidence=confidence,
            profile=profile_name,
            point_xy=point_xy,
            diagnostics={
                "color": normalized_color,
                "area_px": best["area"],
                "bbox_xywh": best["bbox_xywh"],
                "point_distance_px": best["distance_px"],
                "mean_lab_distance": best["mean_color_delta"],
                "candidate_count": len(candidates),
            },
        )

    def _seed_pixels(
        self,
        depth_m: np.ndarray,
        point_xy: tuple[float, float],
        radius: int,
    ) -> list[tuple[int, int]]:
        height, width = depth_m.shape
        px, py = int(round(point_xy[0])), int(round(point_xy[1]))
        y0, y1 = max(0, py - radius), min(height, py + radius + 1)
        x0, x1 = max(0, px - radius), min(width, px + radius + 1)
        ys, xs = np.nonzero(
            np.isfinite(depth_m[y0:y1, x0:x1]) & (depth_m[y0:y1, x0:x1] > 0)
        )
        if not len(xs):
            return []
        xs = xs + x0
        ys = ys + y0
        distances = np.square(xs - px) + np.square(ys - py)
        order = np.argsort(distances)
        # Use spatially and depth-diverse seeds so a slightly inaccurate point
        # can still recover a nearby object instead of only the table plane.
        seeds: list[tuple[int, int]] = []
        seed_depths: list[float] = []
        for index in order:
            seed = (int(xs[index]), int(ys[index]))
            value = float(depth_m[seed[1], seed[0]])
            if any(
                abs(value - previous) < 0.008
                and (seed[0] - old[0]) ** 2 + (seed[1] - old[1]) ** 2 < 15**2
                for old, previous in zip(seeds, seed_depths, strict=True)
            ):
                continue
            seeds.append(seed)
            seed_depths.append(value)
            if len(seeds) == 16:
                break
        return seeds

    def _depth_region(
        self,
        depth_m: np.ndarray,
        seed_xy: tuple[int, int],
        profile: SegmentationProfile,
        point_xy: tuple[float, float],
    ) -> np.ndarray:
        working = depth_m.astype(np.float32, copy=True)
        working[~np.isfinite(working)] = 0
        flood_mask = np.zeros((working.shape[0] + 2, working.shape[1] + 2), np.uint8)
        flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        cv2.floodFill(
            working,
            flood_mask,
            seedPoint=seed_xy,
            newVal=0,
            loDiff=profile.depth_tolerance_m,
            upDiff=profile.depth_tolerance_m,
            flags=flags,
        )
        region = flood_mask[1:-1, 1:-1] > 0
        yy, xx = np.ogrid[: region.shape[0], : region.shape[1]]
        radius = profile.search_radius_px * 2
        roi = (xx - point_xy[0]) ** 2 + (yy - point_xy[1]) ** 2 <= radius**2
        return _morphology(region & roi, profile.morphology_px)

    def _segment_depth(
        self,
        frame: CameraFrame,
        point_xy: tuple[float, float],
        profile_name: str,
        profile: SegmentationProfile,
    ) -> SegmentationResult:
        depth_m = frame.depth_m
        seeds = self._seed_pixels(depth_m, point_xy, profile.search_radius_px)
        candidates: list[dict[str, Any]] = []
        for seed in seeds:
            for component in _components(
                self._depth_region(depth_m, seed, profile, point_xy), profile
            ):
                distance = _distance_to_mask(component["mask"], point_xy)
                component["distance_px"] = distance
                component["seed_xy"] = seed
                component["score"] = distance / max(profile.search_radius_px, 1)
                candidates.append(component)
        if not candidates:
            raise ValueError(
                f"OpenCV found no depth-connected component near {point_xy} for {profile_name}"
            )
        best = min(candidates, key=lambda item: (item["score"], -item["area"]))
        if best["distance_px"] > profile.search_radius_px:
            raise ValueError(
                f"nearest depth component is {best['distance_px']:.1f}px from VLM point"
            )
        confidence = float(np.clip(1.0 - best["score"], 0.05, 1.0))
        return SegmentationResult(
            mask=best["mask"],
            confidence=confidence,
            profile=profile_name,
            point_xy=point_xy,
            diagnostics={
                "area_px": best["area"],
                "bbox_xywh": best["bbox_xywh"],
                "point_distance_px": best["distance_px"],
                "seed_xy": best["seed_xy"],
                "candidate_count": len(candidates),
            },
        )

    def segment_scene_targets(self, frame: CameraFrame) -> tuple[SegmentationResult, ...]:
        """Find all compact, saturated grid targets for the no-point actions."""

        profile_name = "grid_targets"
        profile = self._profile(profile_name)
        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
        valid_depth = np.isfinite(frame.depth_m) & (frame.depth_m > 0)
        foreground = (hsv[..., 1] >= 45) & (hsv[..., 2] >= 45) & valid_depth
        components = _components(
            _morphology(foreground, profile.morphology_px), profile
        )
        if not components:
            raise ValueError("OpenCV found no grid target candidates")
        components.sort(key=lambda item: (item["center_xy"][1], item["center_xy"][0]))
        return tuple(
            SegmentationResult(
                mask=component["mask"],
                confidence=0.5,
                profile=profile_name,
                diagnostics={
                    "area_px": component["area"],
                    "bbox_xywh": component["bbox_xywh"],
                    "center_xy": component["center_xy"],
                    "candidate_count": len(components),
                },
            )
            for component in components
        )
