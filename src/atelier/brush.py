"""Brush engine: every mark on the canvas is a simulated bristle stroke.

A stroke is a polyline with a half-width per point. Each pixel under it is
assigned to a bristle; bristles carry their own load, colour deviation and
ragged start/end, and they run dry along the stroke. Paint also builds a
height map so thick passages catch light when the picture is varnished.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _hash(a, b):
    h = (a * 374761393 + b * 668265263) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777216.0


@njit(cache=True, fastmath=True)
def _seg_d2(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = ((px - ax) * dx + (py - ay) * dy) / l2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    qx = px - ax - t * dx
    qy = py - ay - t * dy
    return qx * qx + qy * qy


@njit(cache=True, fastmath=True)
def stamp(canvas, height, grain, pts, hw, col, opacity, seed,
          dry, rag, jitter, soft, thick):
    """Lay one stroke. pts (n,2) xy pixels, hw (n,) half widths, col (3,)."""
    H = canvas.shape[0]
    W = canvas.shape[1]
    n = pts.shape[0]
    if n < 2:
        return
    cum = np.zeros(n, np.float32)
    wmax = hw[0]
    for i in range(1, n):
        cum[i] = cum[i - 1] + np.sqrt((pts[i, 0] - pts[i - 1, 0]) ** 2 +
                                      (pts[i, 1] - pts[i - 1, 1]) ** 2)
        if hw[i] > wmax:
            wmax = hw[i]
    L = cum[n - 1]
    if L < 0.5:
        return
    nb = int(wmax * 1.6) + 2          # bristles across the brush
    for j in range(n - 1):
        ax = pts[j, 0]
        ay = pts[j, 1]
        bx = pts[j + 1, 0]
        by = pts[j + 1, 1]
        dx = bx - ax
        dy = by - ay
        sl2 = dx * dx + dy * dy
        if sl2 < 1e-8:
            continue
        sl = np.sqrt(sl2)
        ws = max(hw[j], hw[j + 1]) + 1.0
        x0 = max(0, int(min(ax, bx) - ws))
        x1 = min(W - 1, int(max(ax, bx) + ws) + 1)
        y0 = max(0, int(min(ay, by) - ws))
        y1 = min(H - 1, int(max(ay, by) + ws) + 1)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                px = x + 0.5 - ax
                py = y + 0.5 - ay
                t = (px * dx + py * dy) / sl2
                if j == 0 and t < 0.0:
                    continue
                if j == n - 2 and t > 1.0:
                    continue
                tc = min(max(t, 0.0), 1.0)
                qx = px - tc * dx
                qy = py - tc * dy
                d2 = qx * qx + qy * qy
                w = hw[j] + (hw[j + 1] - hw[j]) * tc
                if d2 > (w + 0.5) * (w + 0.5):
                    continue
                if j > 0:
                    if _seg_d2(x + 0.5, y + 0.5, pts[j - 1, 0], pts[j - 1, 1], ax, ay) <= d2:
                        continue
                if j < n - 2:
                    if _seg_d2(x + 0.5, y + 0.5, bx, by, pts[j + 2, 0], pts[j + 2, 1]) < d2:
                        continue
                d = np.sqrt(d2)
                side = px * dy - py * dx
                v = d / max(w, 0.3)
                if side < 0:
                    v = -v
                s = cum[j] + tc * sl
                b = int((v * 0.5 + 0.5) * nb)
                h1 = _hash(seed, b)
                h2 = _hash(seed + 7919, b)
                h3 = _hash(seed + 104729, b)
                h4 = _hash(seed + 1299709, b)
                # ragged lift-on / lift-off
                if s < h2 * rag * w * 2.0 or s > L - h3 * rag * w * 2.0:
                    continue
                g = grain[y, x]
                load = 1.0 - dry * (0.25 + 0.75 * s / L)
                ab = (load + 0.35 - (h1 * 0.6 + g * 0.6)) * 8.0
                if ab <= 0.0:
                    continue
                if ab > 1.0:
                    ab = 1.0
                ae = (1.0 - abs(v)) / soft
                aa = w - d + 0.5
                if aa < ae:
                    ae = aa
                if ae <= 0.0:
                    continue
                if ae > 1.0:
                    ae = 1.0
                a = opacity * ab * ae
                k = 1.0 + jitter * (h4 - 0.5) * 2.0
                canvas[y, x, 0] = canvas[y, x, 0] * (1 - a) + min(1.0, col[0] * k) * a
                canvas[y, x, 1] = canvas[y, x, 1] * (1 - a) + min(1.0, col[1] * k) * a
                canvas[y, x, 2] = canvas[y, x, 2] * (1 - a) + min(1.0, col[2] * k) * a
                av = abs(v)
                ridge = 0.0
                if av > 0.75:
                    ridge = (av - 0.75) * 2.0
                hh = thick * (0.5 + 0.5 * h1 + ridge) * (0.4 + 0.6 * load)
                height[y, x] = height[y, x] * (1 - a * 0.85) + hh * a


@njit(cache=True, fastmath=True)
def _dab(canvas, height, cx, cy, r, col, opacity, hard, thick):
    H = canvas.shape[0]
    W = canvas.shape[1]
    for y in range(max(0, int(cy - r)), min(H - 1, int(cy + r) + 1) + 1):
        for x in range(max(0, int(cx - r)), min(W - 1, int(cx + r) + 1) + 1):
            d = np.sqrt((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2) / r
            if d >= 1.0:
                continue
            a = (1.0 - d) / (1.0 - hard + 1e-3)
            if a > 1.0:
                a = 1.0
            a = a * a * (3 - 2 * a) * opacity
            for c in range(3):
                canvas[y, x, c] = canvas[y, x, c] * (1 - a) + col[c] * a
            height[y, x] = height[y, x] * (1 - a) + thick * a


def smooth_path(points, step=3.0):
    """Catmull-Rom through control points, resampled about every `step` px."""
    p = np.asarray(points, np.float32)
    if len(p) < 3:
        return p
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(len(p) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        m = max(2, int(np.linalg.norm(p2 - p1) / step))
        for t in np.linspace(0, 1, m, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                              (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-1])
    return np.asarray(out, np.float32)


def make_grain(H, W, seed=0):
    """Canvas tooth: a linen weave plus noise, 0..1."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    weave = 0.5 + 0.25 * np.sin(x * 2.1) * np.sin(y * 2.1) + 0.12 * np.sin(x * 0.7 + y * 0.05)
    g = weave + rng.normal(0, 0.16, (H, W)).astype(np.float32)
    return np.clip(g, 0, 1).astype(np.float32)


class Canvas:
    def __init__(self, W, H, ground=(0.5, 0.5, 0.5), seed=0):
        self.W, self.H = W, H
        self.rgb = np.empty((H, W, 3), np.float32)
        self.rgb[:] = np.asarray(ground, np.float32)
        self.height = np.zeros((H, W), np.float32)
        self.grain = make_grain(H, W, seed)
        self.rng = np.random.default_rng(seed + 1)
        self.count = 0

    def stroke(self, points, width, color, opacity=1.0, dry=0.2, rag=0.5,
               jitter=0.06, soft=0.12, thick=0.5, taper=0.0, smooth=True):
        """Hand-placed stroke. width is full width in px (scalar or per point)."""
        p = smooth_path(points) if smooth else np.asarray(points, np.float32)
        n = len(p)
        if n < 2:
            return
        if np.isscalar(width):
            hw = np.full(n, width * 0.5, np.float32)
        else:
            w = np.asarray(width, np.float32) * 0.5
            hw = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(w)), w).astype(np.float32)
        if taper > 0:
            t = np.linspace(0, 1, n)
            hw = hw * (1 - taper + taper * np.sin(np.pi * t) ** 0.6).astype(np.float32)
        stamp(self.rgb, self.height, self.grain, np.ascontiguousarray(p), hw,
              np.asarray(color, np.float32), opacity, int(self.rng.integers(1, 2 ** 30)),
              dry, rag, jitter, soft, thick)
        self.count += 1

    def dab(self, x, y, r, color, opacity=1.0, hard=0.5, thick=0.5):
        """Touch of a round brush."""
        _dab(self.rgb, self.height, float(x), float(y), float(r), np.asarray(color, np.float32),
             opacity, hard, thick)
        self.count += 1

    def varnish(self, strength=0.5, spec=0.10, light=(-0.55, -0.83)):
        """Rake light across the paint relief; returns uint8 image array."""
        h = self.height
        hb = h.copy()
        for _ in range(2):
            hb[1:-1, 1:-1] = (hb[1:-1, 1:-1] * 4 + hb[:-2, 1:-1] + hb[2:, 1:-1] +
                              hb[1:-1, :-2] + hb[1:-1, 2:]) / 8
        hb = hb + (self.grain - 0.5) * 0.05
        gy, gx = np.gradient(hb)
        lam = np.clip(-(gx * light[0] + gy * light[1]) * 3.0, -1, 1)
        out = self.rgb * (1 + strength * lam[..., None]) + spec * np.clip(lam, 0, 1)[..., None] ** 3
        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)

    def save(self, path, **kw):
        from PIL import Image
        Image.fromarray(self.varnish(**kw)).save(path)
