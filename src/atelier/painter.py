"""Painting from a study. A painter looks at a reference (colour, direction of
form, where one thing ends and another begins) and decides, stroke by stroke,
where the picture is still wrong and what mark would fix it.

`Handling` is a painter's hand as data: how loaded the brush is, how far a
stroke runs, how much the colour wanders. Different artists = different
handling + different rules for the direction field + different subjects.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit
from .brush import stamp


@dataclass
class Handling:
    width: float = 12.0         # half-width of the brush, px
    length: float = 8.0         # max stroke length, in brush half-widths
    tol: float = 0.20           # how far the subject may drift before lifting off
    err: float = 0.08           # leave alone anything already this close
    spacing: float = 1.0        # candidate spacing in half-widths
    opacity: float = 0.95
    dry_dark: float = 0.5
    dry_light: float = 0.15
    thick_dark: float = 0.15
    thick_light: float = 0.9
    rag: float = 0.5
    jitter: float = 0.07        # bristle-to-bristle value variation
    soft: float = 0.12
    coljit: float = 0.06        # stroke-to-stroke value wander
    warm: float = 0.05          # stroke-to-stroke temperature wander
    wobble: float = 0.15        # radians of directional looseness


@njit(cache=True, fastmath=True)
def _layer(canvas, height, grain, ref, orient, region, wmap, cand, base_w,
           len_max, tol, err, opacity, dry_dark, dry_light, thick_dark,
           thick_light, rag, jitter, soft, coljit, warm, wobble, seed):
    np.random.seed(seed)
    H, W = ref.shape[0], ref.shape[1]
    fw = np.zeros((40, 2), np.float32)
    bw = np.zeros((40, 2), np.float32)
    col = np.zeros(3, np.float32)
    made = 0
    for ci in range(cand.shape[0]):
        cx = cand[ci, 0]
        cy = cand[ci, 1]
        ix = int(cx)
        iy = int(cy)
        if ix < 0 or iy < 0 or ix >= W or iy >= H:
            continue
        w = base_w * wmap[iy, ix]
        if w <= 0:
            continue
        reg = region[iy, ix]
        ang = orient[iy, ix] + wobble * np.random.randn()
        ca = np.cos(ang)
        sa = np.sin(ang)
        # fit the brush to the shape it is in
        for _ in range(3):
            ok = True
            for sgn in (-1.0, 1.0):
                qx = int(cx - sa * w * sgn)
                qy = int(cy + ca * w * sgn)
                if qx < 0 or qy < 0 or qx >= W or qy >= H:
                    continue
                if region[qy, qx] != reg:
                    ok = False
            if ok:
                break
            w *= 0.62
        # is the picture already right here?
        e = 0.0
        ns = 0
        for k in range(5):
            qx = ix
            qy = iy
            if k == 1:
                qx = int(cx + ca * w * 0.6)
                qy = int(cy + sa * w * 0.6)
            elif k == 2:
                qx = int(cx - ca * w * 0.6)
                qy = int(cy - sa * w * 0.6)
            elif k == 3:
                qx = int(cx - sa * w * 0.5)
                qy = int(cy + ca * w * 0.5)
            elif k == 4:
                qx = int(cx + sa * w * 0.5)
                qy = int(cy - ca * w * 0.5)
            if qx < 0 or qy < 0 or qx >= W or qy >= H:
                continue
            if region[qy, qx] != reg:
                continue
            e += (abs(canvas[qy, qx, 0] - ref[qy, qx, 0]) + abs(canvas[qy, qx, 1] - ref[qy, qx, 1]) +
                  abs(canvas[qy, qx, 2] - ref[qy, qx, 2]))
            ns += 1
        if ns == 0 or e / ns < err:
            continue
        r0 = ref[iy, ix, 0]
        g0 = ref[iy, ix, 1]
        b0 = ref[iy, ix, 2]
        step = max(2.0, w * 0.9)
        maxlen = len_max * w
        # walk forward and backward along the form
        nf = 0
        nbk = 0
        for direction in (1.0, -1.0):
            x = cx
            y = cy
            ddx = ca * direction
            ddy = sa * direction
            run = 0.0
            cnt = 0
            while run < maxlen * 0.5 and cnt < 39:
                x += ddx * step
                y += ddy * step
                qx = int(x)
                qy = int(y)
                if qx < 0 or qy < 0 or qx >= W or qy >= H:
                    break
                if region[qy, qx] != reg:
                    break
                if (abs(ref[qy, qx, 0] - r0) + abs(ref[qy, qx, 1] - g0) + abs(ref[qy, qx, 2] - b0)) > tol:
                    break
                if direction > 0:
                    fw[cnt, 0] = x
                    fw[cnt, 1] = y
                else:
                    bw[cnt, 0] = x
                    bw[cnt, 1] = y
                cnt += 1
                run += step
                a2 = orient[qy, qx]
                ex = np.cos(a2)
                ey = np.sin(a2)
                if ex * ddx + ey * ddy < 0:
                    ex = -ex
                    ey = -ey
                ddx = ddx * 0.5 + ex * 0.5
                ddy = ddy * 0.5 + ey * 0.5
                nn = np.sqrt(ddx * ddx + ddy * ddy) + 1e-6
                ddx /= nn
                ddy /= nn
            if direction > 0:
                nf = cnt
            else:
                nbk = cnt
        n = nf + nbk + 1
        if n < 2:
            # a dab
            n = 2
            pts = np.zeros((2, 2), np.float32)
            pts[0, 0] = cx - ca * w * 0.6
            pts[0, 1] = cy - sa * w * 0.6
            pts[1, 0] = cx + ca * w * 0.6
            pts[1, 1] = cy + sa * w * 0.6
        else:
            pts = np.zeros((n, 2), np.float32)
            for k in range(nbk):
                pts[k, 0] = bw[nbk - 1 - k, 0]
                pts[k, 1] = bw[nbk - 1 - k, 1]
            pts[nbk, 0] = cx
            pts[nbk, 1] = cy
            for k in range(nf):
                pts[nbk + 1 + k, 0] = fw[k, 0]
                pts[nbk + 1 + k, 1] = fw[k, 1]
        hw = np.zeros(n, np.float32)
        for k in range(n):
            t = k / (n - 1)
            hw[k] = w * (0.8 + 0.2 * np.sin(np.pi * t))
        lum = 0.3 * r0 + 0.55 * g0 + 0.15 * b0
        vj = 1.0 + coljit * np.random.randn() * (1.0 - 0.65 * lum)
        tj = warm * np.random.randn()
        col[0] = min(1.0, max(0.0, r0 * vj * (1 + tj)))
        col[1] = min(1.0, max(0.0, g0 * vj))
        col[2] = min(1.0, max(0.0, b0 * vj * (1 - tj)))
        dry = dry_dark + (dry_light - dry_dark) * lum
        thick = thick_dark + (thick_light - thick_dark) * lum
        stamp(canvas, height, grain, pts, hw, col, opacity,
              np.random.randint(1, 1 << 30), dry, rag, jitter, soft, thick)
        made += 1
    return made


def candidates(W, H, spacing, rng, key=None, mask=None):
    """Jittered grid of places to consider. `key` (H,W) sorts them: low first."""
    xs = np.arange(spacing / 2, W, spacing)
    ys = np.arange(spacing / 2, H, spacing)
    gx, gy = np.meshgrid(xs, ys)
    p = np.stack([gx.ravel(), gy.ravel()], 1)
    p += rng.uniform(-spacing / 2, spacing / 2, p.shape)
    p[:, 0] = np.clip(p[:, 0], 0, W - 1)
    p[:, 1] = np.clip(p[:, 1], 0, H - 1)
    rng.shuffle(p)
    ix = p[:, 0].astype(int)
    iy = p[:, 1].astype(int)
    if mask is not None:
        keep = mask[iy, ix]
        p, ix, iy = p[keep], ix[keep], iy[keep]
    if key is not None:
        k = key[iy, ix] + rng.uniform(0, 1e-3, len(p))
        p = p[np.argsort(k, kind="stable")]
    return p.astype(np.float32)


def paint_layer(cv, ref, orient, region, h: Handling, wmap=None, key=None, mask=None):
    """One pass over the canvas with one brush. Returns strokes laid."""
    if wmap is None:
        wmap = np.ones(ref.shape[:2], np.float32)
    cand = candidates(cv.W, cv.H, max(1.5, h.width * h.spacing), cv.rng, key, mask)
    n = _layer(cv.rgb, cv.height, cv.grain, ref, orient, region, wmap, cand,
               h.width, h.length, h.tol, h.err, h.opacity, h.dry_dark, h.dry_light,
               h.thick_dark, h.thick_light, h.rag, h.jitter, h.soft, h.coljit,
               h.warm, h.wobble, int(cv.rng.integers(1, 2 ** 30)))
    cv.count += n
    return n
