"""A small box raytracer. It never makes the picture; it gives a painter
something to look at: where things are, which way they face, how light falls.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _hit(ox, oy, oz, dx, dy, dz, cen, half, cs, sn):
    px = ox - cen[0]
    py = oy - cen[1]
    pz = oz - cen[2]
    lox = cs * px - sn * pz
    loz = sn * px + cs * pz
    ldx = cs * dx - sn * dz
    ldz = sn * dx + cs * dz
    o = (lox, py, loz)
    d = (ldx, dy, ldz)
    tn = -1e30
    tf = 1e30
    ax = -1
    sg = 0.0
    for k in range(3):
        if abs(d[k]) < 1e-9:
            if abs(o[k]) > half[k]:
                return 1e30, -1, 0.0
            continue
        inv = 1.0 / d[k]
        t1 = (-half[k] - o[k]) * inv
        t2 = (half[k] - o[k]) * inv
        s = -1.0
        if t1 > t2:
            t1, t2 = t2, t1
            s = 1.0
        if t1 > tn:
            tn = t1
            ax = k
            sg = s
        if t2 < tf:
            tf = t2
        if tn > tf:
            return 1e30, -1, 0.0
    if tn < 1e-4:
        return 1e30, -1, 0.0
    return tn, ax, sg


@njit(cache=True, fastmath=True, parallel=True)
def trace(org, dirs, cen, half, yaw):
    H, W = dirs.shape[0], dirs.shape[1]
    N = cen.shape[0]
    ids = np.full((H, W), -1, np.int32)
    face = np.zeros((H, W), np.int32)
    pos = np.zeros((H, W, 3), np.float32)
    nrm = np.zeros((H, W, 3), np.float32)
    loc = np.zeros((H, W, 3), np.float32)
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    for y in prange(H):
        for x in range(W):
            dx, dy, dz = dirs[y, x, 0], dirs[y, x, 1], dirs[y, x, 2]
            best = 1e30
            bi = -1
            bax = 0
            bsg = 0.0
            for i in range(N):
                t, ax, sg = _hit(org[0], org[1], org[2], dx, dy, dz, cen[i], half[i], cs[i], sn[i])
                if t < best:
                    best = t
                    bi = i
                    bax = ax
                    bsg = sg
            if bi < 0:
                continue
            ids[y, x] = bi
            face[y, x] = bax * 2 + (1 if bsg > 0 else 0)
            wx = org[0] + dx * best
            wy = org[1] + dy * best
            wz = org[2] + dz * best
            pos[y, x, 0] = wx
            pos[y, x, 1] = wy
            pos[y, x, 2] = wz
            px = wx - cen[bi, 0]
            pz = wz - cen[bi, 2]
            loc[y, x, 0] = (cs[bi] * px - sn[bi] * pz) / half[bi, 0]
            loc[y, x, 1] = (wy - cen[bi, 1]) / half[bi, 1]
            loc[y, x, 2] = (sn[bi] * px + cs[bi] * pz) / half[bi, 2]
            if bax == 1:
                nrm[y, x, 1] = bsg
            elif bax == 0:
                nrm[y, x, 0] = cs[bi] * bsg
                nrm[y, x, 2] = -sn[bi] * bsg
            else:
                nrm[y, x, 0] = sn[bi] * bsg
                nrm[y, x, 2] = cs[bi] * bsg
    return ids, face, pos, nrm, loc


@njit(cache=True, fastmath=True, parallel=True)
def shade(ids, pos, nrm, lpos, lcol, lnrm, cen, half, yaw):
    """Direct irradiance with shadows. lnrm zero = omni, else cosine panel."""
    H, W = ids.shape
    N = cen.shape[0]
    K = lpos.shape[0]
    irr = np.zeros((H, W, 3), np.float32)
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    for y in prange(H):
        for x in range(W):
            if ids[y, x] < 0:
                continue
            nx, ny, nz = nrm[y, x, 0], nrm[y, x, 1], nrm[y, x, 2]
            px = pos[y, x, 0] + nx * 2e-3
            py = pos[y, x, 1] + ny * 2e-3
            pz = pos[y, x, 2] + nz * 2e-3
            for k in range(K):
                lx = lpos[k, 0] - px
                ly = lpos[k, 1] - py
                lz = lpos[k, 2] - pz
                d2 = lx * lx + ly * ly + lz * lz
                d = np.sqrt(d2)
                ndl = (nx * lx + ny * ly + nz * lz) / d
                if ndl <= 0:
                    continue
                em = 1.0
                if lnrm[k, 0] != 0 or lnrm[k, 1] != 0 or lnrm[k, 2] != 0:
                    em = (-(lnrm[k, 0] * lx + lnrm[k, 1] * ly + lnrm[k, 2] * lz) / d + 0.18) / 1.18
                    if em <= 0:
                        continue
                occ = False
                for i in range(N):
                    t, ax, sg = _hit(px, py, pz, lx, ly, lz, cen[i], half[i], cs[i], sn[i])
                    if t < 0.999:
                        occ = True
                        break
                if occ:
                    continue
                f = ndl * em / (d2 + 0.05)
                irr[y, x, 0] += lcol[k, 0] * f
                irr[y, x, 1] += lcol[k, 1] * f
                irr[y, x, 2] += lcol[k, 2] * f
    return irr


class Camera:
    def __init__(self, pos, target, fov, W, H):
        self.pos = np.asarray(pos, np.float32)
        f = np.asarray(target, np.float32) - self.pos
        self.f = f / np.linalg.norm(f)
        r = np.cross(np.array([0, 1, 0], np.float32), self.f)
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.f, self.r)
        self.W, self.H = W, H
        self.fl = (H / 2) / np.tan(np.radians(fov) / 2)

    def rays(self):
        y, x = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        a = (x + 0.5 - self.W / 2) / self.fl
        b = -(y + 0.5 - self.H / 2) / self.fl
        d = self.f[None, None] + a[..., None] * self.r + b[..., None] * self.u
        d /= np.linalg.norm(d, axis=2, keepdims=True)
        return d.astype(np.float32)

    def project(self, P):
        v = np.asarray(P, np.float32) - self.pos
        z = v @ self.f
        return np.stack([self.W / 2 + self.fl * (v @ self.r) / z,
                         self.H / 2 - self.fl * (v @ self.u) / z, z], axis=-1)


class Boxes:
    def __init__(self):
        self.c, self.h, self.y, self.names = [], [], [], {}

    def add(self, name, lo, hi, yaw=0.0):
        lo = np.asarray(lo, np.float32)
        hi = np.asarray(hi, np.float32)
        self.c.append((lo + hi) / 2)
        self.h.append((hi - lo) / 2)
        self.y.append(yaw)
        self.names.setdefault(name, []).append(len(self.c) - 1)
        return len(self.c) - 1

    def arrays(self):
        return (np.asarray(self.c, np.float32), np.asarray(self.h, np.float32),
                np.asarray(self.y, np.float32))

    def mask(self, ids, name):
        return np.isin(ids, self.names[name])


def _h(ix, iy, seed):
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF) / 16777216.0


def vnoise(x, y, seed=0):
    xi = np.floor(x)
    yi = np.floor(y)
    fx = x - xi
    fy = y - yi
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    a = _h(xi, yi, seed)
    b = _h(xi + 1, yi, seed)
    c = _h(xi, yi + 1, seed)
    d = _h(xi + 1, yi + 1, seed)
    return ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy).astype(np.float32)


def fbm(x, y, octaves=4, seed=0):
    out = 0
    amp = 0.5
    for o in range(octaves):
        out = out + amp * vnoise(x * 2 ** o, y * 2 ** o, seed + o * 17)
        amp *= 0.5
    return out / (1 - 0.5 ** octaves)
