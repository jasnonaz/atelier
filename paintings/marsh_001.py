"""Odile Marsh, no. 1 — "The 11:52 Has Gone"

A drinks machine on a rural platform, snow, nobody. Footprints go to the
machine, then to the platform edge, and stop.
"""
import sys, time
import numpy as np
from PIL import Image
from atelier.rt import Camera, Boxes, trace, shade, fbm, vnoise
from atelier.brush import Canvas
from atelier.painter import Handling, paint_layer

W, H = 1600, 2000
SEED = 1152
rng = np.random.default_rng(SEED)
cam = Camera(pos=(0.4, 1.6, -1.0), target=(0.62, 0.72, 8.6), fov=33, W=W, H=H)

# ---------------------------------------------------------------- the place
B = Boxes()
B.add("platform", (-0.9, -1.3, -10), (8, 0, 140))
B.add("bed", (-90, -1.4, -10), (-0.9, -1.0, 500))
B.add("rail", (-2.47, -1.0, -10), (-2.40, -0.87, 500))
B.add("rail", (-3.90, -1.0, -10), (-3.83, -0.87, 500))
B.add("machine", (1.5, 0, 8.6), (2.5, 1.84, 9.4))
B.add("msnow", (1.5, 1.84, 8.6), (2.5, 1.93, 9.4))
B.add("wall", (-0.25, 0, 9.42), (3.9, 2.4, 9.52))
B.add("roof", (-0.5, 2.4, 9.1), (4.2, 2.5, 13))
B.add("rsnow", (-0.5, 2.5, 9.1), (4.2, 2.66, 13))
B.add("post", (-0.80, 0, 5.4), (-0.71, 3.7, 5.49))
B.add("post", (-1.05, 3.7, 5.36), (-0.40, 3.78, 5.54))
B.add("psnow", (-1.05, 3.78, 5.36), (-0.40, 3.85, 5.54))
B.add("bench", (-0.05, 0.42, 8.95), (1.25, 0.47, 9.38))
B.add("bench", (-0.05, 0.47, 9.33), (1.25, 0.95, 9.38))
B.add("bench", (0.0, 0, 9.0), (0.06, 0.42, 9.36))
B.add("bench", (1.14, 0, 9.0), (1.20, 0.42, 9.36))
B.add("bsnow", (-0.05, 0.47, 8.95), (1.25, 0.55, 9.33))
for z in (16, 46, 76, 106):
    B.add("mast", (-5.75, -1, z), (-5.55, 6.2, z + 0.2))
    B.add("mast", (-5.6, 5.3, z + 0.05), (-2.5, 5.38, z + 0.13))
B.add("signal", (-4.75, -1, 40), (-4.65, 3.3, 40.1))
B.add("signal", (-4.92, 3.0, 39.95), (-4.48, 4.0, 40.05))
cen, half, yaw = B.arrays()

t0 = time.time()
dirs = cam.rays()
ids, face, pos, nrm, loc = trace(cam.pos, dirs, cen, half, yaw)

snowy = (B.mask(ids, "platform") | B.mask(ids, "bed") | B.mask(ids, "rsnow") |
         B.mask(ids, "msnow") | B.mask(ids, "bsnow") | B.mask(ids, "psnow")) & (face == 3)
caps = B.mask(ids, "rsnow") | B.mask(ids, "msnow") | B.mask(ids, "bsnow") | B.mask(ids, "psnow")
# drifts: tilt the snow's normal so raking light finds it
wx, wz = pos[..., 0], pos[..., 2]
e = 0.04
hx = fbm(wx * 1.7 + e, wz * 1.3, 4, 3) - fbm(wx * 1.7 - e, wz * 1.3, 4, 3)
hz = fbm(wx * 1.7, wz * 1.3 + e, 4, 3) - fbm(wx * 1.7, wz * 1.3 - e, 4, 3)
k = 0.9
n2 = nrm.copy()
n2[..., 0] = np.where(snowy, -hx / (2 * e) * k * 0.07, nrm[..., 0])
n2[..., 2] = np.where(snowy, -hz / (2 * e) * k * 0.07, nrm[..., 2])
n2 /= np.linalg.norm(n2, axis=2, keepdims=True) + 1e-9
n2 = n2.astype(np.float32)

# the machine is the light
lp, lc, ln = [], [], []
for lx in (1.68, 2.0, 2.32):
    for ly in (0.95, 1.3, 1.62):
        lp.append((lx, ly, 8.5)); lc.append(np.array((1.0, 0.68, 0.34)) * 21 / 9); ln.append((0, 0, -1))
main = shade(ids, pos, n2, np.array(lp, np.float32), np.array(lc, np.float32), np.array(ln, np.float32), cen, half, yaw)
# what the lit snow throws back up onto things
bounce = shade(ids, pos, nrm, np.array([(2.0, 0.25, 7.3), (1.0, 0.2, 7.6)], np.float32),
               np.array([(0.55, 0.42, 0.30), (0.25, 0.19, 0.14)], np.float32), np.zeros((2, 3), np.float32), cen, half, yaw)
ground = B.mask(ids, "platform") | B.mask(ids, "bed")
irr = main + np.where(ground[..., None], 0, bounce)
amb = np.array((0.015, 0.020, 0.060), np.float32) * (0.5 + 0.5 * np.clip(n2[..., 1], 0, 1))[..., None]
irr = irr + amb

# ------------------------------------------------------------- local colour
alb = np.zeros((H, W, 3), np.float32)
emi = np.zeros((H, W, 3), np.float32)
region = (ids * 6 + face).astype(np.int32)
region[ids < 0] = -1

def put(mask, c):
    alb[mask] = c

put(B.mask(ids, "platform"), (0.20, 0.19, 0.19))
put(B.mask(ids, "bed"), (0.10, 0.09, 0.09))
put(snowy | caps, (0.90, 0.93, 0.97))
alb[snowy] *= (0.72 + 0.5 * fbm(wx * 0.35, wz * 0.12, 4, 31))[snowy][..., None]
bedtop = B.mask(ids, "bed") & (face == 3)
ball = np.clip((fbm(wx * 3, wz * 0.6, 3, 9) - 0.42) * 3.0, 0, 1)
between = np.exp(-((wx + 3.15) / 1.1) ** 2)
alb[bedtop] = (alb * (1 - 0.6 * (ball * between))[..., None])[bedtop]
put(B.mask(ids, "rail"), (0.16, 0.16, 0.19))
put(B.mask(ids, "machine"), (0.16, 0.40, 0.36))
put(B.mask(ids, "roof"), (0.07, 0.07, 0.09))
put(B.mask(ids, "post"), (0.07, 0.10, 0.10))
put(B.mask(ids, "mast"), (0.22, 0.23, 0.26))
put(B.mask(ids, "signal"), (0.05, 0.05, 0.06))
wallm = B.mask(ids, "wall")
u = loc[..., 0]; v = loc[..., 1]
plank = np.floor((u + 1) * 13.5)
ptone = 0.8 + 0.4 * vnoise(plank * 7.3, plank * 0 + 2, 5)
gap = np.abs(((u + 1) * 13.5) % 1 - 0.5) > 0.44
wc = np.array((0.30, 0.19, 0.13), np.float32) * ptone[..., None] * np.where(gap, 0.35, 1)[..., None]
alb[wallm] = wc[wallm]
region[wallm] = (region + 5000 + plank.astype(np.int32) * 7)[wallm]
benchm = B.mask(ids, "bench")
put(benchm, (0.34, 0.20, 0.11))

# machine front: window of drinks, header, lower panel
front = B.mask(ids, "machine") & (face == 4)
DRINKS = [(0.93, 0.42, 0.05), (0.20, 0.52, 0.16), (0.80, 0.08, 0.08), (0.10, 0.25, 0.75),
          (0.95, 0.76, 0.08), (0.90, 0.90, 0.84), (0.22, 0.11, 0.06), (0.45, 0.14, 0.52),
          (0.05, 0.50, 0.55), (0.90, 0.40, 0.50)]

def inv(c):
    """display colour -> the linear light that tone-maps to it"""
    return -np.log(1 - np.clip(np.asarray(c, np.float32), 0, 0.985) ** 2.2) / 1.25
fe = np.zeros((H, W, 3), np.float32)       # emitted
fa = np.zeros((H, W, 3), np.float32)       # reflective
fr = np.zeros((H, W), np.int32)
body = np.array((0.16, 0.40, 0.36), np.float32)
fa[:] = body
win = (np.abs(u) < 0.86) & (v > 0.06) & (v < 0.80)
fe[win] = inv((0.97, 0.93, 0.78))
fr[win] = 1
row = np.clip(np.floor((v - 0.06) / 0.74 * 3), 0, 2)
rv = ((v - 0.06) / 0.74 * 3) % 1.0                    # 0 bottom of shelf .. 1
NCOL = 8
colf = (u + 0.86) / 1.72 * NCOL
col = np.clip(np.floor(colf), 0, NCOL - 1)
cu = colf % 1.0 - 0.5                                  # -.5 .. .5 across slot
kind = ((col * 3 + row * 5).astype(int) % 7)
pick = ((col * 7 + row * 3 + (col * row)) .astype(int)) % len(DRINKS)
dcol = np.array(DRINKS, np.float32)[pick]
tall = np.where(row == 2, 0.90, np.where(row == 1, 0.80, 0.62))   # bottles up top, cans below
neck = (rv > 0.12 + tall * 0.62) & (row > 0)
bw_ = np.where(neck, 0.16, 0.33)
item = win & (np.abs(cu) < bw_) & (rv > 0.12) & (rv < 0.12 + tall * 0.86)
shade_ = 0.72 + 0.5 * np.clip(0.5 - np.abs(cu + 0.1) * 2.2, -0.4, 0.5)
ic = dcol * shade_[..., None]
label = item & (rv > 0.12 + tall * 0.2) & (rv < 0.12 + tall * 0.45) & ~neck
ic = np.where(label[..., None], ic * 0.55 + np.array((0.95, 0.93, 0.85), np.float32) * 0.5, ic)
cap = item & (rv > 0.12 + tall * 0.78) & (row > 0)
ic = np.where(cap[..., None], np.array((0.92, 0.9, 0.85), np.float32) * shade_[..., None], ic)
fe[item] = inv(np.clip(ic, 0, 0.97))[item]
fr[item] = (10 + row * 20 + col)[item].astype(np.int32)
shelf = win & (rv < 0.10)
fe[shelf] = (0.10, 0.10, 0.11)
fr[shelf] = (2 + row)[shelf].astype(np.int32)
btn = shelf & (np.abs(cu) < 0.16) & (rv > 0.02) & (rv < 0.08)
hot = ((col + row * 2).astype(int) % 5) == 0
bc = np.where(hot[..., None], np.array((1.6, 0.25, 0.12), np.float32), np.array((0.25, 0.75, 1.7), np.float32))
fe[btn] = bc[btn]
fr[btn] = (100 + row * 20 + col)[btn].astype(np.int32)
head = (np.abs(u) < 0.88) & (v > 0.85) & (v < 0.96)
fe[head] = (1.0, 1.5, 1.6)
fr[head] = 6
hstripe = head & (np.abs(u) < 0.45) & (v > 0.885) & (v < 0.925)
fe[hstripe] = (0.15, 0.65, 0.60)
fr[hstripe] = 7
panel = (u > 0.38) & (u < 0.82) & (v > -0.40) & (v < -0.08)
fa[panel] = (0.06, 0.07, 0.08); fr[panel] = 8
disp = (u > 0.44) & (u < 0.76) & (v > -0.19) & (v < -0.12)
fe[disp] = (0.25, 1.3, 0.45); fr[disp] = 9
slot = (u > 0.52) & (u < 0.68) & (v > -0.33) & (v < -0.27)
fe[slot] = (1.4, 0.6, 0.1); fr[slot] = 200
flap = (u > -0.72) & (u < 0.22) & (v > -0.86) & (v < -0.62)
fa[flap] = (0.03, 0.035, 0.04); fr[flap] = 201
poster = (u > -0.72) & (u < 0.22) & (v > -0.46) & (v < -0.08)
fa[poster] = (0.80, 0.74, 0.55); fr[poster] = 202
pband = poster & (v > -0.30) & (v < -0.22)
fa[pband] = (0.75, 0.20, 0.12); fr[pband] = 203
alb[front] = fa[front]
emi[front] = fe[front]
region[front] = 9000 + fr[front]
# the front is lit by its own spill off the snow
irr[front] += np.array((0.30, 0.24, 0.17), np.float32) * np.clip(0.2 - v, 0, 1.2)[front][..., None]

lin = alb * irr + emi

# ---------------------------------------------------------------------- sky
sky = ids < 0
el = np.arcsin(np.clip(dirs[..., 1], -1, 1))
az = np.arctan2(dirs[..., 0], dirs[..., 2])
tt = np.clip(el / 0.45, 0, 1) ** 0.7
zen = np.array((0.005, 0.009, 0.026), np.float32)
hor = np.array((0.024, 0.026, 0.054), np.float32)
skyc = hor * (1 - tt[..., None]) + zen * tt[..., None]
cloud = fbm(az * 9 + 3, el * 22, 5, 21)
skyc = skyc * (0.55 + 0.9 * cloud)[..., None]
glow = np.exp(-((az + 0.22) / 0.16) ** 2) * np.exp(-np.clip(el, 0, 9) / 0.065)
skyc = skyc + np.array((0.15, 0.07, 0.035), np.float32) * (glow * (0.5 + cloud))[..., None]
treeh = 0.012 + 0.030 * fbm(az * 30, az * 0 + 1, 4, 5) + 0.012 * vnoise(az * 260, az * 0, 8)
trees = sky & (el < treeh)
skyc = np.where(trees[..., None], np.array((0.006, 0.009, 0.014), np.float32), skyc)
lin = np.where(sky[..., None], skyc, lin)
region[trees] = -2

# air: falling snow carries the light a little way out
mc = cam.project(np.array([2.0, 1.25, 8.55]))
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
rr = np.hypot(xx - mc[0], (yy - mc[1]) * 0.9)
halo = 0.085 / (1 + (rr / 150.0) ** 2.2)
infront = (ids < 0) | (pos[..., 2] > 8.4) | ground
lin = lin + np.array((1.0, 0.80, 0.52), np.float32) * (halo * np.where(front, 0.0, 1.0))[..., None]

ref = (1 - np.exp(-lin * 1.25)) ** (1 / 2.2)
ref = np.clip(ref, 0, 1).astype(np.float32)
print("study %.1fs" % (time.time() - t0))
if "--ref" in sys.argv:
    Image.fromarray((ref * 255).astype(np.uint8)).save("out/marsh_001_ref.png")
    sys.exit()

# ============================================================== the painting
# Marsh's rules: a dark violet ground; the dark is laid first, thin, in long
# level strokes; light is built thick and combed outward from its source;
# pure white is used once.
cv = Canvas(W, H, ground=(0.125, 0.085, 0.135), seed=SEED)
lum = ref @ np.array((0.3, 0.55, 0.15), np.float32)
base = cam.project(np.array([2.0, 0.0, 8.6]))
radial = np.arctan2(yy - base[1], xx - base[0])
vp = cam.project(np.array([-3.15, -1.0, 5000.0]))
persp = np.arctan2(yy - vp[1], xx - vp[0])
orient = np.zeros((H, W), np.float32)
upright = (ids >= 0) & (np.abs(nrm[..., 1]) < 0.5)
orient[upright] = np.pi / 2
orient[B.mask(ids, "bed") | B.mask(ids, "rail")] = persp[B.mask(ids, "bed") | B.mask(ids, "rail")]
litg = B.mask(ids, "platform") & (face == 3) & (lum > 0.30 + 0.12 * vnoise(xx / 60, yy / 60, 4))
orient[litg] = radial[litg]
orient[sky] = (0.05 + 0.5 * (cloud - 0.5))[sky]
orient[trees] = np.pi / 2
orient[front & ((fr == 6) | (fr == 7) | ((fr >= 2) & (fr <= 4)))] = 0.0

wmap = np.ones((H, W), np.float32)
wmap[front] = 0.3
wmap[benchm | B.mask(ids, "bsnow") | B.mask(ids, "post") | B.mask(ids, "rail") | B.mask(ids, "mast")] = 0.5
detail = front | benchm | B.mask(ids, "bsnow") | B.mask(ids, "msnow")
depthkey = -lum                      # darks first, lights last

t0 = time.time()
passes = [
    Handling(width=27, length=12, tol=0.30, err=0.0, spacing=0.8, opacity=0.93, dry_dark=0.6, dry_light=0.2,
             thick_dark=0.10, thick_light=0.8, coljit=0.11, warm=0.10, wobble=0.14),
    Handling(width=13, length=10, tol=0.22, err=0.13, spacing=0.9, dry_dark=0.5, dry_light=0.12,
             thick_dark=0.12, thick_light=1.0, coljit=0.08, warm=0.07, wobble=0.16),
    Handling(width=6.5, length=8, tol=0.16, err=0.15, spacing=0.9, dry_dark=0.35, dry_light=0.1,
             thick_dark=0.12, thick_light=1.1, coljit=0.05, warm=0.04, wobble=0.14),
    Handling(width=3.2, length=7, tol=0.12, err=0.16, spacing=0.9, dry_dark=0.2, dry_light=0.05,
             thick_dark=0.1, thick_light=0.9, coljit=0.03, warm=0.02, wobble=0.08, rag=0.3),
]
for i, h in enumerate(passes):
    n = paint_layer(cv, ref, orient, region, h, wmap=wmap, key=depthkey)
    print("pass %d: %d strokes" % (i, n))
fine = Handling(width=1.7, length=5, tol=0.09, err=0.07, spacing=0.9, dry_dark=0.1, dry_light=0.0,
                thick_dark=0.1, thick_light=0.7, coljit=0.02, warm=0.01, wobble=0.05, rag=0.2, soft=0.3)
n = paint_layer(cv, ref, orient, region, fine, key=depthkey, mask=detail)
print("fine: %d strokes" % n)

# ---------------------------------------------------------------- by hand
depth = np.where(ids >= 0, (pos - cam.pos) @ cam.f, 1e9)

def P(x, y, z):
    return cam.project(np.array([x, y, z], np.float32))

def refat(p):
    return ref[int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1))]

def line3(world_pts, wm, col, wmin=0.8, wmax=6, **kw):
    """A line in the world, drawn only where nothing stands in front of it."""
    pts = np.array([P(*w) for w in world_pts])
    run = []
    for p in list(pts) + [None]:
        vis = (p is not None and 0 <= p[0] < W and 0 <= p[1] < H and p[2] > 0.2
               and p[2] < depth[int(p[1]), int(p[0])] + 0.15)
        if vis:
            run.append(p)
        else:
            if len(run) >= 2:
                r = np.array(run)
                cv.stroke(r[:, :2], np.clip(wm * cam.fl / r[:, 2], wmin, wmax), col, smooth=False, **kw)
            run = []

# rails catching what little there is
for rx in (-2.435, -3.865):
    zs = np.concatenate([np.linspace(8, 40, 60), np.linspace(41, 300, 60)])
    line3([(rx, -0.865, z) for z in zs], 0.045, (0.30, 0.34, 0.50), opacity=0.8, dry=0.45, rag=0.2, thick=0.3)
    line3([(rx, -0.865, z) for z in np.linspace(29, 39.5, 12)], 0.045, (0.66, 0.13, 0.09), wmax=4,
          opacity=0.8, dry=0.3, rag=0.2, thick=0.3)

# overhead line
def messenger(z):
    t = ((z - 16) % 30) / 30
    return (-3.15, 5.32 - 0.55 * 4 * t * (1 - t), z)

wcol = (0.17, 0.18, 0.27)
wk = dict(wmax=3, opacity=0.75, dry=0.3, rag=0.1, thick=0.15, jitter=0.02)
line3([(-3.15, 4.62, z) for z in np.linspace(4, 400, 300)], 0.022, wcol, **wk)
line3([messenger(z) for z in np.linspace(4, 400, 500)], 0.022, wcol, **wk)
for z in np.arange(8.5, 110, 5.0):
    line3([messenger(z), (-3.15, 4.62, z)], 0.012, wcol, wmax=2, opacity=0.5, dry=0.3, thick=0.1)

# the far signal, at danger
sg = P(-4.70, 3.62, 39.9)
for r, o in ((46, 0.10), (26, 0.20), (13, 0.40)):
    cv.dab(sg[0], sg[1], r, (0.80, 0.10, 0.06), opacity=o, hard=0.0, thick=0.05)
cv.dab(sg[0], sg[1], 6.5, (1.0, 0.34, 0.18), hard=0.6, thick=1.4)

# footprints: to the machine, a wait, then to the platform edge
Lg = np.array([2.0, 8.5])

def xspan(poly, y):
    xs = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        if (a[1] - y) * (b[1] - y) < 0:
            xs.append(a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]))
    return (min(xs), max(xs)) if len(xs) >= 2 else None

def footprint(x, z, dx, dz, lat):
    d = np.array([dx, dz]); d = d / np.linalg.norm(d)
    n_ = np.array([-d[1], d[0]])
    c = np.array([x, z]) + n_ * lat
    m = P(c[0], 0.0, c[1])
    if not (0 <= m[0] < W and 0 <= m[1] < H + 60):
        return
    th = np.linspace(0, 2 * np.pi, 28, endpoint=False)
    ring = c[None] + np.cos(th)[:, None] * d * 0.14 + np.sin(th)[:, None] * n_ * 0.058
    poly = np.array([P(q[0], 0.0, q[1])[:2] for q in ring])
    basec = refat(m)
    bl = basec @ np.array((0.3, 0.55, 0.15))
    hollow = basec * np.array((0.42, 0.44, 0.66)) + np.array((0.015, 0.025, 0.085))
    y0, y1 = poly[:, 1].min(), poly[:, 1].max()
    step = float(np.clip((y1 - y0) / 5, 1.6, 7))
    for y in np.arange(y0 + step * 0.5, y1, step):
        sp = xspan(poly, y)
        if sp and sp[1] - sp[0] > 1.5:
            cv.stroke([(sp[0], y), (sp[1], y)], step * 1.5, hollow * rng.uniform(0.9, 1.1), opacity=0.95,
                      dry=0.15, rag=0.25, thick=0.12, soft=0.4, smooth=False)
    if bl > 0.2:
        away = c - Lg; away = away / np.linalg.norm(away)
        out = (ring - c) / np.linalg.norm(ring - c, axis=1, keepdims=True)
        sel = np.where(out @ away > 0.25)[0]
        # keep the arc contiguous
        sel = sorted(sel, key=lambda i: np.arctan2(away[0] * out[i][1] - away[1] * out[i][0], away @ out[i]))
        arc = c[None] + (ring[sel] - c) * 1.13
        ap = np.array([P(q[0], 0.0, q[1])[:2] for q in arc])
        rim = np.clip(basec * 1.18 + np.array((0.10, 0.075, 0.03)), 0, 0.97)
        cv.stroke(ap, max(1.6, step * 0.75), rim, opacity=0.92, dry=0.1, rag=0.2, thick=1.3, taper=0.7)

def trail(path, stride=0.66):
    path = np.asarray(path, float)
    side = 1
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        L = np.linalg.norm(b - a)
        for s in np.arange(0, L, stride):
            p = a + (b - a) * s / L
            footprint(p[0] + rng.normal(0, 0.02), p[1] + rng.normal(0, 0.03), *(b - a), 0.10 * side)
            side = -side

trail([(1.95, 7.95), (1.82, 7.3), (1.35, 5.8), (0.85, 4.0), (0.62, 2.3)][::-1])
for _ in range(8):          # shifting from foot to foot, waiting
    ang = rng.uniform(1.0, 2.1)
    footprint(1.95 + rng.normal(0, 0.24), 7.95 + rng.normal(0, 0.13), np.cos(ang), np.sin(ang), 0)
trail([(1.55, 8.02), (0.6, 8.2), (-0.35, 8.3), (-0.86, 8.22)])

# snow in the air: only what the machine reaches can be seen
Lc = np.array([2.0, 1.3, 8.5])
fall = np.array([0.24, 1.0]); fall = fall / np.linalg.norm(fall)
N = 9000
fx = rng.uniform(-8, 5, N); fy = rng.uniform(-0.9, 6.0, N); fz = rng.uniform(0.3, 45, N)
M = 5000                    # and thicker where the light is, because that is where you can see it
fx = np.concatenate([fx, rng.normal(2.0, 1.5, M)]); fy = np.concatenate([fy, rng.uniform(0, 3.4, M)])
fz = np.concatenate([fz, 8.5 - np.abs(rng.normal(0, 2.2, M))])
nfl = 0
for i in np.argsort(-fz):
    p = P(fx[i], fy[i], fz[i])
    if p[2] < 0.35 or not (0 <= p[0] < W and 0 <= p[1] < H):
        continue
    if p[2] > depth[int(p[1]), int(p[0])]:
        continue
    dv = Lc - np.array([fx[i], fy[i], fz[i]])
    d = np.linalg.norm(dv)
    em = max(0.0, (dv[2] / d + 0.18) / 1.18)
    b = 7.0 * em / (d * d + 0.5)
    c = np.array((0.17, 0.20, 0.33)) + np.array((1.0, 0.80, 0.50)) * b
    if b < 0.05 and rng.random() < 0.6:
        continue
    c = np.clip(c, 0, 0.96)
    size = float(np.clip(0.0085 * cam.fl / p[2], 1.0, 12))
    op = 0.9 if size < 4 else 0.9 * (4 / size) ** 1.2
    if size < 2.2:
        cv.dab(p[0], p[1], size * 0.9, c, opacity=op, hard=0.3, thick=0.9)
    else:
        ln_ = size * rng.uniform(0.8, 2.0)
        cv.stroke([p[:2] - fall * ln_ / 2, p[:2], p[:2] + fall * ln_ / 2], size, c, opacity=op, dry=0.0,
                  rag=0.0, soft=0.9, thick=0.9, taper=0.75, smooth=False)
    nfl += 1

# the one white
a = P(1.60, 1.705, 8.59); b = P(2.40, 1.705, 8.59)
cv.stroke([a[:2], b[:2]], 3.2, (1, 1, 1), dry=0.05, rag=0.1, thick=1.6, jitter=0, smooth=False)

print("painted %d strokes in %.1fs (%d flakes)" % (cv.count, time.time() - t0, nfl))
cv.save("out/marsh_001.png")
Image.open("out/marsh_001.png").resize((W // 2, H // 2), Image.LANCZOS).save("out/marsh_001_small.png")
