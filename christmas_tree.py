"""3D Rotating ASCII Christmas Tree animation rendered with NumPy and Pygame."""

import math
from datetime import datetime
import numpy as np
import pygame


# timing & layout constants
TAU = 2.0 * np.pi
GROUND = -1.05
Y0 = -0.74
Y1 = 1.42
ROT_SPEED = 0.31

T_GROW = (1.0, 5.5)
T_GARLAND = (6.3, 7.8)
T_BALLS = (7.5, 9.5)
T_LIGHTS = (8.1, 11.4)
T_PRESENTS = (10.0, 11.1)
T_STAR = 11.6
T_MSG = 13.0

MESSAGE = "M E R R Y   C H R I S T M A S   2 0 2 6"

# colors (HDR linear float RGB)
WHITE = (1.0, 1.0, 1.0)
GOLD = (1.0, 0.90, 0.28)
RED = (1.0, 0.24, 0.26)
BAUBLE_GOLD = (1.0, 0.85, 0.26)
BLUE = (0.34, 0.50, 1.00)
GREEN_BOX = (0.30, 0.90, 0.40)
SILVER = (0.93, 0.95, 1.00)
WARM = (1.0, 0.92, 0.58)
TRUNK_COL = (0.64, 0.34, 0.22)
SKY_BG = (9, 10, 16)

# material IDs & per-material ASCII brightness ramps
NEEDLE, SNOW, INNER, TRUNK, TINSEL, BALL, LIGHT, BOX, RIBBON = range(9)
RAMPS = {
    NEEDLE: ".-;~^vx%&W#",
    SNOW:   ".+*#",
    INNER:  ".-~vx%&",
    TRUNK:  ":|=#",
    TINSEL: "-~=*#@",
    BALL:   ".:o0O%#@",
    LIGHT:  ".+*#@",
    BOX:    ".-=+xX%#@",
    RIBBON: ".-=+xX%#@",
}
RAMP_ARRS = [np.array([ord(c) for c in RAMPS[m]], dtype=np.int32) for m in range(9)]


def unit(v):
    """Normalize vector(s) along the last axis."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def smooth(x):
    """Cubic Hermite smoothstep clamped to [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def envelope(y):
    """Outer radial envelope of the spruce tree cone at height y."""
    u = np.clip((y - Y0) / (Y1 - Y0), 0.0, 1.0)
    return 0.94 * (1.0 - u) ** 0.94 + 0.04


def fib_sphere(n):
    """Distribute n unit points quasi-uniformly on a unit sphere."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    sp = np.sin(phi)
    return np.stack([sp * np.cos(theta), np.cos(phi), sp * np.sin(theta)], axis=1)


class Parts:
    """Accumulator for 3D point cloud geometry and animation metadata."""

    def __init__(self):
        self.pos = []
        self.nrm = []
        self.col = []
        self.mat = []
        self.t0 = []
        self.base = []
        self.phase = []
        self.centre = []
        self.pop = []
        self.bias = []

    def add(self, p, nrm, col, mat, t0, base=1.0, phase=0.0, centre=None, pop=False, bias=0.0):
        p = np.asarray(p, dtype=np.float64)
        n = len(p)
        nrm = unit(np.broadcast_to(nrm, (n, 3)))
        col = np.broadcast_to(np.asarray(col, dtype=np.float64), (n, 3))
        mat = np.broadcast_to(np.asarray(mat, dtype=np.int32), (n,))
        t0 = np.broadcast_to(np.asarray(t0, dtype=np.float64), (n,))
        base = np.broadcast_to(np.asarray(base, dtype=np.float64), (n,))
        phase = np.broadcast_to(np.asarray(phase, dtype=np.float64), (n,))
        bias = np.broadcast_to(np.asarray(bias, dtype=np.float64), (n,))
        centre_arr = p if centre is None else np.broadcast_to(np.asarray(centre, dtype=np.float64), (n, 3))
        pop_arr = np.full(n, bool(pop), dtype=bool)

        self.pos.append(p)
        self.nrm.append(nrm)
        self.col.append(col)
        self.mat.append(mat)
        self.t0.append(t0)
        self.base.append(base)
        self.phase.append(phase)
        self.centre.append(centre_arr)
        self.pop.append(pop_arr)
        self.bias.append(bias)

    def compile(self):
        """Pack accumulated point lists into contiguous float32/int32 NumPy arrays."""
        return {
            "pos": np.concatenate(self.pos, axis=0).astype(np.float32),
            "nrm": np.concatenate(self.nrm, axis=0).astype(np.float32),
            "col": np.concatenate(self.col, axis=0).astype(np.float32),
            "mat": np.concatenate(self.mat, axis=0).astype(np.int32),
            "t0": np.concatenate(self.t0, axis=0).astype(np.float32),
            "base": np.concatenate(self.base, axis=0).astype(np.float32),
            "phase": np.concatenate(self.phase, axis=0).astype(np.float32),
            "centre": np.concatenate(self.centre, axis=0).astype(np.float32),
            "pop": np.concatenate(self.pop, axis=0),
            "bias": np.concatenate(self.bias, axis=0).astype(np.float32),
        }


def grow_time(y):
    return T_GROW[0] + (T_GROW[1] - T_GROW[0]) * np.clip((y - GROUND) / (Y1 - GROUND), 0, 1)


def build():
    rng = np.random.default_rng(7)
    q = Parts()
    tips = []

    # needles: whorls of drooping branches, denser and longer at the bottom
    for i, y in enumerate(np.linspace(Y0 + 0.04, Y1 - 0.05, 21)):
        u = (y - Y0) / (Y1 - Y0)
        R = envelope(y) + 0.04
        nb = int(5 + 10 * (1 - u))
        for b in range(nb):
            a = TAU * (b + 0.5 * rng.random()) / nb + i * 0.77
            L = R * rng.uniform(0.76, 1.12)
            n = int(20 + 165 * L)
            s = rng.random(n) ** 0.65                               # 0 trunk .. 1 tip
            d = np.array([np.cos(a), 0, np.sin(a)])
            side = np.array([-np.sin(a), 0, np.cos(a)])
            spread = (0.022 + 0.085 * s) * rng.normal(0, 1, n)
            sl = s * L
            p = d * sl[:, None] + side * spread[:, None]
            p[:, 1] = y + 0.05 * sl - 0.18 * sl ** 2 + 0.012 * rng.normal(0, 1, n)
            nrm = unit(d * 0.75 + np.array([0, 0.55, 0]) + 0.3 * rng.normal(0, 1, (n, 3)))
            ao = (0.30 + 0.70 * s ** 1.3) * rng.uniform(0.75, 1.1, n)
            snow = (s > 0.5) & (rng.random(n) < 0.06 * s) & (nrm[:, 1] > 0.5)
            green = np.array([0.08, 0.45, 0.16]) + np.array([0.12, 0.38, 0.14]) * s[:, None]
            t_spawn = grow_time(y) + 0.35 * s + 0.22 * rng.random(n)
            q.add(p, nrm, np.where(snow[:, None], WHITE, green), np.where(snow, SNOW, NEEDLE),
                  t_spawn, base=np.where(snow, 1.0, ao))
            tl = 0.93 * L
            tips.append(d * tl + [0, y + 0.05 * tl - 0.18 * tl ** 2, 0])

    # dark inner cone so nothing shows through the branches
    m = 2800
    y = Y0 + (Y1 - 0.1 - Y0) * rng.random(m)
    phi = rng.random(m) * TAU
    out = np.stack([np.cos(phi), np.full(m, 0.3), np.sin(phi)], 1)
    r = 0.56 * envelope(y)
    q.add(np.stack([np.cos(phi) * r, y, np.sin(phi) * r], 1), out, (0.08, 0.38, 0.14), INNER,
          grow_time(y), base=0.70)

    # trunk
    m = 900
    y, phi = GROUND + rng.random(m) * (Y0 + 0.14 - GROUND), rng.random(m) * TAU
    n = np.stack([np.cos(phi), np.zeros(m), np.sin(phi)], 1)
    bark = 0.80 + 0.20 * np.sign(np.sin(phi * 30 + y * 5))
    q.add(n * 0.095 + np.stack([np.zeros(m), y, np.zeros(m)], 1), n, np.array(TRUNK_COL) * bark[:, None],
          TRUNK, grow_time(y), bias=0.01)

    # tinsel garland: delicate gold spiral drooping in swags between hanging points
    u = np.linspace(0, 1, 950)
    turns = 4.1
    yg = Y0 + 0.14 + (Y1 - Y0 - 0.40) * u
    ag = TAU * turns * u + 0.9
    swag = -0.055 * np.abs(np.sin(ag * 2.5))
    rg = envelope(yg) + 0.025
    line = np.stack([np.cos(ag) * rg, yg + swag, np.sin(ag) * rg], 1)
    for _ in range(2):
        nrm = np.stack([np.cos(ag), np.full_like(u, 0.3), np.sin(ag)], 1) + 0.5 * rng.normal(0, 1, line.shape)
        q.add(line + 0.005 * rng.normal(0, 1, line.shape), nrm, GOLD, TINSEL,
              T_GARLAND[0] + u * (T_GARLAND[1] - T_GARLAND[0]), phase=rng.random(len(u)) * TAU, bias=0.025)

    # baubles: red and gold spheres tucked onto branches
    sphere = fib_sphere(90)
    for k in range(34):
        u = (k + rng.random()) / 34
        y = Y0 + 0.08 + u * (Y1 - Y0 - 0.38)
        a = k * 2.39996 + rng.random() * 0.4
        c = np.array([np.cos(a), 0, np.sin(a)]) * (envelope(y) + 0.02) + [0, y - 0.03, 0]
        rad = 0.058 + 0.028 * (1 - u)
        col = RED if k % 2 == 0 else BAUBLE_GOLD             # half red, half gold
        q.add(c + sphere * rad, sphere, col, BALL, T_BALLS[0] + rng.random() * (T_BALLS[1] - T_BALLS[0]),
              centre=c, pop=True, bias=0.05)

    # fairy lights on the branch tips
    tips = np.array(tips)
    pick = rng.choice(len(tips), min(170, len(tips)), replace=False)
    bulb = fib_sphere(12)
    light_t0 = T_LIGHTS[0] + rng.random(len(pick)) * (T_LIGHTS[1] - T_LIGHTS[0])
    for c, t0 in zip(tips[pick], light_t0):
        col = WARM if rng.random() < 0.8 else (1.0, 0.95, 0.85)
        q.add(c + bulb * 0.030, bulb, col, LIGHT, t0, centre=c, phase=rng.random() * TAU, bias=0.07)

    # presents: boxes with a crossed ribbon and a bow
    gifts = [
        (0.3, 1.12, 0.30, 0.24, 0.28, 0.4, RED, GOLD),
        (1.5, 1.18, 0.26, 0.30, 0.24, 0.9, BLUE, GOLD),
        (2.6, 1.10, 0.34, 0.18, 0.26, 1.0, GOLD, RED),
        (3.7, 1.18, 0.24, 0.26, 0.24, 0.6, SILVER, RED),
        (4.7, 1.12, 0.30, 0.22, 0.30, 0.1, GREEN_BOX, RED),
        (5.6, 1.22, 0.22, 0.26, 0.22, 0.7, RED, SILVER),
    ]
    loop = fib_sphere(48)
    for g, (a, dist, w, h, d, rot, body, rib) in enumerate(gifts):
        areas = np.array([w * d, h * d, h * d, h * w, h * w])
        m = int(2600 * areas.sum())
        face = rng.choice(5, m, p=areas / areas.sum())
        fa, fb = rng.random(m) - 0.5, rng.random(m) - 0.5
        lx = np.select([face == 0, face == 1, face == 2], [fa * w, np.full(m, w * 0.5), np.full(m, -w * 0.5)], fb * w)
        ly = np.where(face == 0, h, (fb + 0.5) * h)
        lz = np.select([face == 0, face <= 2, face == 3], [fb * d, fa * d, np.full(m, d * 0.5)], np.full(m, -d * 0.5))
        n = np.array([[0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1.0]])[face]
        on_x, on_z = np.abs(lx) < 0.022, np.abs(lz) < 0.022
        ribbon = np.select([face == 0, face <= 2], [on_x | on_z, on_z], on_x)

        # bow on top of the present
        bow_pts = []
        bow_nrm = []
        for ang in (0.4, -0.4, np.pi + 0.4, np.pi - 0.4):
            ca, sa = np.cos(ang), np.sin(ang)
            lp = loop * [0.034, 0.020, 0.015] + [0.034, 0.014, 0.0]
            rx_l = lp[:, 0] * ca - lp[:, 2] * sa
            rz_l = lp[:, 0] * sa + lp[:, 2] * ca
            bow_pts.append(np.stack([rx_l, h + lp[:, 1], rz_l], 1))
            bnx = loop[:, 0] * ca - loop[:, 2] * sa
            bnz = loop[:, 0] * sa + loop[:, 2] * ca
            bow_nrm.append(np.stack([bnx, loop[:, 1], bnz], 1))
        bow_pts.append(loop * [0.020, 0.018, 0.020] + [0.0, h + 0.010, 0.0])
        bow_nrm.append(loop)
        bow = np.concatenate(bow_pts, 0)
        bn = np.concatenate(bow_nrm, 0)

        pts_local = np.concatenate([np.stack([lx, ly, lz], 1) + n * np.where(ribbon, 0.006, 0.0)[:, None], bow], 0)
        nrm_local = np.concatenate([n, bn], 0)
        is_rib = np.concatenate([ribbon, np.ones(len(bow), dtype=bool)])
        cols = np.where(is_rib[:, None], rib, body)
        mats = np.where(is_rib, RIBBON, BOX)
        biases = np.where(is_rib, 0.035, 0.0)

        cr, sr = np.cos(rot), np.sin(rot)
        center_world = np.array([np.cos(a) * dist, GROUND, np.sin(a) * dist])
        px_w = pts_local[:, 0] * cr - pts_local[:, 2] * sr + center_world[0]
        py_w = pts_local[:, 1] + GROUND
        pz_w = pts_local[:, 0] * sr + pts_local[:, 2] * cr + center_world[2]
        nx_w = nrm_local[:, 0] * cr - nrm_local[:, 2] * sr
        ny_w = nrm_local[:, 1]
        nz_w = nrm_local[:, 0] * sr + nrm_local[:, 2] * cr
        t_gift = T_PRESENTS[0] + (T_PRESENTS[1] - T_PRESENTS[0]) * (g / max(len(gifts) - 1, 1))
        pop_center = center_world + np.array([0.0, h * 0.5, 0.0])
        q.add(
            np.stack([px_w, py_w, pz_w], 1),
            np.stack([nx_w, ny_w, nz_w], 1),
            cols,
            mats,
            t_gift,
            centre=pop_center,
            pop=True,
            bias=biases,
        )

    data = q.compile()
    data["light_t0"] = np.sort(light_t0).astype(np.float32)
    return data


def project(pts, cols, rows, aspect):
    """Perspective projection from rotated world space onto ASCII grid coordinates."""
    d = np.maximum(5.0 - pts[:, 2], 0.2)
    x = cols * 0.5 + (90.0 / aspect) * pts[:, 0] / d
    y = rows * 0.53 - 90.0 * pts[:, 1] / d
    return x, y, d


def put(char, col, y, x, ch, tint, power):
    """Write a single character and HDR color into cell buffers."""
    if 0 <= y < char.shape[0] and 0 <= x < char.shape[1]:
        char[y, x] = ord(ch)
        col[y, x] = np.asarray(tint, dtype=np.float32) * float(power)


def point_in_poly(px, py, poly):
    res = np.zeros_like(px, dtype=bool)
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        res ^= ((y1 > py) != (y2 > py)) & (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1)
    return res


def star(char, col, cx, cy, size, t, power, cell, flash=0.0):
    # thin rays first, then a filled five-pointed star on top;
    # when it lights up it bursts white with long rays, then settles to gold
    tint = np.array(GOLD) + (1 - np.array(GOLD)) * flash
    power = power * (1 + 1.6 * flash)
    for k in range(8):
        ang = k * TAU / 8 + 0.10 * np.sin(t * 0.7)
        if np.sin(ang) < -0.3:                                  # no rays down into tree
            continue
        reach = size * (3.0 + 0.8 * np.sin(t * 3 + k * 1.3)) * (1.0 if k % 2 == 0 else 0.7)
        ch = "-|/\\"[ [0, 2, 1, 3][int(round((np.degrees(ang) % 180) / 45)) % 4] ]
        for s in np.arange(size * 1.0, reach, 0.5):
            y, x = int(round(cy - np.sin(ang) * s)), int(round(cx + np.cos(ang) * s))
            if 0 <= y < char.shape[0] and 0 <= x < char.shape[1] and col[y, x].max() < 0.38:
                put(char, col, y, x, ch, tint, power * 0.85 * (1.0 - 0.72 * s / reach))

    # filled 5-pointed star polygon with 2x2 sub-cell coverage for crisp silhouette tips
    outer = size * 1.05
    inner = outer * 0.46
    vy_scale = cell * 1.05
    poly = []
    for k in range(10):
        ang = -np.pi * 0.5 + k * np.pi / 5.0
        r = outer if k % 2 == 0 else inner
        poly.append((cx + np.cos(ang) * r, cy + np.sin(ang) * r * vy_scale))

    rx = int(math.ceil(outer + 1))
    ry = int(math.ceil(outer * vy_scale + 1))
    x_lo, x_hi = max(0, cx - rx), min(char.shape[1], cx + rx + 1)
    y_lo, y_hi = max(0, cy - ry), min(char.shape[0], cy + ry + 1)
    if x_lo >= x_hi or y_lo >= y_hi:
        return

    gy, gx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    gx_f, gy_f = gx.astype(np.float64), gy.astype(np.float64)
    cov = np.zeros_like(gx_f)
    for ox in (-0.22, 0.22):
        for oy in (-0.22, 0.22):
            cov += point_in_poly(gx_f + ox, gy_f + oy, poly)
    cov *= 0.25

    ys, xs, cv = gy[cov > 0], gx[cov > 0], cov[cov > 0]
    dist = np.hypot(xs - cx, (ys - cy) / max(vy_scale, 1e-6)) / max(outer, 1e-6)
    lum = power * np.clip((1.05 - 0.50 * dist) * (0.45 + 0.55 * cv), 0.35, 1.15)
    for py_i, px_i, lv, c_frac in zip(ys, xs, lum, cv):
        if c_frac >= 0.99 and lv > 0.72:
            ch = "@"
        elif lv > 0.52 and c_frac >= 0.5:
            ch = "#"
        else:
            ch = "*"
        put(char, col, int(py_i), int(px_i), ch, tint, float(lv))


def _make_background_cache(cols, rows, aspect):
    """Precompute static snow ground plane and deterministic snowfall flakes."""
    rng = np.random.default_rng(123)
    ground_char = np.zeros((rows, cols), dtype=np.int32)
    ground_col = np.zeros((rows, cols, 3), dtype=np.float32)

    sky_h = rows * 0.525
    horizon = int(math.floor(sky_h))
    for r in range(horizon, rows):
        dr = (r + 0.45) - sky_h
        if dr <= 0.15:
            continue
        d = 90.0 * (-GROUND) / dr
        fade = 1.0 / (1.0 + 0.014 * d * d)
        row_prob = 0.97 if r >= horizon + 2 else (0.55 if r == horizon + 1 else 0.32)
        for c in range(cols):
            if rng.random() > row_prob:
                continue
            wx = (c - cols * 0.5) * d * aspect / 90.0
            wz = 5.0 - d
            rad2 = wx * wx + wz * wz
            pool = 0.08 * math.exp(-rad2 / 4.0)
            n = rng.random()
            val = (0.08 + 0.095 * fade + pool) * (0.78 + 0.44 * n)
            if val > 0.215 and n > 0.78:
                ch = ord("=")
            elif val > 0.135 and n > 0.42:
                ch = ord("-")
            else:
                ch = ord(".")
            ground_char[r, c] = ch
            ground_col[r, c] = np.array([0.74, 0.76, 1.0], dtype=np.float32) * val

    n_snow = 175
    snow = {
        "x0": rng.uniform(0, cols, n_snow),
        "y0": rng.uniform(0, rows, n_snow),
        "vy": rng.uniform(4.5, 9.0, n_snow),
        "phase": rng.uniform(0, TAU, n_snow),
        "sway": rng.uniform(0.3, 1.2, n_snow),
        "glyph": rng.choice([ord("."), ord("+"), ord("*")], n_snow, p=[0.40, 0.42, 0.18]),
        "bright": rng.uniform(0.55, 0.98, n_snow),
    }
    return ground_char, ground_col, snow


_BG_CACHE = {}


def render(scene, t, cols, rows, aspect):
    """Render one frame of the ASCII Christmas tree at timestamp t."""
    key = (cols, rows, round(aspect, 5))
    if key not in _BG_CACHE:
        _BG_CACHE[key] = _make_background_cache(cols, rows, aspect)
    ground_char, ground_col, snow = _BG_CACHE[key]

    char = ground_char.copy()
    col = ground_col.copy()

    # snowfall (drifting behind tree and over ground)
    sx = np.mod((snow["x0"] + snow["sway"] * np.sin(t * 1.1 + snow["phase"])).round().astype(np.int32), cols)
    sy = np.mod((snow["y0"] + snow["vy"] * t).round().astype(np.int32), rows)
    snow_mask = col[sy, sx].max(axis=1) < snow["bright"]
    if np.any(snow_mask):
        sy_m, sx_m = sy[snow_mask], sx[snow_mask]
        char[sy_m, sx_m] = snow["glyph"][snow_mask]
        col[sy_m, sx_m] = snow["bright"][snow_mask, None]

    # filter points already spawned at time t
    t0 = scene["t0"]
    mask = t >= t0
    rot = ROT_SPEED * t
    info = {
        "rot_deg": float(np.degrees(rot) % 360.0),
        "lights": int(np.searchsorted(scene["light_t0"], t, side="right")),
    }

    if np.any(mask):
        pos = scene["pos"][mask].copy()
        nrm = scene["nrm"][mask]
        pcol = scene["col"][mask]
        mat = scene["mat"][mask]
        age = t - t0[mask]
        base = scene["base"][mask]
        phase = scene["phase"][mask]
        centre = scene["centre"][mask]
        pop = scene["pop"][mask]
        bias = scene["bias"][mask]

        # pop-in scaling from local part centre
        if np.any(pop):
            sc = smooth(age[pop] / 0.25)[:, None]
            pos[pop] = centre[pop] + (pos[pop] - centre[pop]) * sc

        # rotate scene around Y axis
        c, s = math.cos(rot), math.sin(rot)
        rx = pos[:, 0] * c + pos[:, 2] * s
        rz = -pos[:, 0] * s + pos[:, 2] * c
        rpos = np.stack([rx, pos[:, 1], rz], axis=1)

        nx = nrm[:, 0] * c + nrm[:, 2] * s
        ny = nrm[:, 1]
        nz = -nrm[:, 0] * s + nrm[:, 2] * c

        # lighting (warm front-left directional key + rich ambient floor)
        ndotl = np.clip(-0.34 * nx + 0.45 * ny + 0.82 * nz, 0.0, 1.0)
        shade = base * (0.40 + 0.74 * ndotl)

        # per-material shimmer / twinkle / surface lift
        is_tinsel = mat == TINSEL
        if np.any(is_tinsel):
            tw = 0.74 + 0.36 * np.sin(t * 4.0 + phase[is_tinsel])
            shade[is_tinsel] = (0.34 + 0.72 * ndotl[is_tinsel]) * tw

        is_light = mat == LIGHT
        if np.any(is_light):
            pulse = 0.48 + 0.68 * (0.5 + 0.5 * np.sin(t * 4.2 + phase[is_light]))
            shade[is_light] = pulse

        is_ball = mat == BALL
        if np.any(is_ball):
            spec = np.clip(-0.30 * nx[is_ball] + 0.48 * ny[is_ball] + 0.82 * nz[is_ball], 0.0, 1.0) ** 9
            shade[is_ball] = 0.42 + 0.68 * ndotl[is_ball] + 0.55 * spec

        is_box = (mat == BOX) | (mat == RIBBON)
        if np.any(is_box):
            shade[is_box] = 0.52 + 0.62 * ndotl[is_box]

        is_trunk = mat == TRUNK
        if np.any(is_trunk):
            shade[is_trunk] = 0.65 + 0.45 * ndotl[is_trunk]

        # crisp white spawn flash over ~0.36s
        flash = (np.clip(1.0 - age / 0.36, 0.0, 1.0) ** 1.5)[:, None]
        rgb = pcol + (1.0 - pcol) * flash
        lum = shade * (1.0 + 0.85 * flash[:, 0]) + 0.28 * flash[:, 0]

        # project to screen and sort back-to-front (painter's order with subtle material priority)
        px_f, py_f, depth = project(rpos, cols, rows, aspect)
        px = np.rint(px_f).astype(np.int32)
        py = np.rint(py_f).astype(np.int32)
        on_scr = (px >= 0) & (px < cols) & (py >= 0) & (py < rows) & (depth > 0.2)

        px, py = px[on_scr], py[on_scr]
        depth_key = depth[on_scr] - bias[on_scr] - 0.02 * np.minimum(lum[on_scr], 1.2)
        mat, lum, rgb = mat[on_scr], lum[on_scr], rgb[on_scr]

        order = np.argsort(-depth_key, kind="quicksort")
        px, py = px[order], py[order]
        mat, lum, rgb = mat[order], lum[order], rgb[order]

        glyph = np.empty(len(px), dtype=np.int32)
        for m_id, ramp in enumerate(RAMP_ARRS):
            sel = mat == m_id
            if np.any(sel):
                li = np.clip((lum[sel] * len(ramp)).astype(np.int32), 0, len(ramp) - 1)
                glyph[sel] = ramp[li]

        # lift color slightly relative to glyph density so dark branches remain vivid green
        vis_gain = np.power(np.clip(lum, 0.0, 1.35), 0.72)[:, None]
        char[py, px] = glyph
        col[py, px] = np.clip(rgb * vis_gain, 0.0, 1.3)

    # topper star & burst rays
    if t >= T_STAR:
        on = float(smooth((t - T_STAR) / 0.26))
        flash = float(np.clip(1 - (t - T_STAR) / 1.0, 0, 1) ** 2)
        tx, ty, _ = project(np.array([[0, Y1 + 0.13, 0]]), cols, rows, aspect)
        star(char, col, int(round(tx[0])), int(round(ty[0])),
             (2.3 + 2.7 * on) * (1 + 0.05 * np.sin(t * 4.0)), t, 1.12, aspect, flash=flash)

    # typewriter holiday banner near the bottom
    if t > T_MSG:
        row, x0 = rows - 9, (cols - len(MESSAGE)) // 2
        shown = min(len(MESSAGE), int((t - T_MSG) * 10))
        char[row - 1:row + 2, max(x0 - 3, 0):x0 + shown + 3] = 0    # clear a clean strip
        for i, ch in enumerate(MESSAGE[:shown]):
            if ch != " ":
                tint = (RED, GOLD, GREEN_BOX)[(i // 2 + int(t * 1.5)) % 3]
                put(char, col, row, x0 + i, ch, tint, 0.85 + 0.25 * np.sin(t * 5 + i * 0.4))

    info["letters"] = int((char > 0).sum())
    return char, col, info


# window

TITLE = "ASCII Christmas Tree"
CW, CH, FONT_PX = 9, 14, 15
COLS, ROWS = 88, 74
W, H = COLS * CW, ROWS * CH
MONO = "consolas,menlo,dejavusansmono,couriernew,monospace"
HINT = "Space pause · R restart · S screenshot · Esc quit"
DIM, ACCENT, SIGN = (96, 104, 122), (255, 196, 90), (176, 168, 196)


class Renderer:
    """Fast Unicode/ASCII cell blitter with cached ground plane, bloom, and HUD."""

    def __init__(self, cols=COLS, rows=ROWS, cw=CW, ch=CH, font_px=FONT_PX, hud_h=34):
        self.cols, self.rows, self.cw, self.ch = cols, rows, cw, ch
        self.w, self.h = cols * cw, rows * ch
        self.hud_h = hud_h
        font = pygame.font.SysFont(MONO, font_px)
        atlas = []
        for code in range(32, 127):
            s = pygame.Surface((cw, ch))
            g = font.render(chr(code), True, (255, 255, 255))
            s.blit(g, ((cw - g.get_width()) // 2, (ch - g.get_height()) // 2))
            atlas.append(pygame.surfarray.array3d(s)[:, :, 0])
        a_u16 = (np.stack(atlas).astype(np.uint16) * 256) // 255            # glyph, CW, CH (0..256)
        self.bg_u8 = np.array(SKY_BG, dtype=np.uint8)
        self.bg_u16 = self.bg_u8.astype(np.uint16)
        self.atlas_fg = a_u16
        self.atlas_bg = (((256 - a_u16)[..., None] * self.bg_u16) >> 8).astype(np.uint16)

        self.buf = np.empty((self.w, self.h, 3), dtype=np.uint8)
        self.view = self.buf.reshape(cols, cw, rows, ch, 3).swapaxes(1, 2)
        self.art = pygame.Surface((self.w, self.h))
        self.glow_small = pygame.Surface((self.w // 5, self.h // 5))
        self.glow_big = pygame.Surface((self.w, self.h))
        self.glow_big.set_alpha(95)

        self.hud_font = pygame.font.SysFont(MONO, max(12, font_px - 2))
        self.sign_surf = self.hud_font.render("jaeyonni", True, SIGN)
        self.hint_surf = self.hud_font.render(HINT, True, DIM)
        self._ground = None

    def _paint_cells(self, view, xs, ys, char, col):
        codes = char[ys, xs]
        active = codes >= 32
        if np.any(active):
            xa, ya = xs[active], ys[active]
            idx = np.clip(codes[active] - 32, 0, 94)
            rgb = (np.clip(col[ya, xa], 0.0, 1.0) * 255.0).astype(np.uint16)
            view[xa, ya] = (
                self.atlas_bg[idx] + ((self.atlas_fg[idx][..., None] * rgb[:, None, None, :]) >> 8)
            ).astype(np.uint8)
        blank = ~active
        if np.any(blank):
            view[xs[blank], ys[blank]] = self.bg_u8

    def _ensure_ground(self):
        if self._ground is None:
            key = (self.cols, self.rows, round(self.cw / self.ch, 5))
            if key not in _BG_CACHE:
                _BG_CACHE[key] = _make_background_cache(self.cols, self.rows, self.cw / self.ch)
            bg_char, bg_col, _ = _BG_CACHE[key]
            g_buf = np.empty_like(self.buf)
            g_buf[:] = self.bg_u8
            g_view = g_buf.reshape(self.cols, self.cw, self.rows, self.ch, 3).swapaxes(1, 2)
            ys, xs = np.nonzero(bg_char >= 32)
            self._paint_cells(g_view, xs, ys, bg_char, bg_col)
            self._ground = (bg_char, bg_col, g_buf)
        return self._ground

    def draw(self, screen, char, col, t, info, paused=False):
        bg_char, bg_col, g_buf = self._ensure_ground()
        np.copyto(self.buf, g_buf)
        dirty = (char != bg_char) | np.any(col != bg_col, axis=2)
        if np.any(dirty):
            ys, xs = np.nonzero(dirty)
            self._paint_cells(self.view, xs, ys, char, col)
        pygame.surfarray.blit_array(self.art, self.buf)

        screen.blit(self.art, (0, 0))
        if self.hud_h > 0:
            screen.fill(SKY_BG, (0, self.h, self.w, self.hud_h))

        # subtle phosphor glow pass
        pygame.transform.smoothscale(self.art, self.glow_small.get_size(), self.glow_small)
        pygame.transform.smoothscale(self.glow_small, (self.w, self.h), self.glow_big)
        screen.blit(self.glow_big, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

        # top-right live timer indicator
        tr_text = self.hud_font.render(f"t {t:05.2f}s", True, ACCENT)
        tx = self.w - tr_text.get_width() - 16
        ty = 12
        dot_col = DIM if paused else ACCENT
        pygame.draw.circle(screen, dot_col, (tx - 12, ty + tr_text.get_height() // 2), 4)
        screen.blit(tr_text, (tx, ty))

        # bottom-left HUD overlay matching video layout
        pad_x = 14
        base_y = self.h - 76
        stat_surf = self.hud_font.render(
            f"rotation {info['rot_deg']:06.2f}°   lights {info['lights']:3d}", True, DIM
        )
        let_surf = self.hud_font.render(f"letters {info['letters']:,}", True, DIM)
        screen.blit(self.sign_surf, (pad_x, base_y))
        screen.blit(stat_surf, (pad_x, base_y + 21))
        screen.blit(let_surf, (pad_x, base_y + 42))

        # controls bar at bottom
        if self.hud_h > 0:
            screen.blit(
                self.hint_surf,
                ((self.w - self.hint_surf.get_width()) // 2, self.h + (self.hud_h - self.hint_surf.get_height()) // 2),
            )


def main():
    pygame.init()
    renderer = Renderer()
    screen = pygame.display.set_mode((W, H + renderer.hud_h))
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()

    scene = build()
    aspect = CW / CH
    t = 0.0
    paused = False
    running = True

    while running:
        dt = min(clock.tick(30) / 1000.0, 0.1)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    t = 0.0
                    paused = False
                elif ev.key == pygame.K_s:
                    fname = f"christmas_tree_{datetime.now():%Y%m%d_%H%M%S}.png"
                    pygame.image.save(screen, fname)

        if not paused:
            t += dt

        char, col, info = render(scene, t, COLS, ROWS, aspect)
        renderer.draw(screen, char, col, t, info, paused=paused)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
