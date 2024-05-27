import taichi as ti

ti.init(arch=ti.gpu)

res = 512
dt = 0.03
maxfps = 60
time_c = 2
dye_decay = 1 - 1 / (maxfps * time_c)

v1 = ti.Vector.field(2, ti.f32, shape=(res, res))
v2 = ti.Vector.field(2, ti.f32, shape=(res, res))
v_div = ti.field(ti.f32, shape=(res, res))
v_curl = ti.field(ti.f32, shape=(res, res))
p1 = ti.field(ti.f32, shape=(res, res))
p2 = ti.field(ti.f32, shape=(res, res))
d1 = ti.field(ti.f32, shape=(res, res))
d2 = ti.field(ti.f32, shape=(res, res))


class MakePair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


v_p = MakePair(v1, v2)
p_p = MakePair(p1, p2)
d_p = MakePair(d1, d2)


@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = ti.max(0, ti.min(res - 1, I))
    return qf[I]


@ti.func
def lerp(vl, vr, frac):
    return vl + frac * (vr - vl)


@ti.func
def bilerp(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    iu, iv = ti.floor(s), ti.floor(t)
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


@ti.func
def backtrace(vf: ti.template(), p, dt_: ti.template()):
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt_ * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt_ * v2
    v3 = bilerp(vf, p2)
    p -= dt_ * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p


@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5  # center of the cell
        p = backtrace(vf, p, dt)
        new_qf[i, j] = bilerp(qf, p) * dye_decay


@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        vc = sample(vf, i, j)
        if i == 0:
            vl.x = -vc.x
        if i == res - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == res - 1:
            vt.y = -vc.y
        v_div[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5
