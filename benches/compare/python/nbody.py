# The classic N-body benchmark: Sun and the four gas giants, step 0.01.
# usage: nbody.py STEPS
import math
import sys

PI = math.pi
SOLAR_MASS = 4 * PI * PI
DPY = 365.24

# [x, y, z], [vx, vy, vz], mass
BODIES = [
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], SOLAR_MASS),
    (
        [4.84143144246472090e00, -1.16032004402742839e00, -1.03622044471123109e-01],
        [1.66007664274403694e-03 * DPY, 7.69901118419740425e-03 * DPY, -6.90460016972063023e-05 * DPY],
        9.54791938424326609e-04 * SOLAR_MASS,
    ),
    (
        [8.34336671824457987e00, 4.12479856412430479e00, -4.03523417114321381e-01],
        [-2.76742510726862411e-03 * DPY, 4.99852801234917238e-03 * DPY, 2.30417297573763929e-05 * DPY],
        2.85885980666130812e-04 * SOLAR_MASS,
    ),
    (
        [1.28943695621391310e01, -1.51111514016986312e01, -2.23307578892655734e-01],
        [2.96460137564761618e-03 * DPY, 2.37847173959480950e-03 * DPY, -2.96589568540237556e-05 * DPY],
        4.36624404335156298e-05 * SOLAR_MASS,
    ),
    (
        [1.53796971148509165e01, -2.59193146099879641e01, 1.79258772950371181e-01],
        [2.68067772490389322e-03 * DPY, 1.62824170038242295e-03 * DPY, -9.51592254519715870e-05 * DPY],
        5.15138902046611451e-05 * SOLAR_MASS,
    ),
]
PAIRS = [(a, b) for i, a in enumerate(BODIES) for b in BODIES[i + 1 :]]


def advance(dt):
    for (x1, v1, m1), (x2, v2, m2) in PAIRS:
        dx = x1[0] - x2[0]
        dy = x1[1] - x2[1]
        dz = x1[2] - x2[2]
        r2 = dx * dx + dy * dy + dz * dz
        mag = dt / (r2 * math.sqrt(r2))
        v1[0] -= dx * m2 * mag
        v1[1] -= dy * m2 * mag
        v1[2] -= dz * m2 * mag
        v2[0] += dx * m1 * mag
        v2[1] += dy * m1 * mag
        v2[2] += dz * m1 * mag
    for x, v, _ in BODIES:
        x[0] += dt * v[0]
        x[1] += dt * v[1]
        x[2] += dt * v[2]


def energy():
    e = 0.0
    for i, (x1, v1, m1) in enumerate(BODIES):
        e += 0.5 * m1 * (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
        for x2, _, m2 in BODIES[i + 1 :]:
            dx = x1[0] - x2[0]
            dy = x1[1] - x2[1]
            dz = x1[2] - x2[2]
            e -= m1 * m2 / math.sqrt(dx * dx + dy * dy + dz * dz)
    return e


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    p = [0.0, 0.0, 0.0]
    for _, v, m in BODIES:
        for k in range(3):
            p[k] += v[k] * m
    for k in range(3):
        BODIES[0][1][k] = -p[k] / SOLAR_MASS
    e0 = energy()
    for _ in range(steps):
        advance(0.01)
    print(repr(e0), repr(energy()))


main()
