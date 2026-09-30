# Classic RK4 for a damped pendulum, y = (angle, angular velocity), h = 0.001.
# usage: rk4.py STEPS
import math
import sys


def f(y):
    return (y[1], -math.sin(y[0]) - 0.1 * y[1])


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    y = (1.0, 0.0)
    h = 0.001
    for _ in range(steps):
        k1 = f(y)
        k2 = f((y[0] + (h / 2) * k1[0], y[1] + (h / 2) * k1[1]))
        k3 = f((y[0] + (h / 2) * k2[0], y[1] + (h / 2) * k2[1]))
        k4 = f((y[0] + h * k3[0], y[1] + h * k3[1]))
        y = (
            y[0] + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            y[1] + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
        )
    print(repr(y[0]), repr(y[1]))


main()
