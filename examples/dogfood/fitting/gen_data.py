# Generates the synthetic "measurements" pasted into the .sky files in this directory.
# Sky has no random numbers and no file input, so the data are made here and pasted in as
# array literals (see FRICTION.md, "no data in, no noise out").
#   python3 gen_data.py  ->  prints the Sky literals for fit_oscillator*.sky, then the
#   lists used by fit_line.sky and units_pipeline.sky
import math
import random

random.seed(42)
A, g, w, ph, sig = 20.0, 0.18, 3.93, 0.4, 0.6      # deg, 1/s, rad/s, rad, deg
ts = [round(0.25 * i, 2) for i in range(50)]
ys = [round(A * math.exp(-g * t) * math.cos(w * t + ph) + random.gauss(0, sig), 2) for t in ts]


def lit(xs, unit):
    rows = ["    " + ", ".join(f"{x} {unit}" for x in xs[i:i + 5]) + "," for i in range(0, len(xs), 5)]
    return "[\n" + "\n".join(rows) + "\n]"


print("t = " + lit(ts, "s"))
print("theta = " + lit(ys, "deg"))

# fit_line.sky: spring length vs mass (L0 = 120 mm, 0.15 mm/g, sigma_i = 0.4 + 0.08 i mm)
random.seed(7)
xs = [50 * i for i in range(1, 13)]
sigs = [round(0.4 + 0.08 * i, 2) for i in range(12)]
print([round(120 + 0.15 * x + random.gauss(0, s), 2) for x, s in zip(xs, sigs)])

# units_pipeline.sky: 20 swings of a pendulum, g = 9.8123, finite-amplitude correction
random.seed(11)
wire = [31.2, 44.8, 60.1, 75.4, 90.0, 104.7]
amp = [5.0, 7.5, 6.0, 8.0, 5.5, 7.0]
print([round(20 * 1000 * 2 * math.pi * math.sqrt((w_ * 10 + 12.5) / 1000 / 9.8123)
             * (1 + math.radians(a) ** 2 / 16) + random.gauss(0, 40), 0) for w_, a in zip(wire, amp)])
