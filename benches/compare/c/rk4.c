// Classic RK4 for a damped pendulum, y = {angle, angular velocity}, h = 0.001.
// usage: rk4 STEPS
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void f(const double y[2], double out[2]) {
    out[0] = y[1];
    out[1] = -sin(y[0]) - 0.1 * y[1];
}

int main(int argc, char **argv) {
    long steps = argc > 1 ? atol(argv[1]) : 200000;
    double y[2] = {1.0, 0.0};
    const double h = 0.001;
    double k1[2], k2[2], k3[2], k4[2], t[2];
    for (long i = 0; i < steps; i++) {
        f(y, k1);
        for (int j = 0; j < 2; j++) t[j] = y[j] + (h / 2) * k1[j];
        f(t, k2);
        for (int j = 0; j < 2; j++) t[j] = y[j] + (h / 2) * k2[j];
        f(t, k3);
        for (int j = 0; j < 2; j++) t[j] = y[j] + h * k3[j];
        f(t, k4);
        for (int j = 0; j < 2; j++)
            y[j] = y[j] + (h / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
    }
    printf("%.17g %.17g\n", y[0], y[1]);
    return 0;
}
