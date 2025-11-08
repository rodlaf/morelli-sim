#include <math.h>

// fix: round towards zero
static inline int fix(double ele) {
    return (ele > 0.0) ? (int)floor(ele) : (int)ceil(ele);
}

// sign: sign of a number
static inline int sign(double ele) {
    return (ele < 0.0) ? -1 : ((ele == 0.0) ? 0 : 1);
}

// converts velocity (vt) and altitude (alt) to mach number (amach) and dynamic pressure
// (qbar) See pages 63-65 of Stevens & Lewis, "Aircraft Control and Simulation", 2nd
// edition
static inline void adc(double vt, double alt, double *amach, double *qbar) {
    double ro = 2.377e-3;
    double tfac = 1.0 - 0.703e-5 * alt;
    double t = (alt >= 35000.0) ? 390.0 : 519.0 * tfac;
    double rho = ro * pow(tfac, 4.14);
    double a = sqrt(1.4 * 1716.3 * t);
    *amach = vt / a;
    *qbar = 0.5 * rho * vt * vt;
}

// dampp function
static inline void dampp(double alpha, double d[9]) {
    static const double a[12][9] = {
        {-.267, .882, -.108, -8.80, -.126, -.360, -7.21, -.380, .061},
        {-.110, .852, -.108, -25.8, -.026, -.359, -.540, -.363, .052},
        {.308, .876, -.188, -28.9, .063, -.443, -5.23, -.378, .052},
        {1.34, .958, .110, -31.4, .113, -.420, -5.26, -.386, -.012},
        {2.08, .962, .258, -31.2, .208, -.383, -6.11, -.370, -.013},
        {2.91, .974, .226, -30.7, .230, -.375, -6.64, -.453, -.024},
        {2.76, .819, .344, -27.7, .319, -.329, -5.69, -.550, .050},
        {2.05, .483, .362, -28.2, .437, -.294, -6.00, -.582, .150},
        {1.50, .590, .611, -29.0, .680, -.230, -6.20, -.595, .130},
        {1.49, 1.21, .529, -29.8, .100, -.210, -6.40, -.637, .158},
        {1.83, -.493, .298, -38.3, .447, -.120, -6.60, -1.02, .240},
        {1.21, -1.04, -2.27, -35.3, -.330, -.100, -6.00, -.840, .150}
    };

    double s = 0.2 * alpha;
    int k = fix(s);
    if (k <= -2) k = -1;
    if (k >= 9) k = 8;
    double da = s - k;
    int l = k + fix(1.1 * sign(da));
    
    k = k + 3;
    l = l + 3;

    for (int i = 0; i < 9; i++) {
        d[i] = a[k-1][i] + fabs(da) * (a[l-1][i] - a[k-1][i]);
    }
}
