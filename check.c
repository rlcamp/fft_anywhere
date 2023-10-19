/* do a silly thing to validate the real-to-complex forward and complex-to-real inverse fft,
 which call most of the other code, on a sequence of transforms and manipulations that
 yields a trivial expected output relative to the input */

#include "fft_anywhere.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

static uint64_t xorshift64star(void) {
    /* marsaglia et al. generates 64 bits at a time, the most significant bits are the most
     random, but it passes statistical tests even when reversed */
    static uint64_t x = 1; /* must be nonzero */
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * 0x2545F4914F6CDD1DULL;
}

static float frand_minus_frand(void) {
    /* generate 64 random bits, of which we will use the highest 46, in two groups of 23 */
    const uint64_t bits = xorshift64star();

    /* generate two random numbers each uniformly distributed on [1.0f, 2.0f) */
    const union { uint32_t u; float f; } x = { .u = 0x3F800000U | ((bits >> 41) & 0x7FFFFFU) };
    const union { uint32_t u; float f; } y = { .u = 0x3F800000U | ((bits >> 18) & 0x7FFFFFU) };

    /* and subtract them, yielding a triangular distribution on [-1.0f, +1.0f] */
    return x.f - y.f;
}

static float good_enough_unit_gaussian(void) {
    /* expected value of |frand_minus_frand|^2 is exactly 1/6, so just... */
    return (frand_minus_frand() + frand_minus_frand() + frand_minus_frand() +
            frand_minus_frand() + frand_minus_frand() + frand_minus_frand());
}

int main(void) {
    /* transform length exercises all implemented prime factors and is a multiple of 4 */
    const size_t T = 4 * 3 * 5 * 7;

    /* constant offset to add to the input, to make relative error sort of a valid metric */
    const float offset = 4.0f;

    /* allocate memory */
    float * restrict const a = malloc(sizeof(float) * T);
    float complex * restrict const b = malloc(sizeof(float complex) * (T / 2));
    float * restrict const c = malloc(sizeof(float) * T);

    /* attempt to plan transforms of the given length */
    struct planned_real_fft * plan_forward = plan_real_fft_of_length(T);
    if (!plan_forward) return 1;

    struct planned_real_inverse_fft * plan_inverse = plan_real_inverse_fft_of_length(T);
    if (!plan_inverse) return 1;

    /* construct a monotonically increasing signal with an offset */
    for (size_t it = 0; it < T; it++)
        a[it] = good_enough_unit_gaussian() + offset;

    /* do the real-to-complex forward fft */
    fft_evaluate_real(b, a, plan_forward);

    /* negate the  imaginary component of everything but the dc and nyquist bins */
    for (size_t iw = 1; iw < T / 2; iw++)
        b[iw] = conjf(b[iw]);

    /* do the complex-to-real inverse fft */
    fft_evaluate_real_inverse(c, b, plan_inverse);

    /* calculate mean and worst case relative error, which is a sort of okay metric */
    float max_relative_error = 0;
    double sum_of_squared_relative_error = 0;

    for (size_t it = 0; it < T; it++) {
        /* ifft(fft(x)) must be normalized according to tranform length */
        const float value = c[it] / T;

        /* the expected result of the manipulated transform is a reordering of input */
        const float expected = a[(T - it) % T];

        /* somewhat iffy and obviously only works when expected is nonzero */
        const float relative_error = fabsf((value - expected) / expected);
        max_relative_error = fmaxf(max_relative_error, relative_error);
        sum_of_squared_relative_error += relative_error * relative_error;
    }

    /* cleanup */
    destroy_planned_real_inverse_fft(plan_inverse);
    destroy_planned_real_fft(plan_forward);
    free(c);
    free(b);
    free(a);

    /* summarize statistics */
    const float mean_good_bits =  -log2f(sqrtf(sum_of_squared_relative_error / T));
    const float min_good_bits = -log2f(max_relative_error);
    fprintf(stderr, "mean/min bits not in error: %.1f/%.1f\n", mean_good_bits, min_good_bits);

    if (mean_good_bits < 22.0f || min_good_bits < 14.0f) {
        printf("fail\n");
        return 1;
    } else {
        printf("pass\n");
        return 0;
    }
}
