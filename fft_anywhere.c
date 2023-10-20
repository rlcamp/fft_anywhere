/*
 Copyright 2015-2023 Richard Campbell

 Permission to use, copy, modify, and/or distribute this software for any purpose with or without
 fee is hereby granted, provided that the above copyright notice and this permission notice appear
 in all copies.

 THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
 SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE
 AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
 THIS SOFTWARE.

 This is a good-enough FFT implementation which benchmarks within a respectable distance of the
 fastest known fft implementation on platforms of interest, and usually beats the most widely used
 one, when compiled with a modern C compiler. This implementation is known to not be terribly
 cache-efficient, but should work where faster fft implementations do not, hence the name. If you
 are on a SIMD processor and need a permissive-licensed fft implementation, jpommier/pffft is
 considerably faster, and you should prefer it to this or fftw.

 The core of the FFT is a set of functions for the DFTs/FFTs of size 3, 4, 5, 7, and 8. A second set
 of functions implements the Cooley-Tukey decomposition for T / S x S, for S = 2, 3, 4, 5, and 7.
 Each of these functions decomposes a length-T FFT into S FFTs of length T/S, followed by T/S FFTs
 of length S, where the latter are implemented using the relevant primitive DFT/FFT function. In
 this way, all FFTs of length T=2^a 3^b 5^c 7^d, for nonnegative integers a-d, may be computed.

 The twiddle factors are precomputed for a given FFT length and may be reused indefinitely and
 across threads. The transform is out-of-place, with the destination buffer used as scratch space.
 The inverse complex-to-real transform distorts its input, but the other three transforms do not.

 This code (and all high-performance C code using complex arithmetic) expects to be compiled with at
 least one of -ffinite-math-only, -fcx-limited-range, or -fcx-fortran-rules, in order to avoid a
 very significant slowdown due to the default semantics of complex multiplication. More modest
 speedups are obtained via (in descending order of benefit over risk ratio): "-fno-signed-zeros
 -fno-rounding-math -fexcess-precision=fast -fno-trapping-math -fno-math-errno -fassociative-math".
 In other words, this code is expected to perform fastest under -ffast-math semantics, but is almost
 as fast under "-ffast-math -fno-associative-math -fno-reciprocal-math", which should be palatable
 to a wider audience.
 */

#include "fft_anywhere.h"

#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* workaround for newlib and certain combinations of apple libc and gcc */
#ifndef CMPLXF
#define CMPLXF __builtin_complex
#endif

struct planned_forward_fft {
    /* next function in recursive scheme */
    void (* function)(float complex * restrict, const float complex * restrict, size_t, const struct planned_forward_fft *);

    /* next plan in the recursive scheme */
    struct planned_forward_fft * next;

    size_t T;
    size_t pad;

    /* twiddle factors for this step in the recursive scheme */
    float complex twiddles[];
};

struct planned_real_fft {
    struct planned_forward_fft * plan;
    float complex twiddles_r2c[];
};

/* assert that the fft plan structs meet the alignment re alignment of the twiddle factors which follow */
static_assert((sizeof(struct planned_forward_fft) % 8) == 0, "misaligned struct");

static void dft3(float complex * restrict const out, const size_t stride, const float complex in0, const float complex in1, const float complex in2) {
    /* primitive for three-point discrete Fourier transform. this and the other primitives are inlined in several places */
    const float complex d1 = -0.5f - I * 0.866025404f;
    const float complex d2 = -0.5f + I * 0.866025404f;
    out[0 * stride] = in0 + in1 + in2;
    out[1 * stride] = in0 + d1 * in1 + d2 * in2;
    out[2 * stride] = in0 + d2 * in1 + d1 * in2;
}

static void dft5(float complex * restrict const out, const size_t stride, const float complex in0, const float complex in1, const float complex in2, const float complex in3, const float complex in4) {
    /* primitive for five-point discrete Fourier transform */
    const float complex d1 = +0.309016994f - I * 0.951056516f;
    const float complex d2 = -0.809016994f - I * 0.587785252f;
    const float complex d3 = -0.809016994f + I * 0.587785252f;
    const float complex d4 = +0.309016994f + I * 0.951056516f;
    out[0 * stride] = in0 +      in1 +      in2 +      in3 +      in4;
    out[1 * stride] = in0 + d1 * in1 + d2 * in2 + d3 * in3 + d4 * in4;
    out[2 * stride] = in0 + d2 * in1 + d4 * in2 + d1 * in3 + d3 * in4;
    out[3 * stride] = in0 + d3 * in1 + d1 * in2 + d4 * in3 + d2 * in4;
    out[4 * stride] = in0 + d4 * in1 + d3 * in2 + d2 * in3 + d1 * in4;
}

static void dft7(float complex * restrict const out, const size_t stride, const float complex in0, const float complex in1, const float complex in2, const float complex in3, const float complex in4, const float complex in5, const float complex in6) {
    /* primitive for seven-point discrete Fourier transform */
    const float complex d1 = +0.623489802f - I * 0.781831482f;
    const float complex d2 = -0.222520934f - I * 0.974927912f;
    const float complex d3 = -0.900968868f - I * 0.433883739f;
    const float complex d4 = -0.900968868f + I * 0.433883739f;
    const float complex d5 = -0.222520934f + I * 0.974927912f;
    const float complex d6 = +0.623489802f + I * 0.781831482f;
    out[0 * stride] = in0 +      in1 +      in2 +      in3 +      in4 +      in5 +      in6;
    out[1 * stride] = in0 + d1 * in1 + d2 * in2 + d3 * in3 + d4 * in4 + d5 * in5 + d6 * in6;
    out[2 * stride] = in0 + d2 * in1 + d4 * in2 + d6 * in3 + d1 * in4 + d3 * in5 + d5 * in6;
    out[3 * stride] = in0 + d3 * in1 + d6 * in2 + d2 * in3 + d5 * in4 + d1 * in5 + d4 * in6;
    out[4 * stride] = in0 + d4 * in1 + d1 * in2 + d5 * in3 + d2 * in4 + d6 * in5 + d3 * in6;
    out[5 * stride] = in0 + d5 * in1 + d3 * in2 + d1 * in3 + d6 * in4 + d4 * in5 + d2 * in6;
    out[6 * stride] = in0 + d6 * in1 + d5 * in2 + d4 * in3 + d3 * in4 + d2 * in5 + d1 * in6;
}

static void fft4(float complex * restrict const out, const size_t stride, const float complex in0, const float complex in1, const float complex in2, const float complex in3) {
    /* performs an fft of size 4 using four dft's of size 2, which results in 2/3 as many operations as a straight dft of size 4 */

    /* perform two dfts of size 2, one multiplied by a twiddle factor (a -90 degree phase shift) */
    const float complex scratch0 = in0 + in2;
    const float complex scratch1 = in0 - in2;
    const float complex scratch2 = in1 + in3;
    const float complex scratch3 = CMPLXF(cimagf(in1) - cimagf(in3), - crealf(in1) + crealf(in3));

    /* perform two more dfts of size 2 */
    out[0 * stride] = scratch0 + scratch2;
    out[1 * stride] = scratch1 + scratch3;
    out[2 * stride] = scratch0 - scratch2;
    out[3 * stride] = scratch1 - scratch3;
}

static void fft_recursive_3(float complex * restrict const out, const float complex * restrict const in, const size_t istride, const struct planned_forward_fft * const plan __attribute((unused))) {
    /* perform a three-point dft within the recursive framework */
    dft3(out, 1, in[0], in[istride], in[2 * istride]);
}

static void fft_recursive_4(float complex * restrict const out, const float complex * restrict const in, const size_t istride, const struct planned_forward_fft * const plan __attribute((unused))) {
    /* perform a four-point fft within the recursive framework */
    fft4(out, 1, in[0], in[istride], in[2 * istride], in[3 * istride]);
}

static void fft_recursive_5(float complex * restrict const out, const float complex * restrict const in, const size_t istride, const struct planned_forward_fft * const plan __attribute((unused))) {
    /* perform a five-point dft within the recursive framework */
    dft5(out, 1, in[0], in[istride], in[2 * istride], in[3 * istride], in[4 * istride]);
}

static void fft_recursive_7(float complex * restrict const out, const float complex * restrict const in, const size_t istride, const struct planned_forward_fft * const plan __attribute((unused))) {
    /* perform a five-point dft within the recursive framework */
    dft7(out, 1, in[0], in[istride], in[2 * istride], in[3 * istride], in[4 * istride], in[5 * istride], in[6 * istride]);
}

static void fft_recursive_8(float complex * restrict const out, const float complex * restrict const in, const size_t istride, const struct planned_forward_fft * const plan __attribute((unused))) {
    /* perform an eight-point fft within the recursive framework */
    const float complex in0 = in[0], in1 = in[istride], in2 = in[2 * istride], in3 = in[3 * istride], in4 = in[4 * istride], in5 = in[5 * istride], in6 = in[6 * istride], in7 = in[7 * istride];

    /* perform four dfts of size 2, two of which are multiplied by a twiddle factor (a -90 degree phase shift) */
    const float complex a0 = in0 + in4;
    const float complex a1 = in0 - in4;
    const float complex a2 = in2 + in6;
    const float complex a3 = CMPLXF(cimagf(in2) - cimagf(in6), crealf(in6) - crealf(in2));
    const float complex a4 = in1 + in5;
    const float complex a5 = in1 - in5;
    const float complex a6 = in3 + in7;
    const float complex a7 = CMPLXF(cimagf(in3) - cimagf(in7), crealf(in7) - crealf(in3));

    /* perform four more dfts of size 2 */
    const float complex c0 = a0 + a2;
    const float complex c1 = a1 + a3;
    const float complex c2 = a0 - a2;
    const float complex c3 = a1 - a3;
    const float complex c4 = a4 + a6;
    const float complex b5 = a5 + a7;
    const float complex b6 = a4 - a6;
    const float complex b7 = a5 - a7;

    /* apply final twiddle factors */
    const float complex c5 = CMPLXF(cimagf(b5) + crealf(b5),   cimagf(b5) - crealf(b5) ) * (float)M_SQRT1_2;
    const float complex c6 = CMPLXF(cimagf(b6), -crealf(b6));
    const float complex c7 = CMPLXF(cimagf(b7) - crealf(b7), -(crealf(b7) + cimagf(b7))) * (float)M_SQRT1_2;

    /* perform four dfts of length two */
    out[0] = c0 + c4;
    out[1] = c1 + c5;
    out[2] = c2 + c6;
    out[3] = c3 + c7;
    out[4] = c0 - c4;
    out[5] = c1 - c5;
    out[6] = c2 - c6;
    out[7] = c3 - c7;
}

static void fft_recursive_by_2(float complex * restrict const out, const float complex * restrict const in, const size_t istride, const struct planned_forward_fft * const plan) {
    const size_t T = plan->T;

    /* perform two ffts of length T / 2 */
    plan->next->function(out + 0 * T / 2, in + 0 * istride, 2 * istride, plan->next);
    plan->next->function(out + 1 * T / 2, in + 1 * istride, 2 * istride, plan->next);

    /* perform T / 2 dfts of length two, applying twiddle factors to all but the first */
    for (size_t it = 0; it < T / 2; it++) {
        const float complex tmp0 = out[it + 0 * T / 2];
        const float complex tmp1 = out[it + 1 * T / 2] * plan->twiddles[it];

        out[it + 0 * T / 2] = tmp0 + tmp1;
        out[it + 1 * T / 2] = tmp0 - tmp1;
    }
}

static void fft_recursive_by_3(float complex * restrict const out, const float complex * restrict const in, const size_t istride, const struct planned_forward_fft * const plan) {
    const size_t T = plan->T;

    /* perform three ffts of length T / 3 */
    plan->next->function(out + 0 * T / 3, in + 0 * istride, 3 * istride, plan->next);
    plan->next->function(out + 1 * T / 3, in + 1 * istride, 3 * istride, plan->next);
    plan->next->function(out + 2 * T / 3, in + 2 * istride, 3 * istride, plan->next);

    /* perform T / 3 dfts of length three, applying twiddle factors to all but the first */
    for (size_t it = 0; it < T / 3; it++)
        dft3(out + it, T / 3,
             out[it + 0 * T / 3],
             out[it + 1 * T / 3] * plan->twiddles[2 * it + 0],
             out[it + 2 * T / 3] * plan->twiddles[2 * it + 1]);
}

static void fft_recursive_by_4(float complex * restrict const out, const float complex * restrict const in, const size_t istride, const struct planned_forward_fft * const plan) {
    const size_t T = plan->T;

    /* perform four ffts of length T / 4 */
    plan->next->function(out + 0 * T / 4, in + 0 * istride, 4 * istride, plan->next);
    plan->next->function(out + 1 * T / 4, in + 1 * istride, 4 * istride, plan->next);
    plan->next->function(out + 2 * T / 4, in + 2 * istride, 4 * istride, plan->next);
    plan->next->function(out + 3 * T / 4, in + 3 * istride, 4 * istride, plan->next);

    /* perform T / 4 ffts of length four, applying twiddle factors to all but the first */
    fft4(out + 0, T / 4, out[0], out[T / 4], out[T / 2], out[3 * T / 4]);

    for (size_t it = 1; it < T / 4; it++)
        fft4(out + it, T / 4,
             out[it + 0 * T / 4],
             out[it + 1 * T / 4] * plan->twiddles[3 * it + 0],
             out[it + 2 * T / 4] * plan->twiddles[3 * it + 1],
             out[it + 3 * T / 4] * plan->twiddles[3 * it + 2]);
}

static void fft_recursive_by_5(float complex * restrict const out, const float complex * restrict const in, const size_t istride, const struct planned_forward_fft * const plan) {
    const size_t T = plan->T;

    /* perform five ffts of length T / 5 */
    plan->next->function(out + 0 * T / 5, in + 0 * istride, 5 * istride, plan->next);
    plan->next->function(out + 1 * T / 5, in + 1 * istride, 5 * istride, plan->next);
    plan->next->function(out + 2 * T / 5, in + 2 * istride, 5 * istride, plan->next);
    plan->next->function(out + 3 * T / 5, in + 3 * istride, 5 * istride, plan->next);
    plan->next->function(out + 4 * T / 5, in + 4 * istride, 5 * istride, plan->next);

    /* perform T / 5 dfts of length five, applying twiddle factors to all but the first */
    for (size_t it = 0; it < T / 5; it++)
        dft5(out + it, T / 5,
             out[it + 0 * T / 5],
             out[it + 1 * T / 5] * plan->twiddles[4 * it + 0],
             out[it + 2 * T / 5] * plan->twiddles[4 * it + 1],
             out[it + 3 * T / 5] * plan->twiddles[4 * it + 2],
             out[it + 4 * T / 5] * plan->twiddles[4 * it + 3]);
}

static void fft_recursive_by_7(float complex * restrict const out, const float complex * restrict const in, const size_t istride, const struct planned_forward_fft * const plan) {
    const size_t T = plan->T;

    /* perform seven ffts of length T / 7 */
    plan->next->function(out + 0 * T / 7, in + 0 * istride, 7 * istride, plan->next);
    plan->next->function(out + 1 * T / 7, in + 1 * istride, 7 * istride, plan->next);
    plan->next->function(out + 2 * T / 7, in + 2 * istride, 7 * istride, plan->next);
    plan->next->function(out + 3 * T / 7, in + 3 * istride, 7 * istride, plan->next);
    plan->next->function(out + 4 * T / 7, in + 4 * istride, 7 * istride, plan->next);
    plan->next->function(out + 5 * T / 7, in + 5 * istride, 7 * istride, plan->next);
    plan->next->function(out + 6 * T / 7, in + 6 * istride, 7 * istride, plan->next);

    /* perform T / 7 dfts of length seven, applying twiddle factors to all but the first */
    for (size_t it = 0; it < T / 7; it++)
        dft7(out + it, T / 7,
             out[it + 0 * T / 7],
             out[it + 1 * T / 7] * plan->twiddles[6 * it + 0],
             out[it + 2 * T / 7] * plan->twiddles[6 * it + 1],
             out[it + 3 * T / 7] * plan->twiddles[6 * it + 2],
             out[it + 4 * T / 7] * plan->twiddles[6 * it + 3],
             out[it + 5 * T / 7] * plan->twiddles[6 * it + 4],
             out[it + 6 * T / 7] * plan->twiddles[6 * it + 5]);
}

struct planned_forward_fft * plan_forward_fft_of_length(const size_t T) {
    /* plan an fft for a given length. this is a recursive function that calculates all the
     necessary twiddle factors and branch conditions which will be encountered during execution of
     the fft, such that executing the fft only requires addition, multiplication, and following
     function pointers */

    if (T < 3) return NULL;

    static struct planned_forward_fft primitives[] = {
        { .function = fft_recursive_3, .T = 3 },
        { .function = fft_recursive_4, .T = 4 },
        { .function = fft_recursive_5, .T = 5 },
        { .function = fft_recursive_7, .T = 7 },
        { .function = fft_recursive_8, .T = 8 },
    };

    /* if fft is one of the above primitive sizes... */
    for (struct planned_forward_fft * primitive = primitives; primitive < primitives + sizeof(primitives) / sizeof(primitives[0]); primitive++)
        if (primitive->T == T) return primitive;

    /* FFT size is not one of the primitive sizes, and must be divisible by a prime factor not larger than 7 */
    size_t S;

    if (6 == T || 10 == T || 14 == T) S = 2; /* special case so we don't end up at T = 2 */
    else if (!(T % 7)) S = 7;
    else if (!(T % 5)) S = 5;
    else if (!(T % 3)) S = 3;
    /* if T is not a power of 2, nothing we can do, calling code should check for this */
    else if (T & (T - 1)) return NULL;
    /* once reduced to powers of 2, try to get to repeatedly dividing by 4 and ending up at 8 */
    else S = T & (size_t)0xAAAAAAAAAAAAAAAA ? 4 : 2; /* S = 2 if T is a power of 4, else 4 */

    /* recursively plan the next fft size, if we can, before allocating the current one */
    struct planned_forward_fft * const plan_next = plan_forward_fft_of_length(T / S);
    if (!plan_next) return NULL;

    struct planned_forward_fft * const plan = malloc(sizeof(*plan) + sizeof(float complex) * (T / S) * (S - 1));
    plan->function = (S == 7) ? fft_recursive_by_7 : (S == 5) ? fft_recursive_by_5 : (S == 4) ? fft_recursive_by_4 : (S == 3) ? fft_recursive_by_3 : fft_recursive_by_2;
    plan->T = T;
    plan->next = plan_next;

    for (size_t it = 0; it < T / S; it++)
        for (size_t is = 1; is < S; is++)
            plan->twiddles[(is - 1) + (S - 1) * it] = cexpf(-I * 2.0f * it * is * (float)M_PI / (float)T);

    return plan;
}

void destroy_planned_forward_fft(struct planned_forward_fft * plan) {
    /* destroy the given plan */
    while (plan) {
        struct planned_forward_fft * next = plan->next;

        /* primitive plans should not be freed */
        if (next) free(plan);
        plan = next;
    }
}

void destroy_planned_inverse_fft(struct planned_inverse_fft * plan) {
    destroy_planned_forward_fft((void *)plan);
}

void destroy_planned_real_fft(struct planned_real_fft * plan) {
    destroy_planned_forward_fft(plan->plan);
    free(plan);
}

void destroy_planned_real_inverse_fft(struct planned_real_inverse_fft * plan) {
    destroy_planned_real_fft((void *)plan);
}

void fft_evaluate_forward(float complex * restrict const out, const float complex * restrict const in, const struct planned_forward_fft * const plan) {
    plan->function(out, in, 1, plan);
}

struct planned_inverse_fft * plan_inverse_fft_of_length(const size_t T) {
    return (void *)plan_forward_fft_of_length(T);
}

void fft_evaluate_inverse(float complex * restrict const out, const float complex * restrict const in, const struct planned_inverse_fft * const iplan) {
    const struct planned_forward_fft * plan = (void *)iplan;
    const size_t T = plan->T;

    /* first compute the forward fft normally */
    plan->function(out, in, 1, plan);

    /* and then reverse the order of the outputs */
    for (size_t it = 1; it < T / 2; it++) {
        float complex tmp = out[it];
        out[it] = out[T - it];
        out[T - it] = tmp;
    }
}

struct planned_real_fft * plan_real_fft_of_length(const size_t T) {
    if ((T / 4U) * 4U != T) return NULL;
    struct planned_forward_fft * plan_actual = plan_forward_fft_of_length(T / 2);
    if (!plan_actual) return NULL;
    struct planned_real_fft * plan = malloc(sizeof(*plan) + sizeof(float complex) * T / 4);
    plan->plan = plan_actual;

    for (size_t iw = 0; iw < T / 4; iw++)
        plan->twiddles_r2c[iw] = -I * cexpf(-I * (float)M_PI * 2.0f * iw / T);

    return plan;
}

struct planned_real_inverse_fft * plan_real_inverse_fft_of_length(const size_t T) {
    return (void *)plan_real_fft_of_length(T);
}

void fft_evaluate_real(float complex * restrict const out, const float * restrict const in, const struct planned_real_fft * const plan) {
    const size_t Th = plan->plan->T;

    plan->plan->function(out, (void *)in, 1, plan->plan);

    /* handle dc bin and nyquist bins. real component of nyquist bin is stored in imaginary component of dc bin */
    out[0] = CMPLXF(crealf(out[0]) + cimagf(out[0]), crealf(out[0]) - cimagf(out[0]));

    for (size_t iw = 1; iw < Th / 2; iw++) {
        /* this can probably be compactified more */
        const float complex a = out[iw], b = out[Th - iw];
        const float complex conj_b = conjf(b);
        const float complex tmpe = a + conj_b, tmpf = a - conj_b;
        const float complex tmpg = conjf(tmpe), tmph = -conjf(tmpf);
        const float complex tw = plan->twiddles_r2c[iw], conj_tw = conjf(tw);

        out[     iw] = 0.5f * (tmpe +      tw * tmpf);
        out[Th - iw] = 0.5f * (tmpg + conj_tw * tmph);
    }

    /* handle T/4 bin, for which the r2c twiddle factor is just -1.0 */
    out[Th / 2] = conjf(out[Th / 2]);
}

void fft_evaluate_real_inverse(float * restrict const out, float complex * restrict const in, const struct planned_real_inverse_fft * const iplan) {
    const struct planned_real_fft * const plan = (void *)iplan;
    const size_t Th = plan->plan->T, T = 2 * Th;

    in[Th / 2] = 2.0f * in[Th / 2];

    for (size_t iw = 1; iw < Th / 2; iw++) {
        const float complex a = in[iw], b = in[Th - iw];
        const float complex conj_b = conjf(b);
        const float complex tmpe = a + conj_b, tmpf = a - conj_b;
        const float complex tmpg = conjf(tmpe), tmph = conjf(tmpf);
        const float complex conj_tw = plan->twiddles_r2c[iw], tw = conjf(conj_tw);

        in[     iw] = tmpg + conj_tw * tmph;
        in[Th - iw] = tmpe -      tw * tmpf;
    }

    in[0] = CMPLXF(crealf(in[0]) + cimagf(in[0]), cimagf(in[0]) - crealf(in[0]));

    plan->plan->function((void *)out, in, 1, plan->plan);

    for (size_t it = 1; it < T; it += 2)
        out[it] = -out[it];
}
