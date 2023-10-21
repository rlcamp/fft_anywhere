# fft_anywhere

 This is a good-enough FFT implementation which benchmarks within a respectable distance of the fastest known implementations on platforms of interest, and usually beats them on non-SIMD platforms, when compiled with a modern C compiler. This implementation should work where other FFT implementations do not, hence the name. If you are on a SIMD processor and need a permissive-licensed FFT implementation, [jpommier/pffft](https://bitbucket.org/jpommier/pffft) is considerably faster, and you should prefer it to this or fftw.

## How it works

 The core of the implementation is a set of primitives for the DFTs of sizes 3, 5, and 7, and unrolled FFTs of sizes 4 and 8. A second set of functions implements the Cooley-Tukey decomposition for T / S x S, for S = 2, 3, 4, 5, and 7. These are used recursively to decompose a length-T Fourier transform into S transforms of length T/S, followed by T/S transforms of length S, where the latter are implemented using the relevant primitive DFT or FFT function.

 In this way, all FFTs of length T=2^a 3^b 5^c 7^d, for nonnegative integers a-d, may be computed. The real-to-complex and complex-to-real transform variants require that T be a multiple of 4, otherwise there are no further restrictions.

 The recursive function call structure, and the twiddle factors for each stage of the recursion, are precomputed for a given FFT length and may be reused indefinitely and across threads, with no per-use requirement for a memory allocator. The transform is out-of-place, with the destination buffer used as scratch space. The inverse complex-to-real transform distorts its input, but the other three transforms do not.

## Caveats about compilation

 This code (and all high-performance C code using complex arithmetic) expects to be compiled with at least one of `-ffinite-math-only`, `-fcx-limited-range`, or `-fcx-fortran-rules`, in order to avoid a very significant slowdown due to the default semantics of complex multiplication.

 More modest speedups are obtained via (in descending order of benefit over risk ratio): `-fno-signed-zeros -fno-rounding-math -fexcess-precision=fast -fno-trapping-math -fno-math-errno -fassociative-math`. In other words, this code is expected to perform fastest under `-ffast-math` semantics, but is almost as fast under `-ffast-math -fno-associative-math -fno-reciprocal-math`, which should be palatable to a wider audience.
