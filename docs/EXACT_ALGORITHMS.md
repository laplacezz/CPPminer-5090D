# Exact alternatives for original RTX 5090D 32GB

These are experimental implementations for the integer certificate-v3 PEARL
path in this fork, not a driver unlock or a change of consensus. They were
developed here as alternative algorithms; no claim of worldwide novelty is made.
Read the dated experiment report for hardware measurements and adoption limits.

## The range correction

Small *signal* does not imply small *noised operand*. The current dense noise is
`(random & 63)-16`, in `[-16,47]`. A sparse positive/negative pair gives
`[-63,63]`. Binary A signals therefore produce `[-63,64]` noised A; zero B
signals still produce `[-63,63]` noised B. Early microtests on `[-3,3]` inputs
cannot establish live-mining suitability. Converters now reconstruct those
actual values exactly, or fall back to INT8 outside their checked range.

The final XOR is not linear with respect to integer addition. Every individual
cell must be reconstructed before the prescribed milestone XOR/rotation/hash.
All alternatives preserve all 32 cumulative K=128 milestones and the 8x8
interleaved hash layout. There is no approximate acceptance tolerance.

## Modes

| CLI mode | Actual arithmetic | Exactness mechanism |
|---|---|---|
| int8 | S8 x S8 -> S32 | Direct integer baseline |
| fp16, fp16-fast | F16 x F16 -> F32 | Reset every K=128, convert partials to S32 |
| mixed | Alternating INT8 / F16-F32 | Same S32 cumulative cells |
| bf16 | BF16 x BF16 -> F32 | All INT8 inputs exact; K=128 partials |
| tf32 | TF32 x TF32 -> F32 | All INT8 inputs exact; K=128 partials |
| fp8 | E4M3 x E4M3 -> F32 | Three-product radix-8 Karatsuba |
| fp6-karatsuba | E2M3 x E2M3 -> F32 | Three half-scaled products, quarter-unit recovery |
| fp6 | E3M2 x E3M2 -> F32 | Six-product LUT-selected radix-4 |
| fp4 | E2M1 x E2M1 -> F32 | Same six-product reconstruction |
| fp4-fp16, fp6-fp16 | Low-precision inputs -> F16 | Six products; reset each K=32 panel |
| int4 | Legacy S4 x S4 -> S32 | Four radix-16 cross products |
| mxfp6 | Unit-scaled E3M2 -> F32 | Six-product reconstruction |
| mxfp4, nvfp4 | Unit-scaled packed E2M1 -> F32 | Six-product reconstruction |
| project-fp16 | F16 x F16 -> F32 | Checked sparse-prefix reassociation |
| project-bf16 | Two BF16 x BF16 -> F32 | Exact high/residual split of projected A |
| project-tf32 | TF32 x TF32 -> F32 | Checked projection, exact integer inputs |
| fp16-bias | F16 x F16 -> F32 | Cauchy bound; otherwise segmented exact fallback |
| int8-tall / int8-wide | S8 x S8 -> S32 | 256x64 / 64x256 CTA, same canonical hash cells |

FP6/FP4/blocked formats require the included SM120a code and Blackwell hardware.
Unsupported radix modes return an exact INT8 fallback on older hardware.
This is a Windows x64 research build tested on one original RTX 5090D 32GB;
build inclusion of SM89 does not imply every mode has been hardware-tested on 4090.

## Three products instead of four: FP8 and E2M3 FP6

For each integer x in [-64,64], choose
`h=floor((x+4)/8)`, `l=x-8h`. Then `l in [-4,3]`, `h in [-8,8]`,
and `h+l in [-12,11]`. All are exact E4M3 integers. With corresponding
digits of y, form matrix products P=H*H', Q=L*L', R=(H+L)*(H'+L').

```
XY = 56 P - 7 Q + 8 R
```

Three MMA products replace the four naive high/low cross products. Conversions
are cached independently for A and B. This still costs three products: it does
not turn FP8's raw throughput into the same effective INT8 throughput.

E2M3 can represent each of h/2, l/2, (h+l)/2 exactly. Its products are quarters.
Initializing F32 accumulators with 3*2^20 gives ULP=1/4. Subtracting the bias's
bit pattern recovers four times the matrix product, hence the same integer
coefficients above recover XY without a division or precision loss.

## Six products instead of nine: FP4 and E3M2 FP6

Use x=a0+4a1+16a2. An exhaustively checked LUT chooses digits whose six values
`a0,a1,a2,a0+a1,a0+a2,a1+a2` all belong to
`{-6,-4,-3,-2,-1,0,1,2,3,4,6}`. These are exactly representable by E2M1;
E3M2 can also represent them. All 129 integers in [-64,64] are covered.

Let Pi=Ai*Bi and Sij=(Ai+Aj)*(Bi+Bj). Then

```
XY = -19 P0 -52 P1 +176 P2 +4 S01 +16 S02 +64 S12
```

This uses six products rather than nine. F32 partials reset at K=128. F16
accumulators reset at K=32 because `32*6*6=1152 <=2048`; accumulating a full
128-term segment in F16 could exceed its consecutive-integer range.

INT4 instead uses x=16h+l, l in [-8,7], h in [-4,4], and four products with
weights 256,16,16,1. The screenshot's legacy INT4 instruction is itself slow,
so equality between D and full 5090 does not imply competitive throughput.

## New sparse-prefix reassociation

For zero signal B, the generated noisy matrix has the exact structure

```
B'[c,k] = E[c,p(k)] - E[c,q(k)]
T_s[r,j] = sum(k < 128(s+1)) A'[r,k] * ([j=p(k)]-[j=q(k)])
C_s = T_s E^T
```

The GPU builds T for every required prefix. E has only 128 columns rather than
4096. At n=131072 its F16 storage is 32 MiB, vs 512 MiB for dense INT8 noisy B.
Reusing E may reduce external memory traffic and lets the F32-accumulating
Tensor instruction keep only a floating accumulator during the dot product.

This **does not reduce nominal multiply count**: 32 products of width 128
still equal width 4096. It adds projection preparation and storage, and its
advantage, if any, must be measured including them.

Before use, GPU checks require `abs(T_s[r,j]) <=2048` and
`47*sum_j(abs(T_s[r,j])) <2^22` for every row and milestone. Thus F16/TF32
integers and all F32 biased prefix sums are exact. If any row fails, the whole
attempt uses original INT8 rather than silently rounding. Nonzero signal B
and missing noise metadata also select INT8.

The BF16 variant splits T into a rounded BF16 integer high part and an exact
BF16 residual. It validates the residual and the stronger bound using both
absolute parts, then performs two products. This is not an assumption that
BF16 directly represents integers up to 2048.

With accumulator bias 3*2^22, F32 has ULP=1 throughout the strict checked
interval (-2^22,2^22). After XOR of 64 biased encodings, signed 23-bit recovery
gives the exact signed-integer XOR. An even number of cells is essential.

## Data-dependent Cauchy bound

For ordinary noisy A/B, measure the maximum squared row norms EA and EB on
the GPU. If `uint64(EA)*EB <2^44`, Cauchy-Schwarz bounds every subset dot
product by `sqrt(EA*EB) <2^22`. All biased F32 sums then stay in the ULP=1
binade. The fp16-bias mode can avoid the 64 persistent S32 cumulative cells.
If the check fails, it uses the independently tested K=128 segmented path.
The actual input data determine eligibility; a statistical expectation is
never used as a proof of safety.

## Verification and interpretation

`python tests/test_exact_reassociation.py` exhaustively checks 16641 integer
pairs for each reconstruction, the 32 reassociated prefixes, and bias/XOR
recovery. `cp_tensor_test.exe` compares every milestone against independent
scalar INT64 reference results, both layouts and nonzero panel offsets,
including protocol-range and adversarial projection-fallback cases.

`cp_tensor_test --bench-exact-large MODE_ID` uses actual-range operands with
structured zero-B noise, redoes A preparation on every sample, and includes
GEMM/fold/hash. It omits proof/network and is explicitly not pool hashrate.
`--bench-narrow-large` deliberately measures bounded synthetic inputs and
raw instruction alternatives; those rates are not effective live-miner rates.

For adoption use accepted complete proofs and end-to-end work per actual watt,
with aligned measurement windows. A MAC is one multiply-accumulate; standard
TOPS/TFLOPS convention counts two operations per MAC. Do not silently compare
different miners' displayed units or multiply a microbenchmark by a marketing
sparsity factor. No equal-clock full-5090 comparator is available on this PC.

`--binary-a` provides a 0/1-signal INT8 control. The tall/wide modes use this
distribution automatically. Ineligible odd batch shapes retain the 128x128
integer kernel. Multi-CTA tests check their canonical output remapping.
