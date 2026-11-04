CPPminer-5090D exact arithmetic experiments, Windows x64
Source: https://github.com/laplacezz/CPPminer-5090D/tree/research/5090d-exact-tensor

Requires compatible NVIDIA CUDA 12.9 driver support and RTX 4090/5090/D.
Blackwell low-precision formats use SM120a; older hardware uses exact fallbacks.
No Python, CUDA Toolkit, Visual Studio or Rust install is needed to RUN.
Extract ALL files. Run verify-tensor.bat before mining.
The GPU test needs exclusive GPU access; stop other miners during that test.

Edit POOL/WALLET in start-PEARL.bat, then double-click it. It uses LuckyPool
Stratum. Pearlhash's private protocol and PRL+NOCK merged mining are NOT
implemented. Do not copy an @nock=... login suffix into the Pearl wallet.

22 experimental paths are available; see ALGORITHMS.md and EXPERIMENT-REPORT.md.
--tensor-mode project-fp16 is the new checked sparse-prefix reassociation.
--tensor-mode fp16-bias uses a checked Cauchy bound and exact fallback.
--tensor-mode fp8 / fp6-karatsuba use three-product exact reconstruction.
FP4/FP6 modes reconstruct with six products rather than rounding the inputs.
All 32 milestones and exact integer proofs are retained. No driver unlock.

Actual performance and adoption limits are in EXPERIMENT-REPORT.md.
No promise of full-5090 performance or improvement over WildRig is made.
cp_tensor_test benchmarks are not pool-confirmed hashrate; a MAC and two
arithmetic operations use different numerical conventions.

Use --verify --no-fee for controlled tests. No new fee is added; --no-fee
disables upstream fee code. Stop with Ctrl+C. No services or GPU setting
changes are installed. Keep your existing WildRig folder for return.
Personalized LOCAL.bat is local-only and must not be published with a wallet.
The public start-PEARL.bat contains a placeholder wallet.
