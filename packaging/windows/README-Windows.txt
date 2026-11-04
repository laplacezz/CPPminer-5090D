CPPminer-5090D experimental Windows x64 build
Source: https://github.com/laplacezz/CPPminer-5090D

Requires a compatible NVIDIA driver (CUDA 12.9 or newer) and RTX 4090/5090/D.
No Python, CUDA Toolkit, Visual Studio or Rust installation is needed to run.
Extract ALL files to a writable directory. Run verify-tensor.bat first.
Run start-PEARL.bat after setting your pool/wallet. Original WildRig files are untouched.

Modes: --tensor-mode int8 | fp16 | mixed
FP16 means exact signed INT8 inputs, FP32 partials reset every K=128, INT32
reconstruction and the same integer proof. This is not approximate FP16 mining.
The build is experimental. There is NO promise of faster hashrate or full-5090 speed.
cp_tensor_test --bench measures only GEMM+fold+hash, excluding matrix preparation,
proofs and network. Its TMAC/s must NOT be reported as pool-confirmed hashrate.

Upstream CPPminer fee code is retained. See upstream README for its fee policy.
Use --no-fee to disable it for controlled local tests. No new fee is added.
Stop with Ctrl+C. This build does not install services or change GPU settings.
