# Resume checkpoint — 2026-11-05

User scope: a portable Windows miner ZIP, GitHub fork, detailed 5090D32GB vs
5090 analysis, exact PEARL proofs, and a measured attempt to improve the user's
~283.6 TH/s /402W baseline. Pearlhash is preferred; the user accepts another
pool if needed. Preserve and resume the original WildRig BAT after GPU tests.

Completed before resumption:
- Fork `laplacezz/CPPminer-5090D`, branch `research/5090d-exact-tensor`.
- Windows builds through `704fc1c`; CPU and Rust tests pass.
- 15 instruction paths pass bounded synthetic GPU fold tests.
- INT8 miner submitted accepted LuckyPool shares; general FP16 passed a full
  offline proof verification. Earlier versions remain slower than WildRig.
- Raw benchmark logs, per-second GPU telemetry and report draft are in the
  parent lab's `results/` and `research/` directories.
- Original WildRig PID 25132 and watchdog PID 50868 were resumed. Do not assume
  those PIDs indefinitely: the experiment wrapper validates identities.

Actual interruption:
`pool-mxfp4-rect-2b37c27` fails the bounded converter before its first GEMM.
Protocol dense noise is [-16,47]; sparse differences reach [-63,63]. A binary
signal therefore does NOT make the noised operands lie in [-3,3]. The bounded
microtests are not proof that those modes are usable for live mining.

Work to finish:
1. Correct ordinary BF16/TF32/FP16 paths for full integer operands.
2. Implement exact radix reconstruction for low-precision formats. Explore a
   LUT-selected radix-4 / Karatsuba decomposition (six rather than nine matrix
   products), and three-product FP8/FP6 alternatives. Test protocol-range data.
3. Implement a checked sparse-prefix projection for zero-signal B: reuse its
   rank-128 structure while retaining every milestone, with exact conversion
   checks and an integer fallback. Benchmark real preparation cost too.
4. Test complete proof verification and accepted live shares; compare actual
   power and end-to-end rates. Do not promote raw TMAC/s to pool hashrate.
5. Update the Chinese report, remove any stale claims about bounded inputs,
   package licenses/checksums and a ready-to-run local Windows launch script.
6. Finish Git commits and public fork documentation/release. Do not publish the
   user's wallet-bearing scripts, pool captures or private logs.
7. Verify original BAT unchanged and WildRig producing positive hashrate before
   final reply. No Windows driver, clock, power or Office setting changes.

Pearlhash protocol research is optional under the user's relaxed requirement.
Only partial message decoding was achieved. Do not claim compatibility.
