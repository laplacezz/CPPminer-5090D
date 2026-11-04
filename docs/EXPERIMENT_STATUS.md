# Verified checkpoint — 2026-11-05

Performance target remains unmet: no proven improvement over the user's
WildRig baseline and no full-5090 equivalence. This is a runnable experimental
Windows delivery, not a validated production replacement.

Completed:
- 22 selectable exact CUDA paths; native SM89 and SM120a Windows build.
- Corrected the bounded-signal/noised-operand error at the previous interruption.
- Three/six-product exact radix algorithms, checked sparse-prefix reassociation,
  checked Cauchy-bounded F32 bias, and two alternative integer CTA shapes.
- CPU exhaustive algebra, CPU/Rust CI, GPU INT64-reference milestone tests,
  multi-CTA canonical mapping tests, and complete offline proofs.
- Live LuckyPool accepted shares: INT8 binary control, projection-FP16 and
  checked-bias FP16. No local verification failure or pool rejection observed
  in these bounded tests; not a long-run share-rate study.
- Detailed report and sanitized result tables in EXPERIMENT-REPORT.md.
- Binary source commit 803995991eb42f5bf032bf6cf99e55bdd887ffd2;
  GitHub Actions run 41897096683 passed.
- Original WildRig BAT unchanged; original miner/watchdog/window resumed and
  positive API hashrate verified. No driver/clock/power/Office changes.

Pearlhash private protocol and PRL+NOCK merged mining are NOT implemented.
The user authorized another pool if needed. Wallet-bearing LOCAL launch
scripts and private logs are excluded from public source/release packages.
