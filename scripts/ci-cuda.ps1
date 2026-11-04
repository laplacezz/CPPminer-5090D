$ErrorActionPreference = 'Stop'
$dest = Join-Path $env:RUNNER_TEMP 'cuda-12.9'
New-Item $dest -ItemType Directory -Force | Out-Null
$manifest = Invoke-RestMethod 'https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json'
foreach ($component in @('cuda_nvcc','cuda_cudart','cuda_cccl')) {
    $entry = $manifest.$component.'windows-x86_64'
    $zip = Join-Path $env:RUNNER_TEMP "$component.zip"
    Invoke-WebRequest ('https://developer.download.nvidia.com/compute/cuda/redist/' + $entry.relative_path) -OutFile $zip
    if ((Get-FileHash $zip -Algorithm SHA256).Hash.ToLower() -ne $entry.sha256) { throw "Checksum failed: $component" }
    $expanded = Join-Path $env:RUNNER_TEMP $component
    Expand-Archive $zip $expanded -Force
    $root = Get-ChildItem $expanded -Directory | Select-Object -First 1
    Copy-Item (Join-Path $root.FullName '*') $dest -Recurse -Force
}
"CUDA_PATH=$dest" | Out-File $env:GITHUB_ENV -Append -Encoding utf8
"$dest\bin" | Out-File $env:GITHUB_PATH -Append -Encoding utf8
# BLAKE3 C reference; CUDA implementation and proof sources are already vendored.
$zip = Join-Path $env:RUNNER_TEMP 'blake3.zip'
Invoke-WebRequest 'https://github.com/BLAKE3-team/BLAKE3/archive/refs/tags/1.8.2.zip' -OutFile $zip
$expanded = Join-Path $env:RUNNER_TEMP 'blake3-src'
Expand-Archive $zip $expanded -Force
New-Item 'third_party/blake3' -ItemType Directory -Force | Out-Null
Copy-Item "$expanded/BLAKE3-1.8.2/c/*" 'third_party/blake3' -Recurse -Force
