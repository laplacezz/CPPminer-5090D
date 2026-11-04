param()
$ErrorActionPreference='Stop'
$dest='packaging/windows/licenses'
New-Item $dest -ItemType Directory -Force | Out-Null
$metadata=(& cargo metadata --locked --format-version 1 --manifest-path rust/cp-proof-ffi/Cargo.toml) | ConvertFrom-Json
foreach($package in $metadata.packages) {
    $folder=Join-Path $dest ($package.name+'-'+$package.version)
    New-Item $folder -ItemType Directory -Force | Out-Null
    @("Name: $($package.name)","Version: $($package.version)","License: $($package.license)","Repository: $($package.repository)") | Set-Content "$folder/PACKAGE.txt" -Encoding UTF8
    $root=Split-Path $package.manifest_path
    Get-ChildItem $root -File | Where-Object {$_.Name -match '^(LICENSE|COPYING|NOTICE|UNLICENSE)'} | ForEach-Object {Copy-Item $_.FullName $folder}
    if($package.license_file -and (Test-Path (Join-Path $root $package.license_file))) {Copy-Item (Join-Path $root $package.license_file) $folder}
}
foreach($name in @('LICENSE','LICENSE_A2','LICENSE_A2LLVM','LICENSE_CC0','LICENSE_APACHE','LICENSE_MIT')) {
    $p=Join-Path 'third_party/blake3' $name
    if(Test-Path $p){Copy-Item $p (Join-Path $dest ('BLAKE3-'+$name))}
}
