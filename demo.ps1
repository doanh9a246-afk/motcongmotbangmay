# demo.ps1 - Demo N-body truoc lop: bam Enter de chay tung buoc, khong can go lenh.
#   Chay:  .\demo.bat   (hoac: powershell -ExecutionPolicy Bypass -File demo.ps1)
#   Them buoc Newton 3 (khi du gio):  .\demo.bat -Newton3
# Neu mot buoc loi, script in ket qua chay thu ngay 25/11 (ghi ro la so cu) roi di tiep.
param([switch]$Newton3)

Set-Location $PSScriptRoot
$exe = Join-Path $PSScriptRoot 'bin\nbody_cpu.exe'
$env:OMP_NUM_THREADS = '16'   # khong dung -t: libgomp cua MinGW gcc 12.1 co the crash

$steps = @(
    @{ Ten = '1. Tuan tu (1 luong)';       Args = '-n 8192 -s 5 -m serial';
       DuPhong = 'serial, N=8192: 38.237 ms/buoc, 35.10 GFLOP/s' },
    @{ Ten = '2. OpenMP (16 luong)';       Args = '-n 8192 -s 5 -m omp';
       DuPhong = 'omp 16, N=8192: 8.762 ms/buoc, 153.18 GFLOP/s (nhanh hon ~4.4x)' },
    @{ Ten = '3. Kiem tra dung dan';       Args = '-n 2048 -s 5 -m omp -verify';
       DuPhong = "[verify] speedup 1.89x | sai so tuong doi max = 0.000e+00 -> PASS`n      |dE/E0| = 8.783e-07" }
)
if ($Newton3) {
    $steps += @{ Ten = '4. OpenMP Newton 3 (N^2/2 cap)'; Args = '-n 8192 -s 5 -m omp_newton3';
                 DuPhong = 'omp_newton3, N=8192: 13.398 ms/buoc (cham hon all-pairs ~1.5x)' }
}

function Show-Fallback($step, $why) {
    Write-Host "  (!) $why - ket qua chay thu ngay 25/11 tren may nay:" -ForegroundColor Red
    Write-Host "      $($step.DuPhong)" -ForegroundColor DarkYellow
}

Clear-Host
Write-Host '== DEMO N-body: Tuan tu vs OpenMP | i5-12500H, 16 luong ==' -ForegroundColor Cyan
if (-not (Test-Path $exe)) { Write-Host "Khong thay $exe - se chi in ket qua chay thu." -ForegroundColor Red }

foreach ($s in $steps) {
    Write-Host ''
    Write-Host "--- $($s.Ten) ---" -ForegroundColor Cyan
    Write-Host "PS> .\bin\nbody_cpu.exe $($s.Args)" -ForegroundColor Yellow
    [void](Read-Host '    [Enter de chay]')
    if (-not (Test-Path $exe)) { Show-Fallback $s 'Khong co file chay'; continue }
    try {
        & $exe $s.Args.Split(' ')
        if ($LASTEXITCODE -ne 0) { Show-Fallback $s "Chuong trinh tra ve ma loi $LASTEXITCODE" }
    } catch {
        Show-Fallback $s 'Chuong trinh bi loi'
    }
}

Write-Host ''
Write-Host '== Het demo. Cam on thay co va cac ban! ==' -ForegroundColor Cyan
