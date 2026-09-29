# Citrus QA Agent 停止脚本（v8.9）
# 用法: 右键 -> 使用 PowerShell 运行（或在 PowerShell 中执行 .\stop.ps1）
# 作用: 停止占用 8000 端口的服务进程

$ErrorActionPreference = 'SilentlyContinue'

$conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if (-not $conn) {
    Write-Host '✅ 8000 端口无服务在运行，无需停止。' -ForegroundColor Green
    exit 0
}

$pids = @($conn | Select-Object -ExpandProperty OwningProcess -Unique)
foreach ($pid_ in $pids) {
    $p = Get-Process -Id $pid_ -ErrorAction SilentlyContinue
    if (-not $p) { continue }
    Write-Host "发现服务进程: PID=$pid_ $($p.ProcessName) 启动于 $($p.StartTime)"
}

$ans = Read-Host '确认停止以上进程? (y/N)'
if ($ans -ne 'y' -and $ans -ne 'Y') {
    Write-Host '已取消，未停止任何进程。' -ForegroundColor Yellow
    exit 1
}

foreach ($pid_ in $pids) {
    Stop-Process -Id $pid_ -Force
    Write-Host "已停止: PID=$pid_" -ForegroundColor Green
}
Start-Sleep -Seconds 2

if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) {
    Write-Host '⚠️ 8000 端口仍被占用，请手动在任务管理器中结束对应 python 进程。' -ForegroundColor Red
    exit 1
}
Write-Host '✅ 服务已停止，8000 端口已释放。' -ForegroundColor Green
Write-Host '提示: 下次启动请运行 run.ps1（LanceDB 向量库无需锁文件，可直接重启）。'
