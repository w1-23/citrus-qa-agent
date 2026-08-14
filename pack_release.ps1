# ============================================================
#  Citrus QA Agent 发布打包（v8.5.0）
#  ------------------------------------------------------------
#  用法:
#    powershell -File pack_release.ps1               # 主包（不含模型，~几 MB）
#    powershell -File pack_release.ps1 -IncludeModels # 完整包（含模型缓存，~2.5GB）
#  输出: dist/citrus-qa-agent-v8.5.0.zip
#  用户拿到 zip → 解压 → 双击运行 run.ps1 → 浏览器打开即用
# ============================================================
param([switch]$IncludeModels)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Version = '8.5.0'
$Dist = Join-Path $Root 'dist'
$ZipName = "citrus-qa-agent-v$Version" + $(if ($IncludeModels) { '-full' } else { '' }) + '.zip'
$ZipPath = Join-Path $Dist $ZipName
$Stage = Join-Path $env:TEMP ("citrus-pack-" + [guid]::NewGuid().ToString('N'))

$ExcludeDirs = @('state', 'logs', 'workspace', 'data', '__pycache__',
                 '.pytest_cache', '.tmp_runner', '.venv', '.git', '.hf_cache')
$ExcludeFiles = @('.env', '*.pyc', '*.tmp')

Write-Host "打包 Citrus QA Agent v$Version $(if ($IncludeModels) { '(完整版: 含模型)' } else { '(主包)' })" -ForegroundColor Yellow

# ── 暂存区 ──
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
$StageAgent = Join-Path $Stage 'agent'
New-Item -ItemType Directory -Force -Path $StageAgent | Out-Null

# ── 复制 agent/（排除运行时目录与密钥文件；-notlike 对数组失效须逐项判断）──
Get-ChildItem -Path (Join-Path $Root 'agent') -Force | ForEach-Object {
    $skip = $_.Name -in $ExcludeDirs
    if (-not $skip) {
        foreach ($pat in $ExcludeFiles) {
            if ($_.Name -like $pat) { $skip = $true; break }
        }
    }
    if ($skip) { return }
    if ($_.PSIsContainer) {
        Copy-Item -Recurse -Force $_.FullName (Join-Path $StageAgent $_.Name)
    } else {
        Copy-Item -Force $_.FullName (Join-Path $StageAgent $_.Name)
    }
}
# 确保目录结构
New-Item -ItemType Directory -Force -Path (Join-Path $StageAgent 'workspace\output') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $StageAgent 'state') | Out-Null

# ── 完整包: 附带模型缓存 ──
if ($IncludeModels) {
    $HfCache = Join-Path $Root 'agent\.hf_cache'
    if (Test-Path $HfCache) {
        Write-Host "附带模型缓存 (.hf_cache) ..." -ForegroundColor Cyan
        Copy-Item -Recurse -Force $HfCache (Join-Path $StageAgent '.hf_cache')
    } else {
        Write-Host "⚠ agent/.hf_cache 不存在，跳过模型缓存" -ForegroundColor Yellow
    }
}

# ── 根文件 ──
foreach ($f in @('README.md', 'LICENSE', 'requirements.txt', 'run.ps1', '.gitignore')) {
    $src = Join-Path $Root $f
    if (Test-Path $src) { Copy-Item -Force $src (Join-Path $Stage $f) }
}

# ── 打 zip（v8.5.0: 改用 .NET ZipFile——PS5.1 的 Compress-Archive 对 2GB+
#    模型文件 Optimal 压缩极慢/失败；模型权重高熵压缩率低，整体 NoCompression
#    秒级完成，体积差别可忽略）──
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $Stage, $ZipPath,
    [System.IO.Compression.CompressionLevel]::NoCompression, $false)
$sizeMB = [Math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Remove-Item -Recurse -Force $Stage

Write-Host ""
Write-Host "✅ 打包完成: $ZipPath  ($sizeMB MB)" -ForegroundColor Green
Write-Host "   用户使用: 解压 → 运行 run.ps1 → 浏览器自动打开 → 页面填 API Key"
