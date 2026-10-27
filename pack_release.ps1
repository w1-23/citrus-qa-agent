# ============================================================
#  Citrus QA Agent 发布打包（v9.4.0）
#  ------------------------------------------------------------
#  用法:
#    powershell -File pack_release.ps1                # 主包（代码，~2MB）
#    powershell -File pack_release.ps1 -CorpusOnly    # 语料附件（~1.6GB，分 2 卷上传 GitHub Releases）
#    powershell -File pack_release.ps1 -IncludeData   # 主包+语料（~1.6GB，可上传）
#    powershell -File pack_release.ps1 -IncludeModels -IncludeData  # 本地完整包（~3.7GB，仅本地/其他渠道分发）
#  说明: GitHub Releases 单文件上限 2GB——大模型（reranker 2.2GB）不走发布包，
#        首次运行经 HF 镜像自动下载（run.ps1 / prepare_models.py 已内置）；
#        语料（LanceDB 向量库 + chunks.jsonl + metadata.json + _idx_map.json）
#        用 -CorpusOnly 单独打包，run.ps1 按语料版本标记自动全量/增量下载。
#  输出: dist/citrus-qa-agent-v9.4.0[-full[-data]].zip / dist/corpus-v9.4.0-#.zip
#  用户拿到 zip → 解压 → 双击运行 run.ps1 → 浏览器打开即用
# ============================================================
param([switch]$IncludeModels, [switch]$IncludeData, [switch]$CorpusOnly,
      [string]$BatchOnly = '', [int]$PartNo = 0)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Version = '9.4.0'
# v9.4.0: 语料与主包同版本（改为 9.4.0）；语料包含 agent/data/.corpus-version 标记，
#          run.ps1 检测本地标记 != 当前版本时全量重下并整体替换（结构性变更：去重/更名/删除批次）
$CorpusVersion = '9.4.0'
$Dist = Join-Path $Root 'dist'
$Suffix = $(if ($IncludeModels) { '-full' } else { '' }) + $(if ($IncludeData) { '-data' } else { '' })
$ZipName = "citrus-qa-agent-v$Version$Suffix.zip"
$ZipPath = Join-Path $Dist $ZipName
$Stage = Join-Path $env:TEMP ("citrus-pack-" + [guid]::NewGuid().ToString('N'))

# ── 语料批次清单（agent/data 顶层各批次目录 + 版本标记，写入每个分卷）──
function Get-CorpusBatchNames {
    param($DataDir)
    return @(Get-ChildItem $DataDir -Directory | Where-Object {
        $_.Name -ne 'lancedb' -and
        ((Test-Path (Join-Path $_.FullName 'chunks.jsonl')) -or (Test-Path (Join-Path $_.FullName 'chunks\chunks.jsonl')))
    } | ForEach-Object { $_.Name })
}
function Write-CorpusMarkers {
    param($DataStage, $CorpusVersion, $BatchNames)
    New-Item -ItemType Directory -Force -Path $DataStage | Out-Null
    Set-Content -Path (Join-Path $DataStage '.corpus-version') -Value $CorpusVersion -Encoding UTF8
    Set-Content -Path (Join-Path $DataStage '.corpus-batches') -Value ($BatchNames -join "`n") -Encoding UTF8
}
# 批次内容（chunks.jsonl + metadata.json + _idx_map.json）→ 目标批次目录
function Copy-BatchContents {
    param($SrcDir, $DstDir)
    New-Item -ItemType Directory -Force -Path $DstDir | Out-Null
    $cj = if (Test-Path (Join-Path $SrcDir 'chunks.jsonl')) { Join-Path $SrcDir 'chunks.jsonl' } else { Join-Path $SrcDir 'chunks\chunks.jsonl' }
    Copy-Item -Force $cj (Join-Path $DstDir 'chunks.jsonl')
    foreach ($extra in 'metadata.json', '_idx_map.json') {
        $p = Join-Path $SrcDir $extra
        if (Test-Path $p) { Copy-Item -Force $p (Join-Path $DstDir $extra) }
    }
}

# ── 仅语料附件（zip 内路径 agent/data/...，解压到仓库根目录即得 agent/data）──
#  v8.13.0: 只打包 LanceDB 向量库 + 各批次 chunks.jsonl（证据定位必需）；
#  旧 Qdrant 批次目录（向量库本体，1.18GB 冗余）不进发布包——体积减半且远低于
#  GitHub 2GiB 单文件上限；本地仍完整保留，可随时回退 qdrant 后端。
if ($CorpusOnly) {
    $DataDir = Join-Path $Root 'agent\data'
    if (-not (Test-Path $DataDir)) {
        Write-Host "⚠ agent/data 不存在，无法打包语料" -ForegroundColor Red
        exit 1
    }
    # ── v8.14.1: 单批增量分卷（-CorpusOnly -BatchOnly <批次名> [-PartNo <n>]）──
    # 新增批次不打乱既有分卷编号：单独打成 corpus-v$CorpusVersion-<PartNo>.zip
    # （默认 PartNo=1；运行 run.ps1 时自动按序号续接下载）。
    if ($BatchOnly) {
        $name = $BatchOnly
        $srcDir = Join-Path $DataDir $name
        $tbl = Join-Path $DataDir "lancedb\$name.lance"
        if (-not (Test-Path (Join-Path $srcDir 'chunks.jsonl')) -or -not (Test-Path $tbl)) {
            Write-Host "⚠ 批次 $name 缺少 chunks.jsonl 或 lancedb\$name.lance" -ForegroundColor Red
            exit 1
        }
        $pn = if ($PartNo -gt 0) { $PartNo } else { 1 }
        $zipPath = Join-Path $Dist "corpus-v$CorpusVersion-$pn.zip"
        if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
        New-Item -ItemType Directory -Force -Path $Dist | Out-Null
        $binRoot = Join-Path $Stage ("part" + $pn)
        $binData = Join-Path $binRoot 'agent\data'
        New-Item -ItemType Directory -Force -Path (Join-Path $binData 'lancedb') | Out-Null
        Copy-BatchContents -SrcDir $srcDir -DstDir (Join-Path $binData $name)
        robocopy $tbl (Join-Path $binData "lancedb\$name.lance") /E /XF *.lock /R:0 /W:0 /NFL /NDL /NJH /NJS | Out-Null
        # v9.4.0: 每个分卷都带语料版本标记 + 完整批次清单（run.ps1 全量/增量判定依据）
        Write-CorpusMarkers -DataStage $binData -CorpusVersion $CorpusVersion -BatchNames (Get-CorpusBatchNames $DataDir)
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::CreateFromDirectory($binRoot, $zipPath, [System.IO.Compression.CompressionLevel]::NoCompression, $false)
        $mb = [Math]::Round((Get-Item $zipPath).Length / 1MB, 1)
        Write-Host "✅ 单批分卷打包完成: corpus-v$CorpusVersion-$pn.zip  $mb MB（$name）" -ForegroundColor Green
        Remove-Item -Recurse -Force $Stage -ErrorAction SilentlyContinue
        exit 0
    }
    # v8.13.0: 语料分卷打包——GitHub 单附件上限 2GiB，按批次贪心装箱为若干 <1GB 分卷，
    #  run.ps1 按序号循环下载并解压合并（以后数据继续增大也不会超限）
    Write-Host "打包语料分卷 corpus-v$CorpusVersion-#.zip（每个分卷 <1GB）..." -ForegroundColor Yellow
    $PartSpan = 1000MB   # 每卷目标上限（字节）
    # 1) 统计各批次体积（chunks.jsonl + 对应 lancedb 表）
    $sizes = @()
    Get-ChildItem $DataDir -Directory | Where-Object {
        $_.Name -ne 'lancedb' -and
        ((Test-Path (Join-Path $_.FullName 'chunks.jsonl')) -or (Test-Path (Join-Path $_.FullName 'chunks\chunks.jsonl')))
    } | ForEach-Object {
        $name = $_.Name
        $j = if (Test-Path (Join-Path $_.FullName 'chunks.jsonl')) { Join-Path $_.FullName 'chunks.jsonl' } else { Join-Path $_.FullName 'chunks\chunks.jsonl' }
        $sz = (Get-Item $j).Length
        $tbl = Join-Path $DataDir "lancedb\$name.lance"
        if (Test-Path $tbl) { $sz += (Get-ChildItem $tbl -Recurse -File | Measure-Object -Property Length -Sum).Sum }
        $sizes += [pscustomobject]@{ Name = $name; Json = $j; Bytes = [long]$sz }
    }
    if ($sizes.Count -eq 0) { Write-Host "⚠ 未找到任何含 chunks.jsonl 的批次" -ForegroundColor Red; exit 1 }
    # 2) 贪心装箱（体积降序 → 放入首个放得下的卷；ArrayList 规避 PS5.1 嵌套索引 += 坑）
    $bins = New-Object System.Collections.ArrayList
    foreach ($b in ($sizes | Sort-Object Bytes -Descending)) {
        $idx = -1
        for ($i = 0; $i -lt $bins.Count; $i++) {
            $sum = ($bins[$i] | Measure-Object -Property Bytes -Sum).Sum
            if (($sum + $b.Bytes) -le $PartSpan) { $idx = $i; break }
        }
        if ($idx -ge 0) { $null = $bins[$idx].Add($b) }
        else { $nb = New-Object System.Collections.ArrayList; $null = $nb.Add($b); $null = $bins.Add($nb) }
    }
    # 3) 每卷暂存并打包（NoCompression 秒级；内容 = agent/data/<批次>/chunks.jsonl + metadata.json +
    #    _idx_map.json + lancedb/<批次>.lance + .corpus-version / .corpus-batches 标记）
    New-Item -ItemType Directory -Force -Path $Dist | Out-Null
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $partNo = 0; $totalMB = 0
    foreach ($bin in $bins) {
        $partNo++
        $zipPath = Join-Path $Dist "corpus-v$CorpusVersion-$partNo.zip"
        if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
        $binRoot = Join-Path $Stage ("part" + $partNo)
        $binData = Join-Path $binRoot 'agent\data'
        foreach ($b in $bin) {
            Copy-BatchContents -SrcDir (Join-Path $DataDir $b.Name) -DstDir (Join-Path $binData $b.Name)
            $tbl = Join-Path $DataDir "lancedb\$($b.Name).lance"
            if (Test-Path $tbl) { robocopy $tbl (Join-Path $binData "lancedb\$($b.Name).lance") /E /XF *.lock /R:0 /W:0 /NFL /NDL /NJH /NJS | Out-Null }
        }
        Write-CorpusMarkers -DataStage $binData -CorpusVersion $CorpusVersion -BatchNames (Get-CorpusBatchNames $DataDir)
        [System.IO.Compression.ZipFile]::CreateFromDirectory($binRoot, $zipPath, [System.IO.Compression.CompressionLevel]::NoCompression, $false)
        $mb = [Math]::Round((Get-Item $zipPath).Length / 1MB, 1); $totalMB += $mb
        Write-Host "  卷 $partNo/$($bins.Count): corpus-v$CorpusVersion-$partNo.zip  $mb MB  (批次: $(($bin | ForEach-Object Name) -join ', '))" -ForegroundColor Cyan
        Remove-Item -Recurse -Force $binRoot
    }
    Remove-Item -Recurse -Force $Stage -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "✅ 语料分卷打包完成（$partNo 卷，共 $totalMB MB -> $Dist）" -ForegroundColor Green
    Write-Host "   上传到 GitHub Releases v$CorpusVersion；run.ps1 按序号自动循环下载并合并"
    exit 0
}

$ExcludeDirs = @('state', 'logs', 'workspace', 'data', '__pycache__',
                 '.pytest_cache', '.tmp_runner', '.venv', '.git', '.hf_cache',
                 'tests')
$ExcludeFiles = @('.env', '*.pyc', '*.tmp', '*.lock', 'MODEL_ROUTING.md')
$DataDir = Join-Path $Root 'agent\data'

Write-Host "打包 Citrus QA Agent v$Version $(if ($IncludeModels) { '(含模型) ' } else { '(主包) ' })$(if ($IncludeData) { '(含示例语料) ' } else { '' })" -ForegroundColor Yellow

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
        $HfStage = Join-Path $StageAgent '.hf_cache'
        if (Test-Path $HfStage) { Remove-Item -Recurse -Force $HfStage }
        # robocopy: 跳过被占用文件（模型被运行中服务加载）+ 排除锁文件
        robocopy $HfCache $HfStage /E /XF *.lock /R:0 /W:0 /NFL /NDL /NJH /NJS | Out-Null
    } else {
        Write-Host "⚠ agent/.hf_cache 不存在，跳过模型缓存" -ForegroundColor Yellow
    }
}

# ── 完整包: 附带示例语料（LanceDB 向量库 + chunks.jsonl + metadata，开箱可测检索/引用/写作全链路）──
if ($IncludeData) {
    if (Test-Path $DataDir) {
        Write-Host "附带示例语料 (data/lancedb + chunks.jsonl + metadata.json) ..." -ForegroundColor Cyan
        $DataStage = Join-Path $StageAgent 'data'
        if (Test-Path $DataStage) { Remove-Item -Recurse -Force $DataStage }
        robocopy (Join-Path $DataDir 'lancedb') (Join-Path $DataStage 'lancedb') /E /XF *.lock /R:0 /W:0 /NFL /NDL /NJH /NJS | Out-Null
        # 各批次：chunks.jsonl + metadata.json + _idx_map.json（与 CorpusOnly 同口径）
        Get-ChildItem $DataDir -Directory | Where-Object {
            (Test-Path (Join-Path $_.FullName 'chunks.jsonl')) -or (Test-Path (Join-Path $_.FullName 'chunks\chunks.jsonl'))
        } | ForEach-Object {
            Copy-BatchContents -SrcDir $_.FullName -DstDir (Join-Path $DataStage $_.Name)
        }
        Write-CorpusMarkers -DataStage $DataStage -CorpusVersion $CorpusVersion -BatchNames (Get-CorpusBatchNames $DataDir)
    } else {
        Write-Host "⚠ agent/data 不存在，跳过语料" -ForegroundColor Yellow
    }
}

# ── 清理暂存区嵌套产物（递归复制不过滤，逐层清理 __pycache__/.pyc/.env/.tmp_runner）──
Get-ChildItem -Path $Stage -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $Stage -Recurse -Directory -Filter '.tmp_runner' -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $Stage -Recurse -File -Include '*.pyc', '*.tmp' -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $Stage -Recurse -File -Filter '.env' -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

# ── 根文件 ──
foreach ($f in @('README.md', 'LICENSE', 'requirements.txt', 'run.ps1', '.gitignore')) {
    $src = Join-Path $Root $f
    if (Test-Path $src) { Copy-Item -Force $src (Join-Path $Stage $f) }
}

# ── .ps1 统一强制 UTF-8 BOM（v8.13.0 防呆：Windows PowerShell 5.1 读取无 BOM UTF-8 会乱码）──
Get-ChildItem $Stage -Recurse -Filter '*.ps1' | ForEach-Object {
    $t = [System.IO.File]::ReadAllText($_.FullName, [System.Text.Encoding]::UTF8)
    [System.IO.File]::WriteAllText($_.FullName, $t, (New-Object System.Text.UTF8Encoding($true)))
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
