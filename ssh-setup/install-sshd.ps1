# Install and start OpenSSH Server (run elevated). Writes a transcript log.
$ErrorActionPreference = 'Stop'
$log = Join-Path $env:TEMP 'dsh-sshd-install.log'
Start-Transcript -Path $log -Force

try {
    # 1. Install the OpenSSH.Server capability (downloads from Windows Update)
    $cap = Get-WindowsCapability -Online -Name 'OpenSSH.Server*'
    Write-Output "OpenSSH.Server capability state: $($cap.State)"
    if ($cap.State -ne 'Installed') {
        Add-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0'
        Write-Output 'Capability installed.'
    }

    # 2. Enable and start the sshd service
    Set-Service -Name sshd -StartupType Automatic
    Start-Service sshd
    Write-Output 'sshd service started.'

    # 3. Ensure the inbound firewall rule exists for port 22
    $rule = Get-NetFirewallRule -Service sshd -ErrorAction SilentlyContinue
    if (-not $rule) {
        New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' `
            -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
        Write-Output 'Firewall rule created.'
    } else {
        Write-Output 'Firewall rule already present.'
    }

    # 4. Summary
    Get-Service sshd | Select-Object Name, Status, StartType | Format-List
} catch {
    Write-Output "ERROR: $_"
    Write-Output $_.ScriptStackTrace
} finally {
    Stop-Transcript
}
