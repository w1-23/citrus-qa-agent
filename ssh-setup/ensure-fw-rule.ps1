# Ensure inbound TCP 22 firewall rule exists (run elevated, idempotent).
$ErrorActionPreference = 'Stop'
$log = 'C:\Users\Administrator\AppData\Local\Temp\dsh-sshd-fw.log'

try {
    $rule = Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue
    if (-not $rule) {
        # also check whether any rule already allows TCP 22 inbound
        $portRules = Get-NetFirewallPortFilter -Protocol TCP -ErrorAction SilentlyContinue |
            Where-Object { $_.LocalPort -eq 22 }
        $existing = @()
        foreach ($pf in $portRules) {
            $existing += Get-NetFirewallRule -AssociatedNetFirewallPortFilter $pf -ErrorAction SilentlyContinue
        }
        if ($existing) {
            $result = "EXISTS-AS: $($existing[0].Name)"
        } else {
            New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' `
                -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
            $result = 'CREATED'
        }
    } else {
        $result = "EXISTS ($($rule.Name))"
    }
    $check = Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue
    if ($check) { $state = "enabled=$($check.Enabled) action=$($check.Action) profile=$($check.Profile)" } else { $state = 'NOT-FOUND-AFTER-CREATE' }
    "fw=$result | $state" | Out-File -FilePath $log -Encoding UTF8
} catch {
    "ERROR: $($_.Exception.Message)" | Out-File -FilePath $log -Encoding UTF8
    exit 1
}
exit 0
