# 手机远程访问 DSH Web GUI — SSH 隧道方案（方案 B）

## 电脑端状态（已完成 ✅，无需任何操作）

| 项目 | 状态 |
|---|---|
| OpenSSH 服务器 | 已安装（sshd.exe） |
| sshd 服务 | **Running**，开机自动启动 |
| 监听端口 | 0.0.0.0:22（IPv4 + IPv6） |
| 防火墙 | 规则 `OpenSSH-Server-In-TCP` 已启用（Private 网络放行 22） |
| 网络类别 | Private（网络名：403_405_5G） |
| 电脑局域网 IP | **192.168.31.172** |
| 登录账户 | Administrator（有密码，用你的 Windows 登录密码） |
| DSH harness | **未重启、未改动**，仍在 127.0.0.1:3080 |

## 手机端步骤

### 1. 前提
- 手机连接与电脑同一个路由器/热点的 Wi-Fi（同一局域网，192.168.31.x 网段）。
- 手机安装 SSH 客户端：**Termius**（iOS / Android 都有，免费版支持端口转发）。

### 2. 在 Termius 中创建主机
- **Host / 地址**：`192.168.31.172`
- **Port**：`22`
- **Username**：`Administrator`
- **Password**：你的 Windows 登录密码
- 先点连接测试，能登录成功即可。

### 3. 配置本地端口转发（关键步骤）
在 Termius 中打开该主机的 **Port Forwarding（端口转发）** 设置，添加一条：

| 字段 | 值 |
|---|---|
| 类型 | Local（本地转发） |
| 监听地址 | 127.0.0.1（手机本机） |
| 监听端口 | **3080** |
| 目标地址 | **127.0.0.1**（电脑上的 harness） |
| 目标端口 | **3080** |

保存后保持该 SSH 连接**处于已连接状态**。

### 4. 在手机浏览器访问
打开手机浏览器（Chrome / Safari），访问：

```
http://127.0.0.1:3080
```

注意：SSH 连接必须保持不断开，隧道才有效。

## 原理说明

- DSH 的 Web 服务只监听电脑本机 `127.0.0.1:3080`，局域网直连行不通（CLI 也禁止 `--host 0.0.0.0`）。
- SSH 隧道 = 手机上的 `127.0.0.1:3080` → 加密通道 → 电脑上的 `127.0.0.1:3080`。
- 请求的 Host 头仍是回环地址，harness 的浏览器信任栅栏直接放行，无需改任何配置。
- 全程加密 + SSH 密码认证，局域网内其他人无法访问。

## 维护与安全建议

- **用完可关闭** sshd（需要管理员 PowerShell）：
  ```powershell
  Stop-Service sshd
  ```
  再次使用：`Start-Service sshd`
- **更安全**：改用 SSH 密钥登录（Termius 支持），然后在服务器上
  `C:\ProgramData\ssh\sshd_config` 把 `PasswordAuthentication` 改为 `no`。
- 如果电脑网络类别被系统改成「公用网络」，防火墙会拦 22 端口，需要把该网络改回「专用」。
- 手机不在家时（不同网络）无法用此方案——那是 Tailscale（方案 C）的用途。

## 本次修改涉及的文件/脚本（可留作记录）

- `E:\codex_WORKSPACES\Citrus_QA_Agent\ssh-setup\install-sshd.ps1` — 安装 OpenSSH 服务器
- `E:\codex_WORKSPACES\Citrus_QA_Agent\ssh-setup\ensure-fw-rule.ps1` — 确保防火墙规则存在
