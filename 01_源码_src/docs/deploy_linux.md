# CitrusQA Linux 部署

此部署方案先在现有 Linux/NVIDIA 主机上验证。网页服务、Agent、RAG 索引与模型运行在服务器；用户只需浏览器。每个账号有独立会话、输出目录和下载权限。管理员统一配置模型与联网 Key。

## 首次上线顺序

1. 管理员账号负责安装缺失的系统 Python 组件和配置系统服务；普通 Linux 账号 `wangziqiang` 运行应用。现有 Anaconda 为该账号所有，应用环境单独放在数据盘项目目录。
2. 项目统一部署在 `/home/WZQ/CitrusQA/`：`01_源码_src/` 是代码和网页、`02_配置_config/` 是配置、`agent/` 保存语料/模型缓存/状态/日志/用户文件、`.venv/` 是 Python 3.12 环境，`tmp/` 是临时空间。启动前确认 `/home` 挂载在数据盘。
3. 应用只绑定 `127.0.0.1:8000`。**账号库目前为空，第一个注册账号会成为管理员**。首次初始化不要开放公网注册：先 SSH 登录普通账号并运行 `/home/WZQ/CitrusQA/.venv/bin/python /home/WZQ/CitrusQA/01_源码_src/scripts/create_admin.py`，在终端提示中输入用户名和密码；之后管理员通过 HTTPS 域名登录。
4. 管理员登录后配置主模型和联网服务 Key；Key 保存在服务器 `agent/state/`，不进 GitHub。管理员面板可管理账号、每日限额、查看全局队列和请求审计记录。
5. 默认全局最多同时处理 10 个聊天请求，其他请求按 FIFO 排队；等待队列上限 100，单次最长等 15 分钟。审计保存问题、回答、工具名/状态摘要及附件元信息，保留 365 天；不保存附件原文或模型隐藏推理。记录不随用户删除会话而删除。
6. 购买域名后将 DNS 指向服务器，在 Nginx 或 Caddy 配置 HTTPS，并启用 `CITRUS_COOKIE_SECURE=1`。公网只开放 80/443；不要先开放应用端口或公网注册。

## 主机资源建议

现有机器有 96 核、128 GB 内存和两张 RTX PRO 6000D（每张显存单独为 84 GB）。当前 RAG 向量模型与重排模型使用 CPUExecutionProvider，预热完成后已做本地检索试跑；五次耗时中位数约 3.53 秒，GPU 保持空闲。先按 CPU 模式运行，只有真实并发和响应时间显示 CPU 不足时再评估启用 GPU。代码、当前模型缓存和文献数据约 6.5 GiB，另预留索引增长、用户上传和备份空间。不要把 GPU 数量误当作共享显存。

保持单 Uvicorn worker，使活跃请求共享同一服务进程中的检索模型实例，不会每个请求加载一套模型。聊天并发上限 10，额外请求排队；当前 CPU 路径不占用 GPU 显存。若后续启用 GPU，再依据实测设置显存隔离与更细粒度调度。

## 进程与公网入口

当前服务由普通账号临时启动，systemd 服务尚未安装。要启用开机自启，请管理员账号执行：

```bash
sudo cp /home/WZQ/CitrusQA/01_源码_src/scripts/citrusqa.service /etc/systemd/system/citrusqa.service
sudo cp /home/WZQ/CitrusQA/01_源码_src/scripts/citrusqa-tunnel.service /etc/systemd/system/citrusqa-tunnel.service
sudo systemctl daemon-reload
sudo systemctl enable --now citrusqa
sudo systemctl status citrusqa --no-pager
```

系统服务以 `wangziqiang` 身份运行，监听 `127.0.0.1:8000`，只信任来自本机 Tunnel 的转发头，并为公网 HTTPS 设置安全 Cookie。首次启用时如果临时服务仍占用 8000 端口，先用 `sudo ss -ltnp 'sport = :8000'` 确认 PID，再停止已确认的 CitrusQA 旧进程，然后启动 systemd 服务。

域名注册并接入 Cloudflare 后，在 Cloudflare 控制台创建**命名 Tunnel**，将公开主机名指向 `http://127.0.0.1:8000`。在服务器下载 `cloudflared` 到数据盘 `/home/WZQ/CitrusQA/04_脚本_scripts/cloudflared` 并赋予执行权限。将控制台给出的 Tunnel Token 写入 `/home/WZQ/CitrusQA/agent/state/cloudflare-tunnel.env`，格式为 `TUNNEL_TOKEN=...`，文件权限设为 `600`、目录权限 `700`；不要把 Token 发到聊天、命令历史或 Git。

先完成学校网络/信息化部门审批及适用的网站备案和 AI 服务要求，再启用公网 Tunnel：`sudo systemctl enable --now citrusqa-tunnel`。Quick Tunnel 不适用于本服务的流式回答。公网自助注册会在 Tunnel 可达后开放；管理员账号必须先通过上面的 CLI 脚本建立。

Cloudflare Tunnel 提供 HTTPS 入口并转发到本机 Uvicorn。公开链接启用前须确认校园网络允许该 Tunnel 和公众服务；无需对公网开放 Uvicorn 端口。将来迁移到学校子域名时，重新配置同一 Tunnel 的公开主机名并完成外部访问验收。

## 多用户数据与维护

- 账号数据使用 SQLite，密码为 PBKDF2 哈希；无需收集邮箱。管理员能停用账号、重置临时密码和调整普通账号每日请求上限。默认上限为每人每天 30 轮，管理员不受该上限限制。
- 会话和生成文件按账号隔离；聊天内容和附件可能发送给管理员配置的模型及联网服务。上线前应在首页/隐私说明中告知用户，并用 HTTPS 保护传输。
- 附件只在当前会话处理，不加入共享语料。导出文件有期限清理；定期备份 `state/`、本地语料和模型配置，备份同样需要加密并限制管理员访问。
- 首期没有邮箱验证和自助找回密码；用户忘记密码时由管理员重置，并通过安全渠道交付一次性临时密码。
- 上线前验证两个普通账号互相访问对方会话、任务状态、引用、工作区文件和下载链接均被拒绝；确认普通账号无法访问管理设置。

## 当前进度与公网开放前

服务器应用已运行并完成 CPU RAG 预热；目前无账号、无 Cloudflare Tunnel、无临时域名，且 systemd 单元尚未安装。Cloudflare DNS、域名、Tunnel Token 和学校审批是公开上线的前置条件。公网发布前还需完成两个账号的数据隔离验证、隐私说明、备份和外部网络访问验收。不要在聊天中发送服务器密码、私钥、API Key 或 Tunnel Token。
