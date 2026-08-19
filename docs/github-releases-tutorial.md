# GitHub Releases 操作教程（本项目专用）

面向：不熟悉 GitHub 网页操作、只想把 `dist/` 里的发布包传到网上下发的人。
全程浏览器操作，不需要记命令（`gh` CLI 可选装）。

---

## 0. 先搞懂两个概念

- **Release（发布页）**：GitHub 上的一个下载页面，比如 v8.13.0、v8.14.1。
  每个 Release 可以挂多个**附件（Assets）**，别人在那里点附件就能下载。
- **Tag（标签）**：给某次代码快照起的名字（v8.14.1）。删 Release **不会**删 Tag，
  只要 Tag 还在，重建 Release 不会丢任何东西。

本项目的约定：

| 东西 | 放哪个 Release |
|---|---|
| 主包 `citrus-qa-agent-v8.14.1.zip`（代码） | 挂到对应的新 Release（如 v8.14.1） |
| 语料附件 `corpus-v8.13.0-1/2/3/4.zip`（数据） | **统一挂在 v8.13.0 Release**，与主包版本解耦，旧卷不重传 |

---

## 1. 给「已有 Release」追加附件（比如给 v8.13.0 补 corpus 卷 4）

1. 打开 https://github.com/w1-23/citrus-qa-agent/releases
2. 找到目标 Release 那一行，点右侧 **✏️ Edit**
3. 页面向下，找到 **Attach binaries**（拖拽区），把本地文件拖进去
4. 点底部绿色 **Update release** 保存

## 2. 「新建」一个 Release（比如发布 v8.14.1 主包）

最快方式：直接打开直链 **https://github.com/w1-23/citrus-qa-agent/releases/new**
（不用在页面上找按钮）

1. **Choose a tag**：输入版本号（如 `v8.14.1`）→ 出现蓝色小字 *Create new tag on publish*，点它确认
2. **Target**：选分支（本项目发主包时选 `feature/oa-fulltext-evidence`）
3. **Release title**：写一个标题（如 `Citrus QA Agent v8.14.1`）
4. **Describe this release**：粘贴 `dist\RELEASE-<版本>.md` 的内容（记事本打开全选复制）
5. **Attach binaries**：拖入主包 zip
6. 点绿色 **Publish release** 发布

## 3. 删错了怎么办（本教程最常见的场景）

- **只删错了某个附件**：进 Edit 页 → Attach binaries 里每个文件右侧有个小垃圾桶图标 → 点它删除对应文件 → Update release。删附件不影响其他附件和 Release。
- **误删了整条 Release**：没关系。Tag 不会丢，回到「新建 Release」流程，输入同一个 tag 名，
  重新填标题、正文、传附件再 Publish 即可——新 Release 会自动挂回原 tag 指向的代码版本。
- **特例：想废掉一个附件不让别人下载**（如传错了 Release）：到正确 Release 里重新上传一份，
  错的那份进 Edit 删掉。

## 4. 发布后自查（30 秒）

打开 Release 页面，核对下表，符合预期即完成：

| 版本 | 应包含 |
|---|---|
| v8.13.0 | `citrus-qa-agent-v8.13.0.zip` + `corpus-v8.13.0-1/2/3/4.zip`（共 5 个附件） |
| v8.14.1 及以后 | `citrus-qa-agent-v<版本>.zip`（一般 1 个附件；语料卷照旧挂 v8.13.0） |

也可用命令行核对（不用登录，公开数据）：

```
curl.exe -sS https://api.github.com/repos/w1-23/citrus-qa-agent/releases
```

## 5. 常见坑

- **附件传错 Release**：不影响下载，但会误导；按第 3 节删掉错的、重传对的。
- **版本号写错**：tag 名务必和附件名里的版本一致，否则用户下载的脚本和自检对不上。
- **语料卷重传/改名**：不要做。卷 1-3 内容一直没变，永远留在 v8.13.0；以后新增语料只追加新卷号（4、5…）。
- **run.ps1 下载规则（知道即可）**：全新安装按 1→N 顺序下载全部卷，遇 404 停止；
  存量部署按清单（`$NewLanceBatches`）检查缺失批次，只下载增量卷。

## 6. 一劳永逸：装 gh 命令行（可选）

以后想用命令发布：

```
winget install --id GitHub.cli -e
gh auth login        # 跟着提示在浏览器登录一次，之后免密
gh release create vX.Y.Z --title "..." --target feature/oa-fulltext-evidence --notes-file dist\RELEASE-X.Y.Z.md dist\citrus-qa-agent-vX.Y.Z.zip
gh release upload v8.13.0 dist\corpus-v8.13.0-4.zip   # 给已有 Release 追加附件
```

---

*配套代码：`run.ps1`（下载/自检）、`pack_release.ps1`（打 zip）、`dist/`（产物目录）。*