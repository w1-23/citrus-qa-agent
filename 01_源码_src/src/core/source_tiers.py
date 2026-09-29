"""来源可信度分级（P2 架构约束）—— 域名/URL 确定性分级，零 LLM 成本。

分级语义（对应 feedback P2"一亩田公众号幻觉内容被引用"）：
  T1  同行评议 / 官方 / 学术数据库 —— 可单独支撑机制、数值、结论
  T2  权威媒体 / 行业媒体 / 机构新闻稿 —— 可支撑时效类事实；数值/机制结论需与
      本地 [n] 交叉或显式标注
  T3  自媒体 / B2B 导购 / UGC 百科 / 论坛 / 公司营销 / 未知域名 —— **不得单独**
      支撑机制、数值、结论；保留时必须打"低可信来源"标并要求与 [n] 交叉

判定顺序（先命中先赢，全部大小写不敏感）：
  1. t3_url_patterns   —— 页面形态级降级（论坛/问答/文库/公众号等，即使域名校威）
  2. t3_domains        —— 显式不可信域名（长后缀匹配）
  3. t2_domains        —— 媒体/行业
  4. t1_domains        —— 期刊/数据库/机构
  5. t1_suffixes       —— 制度性后缀（.gov.cn/.edu.cn/.ac.*/.go.jp/...）
  6. doi 信号          —— 证据带 DOI → 升至 T2（域名未验但有文献标识）
  7. 默认 T3           —— 宁可错杀；后果是"低可信"标注+要求交叉，非丢弃

冷启动表由 beta pack `session_evidence.csv` 的 193 个真实域名引导生成（2026-09-11）。
维护约定：审计/反馈中发现误判 → 在 yaml 的对应清单加一行即可，重启生效；
yaml 缺失或字段缺失时使用本文件内嵌默认值。

**v10.5 审计（2026-09-23）**：用联网臂 48 题 480 条真实证据复测，补齐 9 处缺口
（T1: doi.org / eppo.int / springer.com / caas.cn；T2: freshplaza.com / freshfruitportal.com /
japanfruit.jp / fruit-processing.com；后缀: .go.kr）。完整方法与数字见内部审计报告
`20_source_tiers_audit.md`（本地维护，不随本仓库分发）。

⚠️ **tier 不得用作丢弃闸门**：同批实测 T3 占 60.2%，其中 **95.5% 是 `default_unknown`
（只是没进过冷启动表），真正命中显式 T3 规则的仅 13/480 = 2.7%**。按 tier 硬删 T3 会以
"没听说过"为由丢掉六成证据。正确用法是**降权 + 要求交叉印证**（见下方三档语义）。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── 页面形态级降级：无论域名是什么，这类页面形态不可信 ──────────────────
T3_URL_PATTERNS: tuple[str, ...] = (
    r"baijiahao\.baidu\.com",
    r"mp\.weixin\.qq\.com",
    r"tieba\.baidu\.com",
    r"wenku\.baidu\.com",
    r"zhidao\.baidu\.com",
    r"mall\.cnki\.net",
    r"zjzx\.cnki\.net",        # cnki 问答/BBS
    r"/bbs[/?#]",
    r"/ask[-_/]",
    r"/zhuanlan/",
    r"baike\.",
    r"wiki/",
)

# ── T3：UGC / 自媒体 / B2B 导购 / 公司营销 / 不可溯源 ───────────────────
T3_DOMAINS: frozenset[str] = frozenset({
    "ymt.com",            # 一亩田（feedback P2 点名：幻觉错误内容）
    "cnhnb.com",          # 惠农网
    "gengzhongbang.com",  # 耕种帮
    "enongzi.com",        # 农资 B2B
    "chinaseed.com",
    "qq.com", "sohu.com", "163.com",        # 门户自媒体聚合
    "hujiang.com",
    "baike.baidu.com",
    "baijiahao.baidu.com",
    "mp.weixin.qq.com",   # 公众号
    "mall.cnki.net", "zjzx.cnki.net",
    "zhangqiaokeyan.com", "wetp.vip",       # 文档聚合/镜像站
    "pubcompare.ai",
    "astracash.com",      # 与主题无关站点（数据实测混入）
    "59.110.221.111",     # 裸 IP，无溯源
    "towinbio.com", "genepioneer.com", "igenebook.com",  # 公司营销/科普
    "blooge.cn",          # 个人博客
    "huayueyang.com.cn", "amole.com.cn", "zxjpsg.cn", "ysfg.net", "kjzz.org",
    "eureka.patsnap.com",
})

# ── T2：权威/行业媒体、机构新闻、可溯源但非同行评议 ─────────────────────
T2_DOMAINS: frozenset[str] = frozenset({
    "thepaper.cn", "sciencenet.cn", "people.com.cn", "chinanews.com.cn",
    "stdaily.com", "nkb.com.cn", "nfncb.cn", "jingjiribao.cn",
    "btwb.baotounews.com.cn", "hbtv.com.cn", "xiancn.com",
    "agrijournal.com.cn", "agri.nais.net.cn",
    "freshplaza.cn", "agropages.com", "asiafruitchina.net",
    "ebiotrade.com", "bioon.com.cn", "antpedia.com",
    "eurekalert.org", "scholar.google.com",
    "citrusindustry.net", "citrusresearch.org", "californiaagnet.com",
    # v10.5 审计补充（实测高频、此前均被判 default_unknown→T3）：
    #   freshplaza.com 18 条——注意 freshplaza.cn 早已在 T2，属同源 .com 镜像漏收
    "freshplaza.com", "freshfruitportal.com", "japanfruit.jp",
    "fruit-processing.com",
    "revistacultivar.com", "dcz-china.org", "farmchina.org.cn",
    "guojiguoshu.com", "lifescience.net", "visualize.jove.com",
    "coial.org", "bv.fapesp.br", "plantsuccess.org", "agriir.cn",
    "zh.wikisource.org", "wikisource.org",
    "planthealth2025.eventscribe.net", "guidelinecentral.com",
    "shflora.ibiodiversity.cn",
})

# ── T1：期刊/出版社/学术数据库/官方数据库/科研机构 ─────────────────────
T1_DOMAINS: frozenset[str] = frozenset({
    # 出版社/期刊平台
    "sciencedirect.com", "link.springer.com", "springer.com",
    "onlinelibrary.wiley.com",
    "nature.com", "academic.oup.com", "frontiersin.org", "mdpi.com",
    "mdpi-res.com", "tandfonline.com", "biomedcentral.com",
    "journals.pagepress.net", "apsjournals.apsnet.org",
    "cdnsciencepub.com", "journals.ashs.org", "oaj.fupress.net",
    "xml-journal.net", "afrjournals.org", "geneticagraria.it",
    "journal.hep.com.cn", "sciengine.com", "jstage.jst.go.jp",
    # 中文期刊/学位库
    "cnki.net", "wanfangdata.com.cn", "cqvip.com",
    "plantscience.cn", "chinaagrisci.com", "ahs.ac.cn",
    "fruitsci.zzgss.cn", "molplantbreed.org", "casb.org.cn",
    "cgl.org.cn", "jtsb.ijournals.cn", "gdnykx.cnjournals.org",
    "nfnyxb.xml-journal.net", "cnjournals.org",
    # 学术索引/预印本
    "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov", "europepmc.org", "biorxiv.org",
    "semanticscholar.org", "scilit.com", "scilit.net", "x-mol.com",
    "scienceopen.com", "mendeley.com", "research.wur.nl",
    "escholarship.org", "scholar.google.com", "webofscience.com",
    "doaj.org", "read.qxmd.com", "discovery.researcher.life",
    "scispace.com", "agris.fao.org",
    # 官方数据库/政府/科研院所
    # v10.5 审计补充（依据：联网臂 480 条真实证据实测，见 reports/20_source_tiers_audit.md）
    #   doi.org    — DOI 官方解析器，溯源第一跳，此前被判 default_unknown→T3
    #   eppo.int   — 欧洲地中海植物保护组织，植检名录权威（与语料病害实体同源）
    #   caas.cn    — 中国农业科学院；注意 `.ac.cn` 后缀**不覆盖** caas.cn，属显式漏收
    "doi.org", "eppo.int", "caas.cn",
    "iplant.cn", "greeningdb.org", "citrusgenomedb.org",
    "citrusvariety.ucr.edu", "cric.cn", "agri.cn", "nsfc.gov.cn",
    "ars.usda.gov", "crec.ifas.ufl.edu", "ivia.gva.es",
    "umr-agap.cirad.fr", "fundecitrus.com.br", "shigen.nig.ac.jp",
    "itak.feilab.net", "registry.seqco.de", "hz.nstl.gov.cn",
    "cae.cn", "patentimages.storage.googleapis.com",
    "ufdcimages.uflib.ufl.edu",
    # 高校/院所（示例，后缀规则会兜底 *.edu.cn 等）
    "hzau.edu.cn", "swu.edu.cn", "pku.edu.cn", "htu.edu.cn",
    "cas.cn", "kib.ac.cn",
})

# ── 制度性后缀（任何未显式列出的子域名按此兜底） ────────────────────────
T1_SUFFIXES: tuple[str, ...] = (
    ".gov.cn", ".gov", ".edu.cn", ".edu", ".ac.cn",
    ".cas.cn", ".ac.jp", ".go.jp", ".go.kr", ".ac.uk", ".edu.au", ".edu.sa",
    # v10.5: 补 .go.kr（韩国政府；实测 lib.qia.go.kr 8 条此前被判 default_unknown→T3。
    #        原表已有 .go.jp 却缺 .go.kr，属明显不对等）
    ".gva.es", ".cirad.fr", ".inrae.fr", ".fao.org", ".uni-bonn.de",
)
T3_SUFFIXES: tuple[str, ...] = (".vip", ".icu", ".xyz", ".top", ".site")

DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+", re.IGNORECASE)


@dataclass(frozen=True)
class TierResult:
    tier: str          # "T1" | "T2" | "T3"
    rule: str          # 命中规则名（日志/审计留痕）
    detail: str = ""   # 命中细节


def _normalize_host(url: str) -> str:
    try:
        host = urlparse(url if "//" in url else "//" + url, scheme="http").hostname or ""
    except ValueError:
        host = re.sub(r"^[a-z]+://", "", url.lower()).split("/")[0]
    return (host or url.lower()).strip().lower().rstrip(".")


def _match_domain(host: str, domains: frozenset[str]) -> Optional[str]:
    for d in sorted(domains, key=len, reverse=True):  # 长后缀优先（严格 host==d 或 host 以 .d 结尾）
        if host == d or host.endswith("." + d):
            return d
    return None


def classify_url(url: str, doi: str = "", *, overrides: Optional[dict] = None) -> TierResult:
    """确定性分级：URL → (tier, rule, detail)。任何异常都安全降级为 T3。"""
    try:
        raw = (url or "").strip()
        low = raw.lower()
        host = _normalize_host(raw)

        for pat in T3_URL_PATTERNS:
            if re.search(pat, low):
                return TierResult("T3", "url_pattern", pat)

        d = _match_domain(host, T3_DOMAINS)
        if d:
            return TierResult("T3", "t3_domain", d)
        d = _match_domain(host, T2_DOMAINS)
        if d:
            return TierResult("T2", "t2_domain", d)
        d = _match_domain(host, T1_DOMAINS)
        if d:
            return TierResult("T1", "t1_domain", d)

        for suf in T1_SUFFIXES:
            if host.endswith(suf) or host == suf.lstrip("."):
                return TierResult("T1", "t1_suffix", suf)

        for suf in T3_SUFFIXES:
            if host.endswith(suf):
                return TierResult("T3", "t3_suffix", suf)

        if (doi or "") and DOI_RE.search(doi):
            return TierResult("T2", "doi_signal", doi[:40])

        return TierResult("T3", "default_unknown", host or raw[:60])
    except Exception as e:  # 永不抛出——分级失败按最保守处理
        logger.debug("[source_tiers] classify failed: %s (%s)", url, e)
        return TierResult("T3", "error_fallback", str(e)[:60])


def grade_web_items(items: list[dict]) -> list[dict]:
    """给 web 证据条目批量打标：新增 tier / tier_rule 字段（就地返回新 list）。"""
    out = []
    for it in items:
        res = classify_url(str(it.get("url", "")), str(it.get("doi", "")))
        out.append({**it, "source_tier": res.tier, "tier_rule": res.rule})
    return out
