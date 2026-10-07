"""失败聚类：把一次构建的失败用例按失败原因聚成簇，并跨构建追踪失败点。

背景
----
一场构建红了几十个用例，逐个点开看报错非常费时。绝大多数失败是少数几个
根因的重复表现——同一个接口挂了、同一条断言不过、同一种报错（超时 /
500 / 连接拒绝）在多个用例里出现。失败聚类要做三件事：

1. **簇内聚因**：从每条失败结果里抽取三类失败特征——
   - 接口特征：``METHOD + 归一化 path + HTTP 状态码``
   - 断言特征：``断言类型 + 期望值``（期望值里的数字归一化，避免噪声拆散）
   - 报错关键词：异常类名 / 已知错误文案 / HTTP 状态码（如
     ``TimeoutError``、``HTTP 500 Server Error``、``Rate Limit Exceeded``）

   聚类按优先级分簇：先归「同一接口」，剩余里同强报错根因（HTTP 4xx/5xx、
   限流、超时、连接错误、异常类名）的归「同一报错关键词」，再归「同一断言」，
   弱关键词随后，最后都不归集的成为单点簇。``status`` 断言的期望值（通常是
   200）太泛，签名会带上实际状态码，避免 404/429/500 被同一条「期望 200」
   错误并簇。每个簇带一个稳定的 ``fingerprint``（与构建无关的特征哈希），
   它是跨构建识别同一失败点的钥匙。

2. **关联缺陷**：跨构建的失败点（:class:`FailureClusterManager` 维护的
   ``failure_points`` 实体）记录自己关联的缺陷。新构建里的失败：

   - 命中未关闭缺陷的失败特征 → 自动归并到该缺陷，不再重复建单；
   - 命中 ``fixed`` / ``verified`` 的缺陷 → 自动 reopen（回归）；
   - 命中 ``closed`` 缺陷或无主失败点且项目开启自动建单 → 新建缺陷。

3. **人工可改、跨构建可见**：簇支持合并、拆分、改关联，调整结果持久化；
   重新聚类时保留所有人工调整过的簇。失败点实体里累积每次构建的出现记录，
   可以直接看到同一失败点在哪些构建里反复出现。
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from typing import Any, Optional

from .models import new_id

FAILED_STATUSES = ("failed", "error", "timeout")

# 每个失败点最多保留的跨构建出现条数（超出丢最早的）
MAX_OCCURRENCES = 50

# ---------------------------------------------------------------------------
# 文本归一化与特征抽取
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d+")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?"
                      r"[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]{12,}$")
# 请求步骤消息形如 "POST /api/users -> 500 (23ms) | error: ..."
_REQUEST_RE = re.compile(r"^([A-Z]+)\s+(\S+)\s+->\s+(\d{3})")
_EXCEPTION_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|TimeoutExpired))\b")
# 错误文案里的短语，如 "rate limit exceeded" / "connection refused"
_PHRASE_RE = re.compile(
    r"([a-z][a-z]+(?:\s+[a-z]+){0,2}?)\s+"
    r"(exceeded|refused|denied|failed|failure|timed\s?out)", re.IGNORECASE)

# 已知错误文案 → (匹配关键词, 标准标签)。标签里的 ``{prefix}`` 会被替换成
# 错误文案里的业务前缀（如 shop / products），使「同业务的一类报错」聚簇，
# 又不至于把毫不相干的所有 500 并到一起。
_KEYWORD_MARKERS = [
    ("gateway timeout", "HTTP 504 Gateway Timeout"),
    ("gateway-timeout", "HTTP 504 Gateway Timeout"),
    ("bad gateway", "HTTP 502 Bad Gateway"),
    ("internal server error", "HTTP 500 Server Error"),
    ("server error", "HTTP 500 Server Error"),
    ("rate limit", "Rate Limit Exceeded"),
    ("connection refused", "Connection Refused"),
    ("not found", "HTTP 404 Not Found"),
    ("unauthorized", "HTTP 401 Unauthorized"),
    ("forbidden", "HTTP 403 Forbidden"),
    ("超时", "Timeout"),
    ("timeout", "Timeout"),
    ("timed out", "Timeout"),
    ("null pointer", "NullPointerException"),
]

# 强根因信号对应的标准标签集合（这些关键词优先于泛化断言成簇）
_STRONG_LABELS = {
    "HTTP 500 Server Error", "HTTP 502 Bad Gateway", "HTTP 504 Gateway Timeout",
    "HTTP 404 Not Found", "HTTP 401 Unauthorized", "HTTP 403 Forbidden",
    "Rate Limit Exceeded", "Connection Refused", "Timeout",
}

_DIM_LABELS = {
    "endpoint": "接口",
    "assertion": "断言",
    "keyword": "报错关键词",
    "single": "单点",
    "custom": "自定义",
}


def normalize_path(url: str) -> str:
    """归一化 URL path：去 query、数字/UUID/长 hex 段替换为 ``{id}``。

    这样 ``/api/users/123/orders`` 与 ``/api/users/456/orders`` 视为同一接口，
    而有业务含义的静态段（``/api/health``）原样保留。
    """
    path = (url or "").split("?")[0].strip()
    if not path:
        return ""
    segs = []
    for seg in path.split("/"):
        if not seg:
            segs.append(seg)
            continue
        if _NUMBER_RE.fullmatch(seg) or _UUID_RE.match(seg) or _HEX_RE.match(seg):
            segs.append("{id}")
        else:
            segs.append(seg)
    return "/".join(segs)


def normalize_expected(value: Any) -> str:
    """断言期望值归一化：数字串压成 ``#``，让「期望 200 / 期望 201」之外的
    同类断言保持稳定，同时屏蔽实际值（实际值噪声大，不作为聚类依据）。"""
    text = str(value).strip()
    return _NUMBER_RE.sub("#", text)


def _request_step(result: dict) -> Optional[dict]:
    """从步骤里找请求步骤，解析 method/path/status（取最后一个请求）。"""
    found = None
    for step in result.get("steps", []):
        if step.get("action") != "request":
            continue
        m = _REQUEST_RE.match(step.get("message", ""))
        if m:
            method, url, status = m.group(1), m.group(2), int(m.group(3))
            found = {"method": method, "path": normalize_path(url),
                    "raw_path": url.split("?")[0], "status": status,
                    "message": step.get("message", "")}
    return found


def _failure_texts(result: dict, req: Optional[dict]) -> list[str]:
    texts = []
    failing_assert = next((a for a in result.get("assertions", [])
                           if not a.get("ok")), None)
    failing_step = next((s for s in result.get("steps", [])
                         if s.get("status") in ("failed", "error")), None)
    if failing_assert:
        texts.append(str(failing_assert.get("message", "")))
    if failing_step:
        texts.append(str(failing_step.get("message", "")))
    if req:
        texts.append(req["message"])
    texts.extend(str(x) for x in result.get("logs", [])[-3:])
    if result.get("message"):
        texts.append(str(result["message"]))
    return [t for t in texts if t]


def _classify_phrase(phrase: str) -> Optional[str]:
    """把错误文案归到标准标签；允许保留业务前缀，如
    ``"shop Rate Limit Exceeded"``。无法识别返回 None。"""
    low = phrase.lower()
    for marker, label in _KEYWORD_MARKERS:
        idx = low.find(marker)
        if idx < 0:
            continue
        prefix = phrase[:idx].strip(" -:|，,")
        # 前缀只保留简单标识符（避免把整句消息带进签名）
        if prefix and re.fullmatch(r"[A-Za-z0-9_./-]+", prefix):
            return f"{prefix} {label}"
        return label
    return None


def extract_keyword(result: dict, req: Optional[dict]) -> Optional[str]:
    """按优先级抽取一个稳定的报错关键词。"""
    texts = _failure_texts(result, req)
    joined = "\n".join(texts)
    low = joined.lower()

    # 1) 超时：结果态 / 文案
    if result.get("status") == "timeout":
        return "Timeout"

    # 2) 显式异常类名（如 AssertionError / KeyError）
    m = _EXCEPTION_RE.search(joined)
    if m:
        return m.group(1).split(".")[-1]

    # 3) 请求/步骤消息里带出的明确错误文案，归到标准标签
    for text in texts:
        m = re.search(r"error:\s*(.+?)\s*$", text.strip(), flags=re.MULTILINE)
        if m:
            phrase = m.group(1).strip()
            if "injected failure" in phrase.lower():
                continue  # 环境注入噪声，无法区分接口
            label = _classify_phrase(phrase)
            if label:
                return label

    # 4) 已知错误文案（直接出现在断言/日志里）
    for marker, label in _KEYWORD_MARKERS:
        if marker in low:
            return label

    # 5) 请求本身非 2xx
    if req and req["status"] >= 400:
        return f"HTTP {req['status']}"

    # 6) 错误文案短语（"... exceeded/refused/failed ..."）
    m = _PHRASE_RE.search(joined)
    if m:
        return " ".join(m.group(0).lower().split())

    return None


def extract_features(result: dict) -> Optional[dict]:
    """从一条失败用例结果抽取聚类特征。非失败结果返回 None。"""
    if result.get("status") not in FAILED_STATUSES:
        return None

    req = _request_step(result)
    failing_assert = next((a for a in result.get("assertions", [])
                           if not a.get("ok")), None)
    failing_step = next((s for s in result.get("steps", [])
                         if s.get("status") in ("failed", "error")), None)

    endpoint_sig = None
    if req:
        endpoint_sig = f"{req['method']} {req['path']} @{req['status']}"

    assert_sig = None
    if failing_assert:
        atype = failing_assert.get("type", "equals")
        assert_sig = f"{atype}~{normalize_expected(failing_assert.get('expected'))}"
        # status 断言的期望值（通常是 200）太泛，实际状态码才是真正的区分点，
        # 否则 404/429/500 会被同一条「期望 200」错误地并成一簇
        if atype == "status":
            assert_sig += f"~actual{normalize_expected(failing_assert.get('actual'))}"

    keyword = extract_keyword(result, req)

    reason = ""
    if failing_assert:
        reason = str(failing_assert.get("message", ""))
    elif failing_step:
        reason = str(failing_step.get("message", ""))
    elif texts := _failure_texts(result, req):
        reason = texts[0]

    return {
        "case_id": result.get("case_id"),
        "case_name": result.get("case_name") or result.get("case_id"),
        "status": result.get("status"),
        "group": result.get("group") or "默认",
        "priority": result.get("priority") or "P3",
        "duration": result.get("duration", 0.0),
        "order": result.get("order", 0),
        "reason": reason,
        "endpoint_sig": endpoint_sig,
        "assert_sig": assert_sig,
        "keyword": keyword,
        "req": req,
        "assert_type": failing_assert.get("type") if failing_assert else None,
        "assert_expected": failing_assert.get("expected") if failing_assert else None,
    }


def fingerprint_for(dim: str, signature: str) -> str:
    """由「维度 + 稳定签名」生成与构建无关的失败点指纹。"""
    digest = hashlib.sha1(f"{dim}|{signature}".encode("utf-8")).hexdigest()[:16]
    return f"fp_{digest}"


# ---------------------------------------------------------------------------
# 聚类
# ---------------------------------------------------------------------------

def _cluster_label(dim: str, feats: list[dict]) -> str:
    rep = feats[0]
    if dim == "endpoint":
        req = rep.get("req") or {}
        return f"{req.get('method', 'HTTP')} {req.get('path') or req.get('raw_path', '')} · HTTP {req.get('status', '?')}"
    if dim == "assertion":
        return f"断言 {rep.get('assert_type') or 'equals'}：期望 {rep.get('assert_expected')!r}"
    if dim == "keyword":
        return f"报错：{rep.get('keyword')}"
    return f"单点：{rep.get('case_name')}"


def _is_strong_keyword(keyword: Optional[str]) -> bool:
    """强关键词：明确的根因信号（HTTP 4xx/5xx、限流、超时、连接类错误、
    异常类名）。这类共性比泛化的「断言不过」更能代表根因，应优先成簇。
    标签可能带业务前缀（如 ``shop Rate Limit Exceeded``），按后缀识别。"""
    if not keyword:
        return False
    if keyword == "Timeout":
        return True
    if keyword in _STRONG_LABELS:
        return True
    if any(keyword.endswith(" " + label) for label in _STRONG_LABELS):
        return True
    head = keyword.split()[0] if keyword else ""
    return bool(head[:1].isupper()) and (head.endswith("Error")
                                         or head.endswith("Exception"))


def cluster_features(features: list[dict]) -> list[dict]:
    """把特征分成若干簇，优先级：接口 → 强关键词 → 断言 → 弱关键词 → 单点。

    每一步只收走「2 条以上同特征」的用例，单条同特征的留给后面的维度，
    避免一个大簇被过早的泛化特征整体吞掉。每个特征只进入一个簇，不重不漏。
    返回 ``{"dim", "signature", "fingerprint", "label", "features"}`` 列表，
    大簇排前面。
    """
    remaining = list(features)
    clusters: list[dict] = []

    def _take(dim: str, key_func, predicate=None):
        nonlocal remaining
        groups: dict[str, list[dict]] = {}
        for f in remaining:
            if predicate is not None and not predicate(f):
                continue
            key = key_func(f)
            if key:
                groups.setdefault(key, []).append(f)
        for sig, feats in groups.items():
            if len(feats) < 2:
                continue
            taken = {id(x) for x in feats}
            remaining = [f for f in remaining if id(f) not in taken]
            clusters.append({
                "dim": dim, "signature": sig,
                "fingerprint": fingerprint_for(dim, sig),
                "label": _cluster_label(dim, feats), "features": feats,
            })

    _take("endpoint", lambda f: f.get("endpoint_sig"))
    _take("keyword", lambda f: f.get("keyword"),
          predicate=lambda f: _is_strong_keyword(f.get("keyword")))
    _take("assertion", lambda f: f.get("assert_sig"))
    _take("keyword", lambda f: f.get("keyword"))

    for f in remaining:
        dim = "keyword" if f.get("keyword") else "single"
        sig = f.get("keyword") or f.get("case_id") or new_id("s")
        clusters.append({
            "dim": dim, "signature": sig,
            "fingerprint": fingerprint_for(dim, sig),
            "label": _cluster_label(dim, [f]), "features": [f],
        })

    clusters.sort(key=lambda c: (-len(c["features"]), c["dim"], c["signature"]))
    return clusters


# ---------------------------------------------------------------------------
# 失败聚类管理器（构建簇 + 跨构建失败点 + 缺陷关联）
# ---------------------------------------------------------------------------

class FailureClusterManager:
    """失败聚类与跨构建失败点追踪。"""

    def __init__(self, registry, build_registry, defect_manager):
        self.registry = registry
        self.builds = build_registry
        self.defects = defect_manager
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # -- 内部工具 ---------------------------------------------------------
    def _lock_for(self, build_id: str) -> threading.Lock:
        with self._locks_guard:
            lk = self._locks.get(build_id)
            if lk is None:
                lk = threading.Lock()
                self._locks[build_id] = lk
            return lk

    def _fp_store(self):
        return self.registry.store("failure_points")

    def _store_for(self, project_id: str):
        return self.builds.for_project(project_id)

    @staticmethod
    def _member(f: dict) -> dict:
        return {
            "case_id": f.get("case_id"),
            "case_name": f.get("case_name"),
            "status": f.get("status"),
            "group": f.get("group"),
            "priority": f.get("priority"),
            "duration": f.get("duration"),
            "order": f.get("order"),
            "reason": f.get("reason", ""),
        }

    def _cluster_doc(self, project_id: str, build_id: str, c: dict,
                     auto: bool = True) -> dict:
        now = time.time()
        return {
            "id": new_id("cl"),
            "project_id": project_id,
            "build_id": build_id,
            "dim": c["dim"],
            "dim_label": _DIM_LABELS.get(c["dim"], c["dim"]),
            "label": c["label"],
            "signature": c["signature"],
            "fingerprint": c["fingerprint"],
            "members": [self._member(f) for f in c["features"]],
            "defect_id": None,
            "defect_source": None,  # auto / manual
            "auto": auto,
            "created_at": now,
            "updated_at": now,
        }

    # -- 聚类入口 ---------------------------------------------------------
    def get_or_analyze(self, project_id: str, build_id: str,
                       force: bool = False,
                       create_defects: Optional[bool] = None) -> dict:
        """返回构建的聚类结果。

        - 已聚过且不强制：直接返回缓存（含人工调整）；
        - 构建已结束：即时执行一次聚类并持久化；
        - 构建仍在运行：只返回已有缓存（可能为空），避免运行中途产生
          只覆盖部分失败、且随后被报告反复触发的半成品。
        """
        store = self._store_for(project_id)
        existing = store.read_clusters(build_id)
        build = store.get(build_id)
        finished = bool(build and build.get("status") not in ("pending", "running"))
        if existing is not None and not force:
            return self._enrich(project_id, existing)
        if not finished:
            return self._enrich(project_id,
                                existing or {"build_id": build_id, "clusters": []})
        return self.analyze_build(project_id, build_id,
                                  create_defects=create_defects)

    def analyze_build(self, project_id: str, build_id: str,
                      create_defects: Optional[bool] = None) -> dict:
        """对一场构建执行完整聚类 + 失败点归并 + 缺陷关联。"""
        with self._lock_for(build_id):
            store = self._store_for(project_id)
            build = store.get(build_id)
            if build is None:
                return {"error": "构建不存在"}
            if create_defects is None:
                project = self.registry.store("projects").get(project_id)
                create_defects = bool(project and project.get("auto_create_defects"))

            results = store.results(
                build_id, where=[("status", "in", list(FAILED_STATUSES))],
                order_by="order", order="asc")
            features = [f for f in (extract_features(r) for r in results) if f]

            # 保留人工调整过的簇（其成员不参与自动重聚）
            prev = store.read_clusters(build_id) or {"clusters": []}
            kept = [c for c in prev.get("clusters", []) if not c.get("auto", True)]
            kept_case_ids = {m.get("case_id") for c in kept for m in c.get("members", [])}
            auto_features = [f for f in features if f.get("case_id") not in kept_case_ids]

            auto_docs = [self._cluster_doc(project_id, build_id, c, auto=True)
                         for c in cluster_features(auto_features)]
            clusters = kept + auto_docs

            # 跨构建失败点归并 + 缺陷关联
            for c in clusters:
                self._register_cluster(c, build, create_defects=create_defects and c.get("auto", True))

            payload = {
                "build_id": build_id,
                "project_id": project_id,
                    "build_status": build.get("status"),
                "cluster_count": len(clusters),
                "failure_count": len(features),
                "generated_at": time.time(),
                "clusters": clusters,
            }
            store.write_clusters(build_id, payload)
            return self._enrich(project_id, payload)

    # -- 跨构建失败点 -----------------------------------------------------
    def _register_cluster(self, cluster: dict, build: dict,
                          create_defects: bool) -> dict:
        """把一个簇登记到跨构建失败点，并完成缺陷匹配 / 归并 / 建单。"""
        fp_id = cluster["fingerprint"]
        store = self._fp_store()
        point = store.get(fp_id)
        now = time.time()
        build_id = build["id"]
        count = len(cluster.get("members", []))
        occurrence = {
            "build_id": build_id,
            "build_name": build.get("name") or build_id,
            "finished_at": build.get("finished_at") or now,
            "cluster_id": cluster.get("id"),
            "count": count,
        }
        if point is None:
            point = {
                "id": fp_id,
                "project_id": cluster["project_id"],
                "fingerprint": fp_id,
                "dim": cluster["dim"],
                "dim_label": cluster.get("dim_label") or _DIM_LABELS.get(cluster["dim"]),
                "label": cluster["label"],
                "signature": cluster["signature"],
                "defect_id": None,
                "occurrences": [occurrence],
                "build_count": 1,
                "first_seen": now,
                "last_seen": now,
                "status": "active",
                "created_at": now,
            }
            store.insert(point)
        else:
            occs = point.get("occurrences", [])
            for occ in occs:
                if occ.get("build_id") == build_id:
                    occ.update(occurrence)
                    break
            else:
                occs.append(occurrence)
            occs.sort(key=lambda o: o.get("finished_at", 0))
            point["occurrences"] = occs[-MAX_OCCURRENCES:]
            point["build_count"] = len({o.get("build_id") for o in point["occurrences"]})
            point["last_seen"] = now
            point["label"] = cluster["label"]
            point["status"] = "active"
            store.update(fp_id, point)

        # 缺陷关联
        defect_id = self._match_defect(point, cluster, build, create_defects)
        if defect_id:
            point["defect_id"] = defect_id
            store.update(fp_id, {"defect_id": defect_id, "last_seen": now})
            cluster["defect_id"] = defect_id
            cluster["defect_source"] = cluster.get("defect_source") or "auto"
            # 每构建每簇累计一次命中（用于缺陷页展示该特征撞了多少次）
            defect = self.defects.get(defect_id)
            if defect is not None:
                self.defects.update(defect_id,
                                    {"hit_count": int(defect.get("hit_count", 0)) + 1})
        return point

    def _match_defect(self, point: dict, cluster: dict, build: dict,
                      create_defects: bool) -> Optional[str]:
        """为失败点找最可能的缺陷：已绑定 → 同指纹未关闭缺陷 → （可选）新建。"""
        pid = cluster["project_id"]
        existing_id = point.get("defect_id")
        if existing_id:
            defect = self.defects.get(existing_id)
            if defect and defect.get("status") != "closed":
                # fixed/verified 又撞上同一失败特征 → 回归，自动重开
                if defect.get("status") in ("fixed", "verified"):
                    self.defects.update(existing_id, {"status": "reopened"})
                return existing_id
            # 缺陷已关闭 / 被删除：落入下面的重新匹配 / 建单流程

        for defect in self.defects.list(pid):
            if defect.get("status") == "closed":
                continue
            if point["fingerprint"] in (defect.get("fingerprints") or []):
                if defect.get("status") in ("fixed", "verified"):
                    self.defects.update(defect["id"], {"status": "reopened"})
                return defect["id"]

        if create_defects:
            members = cluster.get("members", [])
            first = members[0] if members else {}
            severity = "major" if first.get("priority") in ("P0", "P1") else "minor"
            desc_lines = [
                f"失败特征（{cluster.get('dim_label') or cluster['dim']}）：{cluster['signature']}",
                f"首次构建：{build.get('name') or build['id']}（{build['id']}）",
                f"本构建命中用例 {len(members)} 个：",
            ]
            desc_lines += [f"- {m.get('case_name')}：{m.get('reason', '')}"
                           for m in members[:10]]
            defect = self.defects.create(pid, {
                "title": f"[聚类] {cluster['label']}",
                "description": "\n".join(desc_lines),
                "severity": severity,
                "source_build_id": build["id"],
                "source_case_id": first.get("case_id"),
                "fingerprints": [point["fingerprint"]],
            })
            return defect["id"]
        return None

    # -- 人工调整：合并 / 拆分 / 改关联 -----------------------------------
    def get_clusters(self, project_id: str, build_id: str) -> Optional[dict]:
        data = self._store_for(project_id).read_clusters(build_id)
        return self._enrich(project_id, data) if data else None

    def _save(self, project_id: str, build_id: str, clusters: list[dict]) -> dict:
        store = self._store_for(project_id)
        prev = store.read_clusters(build_id) or {}
        prev.update({
            "build_id": build_id, "project_id": project_id,
            "cluster_count": len(clusters),
            "generated_at": time.time(),
            "clusters": clusters,
        })
        store.write_clusters(build_id, prev)
        return self._enrich(project_id, prev)

    def merge(self, project_id: str, build_id: str,
              target_id: str, source_id: str) -> dict:
        """把 source 簇并入 target 簇（跨构建失败点也随之合并）。"""
        with self._lock_for(build_id):
            data = self.get_clusters(project_id, build_id)
            if data is None:
                return {"error": "该构建还没有聚类结果"}
            clusters = data["clusters"]
            target = next((c for c in clusters if c["id"] == target_id), None)
            source = next((c for c in clusters if c["id"] == source_id), None)
            if target is None or source is None:
                return {"error": "簇不存在"}
            if target_id == source_id:
                return {"error": "不能与自身合并"}

            existing_ids = {m.get("case_id") for m in target.get("members", [])}
            for m in source.get("members", []):
                if m.get("case_id") not in existing_ids:
                    target.setdefault("members", []).append(m)
            target["members"].sort(key=lambda m: m.get("order", 0))
            target["auto"] = False
            target["dim"] = "custom"
            target["dim_label"] = _DIM_LABELS["custom"]
            target["label"] = f"{target['label']} + {source['label']}"[:200]
            target["updated_at"] = time.time()

            self._merge_failure_points(project_id, target["fingerprint"],
                                       source["fingerprint"],
                                       defect_id=target.get("defect_id"))
            clusters = [c for c in clusters if c["id"] != source_id]
            return self._save(project_id, build_id, clusters)

    def _merge_failure_points(self, project_id: str, target_fp: str,
                              source_fp: str, defect_id: Optional[str]) -> None:
        if target_fp == source_fp:
            return
        store = self._fp_store()
        target = store.get(target_fp)
        source = store.get(source_fp)
        if target is None:
            return
        if source is not None:
            seen = {(o.get("build_id"), o.get("cluster_id")) for o in target.get("occurrences", [])}
            merged = list(target.get("occurrences", []))
            for occ in source.get("occurrences", []):
                key = (occ.get("build_id"), occ.get("cluster_id"))
                if key not in seen:
                    merged.append(occ)
                    seen.add(key)
            merged.sort(key=lambda o: o.get("finished_at", 0))
            target["occurrences"] = merged[-MAX_OCCURRENCES:]
            target["build_count"] = len({o.get("build_id") for o in target["occurrences"]})
            store.delete(source_fp)
        patch = {"occurrences": target["occurrences"], "build_count": target["build_count"]}
        if defect_id:
            patch["defect_id"] = defect_id
        store.update(target_fp, patch)
        if defect_id and source is not None:
            self._bind_fingerprint(defect_id, source_fp)
            self._bind_fingerprint(defect_id, target_fp)

    def split(self, project_id: str, build_id: str, cluster_id: str,
              case_ids: list[str]) -> dict:
        """把指定用例从簇中拆出，成立一个新的人工簇。"""
        with self._lock_for(build_id):
            data = self.get_clusters(project_id, build_id)
            if data is None:
                return {"error": "该构建还没有聚类结果"}
            clusters = data["clusters"]
            src = next((c for c in clusters if c["id"] == cluster_id), None)
            if src is None:
                return {"error": "簇不存在"}
            wanted = set(case_ids)
            moved = [m for m in src.get("members", []) if m.get("case_id") in wanted]
            if not moved:
                return {"error": "请选择要拆出的用例"}
            kept = [m for m in src.get("members", []) if m.get("case_id") not in wanted]
            if not kept:
                return {"error": "不能把簇全部拆空，请改用合并或删除"}

            src["members"] = kept
            src["auto"] = False
            src["updated_at"] = time.time()

            new_c = self._custom_cluster(project_id, build_id, moved, src)
            # 新失败点只做匹配，不自动建单（人工操作不制造意外缺陷）
            store = self._store_for(project_id)
            build = store.get(build_id)
            self._register_cluster(new_c, build, create_defects=False)
            clusters.append(new_c)
            return self._save(project_id, build_id, clusters)

    def _custom_cluster(self, project_id: str, build_id: str,
                        members: list[dict], ref: dict) -> dict:
        # 拆出的成员若仍共享同一接口/断言/关键词特征，沿用该特征的指纹
        sigs: dict[str, set] = {"endpoint": set(), "assertion": set(), "keyword": set()}
        for m in members:
            f = extract_features(self._member_to_result(m))
            if not f:
                continue
            if f.get("endpoint_sig"):
                sigs["endpoint"].add(f["endpoint_sig"])
            if f.get("assert_sig"):
                sigs["assertion"].add(f["assert_sig"])
            if f.get("keyword"):
                sigs["keyword"].add(f["keyword"])
        now = time.time()
        if len(sigs["endpoint"]) == 1:
            dim, sig = "endpoint", next(iter(sigs["endpoint"]))
            label = sig
        elif len(sigs["assertion"]) == 1:
            dim, sig = "assertion", next(iter(sigs["assertion"]))
            label = f"断言 {sig}"
        elif len(sigs["keyword"]) == 1:
            dim, sig = "keyword", next(iter(sigs["keyword"]))
            label = f"报错：{sig}"
        else:
            dim = "custom"
            sig = "custom~" + hashlib.sha1(
                ",".join(sorted(m.get("case_id") or "" for m in members)).encode()
            ).hexdigest()[:12]
            label = f"拆分自：{ref.get('label', '')[:80]}"
        return {
            "id": new_id("cl"),
            "project_id": project_id,
            "build_id": build_id,
            "dim": dim,
            "dim_label": _DIM_LABELS.get(dim, dim),
            "label": label,
            "signature": sig,
            "fingerprint": fingerprint_for(dim, sig),
            "members": members,
            "defect_id": None,
            "defect_source": None,
            "auto": False,
            "created_at": now,
            "updated_at": now,
        }

    @staticmethod
    def _member_to_result(m: dict) -> dict:
        """把簇成员还原成 extract_features 可消费的最小结果结构。"""
        reason = m.get("reason") or ""
        step = {"action": "request", "status": "failed", "message": reason} \
            if reason else None
        return {
            "case_id": m.get("case_id"), "case_name": m.get("case_name"),
            "status": m.get("status"), "group": m.get("group"),
            "priority": m.get("priority"), "duration": m.get("duration"),
            "order": m.get("order"),
            "steps": [step] if step else [],
            "assertions": [], "logs": [],
        }

    def set_defect(self, project_id: str, build_id: str, cluster_id: str,
                   defect_id: Optional[str]) -> dict:
        """修改簇关联的缺陷；``None`` 解除关联，``"new"`` 立即建单。"""
        with self._lock_for(build_id):
            data = self.get_clusters(project_id, build_id)
            if data is None:
                return {"error": "该构建还没有聚类结果"}
            cluster = next((c for c in data["clusters"] if c["id"] == cluster_id), None)
            if cluster is None:
                return {"error": "簇不存在"}

            if defect_id == "new":
                store = self._store_for(project_id)
                build = store.get(build_id)
                members = cluster.get("members", [])
                first = members[0] if members else {}
                defect = self.defects.create(project_id, {
                    "title": cluster["label"],
                    "description": "人工为失败簇创建的缺陷。\n"
                                   f"失败特征：{cluster['signature']}\n"
                                   f"来源构建：{build_id}",
                    "severity": "major" if first.get("priority") in ("P0", "P1") else "minor",
                    "source_build_id": build_id,
                    "source_case_id": first.get("case_id"),
                    "fingerprints": [cluster["fingerprint"]],
                })
                defect_id = defect["id"]
            elif defect_id:
                defect = self.defects.get(defect_id)
                if defect is None or defect.get("project_id") != project_id:
                    return {"error": "缺陷不存在"}

            cluster["defect_id"] = defect_id
            cluster["defect_source"] = "manual"
            cluster["auto"] = False
            cluster["updated_at"] = time.time()

            point = self._fp_store().get(cluster["fingerprint"])
            if point is not None:
                self._fp_store().update(cluster["fingerprint"], {"defect_id": defect_id})
            if defect_id:
                self._bind_fingerprint(defect_id, cluster["fingerprint"])
            return self._save(project_id, build_id, data["clusters"])

    def _bind_fingerprint(self, defect_id: str, fingerprint: str) -> None:
        defect = self.defects.get(defect_id)
        if defect is None:
            return
        fps = defect.get("fingerprints") or []
        if fingerprint not in fps:
            fps.append(fingerprint)
            self.defects.update(defect_id, {"fingerprints": fps})

    # -- 查询：簇富化 / 跨构建失败点 --------------------------------------
    def _enrich(self, project_id: str, payload: dict) -> dict:
        if not payload:
            return payload
        defects_cache: dict[str, Optional[dict]] = {}

        def _defect(did):
            if did not in defects_cache:
                defects_cache[did] = self.defects.get(did) if did else None
            return defects_cache[did]

        for c in payload.get("clusters", []):
            d = _defect(c.get("defect_id"))
            c["defect"] = {"id": d["id"], "title": d["title"],
                           "status": d["status"], "severity": d["severity"]} if d else None
            c["member_count"] = len(c.get("members", []))
        return payload

    def failure_points(self, project_id: str,
                       status: Optional[str] = None) -> list[dict]:
        """跨构建失败点列表（附出现历史与关联缺陷摘要），最近出现的排前。"""
        points = self._fp_store().query(
            where=[("project_id", "eq", project_id)],
            order_by="last_seen", order="desc")
        out = []
        for p in points:
            defect = self.defects.get(p.get("defect_id")) if p.get("defect_id") else None
            # 已关单且最近没有再撞的失败点视为已解决
            pstatus = p.get("status", "active")
            if defect and defect.get("status") == "closed":
                pstatus = "resolved"
            if status and pstatus != status:
                continue
            occs = sorted(p.get("occurrences", []),
                          key=lambda o: o.get("finished_at", 0), reverse=True)
            recent_states = [o.get("build_id") for o in occs[:3]]
            out.append({
                "fingerprint": p["fingerprint"],
                "dim": p["dim"],
                "dim_label": p.get("dim_label") or _DIM_LABELS.get(p["dim"]),
                "label": p["label"],
                "signature": p["signature"],
                "status": pstatus,
                "build_count": p.get("build_count", len(occs)),
                "first_seen": p.get("first_seen"),
                "last_seen": p.get("last_seen"),
                "recurring": p.get("build_count", len(occs)) >= 2,
                "defect": {"id": defect["id"], "title": defect["title"],
                           "status": defect["status"],
                           "severity": defect["severity"]} if defect else None,
                "occurrences": occs,
                "recent_build_ids": recent_states,
            })
        return out

    def failure_point(self, fingerprint: str) -> Optional[dict]:
        p = self._fp_store().get(fingerprint)
        if p is None:
            return None
        for item in self.failure_points(p["project_id"]):
            if item["fingerprint"] == fingerprint:
                return item
        return None
