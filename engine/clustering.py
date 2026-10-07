"""失败特征抽取与失败聚类（纯函数部分）。

同一次构建里几十条失败用例，逐条看报错成本很高。观察失败用例的结构化
结果（步骤 + 断言 + 日志），失败原因基本落在三个维度上：

- **同一断言**：断言类型 + 期望值相同，例如 10 个用例都在
  ``status == 200`` 上挂掉，往往是同一个接口集体 500；
- **同一接口**：请求方法 + 路径（归一化后）+ HTTP 状态码相同，
  例如都打在 ``POST /api/orders`` 的 500 上；
- **同一报错关键词**：异常类型或归一化后的报错文案相同，
  例如一批 ``ConnectionError`` / ``TimeoutError``。

这里只做**确定性的特征工程**，不引入任何外部依赖：从一条失败结果里
抽出若干「签名」，再按层优先级选出主签名作为聚类键。签名刻意设计成
跨构建稳定（去掉时间戳、自增 id、引号里的动态值），这样同一下游故障
在不同构建里产生的签名一致，才能做跨构建归并与缺陷自动关联。

管理（持久化 / 人工调整 / 缺陷关联）在 :mod:`engine.clusters`。
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

FAILED_STATUSES = ("failed", "error", "timeout")

# 维度标识，同时用于簇的 dim 字段与失败点统计
DIM_ASSERTION = "assertion"
DIM_ENDPOINT = "endpoint"
DIM_KEYWORD = "keyword"
DIM_STATUS = "status"
DIM_MERGED = "merged"  # 人工把不同签名的簇合并到一起

# 请求步骤 message 形如："GET /api/orders?x=1 -> 500 (23.4ms)"
_REQUEST_MSG_RE = re.compile(r"^([A-Z]+)\s+(\S+)\s+->\s+(\d{3})\b")

# 异常类名：ValueError / ConnectionError / ExecutionError / TimeoutError ...
_EXC_RE = re.compile(r"\b([A-Za-z_]\w*(?:Error|Exception|Fault|Timeout))\b")

_NUMERIC_ID_RE = re.compile(r"/\d+(?=/|$)")
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                      r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_QUOTED_RE = re.compile(r"(['\"])(?:\\.|(?!\1).)*\1")
_WS_RE = re.compile(r"\s+")
_LONG_NUM_RE = re.compile(r"\b\d{3,}\b")

# 中文断言文案里的动态尾巴："，实际 xxx"
_ACTUAL_TAIL_RE = re.compile(r"[，,]\s*实际.*$")

ASSERTION_DIM_LABELS = {
    "equals": "相等断言", "not_equals": "不等断言", "contains": "包含断言",
    "regex": "正则断言", "gt": "大于断言", "gte": "大于等于断言",
    "lt": "小于断言", "lte": "小于等于断言", "between": "区间断言",
    "in": "属于断言", "status": "状态码断言", "truthy": "真值断言",
    "json_path": "JSON路径断言",
}


# ---------------------------------------------------------------------------
# 归一化：把易变的报错文本变成稳定签名
# ---------------------------------------------------------------------------

def normalize_path(path: str) -> str:
    """归一化 URL 路径：去查询串、把数字 id / uuid 替换成占位符。"""
    path = path.split("?", 1)[0]
    path = _UUID_RE.sub("/{id}", path)
    path = _NUMERIC_ID_RE.sub("/{id}", path)
    return path.rstrip("/") or "/"


def normalize_expected(value) -> str:
    """归一化断言期望值：去空白、限制长度，使签名稳定。"""
    text = str(value)
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > 80:
        # 超长期望值取哈希，避免签名膨胀，同时保持稳定
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        return f"len={len(text)}:{digest}"
    return text


def normalize_message(message: str) -> str:
    """把一条报错文案归一化成稳定的关键词。

    去掉引号内的动态值、内存地址、长数字、时间戳尾巴，折叠空白，截断。
    """
    text = message or ""
    text = _ACTUAL_TAIL_RE.sub("", text)
    text = _HEX_RE.sub("0x…", text)
    text = _UUID_RE.sub("<uuid>", text)
    text = _QUOTED_RE.sub("<…>", text)
    text = _LONG_NUM_RE.sub("<n>", text)
    text = _WS_RE.sub(" ", text).strip()
    return text[:60] or "未知错误"


def _hash_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# 单条失败结果的特征
# ---------------------------------------------------------------------------

def failure_reason(result: dict) -> str:
    """提取失败结果最有代表性的报错文案（与报告页口径一致）。"""
    failing_assert = next((a for a in result.get("assertions", [])
                           if not a.get("ok")), None)
    if failing_assert:
        msg = failing_assert.get("message", "")
        actual = failing_assert.get("actual")
        if actual is not None:
            return f"{msg}，实际 {actual!r}"
        return msg
    failing_step = next((s for s in result.get("steps", [])
                         if s.get("status") in ("failed", "error")), None)
    if failing_step:
        return failing_step.get("message", "")
    return result.get("message", "") or ""


def _assertion_key(failing: dict) -> tuple[str, str]:
    """非状态码失败断言的 (签名键, 展示标签)。"""
    atype = failing.get("type", "equals")
    expected_norm = normalize_expected(failing.get("expected"))
    key = f"{DIM_ASSERTION}|{atype}|{expected_norm}"
    label = (f"{ASSERTION_DIM_LABELS.get(atype, atype)}失败："
             f"期望 {str(failing.get('expected'))[:40]}")
    return key, label


def _request_step(result: dict) -> Optional[tuple[str, str, Optional[int]]]:
    """取失败前最后一个请求步骤的 (method, raw_path, code)。

    本平台执行器的语义是：请求步骤本身恒为 passed，HTTP 失败由后续
    针对 ``${resp.status}`` 的断言表达。因此：
    - 请求步骤 message 里能解析出状态码（``"POST /x -> 500 ..."``）时直接用；
    - 否则状态码交给失败断言的 actual 补（见 :func:`_endpoint_feature`）。
    """
    last = None
    for step in result.get("steps", []):
        if step.get("action") == "request":
            last = step
    if not last:
        return None
    m = _REQUEST_MSG_RE.match((last.get("message") or "").strip())
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    # message 未带状态码时，尝试从步骤里记录的 url（部分结果不带 method）
    url = last.get("url") or ""
    method = (last.get("method") or "GET")
    return method, url, None


def _endpoint_feature(result: dict, failing_assert: Optional[dict] = None) -> Optional[dict]:
    """接口失败特征：方法 + 归一化路径 + HTTP 状态码。

    状态码来源有两个：失败的状态码断言的实际值（``${resp.status}``），
    或请求步骤 message 自带的码；仅 4xx/5xx 才算接口致因。
    """
    parsed = _request_step(result)
    if not parsed:
        return None
    method, raw_path, code = parsed

    if failing_assert and failing_assert.get("type") == "status":
        try:
            code = int(failing_assert.get("actual"))
        except (TypeError, ValueError):
            pass

    if not raw_path:
        return None
    if code is None or code < 400:
        return None
    path = normalize_path(raw_path)
    key = f"{DIM_ENDPOINT}|{method}|{path}|{code}"
    label = f"接口异常：{method} {path} → {code}"
    return {"dim": DIM_ENDPOINT, "key": key, "label": label,
            "method": method, "path": path, "status_code": code}


def _keyword_feature(result: dict, reason: str) -> Optional[dict]:
    text = reason or ""
    m = _EXC_RE.search(text)
    if m:
        keyword = m.group(1)
        label = f"报错 {keyword}"
    else:
        keyword = normalize_message(text)
        label = keyword
    key = f"{DIM_KEYWORD}|{keyword}"
    return {"dim": DIM_KEYWORD, "key": key, "label": label, "keyword": keyword}


def failure_features(result: dict) -> dict:
    """抽取一条失败结果的全部特征，并按层优先级选出主签名。

    优先级：超时/取消等状态 > 接口（含对 ``resp.status`` 的失败断言）
    > 同一断言 > 同一报错关键词。接口最能圈定故障面（下游挂一片时
    最先看到的就是同一个 500），其它业务断言失败则归断言层；
    关键词兜底连断言都没跑出来的环境类 / 框架类错误。
    """
    reason = failure_reason(result)
    failing_assert = next((a for a in result.get("assertions", [])
                           if not a.get("ok")), None)
    features = {
        "case_id": result.get("case_id"),
        "case_name": result.get("case_name"),
        "status": result.get("status"),
        "priority": result.get("priority"),
        "group": result.get("group"),
        "reason": reason,
        "assertion": None,
        "endpoint": _endpoint_feature(result, failing_assert),
        "keyword": _keyword_feature(result, reason),
    }
    # 非状态码的失败断言才作为「同断言」特征；状态码断言已归入接口层
    if failing_assert and failing_assert.get("type") != "status":
        key, label = _assertion_key(failing_assert)
        features["assertion"] = {"dim": DIM_ASSERTION, "key": key,
                                 "label": label,
                                 "atype": failing_assert.get("type", "equals"),
                                 "expected": failing_assert.get("expected")}

    primary: dict
    if result.get("status") == "timeout":
        primary = {"dim": DIM_STATUS, "key": f"{DIM_STATUS}|timeout",
                   "label": "用例执行超时"}
    elif features["endpoint"]:
        primary = features["endpoint"]
    elif features["assertion"]:
        primary = features["assertion"]
    else:
        primary = features["keyword"] or {"dim": DIM_KEYWORD,
                                          "key": f"{DIM_KEYWORD}|unknown",
                                          "label": "未知错误"}
    features["primary"] = {"dim": primary["dim"], "key": primary["key"],
                           "label": primary["label"]}
    return features


def group_id_for(key: str) -> str:
    """根据签名生成确定性的簇 id（同构建 / 跨构建都稳定）。"""
    return f"clu_{_hash_key(key)}"


def cluster_results(results: list[dict]) -> list[dict]:
    """把一批失败用例结果按主签名聚成簇，按规模倒序返回。

    每簇结构::

        {group_id, dim, signature, title, reason, count, case_ids, cases,
         members: [features...]}
    """
    buckets: dict[str, list[dict]] = {}
    for result in results:
        if result.get("status") not in FAILED_STATUSES:
            continue
        features = failure_features(result)
        buckets.setdefault(features["primary"]["key"], []).append(features)

    clusters = []
    for key, members in buckets.items():
        dims = {m["primary"]["dim"] for m in members}
        dim = next(iter(dims)) if len(dims) == 1 else DIM_KEYWORD
        # 标题取该签名下出现最多的主标签，避免被个别文案带偏
        labels: dict[str, int] = {}
        for m in members:
            labels[m["primary"]["label"]] = labels.get(m["primary"]["label"], 0) + 1
        title = sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        reason = members[0]["reason"]
        cases = [{"case_id": m["case_id"], "case_name": m["case_name"],
                  "status": m["status"], "priority": m["priority"],
                  "reason": m["reason"]} for m in members]
        clusters.append({
            "group_id": group_id_for(key),
            "dim": dim,
            "signature": key,
            "title": title,
            "reason": reason,
            "count": len(members),
            "case_ids": [m["case_id"] for m in members],
            "cases": cases,
            "auto": True,
        })
    clusters.sort(key=lambda c: (-c["count"], c["signature"]))
    return clusters
