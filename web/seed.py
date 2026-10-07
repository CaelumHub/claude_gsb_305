"""演示 / 初始数据生成。

应用启动时，若数据目录里还没有任何项目，会自动调用 :func:`seed_demo_data`
生成一份演示数据（项目 + 用例 + 套件 + 环境 + 计划 + 集成），让各个页面
一打开就有内容可点、可测。HTTP 接口 ``POST /api/seed/demo`` 也复用这里，
供前端「生成演示项目」按钮调用。
"""

from __future__ import annotations

import time

from engine import new_id


def _status_200_step(name="期望 200"):
    return {"action": "assert", "type": "status", "actual": "${resp.status}",
            "expected": 200, "name": name}


# 失败聚类演示用例：在零失败率的 dev 环境也稳定失败，分别形成三类簇——
# 同一接口 404 ×2（接口簇）、同一断言 ×2（断言簇）、同一限流关键词 ×3（关键词簇）
CLUSTER_DEMO_CASES = [
    {"name": "用户详情-不存在的用户", "priority": "P1", "tags": ["api", "users"],
     "steps": [{"action": "request", "method": "GET", "url": "/api/users/1001",
                "name": "查询不存在用户"}, _status_200_step()]},
    {"name": "用户详情-另一个不存在的用户", "priority": "P2", "tags": ["api", "users"],
     "steps": [{"action": "request", "method": "GET", "url": "/api/users/2002",
                "name": "查询不存在用户"}, _status_200_step()]},
    {"name": "算术校验 A", "priority": "P2", "tags": ["unit"],
     "steps": [{"action": "script", "expr": "2 + 3 * 4", "save_as": "result",
                "name": "算术"},
               {"action": "assert", "type": "equals", "actual": "${result}",
                "expected": 15, "name": "结果等于 15"}]},
    {"name": "算术校验 B", "priority": "P3", "tags": ["unit"],
     "steps": [{"action": "script", "expr": "2 + 3 * 4", "save_as": "r2",
                "name": "算术"},
               {"action": "assert", "type": "equals", "actual": "${r2}",
                "expected": 15, "name": "结果等于 15"}]},
]
for _cname, _cprio, _ctags, _url in [
    ("限流-查询商品", "P1", ["api", "products"], "/api/shop/products/ratelimit"),
    ("限流-查询库存", "P2", ["api", "stock"], "/api/shop/stock/ratelimit"),
    ("限流-查询评论", "P3", ["api", "comments"], "/api/shop/comments/ratelimit"),
]:
    CLUSTER_DEMO_CASES.append({
        "name": _cname, "priority": _cprio, "tags": _ctags,
        "steps": [{"action": "request", "method": "GET", "url": _url,
                   "name": "请求被限流的接口"}, _status_200_step()],
    })


def ensure_cluster_demo_cases(registry) -> int:
    """把失败聚类演示用例幂等补进旧版演示数据的冒烟套件。

    已存在同名用例则跳过；返回新补入的用例数。仓库自带的旧 ``data`` 目录
    （8 用例版套件）启动时会被补成 15 用例版，失败聚类页一打开就有内容。
    """
    projects = registry.store("projects").all()
    added = 0
    for project in projects:
        pid = project["id"]
        if "演示项目" not in project.get("name", ""):
            continue
        cases_store = registry.store("cases")
        existing_names = {c.get("name") for c in
                          cases_store.query(where=[("project_id", "eq", pid)])}
        new_ids = []
        for spec in CLUSTER_DEMO_CASES:
            if spec["name"] in existing_names:
                continue
            new_ids.append(cases_store.insert({
                "id": new_id("case"), "project_id": pid,
                "name": spec["name"], "description": "演示用例",
                "priority": spec["priority"], "tags": spec["tags"],
                "timeout": 60, "enabled": True, "steps": spec["steps"],
                "created_at": time.time(),
            }))
            added += 1
        if new_ids:
            suites_store = registry.store("suites")
            for suite in suites_store.query(where=[("project_id", "eq", pid)]):
                cids = suite.get("case_ids") or []
                if cids and suite.get("name") == "冒烟测试套件":
                    suites_store.update(suite["id"], {"case_ids": cids + new_ids})
    return added


def seed_demo_data(registry, env_mgr, notify_mgr) -> dict:
    """生成演示项目，返回 ``{"project": ..., "env_id": ..., "suite_id": ...}``。"""
    proj = {
        "id": new_id("proj"),
        "name": "演示项目 · 测试与CI",
        "description": "内置示例用例、套件、环境与通知集成的演示项目。",
        "repo_url": "https://example.com/demo",
        "auto_create_defects": True,
        "created_at": time.time(),
    }
    registry.store("projects").insert(proj)
    pid = proj["id"]

    env = env_mgr.create(pid, {
        "name": "dev 开发环境",
        "python_version": "3.11",
        "base_image": "python:3.11-slim",
        "variables": {"BASE_URL": "http://dev.mock.local", "REGION": "dev"},
        "config": {"base_url": "http://dev.mock.local", "latency_ms": 15, "fail_rate": 0.0},
        "dependencies": [
            {"name": "requests", "constraint": ">=2.28"},
            {"name": "pytest", "constraint": ">=7.0"},
            {"name": "flask", "constraint": ">=3.0"},
        ],
    })
    env2 = env_mgr.create(pid, {
        "name": "staging 预发环境",
        "python_version": "3.12",
        "base_image": "python:3.12-slim",
        "variables": {"BASE_URL": "http://staging.mock.local", "REGION": "staging"},
        "config": {"base_url": "http://staging.mock.local", "latency_ms": 45, "fail_rate": 0.15},
        "dependencies": [
            {"name": "requests", "constraint": ">=2.30"},
            {"name": "django", "constraint": ">=4.2"},
            {"name": "numpy", "constraint": ">=1.24"},
        ],
    })

    def _case(name, priority, tags, steps):
        return registry.store("cases").insert({
            "id": new_id("case"),
            "project_id": pid,
            "name": name,
            "description": "演示用例",
            "priority": priority,
            "tags": tags,
            "timeout": 60,
            "enabled": True,
            "steps": steps,
            "created_at": time.time(),
        })

    c1 = _case("健康检查接口", "P0", ["smoke", "api"], [
        {"action": "request", "method": "GET", "url": "/api/health", "name": "请求健康检查"},
        {"action": "assert", "type": "status", "actual": "${resp.status}", "expected": 200, "name": "状态码 200"},
        {"action": "assert", "type": "truthy", "actual": "${resp.body.ok}", "expected": True, "name": "返回 ok"},
    ])
    c2 = _case("登录接口", "P0", ["smoke", "auth"], [
        {"action": "set", "key": "user", "value": "admin", "name": "准备用户名"},
        {"action": "request", "method": "POST", "url": "/api/login", "name": "请求登录"},
        {"action": "assert", "type": "status", "actual": "${resp.status}", "expected": 200, "name": "登录成功"},
        {"action": "assert", "type": "contains", "actual": "${resp.body}", "expected": "ok", "name": "返回体含 ok"},
    ])
    c3 = _case("用户列表查询", "P1", ["api", "users"], [
        {"action": "request", "method": "GET", "url": "/api/users", "name": "查询用户列表"},
        {"action": "script", "expr": "len([1,2,3])", "save_as": "count", "name": "计算数量"},
        {"action": "assert", "type": "gte", "actual": "${count}", "expected": 3, "name": "数量 >= 3"},
    ])
    c4 = _case("创建项目", "P1", ["api", "projects"], [
        {"action": "request", "method": "POST", "url": "/api/projects", "name": "创建项目"},
        {"action": "assert", "type": "status", "actual": "${resp.status}", "expected": 200, "name": "状态码 200"},
    ])
    c5 = _case("慢接口（性能）", "P2", ["perf"], [
        {"action": "request", "method": "GET", "url": "/api/slow", "name": "请求慢接口"},
        {"action": "assert", "type": "status", "actual": "${resp.status}", "expected": 200, "name": "状态码 200"},
    ])
    c6 = _case("失败注入接口", "P2", ["chaos"], [
        {"action": "request", "method": "GET", "url": "/api/error", "name": "请求失败接口"},
        {"action": "assert", "type": "status", "actual": "${resp.status}", "expected": 200, "name": "期望 200"},
    ])
    c7 = _case("字符串断言", "P2", ["unit"], [
        {"action": "script", "expr": "2 + 3 * 4", "save_as": "result", "name": "算术"},
        {"action": "assert", "type": "equals", "actual": "${result}", "expected": 14, "name": "结果等于 14"},
        {"action": "assert", "type": "between", "actual": "${result}", "expected": [10, 20], "name": "结果在 10~20"},
    ])
    c8 = _case("正则断言", "P3", ["unit"], [
        {"action": "set", "key": "text", "value": "release-2.31.0", "name": "设置文本"},
        {"action": "assert", "type": "regex", "actual": "${text}", "expected": r"^\d+\.\d+", "name": "匹配版本号"},
    ])
    # —— 失败聚类演示用例：下列用例在 dev 环境也稳定失败，形成三类簇 ——
    cluster_ids = []
    for spec in CLUSTER_DEMO_CASES:
        cluster_ids.append(_case(spec["name"], spec["priority"],
                                 spec["tags"], spec["steps"]))

    suite = {
        "id": new_id("suite"),
        "project_id": pid,
        "name": "冒烟测试套件",
        "description": "核心链路冒烟",
        "group": "smoke",
        "env_id": env["id"],
        "case_ids": [c1, c2, c3, c4, c5, c6, c7, c8] + cluster_ids,
        "created_at": time.time(),
    }
    registry.store("suites").insert(suite)

    registry.store("schedules").insert({
        "id": new_id("sch"),
        "project_id": pid,
        "name": "每 10 分钟跑一次冒烟",
        "cron": "*/10 * * * *",
        "suite_id": suite["id"],
        "env_id": env["id"],
        "enabled": False,
        "last_fired_minute": None,
        "created_at": time.time(),
    })

    notify_mgr.create(pid, {
        "type": "webhook",
        "name": "CI Webhook",
        "config": {"url": "https://example.com/hooks/ci"},
        "events": ["build.finished", "build.failed"],
    })
    notify_mgr.create(pid, {
        "type": "email",
        "name": "团队邮件",
        "config": {"address": "qa@example.com"},
        "events": ["build.failed"],
    })

    return {"project": proj, "env_id": env["id"], "suite_id": suite["id"]}
