"""失败聚类测试。

覆盖：
- 特征抽取（接口归一化 / 断言 / 报错关键词）
- 三类簇（同一接口 / 同一断言 / 同一报错关键词）与优先级
- 跨构建失败点归并：命中未关闭缺陷不重复建单；fixed 自动重开；
  closed 重新建单
- 人工调整：合并、拆分、改关联、重新聚类保留人工簇
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (CoverageAnalyzer, DefectManager, EnvironmentManager,
                    FailureClusterManager, NotificationManager,
                    ReportGenerator, Scheduler, TestExecutor)
from engine.clustering import (cluster_features, extract_features,
                               fingerprint_for, normalize_path)
from storage import BuildStoreRegistry, StoreRegistry


def _failing_result(case_id, name, steps, status="failed", priority="P2"):
    case = {"id": case_id, "name": name, "priority": priority, "tags": ["g"],
            "timeout": 30, "steps": steps}
    return TestExecutor().execute_case(case, {"latency_ms": 0, "fail_rate": 0.0})


def _req_step(url, method="GET"):
    return {"action": "request", "method": method, "url": url, "name": "r"}


def _status_assert(expected=200):
    return {"action": "assert", "type": "status",
            "actual": "${resp.status}", "expected": expected, "name": "状态码"}


class TestFeatureExtraction(unittest.TestCase):
    def test_normalize_path_replaces_ids(self):
        self.assertEqual(normalize_path("/api/users/123/orders"),
                         "/api/users/{id}/orders")
        self.assertEqual(normalize_path("/api/users/123?x=1"), "/api/users/{id}")
        self.assertEqual(normalize_path("/api/health"), "/api/health")

    def test_endpoint_feature(self):
        r = _failing_result("c1", "用户", [_req_step("/api/users/1001"),
                                          _status_assert()])
        f = extract_features(r)
        self.assertEqual(f["endpoint_sig"], "GET /api/users/{id} @404")
        self.assertEqual(f["keyword"], "HTTP 404 Not Found")

    def test_keyword_rate_limit(self):
        r = _failing_result("c2", "限流", [_req_step("/api/products/ratelimit"),
                                           _status_assert()])
        f = extract_features(r)
        self.assertEqual(f["keyword"], "products Rate Limit Exceeded")

    def test_assertion_feature(self):
        r = _failing_result("c3", "算术", [
            {"action": "script", "expr": "2 + 3 * 4", "save_as": "r"},
            {"action": "assert", "type": "equals", "actual": "${r}",
             "expected": 15, "name": "等于15"},
        ])
        f = extract_features(r)
        self.assertEqual(f["assert_sig"], "equals~#")
        self.assertIsNone(f["endpoint_sig"])

    def test_timeout_keyword(self):
        case = {"id": "c4", "name": "超时", "timeout": 0.1, "steps": [
            {"action": "sleep", "seconds": 0.5}]}
        r = TestExecutor().execute_case(case, {"latency_ms": 0})
        f = extract_features(r)
        self.assertEqual(f["keyword"], "Timeout")

    def test_passed_returns_none(self):
        r = _failing_result("c5", "健康", [_req_step("/api/health"),
                                           _status_assert()])
        # /api/health 返回 200，用例通过
        self.assertIsNone(extract_features(r))


class TestClusterFeatures(unittest.TestCase):
    def _feats(self):
        cases = [
            ("u1", "/api/users/1001"), ("u2", "/api/users/2002"),
            ("p1", "/api/shop/products/ratelimit"),
            ("s1", "/api/shop/stock/ratelimit"),
            ("c1", "/api/shop/comments/ratelimit"),
            ("e1", "/api/error"),
        ]
        out = []
        for cid, url in cases:
            out.append(extract_features(
                _failing_result(cid, cid, [_req_step(url), _status_assert()])))
        return out

    def test_three_cluster_types(self):
        clusters = cluster_features(self._feats())
        dims = {}
        for c in clusters:
            dims.setdefault(c["dim"], []).append(len(c["features"]))
        # 接口簇：2；强关键词簇（限流）：3；500 单点：1
        self.assertEqual(dims["endpoint"], [2])
        self.assertEqual(dims["keyword"], [3, 1])

    def test_assertion_cluster(self):
        feats = []
        for cid in ("a1", "a2"):
            feats.append(extract_features(_failing_result(cid, cid, [
                {"action": "script", "expr": "2 + 3 * 4", "save_as": "r"},
                {"action": "assert", "type": "equals", "actual": "${r}",
                 "expected": 15},
            ])))
        clusters = cluster_features(feats)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["dim"], "assertion")

    def test_fingerprint_stable_across_numbers(self):
        fp1 = fingerprint_for("endpoint", "GET /api/users/{id} @404")
        fp2 = fingerprint_for("endpoint", "GET /api/users/{id} @404")
        self.assertEqual(fp1, fp2)


class _ClusterFixture:
    """构造一套接好 FailureClusterManager 的真实存储 + 执行器环境。"""

    def __init__(self, auto_defects=True):
        self.tmp = tempfile.TemporaryDirectory()
        # 每个夹具一个唯一前缀，让不同测试方法的失败特征指纹互不相同，
        # 避免“命中未关闭缺陷自动归并”把测试彼此串起来
        self.token = uuid.uuid4().hex[:8]
        self.registry = StoreRegistry(os.path.join(self.tmp.name, "store"))
        self.builds = BuildStoreRegistry(os.path.join(self.tmp.name, "builds"))
        # 显式唯一 project_id：BuildStoreRegistry 按 project_id 缓存 BuildStore，
        # 若都用自增 id（projects_1），跨测试会复用到指向旧临时目录的实例
        self.project_id = "proj_" + self.token
        self.registry.store("projects").insert(
            {"id": self.project_id, "name": "P",
             "auto_create_defects": auto_defects})
        self.env_mgr = EnvironmentManager(self.registry, self.tmp.name)
        self.env = self.env_mgr.create(
            self.project_id,
            {"name": "dev", "config": {"latency_ms": 0, "fail_rate": 0.0}})
        self.defects = DefectManager(self.registry)
        self.clusterer = FailureClusterManager(self.registry, self.builds,
                                               self.defects)
        self.scheduler = Scheduler(
            self.registry, self.builds, TestExecutor(), self.env_mgr,
            ReportGenerator(self.builds), CoverageAnalyzer(self.builds),
            self.defects, NotificationManager(self.registry),
            cluster_manager=self.clusterer)
        self._case_seq = 0

    def add_case(self, steps, case_id=None, priority="P2", name=None):
        self._case_seq += 1
        cid = case_id or f"case_{self._case_seq}"
        self.registry.store("cases").insert({
            "id": cid, "project_id": self.project_id,
            "name": name or cid, "priority": priority, "tags": ["g"],
            "timeout": 30, "steps": steps,
        })
        return cid

    def add_rate_case(self, key, case_id=None, priority="P2"):
        """限流用例：同夹具内共享业务前缀（错误文案相同 → 关键词簇），
        但子路径不同（不会被更高优先级的接口簇收走）；夹具前缀唯一，
        保证不同测试方法的失败指纹互不干扰。"""
        return self.add_case(
            _rate_case(f"/api/{self.token}/{key}/ratelimit"),
            case_id=case_id, priority=priority, name=case_id or key)

    def add_error_case(self, case_id=None):
        return self.add_case(
            [_req_step(f"/api/{self.token}_error"), _status_assert()],
            case_id=case_id, name=case_id or "error")

    def make_suite(self, case_ids, suite_id="suite_1"):
        self.registry.store("suites").insert({
            "id": suite_id, "project_id": self.project_id, "name": "s",
            "env_id": self.env["id"], "case_ids": case_ids,
        })

    def run(self, suite_id="suite_1", wait=True):
        build = self.scheduler.submit_build(self.project_id, suite_id)
        bid = build["id"]
        if wait:
            deadline = time.time() + 15
            while time.time() < deadline:
                running = {r["build_id"] for r in self.scheduler.running()}
                if bid not in running:
                    break  # _finalize（聚类 + 缺陷归并/建单）已全部完成
                time.sleep(0.01)
        return bid

    def clusters_of(self, build_id):
        return self.clusterer.get_or_analyze(self.project_id, build_id)["clusters"]

    def cleanup(self):
        self.scheduler.shutdown()
        self.tmp.cleanup()


def _rate_case(url):
    return [_req_step(url), _status_assert()]


class TestDefectLinking(unittest.TestCase):
    def setUp(self):
        self.fx = _ClusterFixture(auto_defects=True)

    def tearDown(self):
        self.fx.cleanup()

    def _seed_and_run(self):
        ids = [
            self.fx.add_rate_case("products", "p1"),
            self.fx.add_rate_case("stock", "s1"),
            self.fx.add_rate_case("comments", "c1"),
        ]
        self.fx.make_suite(ids)
        return self.fx.run()

    def test_first_build_creates_one_defect_per_cluster(self):
        bid = self._seed_and_run()
        clusters = self.fx.clusters_of(bid)
        rle = next(c for c in clusters if c["dim"] == "keyword")
        self.assertEqual(len(rle["members"]), 3)
        self.assertIsNotNone(rle["defect_id"])
        self.assertEqual(self.fx.defects.stats(self.fx.project_id)["total"], 1)

    def test_second_build_merges_into_open_defect(self):
        b1 = self._seed_and_run()
        before = self.fx.defects.stats(self.fx.project_id)["total"]
        b2 = self.fx.run()
        after = self.fx.defects.stats(self.fx.project_id)["total"]
        self.assertEqual(before, after)  # 未关闭缺陷被命中 → 不重复建单
        # 跨构建失败点记录了 2 次出现
        points = self.fx.clusterer.failure_points(self.fx.project_id)
        self.assertEqual(points[0]["build_count"], 2)
        self.assertTrue(points[0]["recurring"])
        # 两场构建的簇指向同一个缺陷
        d1 = {c["fingerprint"]: c["defect_id"] for c in self.fx.clusters_of(b1)}
        d2 = {c["fingerprint"]: c["defect_id"] for c in self.fx.clusters_of(b2)}
        self.assertEqual(d1, d2)

    def test_fixed_defect_reopened_on_hit(self):
        self._seed_and_run()
        b2 = self.fx.run()
        defect_id = self.fx.clusters_of(b2)[0]["defect_id"]
        self.fx.defects.update(defect_id, {"status": "fixed"})
        b3 = self.fx.run()
        linked = self.fx.clusters_of(b3)[0]
        self.assertEqual(linked["defect"]["status"], "reopened")
        # 仍然没有新建缺陷
        self.assertEqual(self.fx.defects.stats(self.fx.project_id)["total"], 1)

    def test_closed_defect_creates_new(self):
        self._seed_and_run()
        points = self.fx.clusterer.failure_points(self.fx.project_id)
        old_id = points[0]["defect"]["id"]
        self.fx.defects.update(old_id, {"status": "closed"})
        self.fx.run()
        defects = self.fx.defects.list(self.fx.project_id)
        open_defs = [d for d in defects if d["id"] != old_id]
        self.assertEqual(len(open_defs), 1)
        self.assertEqual(open_defs[0]["status"], "open")

    def test_no_auto_defects_leaves_unlinked(self):
        fx = _ClusterFixture(auto_defects=False)
        try:
            ids = [fx.add_rate_case("products", "p1"),
                   fx.add_rate_case("stock", "s1")]
            fx.make_suite(ids, "suite_x")
            bid = fx.run("suite_x")
            c = fx.clusters_of(bid)[0]
            self.assertIsNone(c["defect_id"])
            # 但失败点仍被记录
            self.assertEqual(len(fx.clusterer.failure_points(fx.project_id)), 1)
        finally:
            fx.cleanup()


class TestManualAdjustments(unittest.TestCase):
    def setUp(self):
        self.fx = _ClusterFixture(auto_defects=True)
        ids = [
            self.fx.add_rate_case("products", "p1"),
            self.fx.add_rate_case("stock", "s1"),
            self.fx.add_error_case("e1"),
        ]
        self.fx.make_suite(ids)
        self.bid = self.fx.run()

    def tearDown(self):
        self.fx.cleanup()

    def _find(self, pred):
        return next(c for c in self.fx.clusters_of(self.bid) if pred(c))

    def test_merge_clusters_and_points(self):
        rle = self._find(lambda c: c["member_count"] == 2)
        single = self._find(lambda c: c["member_count"] == 1)
        result = self.fx.clusterer.merge(
            self.fx.project_id, self.bid, rle["id"], single["id"])
        merged = next(c for c in result["clusters"] if c["id"] == rle["id"])
        self.assertEqual(merged["member_count"], 3)
        self.assertEqual(merged["dim"], "custom")
        self.assertFalse(merged["auto"])
        # 两个失败点也合并了
        points = self.fx.clusterer.failure_points(self.fx.project_id)
        self.assertEqual(len(points), 1)

    def test_split_cluster(self):
        rle = self._find(lambda c: c["member_count"] == 2)
        result = self.fx.clusterer.split(
            self.fx.project_id, self.bid, rle["id"], ["p1"])
        counts = sorted(c["member_count"] for c in result["clusters"])
        # 原簇剩 1、新簇 1、另有 500 单点 1
        self.assertEqual(counts, [1, 1, 1])
        newc = next(c for c in result["clusters"]
                    if c["member_count"] == 1
                    and c["members"][0]["case_id"] == "p1")
        self.assertFalse(newc["auto"])
        # 不能把簇成员全部拆空
        err = self.fx.clusterer.split(
            self.fx.project_id, self.bid, rle["id"], ["s1"])
        self.assertIn("error", err)

    def test_recluster_keeps_manual_clusters(self):
        rle = self._find(lambda c: c["member_count"] == 2)
        single = self._find(lambda c: c["member_count"] == 1)
        self.fx.clusterer.merge(self.fx.project_id, self.bid,
                                rle["id"], single["id"])
        result = self.fx.clusterer.analyze_build(
            self.fx.project_id, self.bid, create_defects=True)
        manual = [c for c in result["clusters"] if not c["auto"]]
        self.assertTrue(manual)
        self.assertEqual(sum(c["member_count"] for c in manual), 3)

    def test_change_defect_link_binds_fingerprint(self):
        rle = self._find(lambda c: c["member_count"] == 2)
        # 解除关联
        result = self.fx.clusterer.set_defect(
            self.fx.project_id, self.bid, rle["id"], None)
        c = next(x for x in result["clusters"] if x["id"] == rle["id"])
        self.assertIsNone(c["defect_id"])
        # 手工新建并关联：指纹写进缺陷，下次构建自动命中
        result = self.fx.clusterer.set_defect(
            self.fx.project_id, self.bid, rle["id"], "new")
        if "clusters" not in result:
            raise AssertionError(f"set_defect 返回错误: {result}")
        c = next(x for x in result["clusters"] if x["id"] == rle["id"])
        defect = self.fx.defects.get(c["defect_id"])
        self.assertIn(rle["fingerprint"], defect["fingerprints"])
        b2 = self.fx.run()
        hits = [x for x in self.fx.clusters_of(b2)
                if x.get("defect_id") == defect["id"]]
        self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main()
