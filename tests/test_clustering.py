"""失败聚类测试。

覆盖：
- 特征抽取（同接口 / 同断言 / 同报错关键词 / 超时）与自动聚类；
- 构建簇持久化与幂等重算；
- 人工合并 / 拆分 / 改关联 / 恢复自动；
- 命中未关闭缺陷的失败特征时自动归并，而不是新建；
- 跨构建同一失败点反复出现的聚合。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import DefectManager, FailureClusterManager
from engine.clustering import cluster_results, failure_features
from storage import BuildStoreRegistry, StoreRegistry


def _failed(cid, name, *, status="failed", priority="P2", group="g",
            steps=None, assertions=None):
    return {
        "case_id": cid, "case_name": name, "group": group,
        "priority": priority, "status": status, "duration": 0.01,
        "steps": steps or [], "assertions": assertions or [],
        "logs": [], "message": "",
    }


def endpoint_fail(cid, name, path="/api/orders", code=500, method="POST",
                  status="error", **kw):
    return _failed(
        cid, name, status=status,
        steps=[{"action": "request", "name": "req", "status": status,
                "message": f"{method} {path} -> {code} (20.0ms)"}], **kw)


def assert_fail(cid, name, atype="status", expected=200, actual=500):
    return _failed(
        cid, name,
        steps=[{"action": "assert", "name": "a", "status": "failed",
                "message": f"期望 {expected!r}，实际 {actual!r}"}],
        assertions=[{"name": "a", "type": atype, "expected": expected,
                     "actual": actual, "ok": False,
                     "message": f"期望 == {expected!r}"}])


def exc_fail(cid, name, text="ConnectionError: dial tcp 10.0.0.1:80 timeout"):
    return _failed(cid, name, status="error",
                   steps=[{"action": "request", "name": "r", "status": "error",
                           "message": text}])


class TestFeatureExtraction(unittest.TestCase):
    def test_endpoint_normalization(self):
        f1 = failure_features(endpoint_fail("c1", "a", "/api/orders/123/items"))
        f2 = failure_features(endpoint_fail("c2", "b", "/api/orders/987/items?x=1"))
        self.assertEqual(f1["primary"]["dim"], "endpoint")
        self.assertEqual(f1["primary"]["key"], f2["primary"]["key"])
        self.assertIn("{id}", f1["primary"]["key"])

    def test_same_assertion_clusters(self):
        clusters = cluster_results(
            [assert_fail(f"c{i}", f"n{i}", atype="equals", expected="固定值",
                         actual="别的值") for i in range(5)])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["dim"], "assertion")
        self.assertEqual(clusters[0]["count"], 5)

    def test_status_assertion_is_endpoint_dim(self):
        # 对 resp.status 的失败断言 = 接口层失败（执行器语义：请求步骤
        # 本身恒通过，HTTP 失败由状态码断言表达）
        result = _failed(
            "c1", "接口",
            steps=[{"action": "request", "name": "r", "status": "passed",
                    "message": "GET /api/orders/123 -> 500 (20ms)"},
                   {"action": "assert", "name": "a", "status": "failed",
                    "message": "期望状态码 == 200，实际 500"}],
            assertions=[{"name": "a", "type": "status", "expected": 200,
                         "actual": 500, "ok": False,
                         "message": "期望状态码 == 200"}])
        f = failure_features(result)
        self.assertEqual(f["primary"]["dim"], "endpoint")
        self.assertEqual(f["primary"]["key"], "endpoint|GET|/api/orders/{id}|500")

    def test_keyword_and_timeout(self):
        clusters = cluster_results([
            exc_fail("c1", "a"), exc_fail("c2", "b",
                                          "ConnectionError: other host down"),
            _failed("c3", "c", status="timeout",
                    steps=[{"action": "sleep", "status": "timeout",
                            "message": "用例超时"}]),
        ])
        dims = {c["dim"]: c["count"] for c in clusters}
        self.assertEqual(dims["keyword"], 2)
        self.assertEqual(dims["status"], 1)

    def test_passes_ignored(self):
        passed = _failed("c1", "a", status="passed")
        self.assertEqual(cluster_results([passed]), [])


class TestClusterManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry = StoreRegistry(os.path.join(self.tmp.name, "store"),
                                      shard_size=50)
        self.builds = BuildStoreRegistry(os.path.join(self.tmp.name, "builds"))
        self.defects = DefectManager(self.registry)
        self.mgr = FailureClusterManager(self.registry, self.builds, self.defects)
        self.pid = self.registry.store("projects").insert(
            {"name": "P", "auto_create_defects": False})

    def tearDown(self):
        self.tmp.cleanup()

    def _build(self, bid, results):
        store = self.builds.for_project(self.pid)
        store.create(bid, name=bid)
        store.set_total(bid, len(results))
        for r in results:
            store.record_result(bid, r)
        store.finish(bid, "failed")
        return store

    # -- 自动聚类 ---------------------------------------------------------
    def test_cluster_build_groups_failures(self):
        results = [endpoint_fail("c1", "下单1"), endpoint_fail("c2", "下单2"),
                   assert_fail("c3", "值断言", atype="equals",
                               expected="固定值", actual="别的值"),
                   _failed("c4", "超时", status="timeout")]
        self._build("b1", results)
        out = self.mgr.cluster_build(self.pid, "b1")
        self.assertEqual(out["total_failed"], 4)
        by_dim = {c["dim"]: c for c in out["clusters"]}
        self.assertEqual(by_dim["endpoint"]["count"], 2)
        self.assertIn("assertion", by_dim)
        self.assertIn("status", by_dim)

    def test_cluster_build_is_idempotent(self):
        self._build("b1", [endpoint_fail("c1", "a"), endpoint_fail("c2", "b")])
        first = self.mgr.cluster_build(self.pid, "b1")
        second = self.mgr.cluster_build(self.pid, "b1")
        self.assertEqual(first["clusters"][0]["group_id"],
                         second["clusters"][0]["group_id"])
        occ = self.registry.store("failure_occurrences").query(
            where=[("build_id", "eq", "b1")])
        self.assertEqual(len(occ), 2)  # 重算不产生重复发生记录

    # -- 人工合并 / 拆分 --------------------------------------------------
    def test_merge_clusters(self):
        self._build("b1", [endpoint_fail("c1", "a"), assert_fail("c2", "b")])
        snap = self.mgr.cluster_build(self.pid, "b1")
        gids = [c["group_id"] for c in snap["clusters"]]
        merged = self.mgr.merge(self.pid, "b1", gids)
        self.assertEqual(len(merged["clusters"]), 1)
        self.assertEqual(merged["clusters"][0]["count"], 2)
        self.assertTrue(merged["clusters"][0]["manual"])
        self.assertEqual(merged["clusters"][0]["dim"], "merged")
        # 重新自动聚类不覆盖人工结果
        again = self.mgr.cluster_build(self.pid, "b1")
        self.assertEqual(len(again["clusters"]), 1)

    def test_merge_requires_two(self):
        self._build("b1", [endpoint_fail("c1", "a")])
        snap = self.mgr.cluster_build(self.pid, "b1")
        err = self.mgr.merge(self.pid, "b1", [snap["clusters"][0]["group_id"]])
        self.assertIn("error", err)

    def test_split_cluster(self):
        # 同接口但一个是 500、一个是 404，先各自一簇；拆接口簇做成员移动
        results = [
            endpoint_fail("c1", "a", code=500),
            endpoint_fail("c2", "b", code=500),
            endpoint_fail("c3", "c", code=500),
            assert_fail("c4", "d"),
        ]
        self._build("b1", results)
        snap = self.mgr.cluster_build(self.pid, "b1")
        ep = next(c for c in snap["clusters"] if c["dim"] == "endpoint")
        out = self.mgr.split(self.pid, "b1", ep["group_id"], ["c3"])
        # c3 被拆出后成为独立簇（其签名仍是同接口，因此这里回到自动簇，
        # 但因 override 冻结，它与剩余 2 个分开呈现）
        groups = out["clusters"]
        gid_counts = {c["group_id"]: c["count"] for c in groups}
        self.assertEqual(sum(gid_counts.values()), 4)
        remaining = next(c for c in groups if c["group_id"] == ep["group_id"])
        self.assertEqual(remaining["count"], 2)

    def test_split_child_does_not_inherit_or_auto_create_defect(self):
        """回归：合并后拆出的小簇不应继承父簇缺陷，自动建缺陷开启时
        也不该因为它「暂无关联」就偷偷再建一条。"""
        self.registry.store("projects").update(
            self.pid, {"auto_create_defects": True})
        results = [endpoint_fail("c1", "a"), endpoint_fail("c2", "b"),
                   endpoint_fail("c3", "c"),
                   assert_fail("c4", "d", atype="equals",
                               expected="固定", actual="别的")]
        self._build("b1", results)
        snap = self.mgr.cluster_build(self.pid, "b1")
        gids = [c["group_id"] for c in snap["clusters"]]
        merged = self.mgr.merge(self.pid, "b1", gids[:2])
        big = max(merged["clusters"], key=lambda x: x["count"])
        out = self.mgr.split(self.pid, "b1", big["group_id"],
                             [big["cases"][0]["case_id"]])
        child = next(c for c in out["clusters"]
                     if c["group_id"].startswith("clusp"))
        self.assertIsNone(child["defect_id"])
        self.assertTrue(child["manual"])
        # 重新计算仍然无缺陷（没有被自动建缺陷补回）
        again = self.mgr.cluster_build(self.pid, "b1")
        child2 = next(c for c in again["clusters"]
                      if c["group_id"] == child["group_id"])
        self.assertIsNone(child2["defect_id"])

    def test_assign_and_reset_override(self):
        self._build("b1", [endpoint_fail("c1", "a"), endpoint_fail("c2", "b")])
        snap = self.mgr.cluster_build(self.pid, "b1")
        gid = snap["clusters"][0]["group_id"]
        defect = self.defects.create(self.pid, {"title": "下单接口挂了"})
        out = self.mgr.assign_defect(self.pid, "b1", gid, defect["id"])
        self.assertEqual(out["clusters"][0]["defect_id"], defect["id"])
        # 锚点记住了签名
        anchor = self.registry.store("failure_anchor").query(
            where=[("project_id", "eq", self.pid)])
        self.assertTrue(any(a["defect_id"] == defect["id"] for a in anchor))
        # 解除关联 + 恢复自动
        self.mgr.assign_defect(self.pid, "b1", gid, None)
        out = self.mgr.cluster_build(self.pid, "b1")
        self.assertIsNone(out["clusters"][0]["defect_id"])
        self.assertTrue(out["clusters"][0]["manual"])  # 覆盖仍在（冻结状态）
        reset = self.mgr.reset(self.pid, "b1", gid)
        self.assertFalse(reset["clusters"][0]["manual"])

    # -- 缺陷归并 ---------------------------------------------------------
    def test_existing_open_defect_signature_auto_links(self):
        self._build("b1", [endpoint_fail("c1", "a"), endpoint_fail("c2", "b")])
        snap = self.mgr.cluster_build(self.pid, "b1")
        sig = snap["clusters"][0]["signature"]

        # 已存在的未关闭缺陷带有同一失败特征
        self.defects.create(self.pid, {"title": "已知下单故障",
                                       "status": "open", "signatures": [sig]})
        out = self.mgr.cluster_build(self.pid, "b1")
        self.assertEqual(len(out["clusters"]), 1)
        self.assertIsNotNone(out["clusters"][0]["defect_id"])

    def test_closed_defect_signature_does_not_auto_link(self):
        self._build("b1", [endpoint_fail("c1", "a")])
        snap = self.mgr.cluster_build(self.pid, "b1")
        sig = snap["clusters"][0]["signature"]
        self.defects.create(self.pid, {"title": "已关闭", "status": "closed",
                                       "signatures": [sig]})
        out = self.mgr.cluster_build(self.pid, "b1")
        self.assertIsNone(out["clusters"][0]["defect_id"])

    def test_create_defect_for_cluster_carries_signatures(self):
        self._build("b1", [endpoint_fail("c1", "a"), endpoint_fail("c2", "b")])
        snap = self.mgr.cluster_build(self.pid, "b1")
        gid = snap["clusters"][0]["group_id"]
        out = self.mgr.create_defect_for_cluster(self.pid, "b1", gid)
        cluster = out["clusters"][0]
        defect = self.defects.get(cluster["defect_id"])
        self.assertIn(cluster["signature"], defect["signatures"])

    def test_cross_build_same_failure_point_reuses_defect(self):
        # 第一场构建：开启自动建缺陷，簇 -> 缺陷
        self.registry.store("projects").update(
            self.pid, {"auto_create_defects": True})
        self._build("b1", [endpoint_fail("c1", "a"), endpoint_fail("c2", "b")])
        out1 = self.mgr.cluster_build(self.pid, "b1")
        defect_id = out1["clusters"][0]["defect_id"]
        self.assertTrue(defect_id)
        self.assertEqual(len(self.defects.list(self.pid)), 1)

        # 第二场构建：同样的失败，应归并到同一缺陷，不再新建
        self._build("b2", [endpoint_fail("c3", "c"), endpoint_fail("c4", "d"),
                           endpoint_fail("c5", "e")])
        out2 = self.mgr.cluster_build(self.pid, "b2")
        self.assertEqual(out2["clusters"][0]["defect_id"], defect_id)
        self.assertEqual(len(self.defects.list(self.pid)), 1)

    # -- 跨构建失败点 -----------------------------------------------------
    def test_failure_points_aggregates_recurrence(self):
        self._build("b1", [endpoint_fail("c1", "a")])
        self.mgr.cluster_build(self.pid, "b1")
        self._build("b2", [endpoint_fail("c2", "b"), assert_fail("c9", "z")])
        self.mgr.cluster_build(self.pid, "b2")
        self._build("b3", [endpoint_fail("c3", "c")])
        self.mgr.cluster_build(self.pid, "b3")

        pts = self.mgr.failure_points(self.pid)["points"]
        endpoint_point = next(p for p in pts if p["dim"] == "endpoint")
        self.assertEqual(endpoint_point["build_count"], 3)
        self.assertEqual(endpoint_point["occurrences"], 3)
        self.assertEqual(endpoint_point["last_build_id"], "b3")
        # 反复出现的排前面
        self.assertEqual(pts[0]["dim"], "endpoint")


if __name__ == "__main__":
    unittest.main()
