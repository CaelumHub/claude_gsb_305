"""失败簇管理：持久化、人工调整、缺陷关联记忆与跨构建失败点。

数据落在通用分片存储里（三类记录）：

1. **failure_occurrences（失败发生记录）**：一条 = 某构建某簇命中的
   一次失败（签名 + 簇展示 id + 用例 + 时间）。这是「跨构建反复出现」
   的原始序列；重新聚类时按 (build_id, case_id) 幂等覆盖。

2. **failure_anchor（失败点锚点）**：以聚类签名为键（每个项目一条），
   记忆该签名当前归到哪个簇 id、关联哪个缺陷。签名跨构建稳定，因此
   新构建里同样的失败能自动落进同一个失败点，并自动归并到未关闭缺陷。

3. **failure_override（人工调整）**：以簇 id 为键的冻结快照，记录
   人工合并 / 拆分 / 改关联的结果。之后即使重新自动聚类，也以人工
   结果为准（机器不抢人的判断）；显式「恢复自动聚类」才清除覆盖。

簇 id 语义
----------
- 自动簇 id 由签名哈希决定（:func:`engine.clustering.group_id_for`），
  同一签名天然复用同一锚点；
- 合并产生新的人工簇 id（``clum_...``），锚点整体迁移过去；
- 拆分产生新的自动分组（以被拆出用例重新按签名生成），原簇覆盖里
  删除这些用例，签名锚点按各分组重新指向。
"""

from __future__ import annotations

import time
from typing import Optional

from .clustering import DIM_MERGED, FAILED_STATUSES, cluster_results
from .models import new_id

# 缺陷关联后，命中锚点的发生记录会带上该缺陷 id
OPEN_DEFECT_STATUSES = ("open", "in_progress", "reopened")


class FailureClusterManager:
    """失败簇 / 失败点管理。"""

    def __init__(self, registry, build_registry, defect_manager):
        self.registry = registry
        self.builds = build_registry
        self.defects = defect_manager

    # -- 存储句柄 ---------------------------------------------------------
    @property
    def _occ(self):
        return self.registry.store("failure_occurrences")

    @property
    def _anchor(self):
        return self.registry.store("failure_anchor")

    @property
    def _override(self):
        return self.registry.store("failure_override")

    # ================================================================== 构建聚类
    def cluster_build(self, project_id: str, build_id: str) -> dict:
        """对一场构建的失败用例聚类，应用人工调整并尝试关联缺陷。

        流程：自动聚类 → 应用合并/拆分覆盖 → 用锚点记忆补关联 →
        未命中记忆时匹配未关闭缺陷的失败特征 → 写发生记录 →
        项目开启自动建缺陷时按未关联的簇各建一条。结果可直接给报告页。
        """
        store = self.builds.for_project(project_id)
        build = store.get(build_id)
        if build is None:
            return {"error": "构建不存在"}

        records = store.results(build_id,
                                where=[("status", "in", list(FAILED_STATUSES))])
        auto = cluster_results(records)

        case_to_group = self._case_group_index(auto)
        overrides = self._load_overrides(project_id, build_id)
        case_to_group = self._apply_overrides(case_to_group, overrides)
        clusters = self._reassemble(auto, case_to_group, overrides)

        self._attach_defects(project_id, clusters, overrides)

        # 写发生记录（幂等：先清掉本构建旧记录）
        self._replace_occurrences(project_id, build_id,
                                  build.get("finished_at") or time.time(),
                                  clusters)

        # 自动建缺陷：项目开启时，每个未关联缺陷的簇最多建一条，
        # 而不是像旧逻辑那样每条失败用例各建一条。
        project = self.registry.store("projects").get(project_id)
        if project and project.get("auto_create_defects"):
            self._auto_create_for_clusters(project_id, build_id, clusters)

        return {
            "project_id": project_id,
            "build_id": build_id,
            "total_failed": len(records),
            "cluster_count": len(clusters),
            "clusters": clusters,
        }

    # -- 覆盖：索引、应用、重组 ------------------------------------------
    @staticmethod
    def _case_group_index(auto: list[dict]) -> dict[str, str]:
        """case_id -> 自动簇 group_id。"""
        idx: dict[str, str] = {}
        for c in auto:
            for cid in c["case_ids"]:
                idx[cid] = c["group_id"]
        return idx

    def _load_overrides(self, project_id: str, build_id: str) -> dict[str, dict]:
        """加载本构建的全部簇覆盖（以簇 id 为键）。

        覆盖是「某场构建内」的人工调整，不跨构建复用——跨构建的归并
        由签名锚点（failure_anchor）承担。合并 / 拆分产生的人工簇 id
        不在本次自动簇列表里，因此不能按自动簇 id 过滤。
        """
        return {o["group_id"]: o
                for o in self._override.query(
                    where=[("project_id", "eq", project_id),
                           ("build_id", "eq", build_id)])}

    def _apply_overrides(self, case_to_group: dict[str, str],
                         overrides: dict[str, dict]) -> dict[str, str]:
        """把人工合并 / 拆分后的归属应用到 case -> group 映射。"""
        result = dict(case_to_group)

        # 1) 合并：把成员簇的全部用例指向合并簇 id。
        for ov in overrides.values():
            if ov.get("kind") != "merged":
                continue
            target = ov["group_id"]
            for member in ov.get("member_group_ids", []):
                for cid, g in list(result.items()):
                    if g == member:
                        result[cid] = target

        # 2) 拆分：移出的用例按拆分时重算的分组指向新簇。
        #    必须在合并之后应用——合并簇的 member_group_ids 仍含这些
        #    用例的原始自动簇，先合并会把它们拽回合并簇。
        for ov in overrides.values():
            for cid in ov.get("removed_case_ids", []):
                if cid in result:
                    result[cid] = None  # 游离：下面按拆分映射归位
            for cid, target in (ov.get("reassign_case_groups") or {}).items():
                result[cid] = target
        return result

    def _reassemble(self, auto: list[dict], case_to_group: dict[str, str],
                    overrides: dict[str, dict]) -> list[dict]:
        """根据（可能被人工改过的）case->group 映射重组簇结构。

        游离（映射为 None，理论上不出现——拆分接口总会写回精确目标簇）
        的用例退回到其自动签名簇。
        """
        # 原始成员信息：case_id -> (auto_cluster, 行数据)
        member_info: dict[str, tuple[dict, dict]] = {}
        for c in auto:
            for case in c["cases"]:
                member_info[case["case_id"]] = (c, case)

        for cid, gid in list(case_to_group.items()):
            if gid is None and cid in member_info:
                case_to_group[cid] = member_info[cid][0]["group_id"]

        # group_id -> case_ids
        groups: dict[str, list[str]] = {}
        for cid, gid in case_to_group.items():
            groups.setdefault(gid, []).append(cid)

        out: list[dict] = []
        for gid, cids in groups.items():
            ov = overrides.get(gid)
            base = self._first_auto_cluster(auto, cids)
            rows = [member_info[cid][1] for cid in cids if cid in member_info]
            if not rows:
                continue
            rows.sort(key=lambda r: (r.get("priority") != "P0", r.get("case_name") or ""))
            signatures = sorted({member_info[cid][0]["signature"]
                                 for cid in cids if cid in member_info})
            cluster = {
                "group_id": gid,
                "dim": (DIM_MERGED if len(signatures) > 1 else
                        (ov.get("dim") if ov else (base or {}).get("dim", "keyword"))),
                "signature": signatures[0] if len(signatures) == 1 else "",
                "signatures": signatures,
                "title": (ov or {}).get("title")
                or (base or {}).get("title") or "失败簇",
                "reason": rows[0].get("reason", ""),
                "count": len(rows),
                "case_ids": [r["case_id"] for r in rows],
                "cases": rows,
                "auto": ov is None,
                "manual": ov is not None,
                # defect_id 只从「该簇自身」的覆盖取；拆分父簇的关联
                # 不能透传给拆出的子簇（它们必须显式另行关联）。
                "defect_id": (ov.get("defect_id")
                              if ov and ov.get("group_id") == gid else None),
            }
            out.append(cluster)
        out.sort(key=lambda c: (-c["count"], c["group_id"]))
        return out

    @staticmethod
    def _first_auto_cluster(auto: list[dict], cids: list[str]):
        wanted = set(cids)
        for c in auto:
            if wanted & set(c["case_ids"]):
                return c
        return None

    # -- 缺陷关联 ---------------------------------------------------------
    def _attach_defects(self, project_id: str, clusters: list[dict],
                        overrides: dict[str, dict]) -> None:
        """给每个簇补 defect_id：锚点记忆 > 未关闭缺陷的失败特征匹配。"""
        anchors = self._anchor_map(project_id)
        candidates = [d for d in self.defects.list(project_id)
                      if d.get("status") in OPEN_DEFECT_STATUSES]

        for cluster in clusters:
            defect_id = cluster.get("defect_id")

            # 人工冻结过该簇的关联（含显式解除）：以人工判断为准，
            # 不再走锚点 / 特征自动匹配，避免机器把人拆开的又连上。
            ov = overrides.get(cluster["group_id"])
            if ov is not None:
                cluster["defect_id"] = ov.get("defect_id")
            else:
                # 1) 签名锚点记忆（该签名之前已关联过缺陷）
                sigs = cluster.get("signatures") or (
                    [cluster["signature"]] if cluster.get("signature") else [])
                for sig in sigs:
                    anchor = anchors.get(sig)
                    if anchor and anchor.get("defect_id"):
                        defect_id = anchor["defect_id"]
                        break

                # 2) 未命中记忆：匹配未关闭缺陷携带的失败特征
                if not defect_id:
                    for sig in sigs:
                        hit = next((d for d in candidates
                                    if sig in (d.get("signatures") or [])), None)
                        if hit:
                            defect_id = hit["id"]
                            self._remember_anchor(project_id, sig,
                                                  cluster["group_id"], defect_id)
                            break
                cluster["defect_id"] = defect_id

            if cluster.get("defect_id"):
                defect = self.defects.get(cluster["defect_id"])
                if defect:
                    cluster["defect"] = {"id": defect["id"],
                                         "title": defect.get("title"),
                                         "status": defect.get("status"),
                                         "severity": defect.get("severity")}

    def _anchor_map(self, project_id: str) -> dict[str, dict]:
        return {a["signature"]: a
                for a in self._anchor.query(where=[("project_id", "eq", project_id)])}

    def _anchor_signatures_of_other_clusters(self, project_id: str, build_id: str,
                                             group_id: str) -> set[str]:
        """本构建里其它「已关联缺陷且成员不少于当前簇」的簇所持有的签名。

        用于人工改关联时判断锚点归属：共享签名应留在主簇，避免
        合并后拆分出的小簇抢锚点。
        """
        snap = self.cluster_build(project_id, build_id)
        if "error" in snap:
            return set()
        own = next((c for c in snap["clusters"] if c["group_id"] == group_id), None)
        own_size = own.get("count", 0) if own else 0
        occupied: set[str] = set()
        for other in snap.get("clusters", []):
            if other["group_id"] == group_id or not other.get("defect_id"):
                continue
            if other.get("count", 0) >= own_size:
                occupied.update(other.get("signatures")
                                or ([other["signature"]]
                                    if other.get("signature") else []))
        return occupied

    def _remember_anchor(self, project_id: str, signature: str,
                         group_id: str, defect_id: Optional[str]) -> None:
        anchors = self._anchor.query(
            where=[("project_id", "eq", project_id), ("signature", "eq", signature)])
        patch = {"group_id": group_id, "defect_id": defect_id,
                 "updated_at": time.time()}
        if anchors:
            self._anchor.update(anchors[0]["id"], patch)
        else:
            self._anchor.insert({
                "id": new_id("fanc"), "project_id": project_id,
                "signature": signature, **patch,
            })

    def _auto_create_for_clusters(self, project_id: str, build_id: str,
                                  clusters: list[dict]) -> None:
        created_sigs: set[str] = set()
        open_defects = [d for d in self.defects.list(project_id)
                        if d.get("status") in OPEN_DEFECT_STATUSES]
        for cluster in clusters:
            if cluster.get("defect_id"):
                continue
            # 人工冻结过的簇（合并 / 拆分 / 显式解除关联）不自动建缺陷：
            # 用户已经表达过「这个簇怎么处理」，机器不能替他再建一条。
            if cluster.get("manual"):
                continue
            sigs = cluster.get("signatures") or (
                [cluster["signature"]] if cluster.get("signature") else [])

            existing = next((d for d in open_defects
                             if any(s in (d.get("signatures") or []) for s in sigs)), None)
            if existing is not None:
                cluster["defect_id"] = existing["id"]
                cluster["defect"] = {"id": existing["id"],
                                     "title": existing.get("title"),
                                     "status": existing.get("status"),
                                     "severity": existing.get("severity")}
                for sig in sigs:
                    self._remember_anchor(project_id, sig,
                                          cluster["group_id"], existing["id"])
                continue
            if any(s in created_sigs for s in sigs):
                continue

            defect = self.defects.create_from_cluster(project_id, cluster, build_id)
            if defect is None:
                continue
            created_sigs.update(sigs)
            open_defects.append(defect)
            cluster["defect_id"] = defect["id"]
            cluster["defect"] = {"id": defect["id"], "title": defect.get("title"),
                                 "status": defect.get("status"),
                                 "severity": defect.get("severity")}
            for sig in sigs:
                self._remember_anchor(project_id, sig,
                                      cluster["group_id"], defect["id"])

    # -- 发生记录 ---------------------------------------------------------
    def _replace_occurrences(self, project_id: str, build_id: str,
                             finished_at: float, clusters: list[dict]) -> None:
        old = self._occ.query(
            where=[("project_id", "eq", project_id), ("build_id", "eq", build_id)])
        for o in old:
            self._occ.delete(o["id"])
        records = []
        for cluster in clusters:
            for case in cluster["cases"]:
                records.append({
                    "id": new_id("focc"),
                    "project_id": project_id,
                    "build_id": build_id,
                    "build_finished_at": finished_at,
                    "group_id": cluster["group_id"],
                    "dim": cluster.get("dim"),
                    "signature": (cluster.get("signature")
                                  or (cluster.get("signatures") or [""])[0]),
                    "case_id": case["case_id"],
                    "case_name": case.get("case_name"),
                    "status": case.get("status"),
                    "reason": case.get("reason", ""),
                    "defect_id": cluster.get("defect_id"),
                    "created_at": finished_at,
                })
        if records:
            self._occ.insert_many(records)

    # ================================================================== 人工调整
    def merge(self, project_id: str, build_id: str, group_ids: list[str]) -> dict:
        """把同一场构建里的若干簇合并成一个人工簇。"""
        group_ids = list(dict.fromkeys(group_ids))
        if len(group_ids) < 2:
            return {"error": "请至少选择两个簇进行合并"}

        snap = self.cluster_build(project_id, build_id)
        if "error" in snap:
            return snap
        chosen = [c for c in snap["clusters"] if c["group_id"] in group_ids]
        if len(chosen) != len(group_ids):
            return {"error": "部分簇不存在或已没有失败用例"}

        merged_id = new_id("clum")
        all_cases, all_sigs = [], []
        # 关联缺陷取严重级别最高的一个，避免合并后丢掉高优问题
        defect_id = None
        best_rank = 99
        severity_rank = {"blocker": 0, "critical": 1, "major": 2}
        for c in chosen:
            all_cases.extend(c["case_ids"])
            all_sigs.extend(c.get("signatures")
                            or ([c["signature"]] if c.get("signature") else []))
            d = c.get("defect") or {}
            rank = severity_rank.get(d.get("severity"), 3)
            if c.get("defect_id") and rank < best_rank:
                best_rank, defect_id = rank, c["defect_id"]
        all_sigs = sorted(set(all_sigs))
        title = "合并：" + " / ".join(c["title"] for c in chosen[:3])
        if len(chosen) > 3:
            title += f" 等 {len(chosen)} 簇"

        self._override.insert({
            "id": new_id("fovr"), "project_id": project_id,
            "build_id": build_id, "group_id": merged_id,
            "kind": "merged", "title": title,
            "member_group_ids": group_ids, "case_ids": all_cases,
            "removed_case_ids": [], "reassign_case_groups": {},
            "defect_id": defect_id, "created_at": time.time(),
        })
        # 被合并簇自身若有覆盖（例如之前拆过），保留；锚点迁移到新簇
        for sig in all_sigs:
            self._remember_anchor(project_id, sig, merged_id, defect_id)
        return self.cluster_build(project_id, build_id)

    def split(self, project_id: str, build_id: str, group_id: str,
              case_ids: list[str]) -> dict:
        """把若干用例从指定簇中拆出，按各自失败签名重新成簇。"""
        case_ids = list(dict.fromkeys(case_ids))
        if not case_ids:
            return {"error": "请选择要拆出的用例"}

        snap = self.cluster_build(project_id, build_id)
        if "error" in snap:
            return snap
        target = next((c for c in snap["clusters"]
                       if c["group_id"] == group_id), None)
        if target is None:
            return {"error": "簇不存在"}
        movable = [cid for cid in case_ids if cid in target["case_ids"]]
        if not movable:
            return {"error": "所选用例都不在该簇中"}

        # 用例重新按自身失败特征聚类（可能落回原签名，也可能形成新签名簇）。
        # 注意：即便拆出的用例与原簇签名相同，拆分的语义也是「人工认为
        # 它们不是一回事」，因此必须给拆出的每个分组一个新簇 id，
        # 否则按签名重算会立刻又合回原簇。
        store = self.builds.for_project(project_id)
        records = store.results(build_id,
                                where=[("status", "in", list(FAILED_STATUSES))])
        by_case = {r.get("case_id"): r for r in records}
        split_clusters = cluster_results([by_case[cid] for cid in movable
                                          if by_case.get(cid)])
        # 签名相同的新分组需要新 id；不同签名则可沿用其签名簇 id
        old_sigs = set(target.get("signatures")
                       or ([target["signature"]] if target.get("signature") else []))
        reassign = {}
        split_targets: list[tuple[dict, str]] = []
        for sc in split_clusters:
            gid = sc["group_id"]
            if sc["signature"] in old_sigs:
                gid = new_id("clusp")
            split_targets.append((sc, gid))
            for cid in sc["case_ids"]:
                reassign[cid] = gid

        existing = self._override.query(
            where=[("project_id", "eq", project_id), ("group_id", "eq", group_id)])
        if existing:
            ov = existing[0]
            removed = sorted(set(ov.get("removed_case_ids", [])) | set(movable))
            reassigned = dict(ov.get("reassign_case_groups") or {})
            reassigned.update(reassign)
            case_left = [cid for cid in ov.get("case_ids", target["case_ids"])
                         if cid not in movable]
            self._override.update(ov["id"], {
                "removed_case_ids": removed,
                "reassign_case_groups": reassigned,
                "case_ids": case_left,
                "updated_at": time.time(),
            })
        else:
            self._override.insert({
                "id": new_id("fovr"), "project_id": project_id,
                "build_id": build_id, "group_id": group_id,
                "kind": "split", "title": target.get("title"),
                "member_group_ids": [], "case_ids": [cid for cid in target["case_ids"]
                                                     if cid not in movable],
                "removed_case_ids": movable,
                "reassign_case_groups": reassign,
                "defect_id": target.get("defect_id"),
                "created_at": time.time(),
            })
        # 给每个拆出的新簇写自己的覆盖条目：否则 _reassemble 找不到
        # 它的覆盖，会把它当成自动簇，再被同签名锚点牵走 / 与原簇混淆。
        # 默认不携带缺陷关联——拆分表达「另行看待」，让用户显式关联。
        for sc, gid in split_targets:
            exists = self._override.query(
                where=[("project_id", "eq", project_id), ("group_id", "eq", gid)])
            if exists:
                continue
            self._override.insert({
                "id": new_id("fovr"), "project_id": project_id,
                "build_id": build_id, "group_id": gid,
                "kind": "split_child", "title": sc.get("title"),
                "member_group_ids": [], "case_ids": sc["case_ids"],
                "removed_case_ids": [], "reassign_case_groups": {},
                "defect_id": None, "created_at": time.time(),
            })

        # 锚点在合并时指向合并簇；拆分后保持不动——拆出的子簇自带
        # 覆盖条目（defect_id=None），归属由覆盖决定；共享签名的锚点
        # 留在主簇，等用户对子簇显式关联时再由 assign_defect 处理。
        return self.cluster_build(project_id, build_id)

    def assign_defect(self, project_id: str, build_id: str, group_id: str,
                      defect_id: Optional[str]) -> dict:
        """人工修改簇关联的缺陷（传 null/空串解除关联）。"""
        snap = self.cluster_build(project_id, build_id)
        if "error" in snap:
            return snap
        target = next((c for c in snap["clusters"]
                       if c["group_id"] == group_id), None)
        if target is None:
            return {"error": "簇不存在"}
        if defect_id:
            defect = self.defects.get(defect_id)
            if defect is None or defect.get("project_id") != project_id:
                return {"error": "缺陷不存在"}
        else:
            defect_id = None

        existing = self._override.query(
            where=[("project_id", "eq", project_id), ("group_id", "eq", group_id)])
        patch = {"defect_id": defect_id, "updated_at": time.time()}
        if existing:
            self._override.update(existing[0]["id"], patch)
        else:
            # 只改关联也冻结该簇：之后重新聚类不能覆盖人工判断
            self._override.insert({
                "id": new_id("fovr"), "project_id": project_id,
                "build_id": build_id, "group_id": group_id,
                "kind": "assign", "title": target.get("title"),
                "member_group_ids": [], "case_ids": target["case_ids"],
                "removed_case_ids": [], "reassign_case_groups": {},
                **patch,
            })
        # 锚点记忆：这些签名以后都归这个缺陷。
        # 若某签名同时被本构建里另一个仍关联缺陷的簇使用（合并后拆分
        # 的典型场景），锚点必须留在成员更多的主簇上，不能被拆出的
        # 小簇抢走——否则小簇解除关联时会被主簇锚点重新牵回。
        sigs = target.get("signatures") or (
            [target["signature"]] if target.get("signature") else [])
        occupied = self._anchor_signatures_of_other_clusters(
            project_id, build_id, group_id)
        for sig in sigs:
            if defect_id and sig in occupied:
                continue
            self._remember_anchor(project_id, sig, group_id, defect_id)
        if defect_id:
            # 缺陷的失败特征里补上这些签名，供其它簇 / 其它构建匹配
            self.defects.add_signatures(defect_id, sigs)
        elif target.get("defect_id"):
            # 解除关联：把该簇贡献的签名从缺陷特征中拿掉，避免下一轮
            # 聚类又通过缺陷自带签名把它匹回来。签名可能被同场构建里
            # 的另一个簇共享（合并后又拆分），那些仍有簇关联的签名
            # 必须保留。共享判定用本次操作前的快照 snap——不能重新
            # cluster_build，否则旧签名还没移除，会把自己也算成共享。
            shared = set()
            for other in snap.get("clusters", []):
                if other["group_id"] != group_id and other.get("defect_id"):
                    shared.update(other.get("signatures") or
                                  ([other["signature"]]
                                   if other.get("signature") else []))
            removable = [s for s in sigs if s not in shared]
            if removable:
                old = self.defects.get(target["defect_id"])
                if old:
                    kept = [s for s in (old.get("signatures") or [])
                            if s not in removable]
                    self.defects.update(target["defect_id"],
                                        {"signatures": kept})
        return self.cluster_build(project_id, build_id)

    def reset(self, project_id: str, build_id: str, group_id: str) -> dict:
        """清除某簇的人工调整，恢复自动聚类。"""
        existing = self._override.query(
            where=[("project_id", "eq", project_id), ("group_id", "eq", group_id)])
        for ov in existing:
            self._override.delete(ov["id"])
        return self.cluster_build(project_id, build_id)

    def create_defect_for_cluster(self, project_id: str, build_id: str,
                                  group_id: str, payload: Optional[dict] = None) -> dict:
        """把一个簇直接转成缺陷（整簇共用一条，而非每用例一条）。

        若簇签名已命中某条未关闭缺陷，则直接关联到既有缺陷，
        除非显式 ``force=true``——防止人工重复造缺陷。
        """
        payload = payload or {}
        snap = self.cluster_build(project_id, build_id)
        if "error" in snap:
            return snap
        target = next((c for c in snap["clusters"]
                       if c["group_id"] == group_id), None)
        if target is None:
            return {"error": "簇不存在"}
        if target.get("defect_id") and not payload.get("force"):
            return {"error": "该簇已关联缺陷，请先解除关联或改用「关联到其他缺陷」"}
        defect = self.defects.create_from_cluster(project_id, target, build_id,
                                                  payload)
        self.assign_defect(project_id, build_id, group_id, defect["id"])
        return self.cluster_build(project_id, build_id)

    # ================================================================== 跨构建失败点
    def failure_points(self, project_id: str, limit: int = 50) -> dict:
        """跨构建聚合失败点：同一签名在多场构建里反复出现的情况。"""
        occurrences = self._occ.query(
            where=[("project_id", "eq", project_id)],
            order_by="build_finished_at", order="desc")
        anchors = self._anchor_map(project_id)

        points: dict[str, dict] = {}
        for o in occurrences:
            sig = o.get("signature") or ""
            p = points.setdefault(sig, {
                "signature": sig,
                "group_id": o.get("group_id"),
                "dim": o.get("dim"),
                "title": o.get("reason") or sig,
                "occurrences": 0,
                "failed_cases": 0,
                "build_ids": [],
                "builds": [],
                "last_build_id": o.get("build_id"),
                "last_seen": o.get("build_finished_at"),
                "first_seen": o.get("build_finished_at"),
                "defect_id": o.get("defect_id"),
                "case_names": [],
            })
            p["occurrences"] += 1
            p["failed_cases"] += 1
            if o.get("build_id") not in p["build_ids"]:
                p["build_ids"].append(o["build_id"])
                p["builds"].append({
                    "build_id": o["build_id"],
                    "finished_at": o.get("build_finished_at"),
                })
            p["first_seen"] = min(p["first_seen"] or o["build_finished_at"],
                                  o.get("build_finished_at") or 0)
            if (o.get("build_finished_at") or 0) >= (p["last_seen"] or 0):
                p["last_seen"] = o.get("build_finished_at")
                p["last_build_id"] = o.get("build_id")
                p["title"] = o.get("reason") or p["title"]
            if o.get("defect_id"):
                p["defect_id"] = o["defect_id"]
            name = o.get("case_name")
            if name and name not in p["case_names"]:
                p["case_names"].append(name)

        for sig, p in points.items():
            p["build_count"] = len(p["build_ids"])
            anchor = anchors.get(sig)
            if anchor and anchor.get("defect_id"):
                p["defect_id"] = anchor["defect_id"]
            if p.get("defect_id"):
                defect = self.defects.get(p["defect_id"])
                if defect:
                    p["defect"] = {"id": defect["id"], "title": defect.get("title"),
                                   "status": defect.get("status"),
                                   "severity": defect.get("severity")}
            p["case_names"] = p["case_names"][:5]

        rows = sorted(points.values(),
                      key=lambda p: (p["build_count"], p["last_seen"] or 0),
                      reverse=True)
        return {"project_id": project_id, "points": rows[:limit],
                "total_points": len(rows)}
