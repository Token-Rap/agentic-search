"""合并手写评测集并标注本地样本能否被真实检索命中。

为什么要标注而不是直接丢：
路由准确率与检索命中是两个不同的指标，样本要求也不同。
- 路由准确率：问的是「该走哪条通道」，问题只要主题属于内部资料即可，
  语料能不能答上来无关。
- 检索命中：需要目标文档真的能被检出，否则这条样本的命中与否是噪声。

所以本地样本统一保留（用于路由准确率），额外打 `hit_eligible` 标记，
命中率只在标记为真的子集上算，并在报告里写明该子集大小。
"""

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from retriever_env import make_retriever, retrieval_hits  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = [ROOT / "data" / "rl" / "routes.jsonl",
       ROOT / "data" / "rl" / "routes_extra.jsonl"]
OUT = ROOT / "data" / "rl" / "routes_eval.jsonl"
TOP_K = 4


def norm(q):
    return re.sub(r"[\s,，。?？!！、:：;；]+", "", q).lower()


def main():
    rows, seen, dup = [], set(), 0
    for path in SRC:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            k = norm(r["question"])
            if k in seen:
                dup += 1
                continue
            seen.add(k)
            rows.append(r)

    retriever = make_retriever()
    eligible = 0
    for r in rows:
        if r["route"] == "retrieve_docs" and r.get("expected_source"):
            ok = retrieval_hits(retriever, r["question"], r["expected_source"], TOP_K)
            r["hit_eligible"] = bool(ok)
            eligible += ok
        else:
            r["hit_eligible"] = False

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by = {}
    for r in rows:
        by[r["route"]] = by.get(r["route"], 0) + 1
    n_local = by.get("retrieve_docs", 0)
    print(f"合并 {len(rows)} 条 -> {OUT.name}   (去重丢弃 {dup})")
    print(f"分布 {by}")
    print(f"检索命中可用子集 {eligible}/{n_local} 条本地样本")


if __name__ == "__main__":
    main()
