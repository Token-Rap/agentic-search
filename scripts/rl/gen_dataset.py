"""用本地 9b 模型扩充路由训练集。

手写的 data/rl/routes.jsonl 当**评测集**（有依据、可信但只有 72 条）；
这里生成更大的训练集。两者严格去重：训练集里出现过评测集的问题，
前后对比的数字就没有意义了。

用法：
    uv run python scripts/rl/gen_dataset.py --per-class 120
"""

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from agentic_search.llm import make_chat_llm  # noqa: E402
from retriever_env import make_retriever, retrieval_hits  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
# 必须对**实际使用的**评测集去重。merge_eval.py 产出的 routes_eval.jsonl
# 是 216 条，只对旧的 72 条去重会把另外 144 条评测样本漏进训练集。
EVAL_PATH = ROOT / "data" / "rl" / "routes_eval.jsonl"
if not EVAL_PATH.exists():
    EVAL_PATH = ROOT / "data" / "rl" / "routes.jsonl"
OUT_PATH = ROOT / "data" / "rl" / "routes_train.jsonl"

# 本地类必须落到真实文档上，否则检索命中奖励无从算起。
# 纯文本文档直接读原文喂给生成模型；PDF/音频/视频读不了原文，
# 退回描述 —— 这几类生成质量差一些，靠后面的检索验证兜底。
LOCAL_TEXT_DOCS = ("menu.md", "membership.md", "operations.md", "faq.md")
LOCAL_OPAQUE_DOCS = {
    "suppliers.pdf": "供应商名单与供应内容（含 NX-42 之类编号）",
    "meeting.wav": "例会录音纪要：季度决议，涉及新门店选址",
    "review.mp4": "新品评审视频：SP-2026 等新品展示",
}


def load_local_docs(corpus: pathlib.Path) -> dict[str, str]:
    """文档名 -> 喂给生成模型的内容（纯文本读原文，其余用描述）。"""
    docs = {}
    for name in LOCAL_TEXT_DOCS:
        p = corpus / name
        if p.exists():
            docs[name] = p.read_text(encoding="utf-8")[:2000]
    docs.update(LOCAL_OPAQUE_DOCS)
    return docs

PROMPTS = {
    "retrieve_docs": (
        "下面是一家虚构咖啡馆内部知识库中某个文档的**真实内容**：\n"
        "----\n{desc}\n----\n\n"
        "请写 {n} 个员工可能会问的、**答案就在上面这段内容里**的具体问题。"
        "要求：每行一个问题，不要编号，不要解释；"
        "**只能问上面出现过的事物**，不要虚构文档里没有的商品名、日期或规则；"
        "不要写需要联网才能答的问题。"
    ),
    "web_search": (
        "请写 {n} 个**必须联网搜索**才能回答的问题。"
        "包括两类：时效性问题（今天的天气、汇率、股价、最新版本、近期新闻）"
        "和明显属于公共常识但不在某家咖啡馆内部资料里的问题（地理、历史、"
        "科学常识）。要求：每行一个问题，不要编号，不要解释。"
    ),
    "direct": (
        "请写 {n} 句用户对客服助手说的**纯寒暄或闲聊**，"
        "不包含任何需要查资料的事实问题。例如打招呼、道谢、告别、确认收到。"
        "要求：每行一句，不要编号，不要解释。"
    ),
}


def norm(q):
    """归一化后比对，避免只差标点的重复混进训练集。"""
    return re.sub(r"[\s,，。?？!！、:：;；]+", "", q).lower()


def ask(llm, prompt):
    text = llm.invoke(prompt).content
    lines = [re.sub(r"^\s*[-*\d.、)]+\s*", "", ln).strip() for ln in text.splitlines()]
    return [ln for ln in lines if 2 <= len(ln) <= 60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--batch", type=int, default=20, help="每次请求生成条数")
    ap.add_argument("--top-k", type=int, default=4,
                    help="检索验证用的 k，应与训练奖励一致")
    ap.add_argument("--max-rounds", type=int, default=12,
                    help="每类最多请求几轮，防止生成不出新样本时空转")
    ap.add_argument("--no-verify", action="store_true",
                    help="跳过检索验证（不建议：会放进检不到的样本）")
    a = ap.parse_args()

    seen = set()
    for line in EVAL_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            seen.add(norm(json.loads(line)["question"]))
    print(f"评测集 {len(seen)} 条，训练集将与之去重", flush=True)

    llm = make_chat_llm()
    retriever = None if a.no_verify else make_retriever()
    rows, dropped, unretrievable = [], 0, 0

    def collect(route, prompt, target, source=None, desc=""):
        nonlocal dropped, unretrievable
        got = 0
        rounds = 0
        while got < target and rounds < a.max_rounds:
            rounds += 1
            n = min(a.batch, target - got)
            try:
                cands = ask(llm, prompt.format(n=n, desc=desc))
            except Exception as exc:  # 生成失败不该中断整轮
                print(f"  生成失败({route}): {exc}", flush=True)
                break
            new = 0
            for q in cands:
                k = norm(q)
                if k in seen:
                    dropped += 1
                    continue
                # 本地类必须真能检出目标文档。9b 会虚构语料里没有的商品名和
                # 规则（实测），那种样本的「检索命中」奖励是算不出来的噪声。
                if source and retriever is not None:
                    if not retrieval_hits(retriever, q, source, a.top_k):
                        unretrievable += 1
                        continue
                seen.add(k)
                row = {"question": q, "route": route}
                if source:
                    row["expected_source"] = source
                rows.append(row)
                got += 1
                new += 1
                if got >= target:
                    break
            if new == 0:  # 已经生成不出新问题，别空转
                break
        return got

    local_docs = load_local_docs(ROOT / "sample_docs")
    per_doc = max(1, a.per_class // len(local_docs))
    for doc, desc in local_docs.items():
        n = collect("retrieve_docs", PROMPTS["retrieve_docs"], per_doc,
                    source=doc, desc=desc)
        print(f"  retrieve_docs/{doc}: {n}", flush=True)

    for route in ("web_search", "direct"):
        n = collect(route, PROMPTS[route], a.per_class)
        print(f"  {route}: {n}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by = {}
    for r in rows:
        by[r["route"]] = by.get(r["route"], 0) + 1
    print(f"\n训练集 {len(rows)} 条 -> {OUT_PATH}")
    print(f"分布 {by}")
    print(f"丢弃：与评测集重复 {dropped} 条，检索验证不通过 {unretrievable} 条")


if __name__ == "__main__":
    main()
