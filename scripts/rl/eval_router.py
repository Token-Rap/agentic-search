"""评测路由决策：基线（9b）与微调后的小模型走同一套判据。

两个指标：
  路由准确率     —— 三分类是否选对通道
  端到端检索命中 —— 选了 retrieve_docs 时，它生成的 query 能否检出目标文档
                    （只在本地类样本上算；这一项才是「路由有用」的实证）

用法：
    uv run python scripts/rl/eval_router.py --backend ollama          # 9b 基线
    uv run python scripts/rl/eval_router.py --backend hf --model out/router
"""

import argparse
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from retriever_env import ROUTES, make_retriever, retrieval_hits  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
# merge_eval.py 产出的合并集；缺失时退回手写小集。
EVAL_PATH = ROOT / "data" / "rl" / "routes_eval.jsonl"
if not EVAL_PATH.exists():
    EVAL_PATH = ROOT / "data" / "rl" / "routes.jsonl"


def wilson(k, n, z=1.96):
    """Wilson 区间。n 小或比例贴近 1 时，正态近似会给出越界且过窄的区间。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)

ROUTER_PROMPT = """你是问答系统的路由器。给定用户问题，只输出该走哪条通道。

通道：
- retrieve_docs：本组织内部知识库能回答的问题（咖啡馆的菜单、价格、会员制度、\
营业时间、活动规则、供应商、会议纪要、新品评审）
- web_search：时效性问题（天气、汇率、股价、最新版本、新闻）或公共常识
- direct：纯寒暄闲聊，不需要查任何资料

严格按下面两行格式输出，不要解释：
route: <retrieve_docs|web_search|direct>
query: <检索用的关键词；route 为 direct 时留空>

问题：{question}"""


def parse(text):
    """从模型输出里抠出 route 与 query。格式不合法返回 (None, "")。"""
    route, query = None, ""
    for line in text.splitlines():
        m = re.match(r"\s*route\s*[:：]\s*(\S+)", line, re.I)
        if m and route is None:
            cand = m.group(1).strip().strip("`\"'")
            if cand in ROUTES:
                route = cand
        m = re.match(r"\s*query\s*[:：]\s*(.+)", line, re.I)
        if m and not query:
            query = m.group(1).strip()
    return route, query


def make_ollama_fn():
    from agentic_search.llm import make_chat_llm

    llm = make_chat_llm()
    return lambda q: llm.invoke(ROUTER_PROMPT.format(question=q)).content


def make_hf_fn(model_dir, base_model, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if model_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_dir)
    model.eval()

    def fn(q):
        msgs = [{"role": "user", "content": ROUTER_PROMPT.format(question=q)}]
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("ollama", "hf"), default="ollama")
    ap.add_argument("--model", default=None, help="hf 后端的 LoRA 目录")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in EVAL_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    router = (make_ollama_fn() if a.backend == "ollama"
              else make_hf_fn(a.model, a.base_model, a.max_new_tokens))
    retriever = make_retriever()

    n_route_ok = n_parse_fail = n_infer_err = 0
    per_class = collections.defaultdict(lambda: [0, 0])   # route -> [对, 总]
    confusion = collections.Counter()
    local_total = local_hit = 0

    for r in rows:
        gold = r["route"]
        try:
            pred, query = parse(router(r["question"]))
            n_infer_err = 0
        except Exception as exc:
            # 逐条吞异常会让「后端根本没起来」跑成一份看似完整的报告
            # （实测：Ollama 未启动时 216 条全失败仍正常结束）。连续失败就停。
            n_infer_err += 1
            print(f"  推理失败({n_infer_err}): {exc}", flush=True)
            if n_infer_err >= 5:
                raise SystemExit(
                    f"连续 {n_infer_err} 条推理失败，后端多半没起来 —— 中止，"
                    f"不产出半份结果"
                )
            pred, query = None, ""
        if pred is None:
            n_parse_fail += 1
        per_class[gold][1] += 1
        confusion[(gold, pred or "PARSE_FAIL")] += 1
        if pred == gold:
            n_route_ok += 1
            per_class[gold][0] += 1
        # 命中率只在「目标文档确实检得到」的子集上算；merge_eval.py 标注。
        # 未标注的老格式默认参与，保持向后兼容。
        if gold == "retrieve_docs" and r.get("hit_eligible", True):
            local_total += 1
            # 用模型自己生成的 query 检索；没给 query 就退回原问题，
            # 否则「不写 query」反而占便宜。
            if pred == gold and retrieval_hits(
                retriever, query or r["question"], r["expected_source"], a.top_k
            ):
                local_hit += 1

    label = a.label or (f"ollama:{a.backend}" if a.backend == "ollama"
                        else f"hf:{a.model or a.base_model}")
    n = len(rows)
    print("=" * 62)
    print(f"路由器        : {label}")
    print(f"评测集        : {n} 条  ({EVAL_PATH.name})")
    print("-" * 62)
    lo, hi = wilson(n_route_ok, n)
    print(f"路由准确率     {n_route_ok}/{n} = {n_route_ok / n:.1%}"
          f"   95% CI [{lo:.1%}, {hi:.1%}]")
    for route in ROUTES:
        ok, tot = per_class[route]
        if tot:
            print(f"  {route:<15} {ok}/{tot} = {ok / tot:.1%}")
    print(f"格式解析失败   {n_parse_fail}/{n}")
    hlo, hhi = wilson(local_hit, local_total)
    print(f"端到端检索命中 {local_hit}/{local_total} = "
          f"{local_hit / max(local_total, 1):.1%}"
          f"   95% CI [{hlo:.1%}, {hhi:.1%}]   (k={a.top_k})")
    # 说清这个样本量能分辨到什么程度：CI 下界就是「不比它更差」的最强断言。
    print(f"分辨率上限     n={n} 时，准确率低于 {lo:.1%} 才可能被判定为显著变差；"
          f"更小的差距只能说「未测出差异」")
    print("-" * 62)
    print("混淆（真实 -> 预测，只列出错的）：")
    for (gold, pred), c in sorted(confusion.items(), key=lambda x: -x[1]):
        if gold != pred:
            print(f"  {gold:>14} -> {pred:<14} {c}")
    print("=" * 62)


if __name__ == "__main__":
    main()
