"""用 GRPO 微调一个小模型做检索路由。

奖励里含一次**真实检索**，不是只比对标签：模型除了选通道，还要为
retrieve_docs 生成检索 query，奖励取决于那个 query 能否检出目标文档。
纯标签匹配的话这就是个分类任务，用 SFT 即可，GRPO 没有立足点。

    路由错 / 格式不合法        0.0 ~ 0.2
    路由对（web / direct）      1.0
    路由对（本地）+ 检索未命中  1.0
    路由对（本地）+ 检索命中    2.0

用法：
    uv run python scripts/rl/train_grpo.py --steps 300
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from eval_router import ROUTER_PROMPT, parse  # noqa: E402
from retriever_env import make_retriever, retrieval_hits  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "rl" / "routes_train.jsonl"

R_PARSE_FAIL = 0.0
R_WRONG_ROUTE = 0.2
R_ROUTE_OK = 1.0
R_RETRIEVAL_HIT = 2.0


def build_reward(retriever, top_k, stats):
    """GRPO 的奖励函数。

    TRL 按数据集列名把额外字段传进 kwargs，这里要的是 route 与
    expected_source —— 二者必须和 completions 一一对应，顺序不能乱。
    """

    def reward(completions, route, expected_source, **_):
        out = []
        for comp, gold, src in zip(completions, route, expected_source):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            pred, query = parse(text)
            stats["n"] += 1
            if pred is None:
                stats["parse_fail"] += 1
                out.append(R_PARSE_FAIL)
                continue
            if pred != gold:
                stats["wrong_route"] += 1
                out.append(R_WRONG_ROUTE)
                continue
            stats["route_ok"] += 1
            if gold != "retrieve_docs" or not src:
                out.append(R_ROUTE_OK)
                continue
            # query 为空就退回原问题会让「不写 query」白占便宜，这里直接判未命中。
            hit = bool(query) and retrieval_hits(retriever, query, src, top_k)
            stats["hit"] += int(hit)
            stats["local"] += 1
            out.append(R_RETRIEVAL_HIT if hit else R_ROUTE_OK)
        return out

    return reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "models" / "router-grpo"))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    a = ap.parse_args()

    if not TRAIN_PATH.exists():
        raise SystemExit(f"训练集不存在：{TRAIN_PATH}\n"
                         f"先跑 uv run python scripts/rl/gen_dataset.py")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用 —— 装的多半是 CPU-only 的 torch，"
                         "见 pyproject 里 [project.optional-dependencies].rl 的说明")

    rows = [json.loads(l) for l in TRAIN_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    ds = Dataset.from_list([
        {
            "prompt": [{"role": "user",
                        "content": ROUTER_PROMPT.format(question=r["question"])}],
            "route": r["route"],
            # 列必须定长且类型一致，非本地类补空串而不是缺列。
            "expected_source": r.get("expected_source", ""),
        }
        for r in rows
    ])
    print(f"训练集 {len(ds)} 条", flush=True)

    retriever = make_retriever()
    stats = {"n": 0, "parse_fail": 0, "wrong_route": 0,
             "route_ok": 0, "hit": 0, "local": 0}

    cfg = GRPOConfig(
        output_dir=a.out,
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch,
        num_generations=a.num_generations,
        max_completion_length=a.max_new_tokens,
        max_steps=a.steps,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=a.base_model,
        reward_funcs=build_reward(retriever, a.top_k, stats),
        args=cfg,
        train_dataset=ds,
        peft_config=LoraConfig(r=a.lora_r, lora_alpha=a.lora_r * 2,
                               lora_dropout=0.05, task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(a.out)

    n = max(stats["n"], 1)
    print("=" * 56)
    print(f"采样总数 {stats['n']}")
    print(f"  格式失败 {stats['parse_fail']} ({stats['parse_fail'] / n:.1%})")
    print(f"  路由错   {stats['wrong_route']} ({stats['wrong_route'] / n:.1%})")
    print(f"  路由对   {stats['route_ok']} ({stats['route_ok'] / n:.1%})")
    if stats["local"]:
        print(f"  本地类检索命中 {stats['hit']}/{stats['local']} "
              f"= {stats['hit'] / stats['local']:.1%}")
    print(f"模型 -> {a.out}")
    print("=" * 56)


if __name__ == "__main__":
    main()
