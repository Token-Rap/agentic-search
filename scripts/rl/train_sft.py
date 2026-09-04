"""GRPO 之前的冷启动 SFT。

为什么需要这一步：0.5B 基础模型零样本路由错 70%（实测），于是 GRPO 每组
4 个采样几乎总是同分（都判「路由错」），组内无差异 → 优势为零 → 梯度为零，
`frac_reward_zero_std` 实测 0.8~0.9。RL 需要正样本才有梯度，模型太弱时探不到。

分工：SFT 负责把输出格式和基本路由学会；GRPO 负责 SFT 给不了的东西 ——
检索 query 的质量（SFT 的监督目标只能是「照抄问题」，而 GRPO 的奖励来自
那个 query 真实检索的命中与否，能找到比原问题更好的检索式）。

用法：
    uv run python scripts/rl/train_sft.py --epochs 3
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from eval_router import ROUTER_PROMPT  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "rl" / "routes_train.jsonl"


def target_text(row):
    """监督目标。本地/web 的 query 用问题本身 —— 本地类已由 gen_dataset
    的检索验证确认「用原问题能检出目标文档」，所以这是个可达的目标。"""
    if row["route"] == "direct":
        return "route: direct\nquery:"
    return f"route: {row['route']}\nquery: {row['question']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "models" / "router-sft"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")

    rows = [json.loads(l) for l in TRAIN_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    ds = Dataset.from_list([
        {"messages": [
            {"role": "user",
             "content": ROUTER_PROMPT.format(question=r["question"])},
            {"role": "assistant", "content": target_text(r)},
        ]}
        for r in rows
    ])
    print(f"SFT 训练集 {len(ds)} 条", flush=True)

    cfg = SFTConfig(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        learning_rate=a.lr,
        logging_steps=20,
        save_strategy="no",
        bf16=True,
        report_to=[],
        max_length=512,
    )
    trainer = SFTTrainer(
        model=a.base_model,
        args=cfg,
        train_dataset=ds,
        peft_config=LoraConfig(r=a.lora_r, lora_alpha=a.lora_r * 2,
                               lora_dropout=0.05, task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(a.out)
    print(f"模型 -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
