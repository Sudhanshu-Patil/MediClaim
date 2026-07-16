"""QLoRA fine-tuning of Llama 3.2 3B Instruct — sized for a free Colab T4.

QLoRA recipe (README §4: HF PEFT + QLoRA + bitsandbytes):
  * base weights frozen in 4-bit NF4 with double quantization,
  * fp16 compute (T4 has no bf16),
  * LoRA r=16 / alpha=32 on all attention + MLP projections,
  * gradient checkpointing + paged 8-bit AdamW so 3B fits in 16 GB VRAM
    with headroom.

Run (in Colab, after finetuning/data/*.jsonl exist):

    python finetuning/qlora_train.py \
        --base-model meta-llama/Llama-3.2-3B-Instruct \
        --train-file finetuning/data/train.jsonl \
        --val-file finetuning/data/val.jsonl \
        --output-dir outputs/medclaim-lora

The output directory holds only the LoRA adapter (~100 MB) — merging into
the base model and GGUF conversion happen in merge_and_convert.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--train-file", default="finetuning/data/train.jsonl")
    parser.add_argument("--val-file", default="finetuning/data/val.jsonl")
    parser.add_argument("--output-dir", default="outputs/medclaim-lora")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push-to-hub", default=None,
                        help="optional HF repo id, e.g. user/medclaim-llama3.2-3b-lora")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )
    print(f"train={len(dataset['train'])} val={len(dataset['validation'])}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # T4: fp16, not bf16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_seq_length=args.max_seq_len,
        packing=False,
        fp16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = {
        "train_loss": train_result.metrics.get("train_loss"),
        "eval": trainer.evaluate(),
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "epochs": args.epochs,
    }
    Path(args.output_dir, "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    if args.push_to_hub:
        trainer.model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"Adapter pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
