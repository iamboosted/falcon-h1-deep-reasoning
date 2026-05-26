import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = "tiiuae/Falcon-H1-1.5B-Deep-Instruct"
OUTPUT_DIR = "/workspace/falcon-h1-1.5b-deep-reasoning"

# === Load model in 4-bit ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["mamba.out_proj"],  # Falcon-H1 requirement
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# === LoRA config targeting both attention AND mamba layers ===
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",       # feed_forward
        "in_proj",                                     # mamba
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === Dataset: MetaMathQA (MIT license, math reasoning) ===
print("Loading dataset...")
dataset = load_dataset("meta-math/MetaMathQA", split="train")
dataset = dataset.shuffle(seed=42).select(range(5000))  # Small subset for quick training

def format_prompt(example):
    return {
        "text": f"<|user|>\n{example['query']}\n<|assistant|>\n{example['response']}{tokenizer.eos_token}"
    }

dataset = dataset.map(format_prompt)

# === Training config ===
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    bf16=True,
    max_length=1024,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="text",
    report_to="none",
)

# === Train ===
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()

# === Save ===
print("Saving adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done! Adapter saved to {OUTPUT_DIR}")
