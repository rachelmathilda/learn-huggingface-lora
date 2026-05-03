from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import gradio as gr
import torch

model_id = "gpt2"

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_id,
    peft_config=lora_config,
)

def predict(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

gradio_app = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Prompt", placeholder="Enter code or text..."),
    outputs=gr.Textbox(label="Generated Output"),
    title="GPT-2 + LoRA Code Assistant",
)

if __name__ == "__main__":
    gradio_app.launch()