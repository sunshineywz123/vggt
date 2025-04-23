import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./"  # 替换为你使用的模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 打印模型中使用的注意力函数
for name, module in model.named_modules():
    if 'attention' in name.lower():
        print(f"{name}: {module}")
