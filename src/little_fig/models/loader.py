from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class FigLanguageModel:
    @staticmethod
    def from_pretrained(model_name):
        print(f"🍐 Little Fig: Loading {model_name} onto CPU...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        return model, tokenizer
