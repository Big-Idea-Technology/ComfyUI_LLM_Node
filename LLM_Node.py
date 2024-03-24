from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig
)
import torch
import os
import folder_paths

GLOBAL_MODELS_DIR = os.path.join(folder_paths.models_dir, "LLM_checkpoints")

class LLM_Node:
    def __init__(self, device="cuda"):
        self.device = device
        # Check if bfloat16 is supported by the device
        self.supports_bfloat16 = 'cuda' in device and torch.cuda.is_bf16_supported()

    @classmethod
    def INPUT_TYPES(cls):
        # Get a list of directories in the checkpoints_path
        model_options = [name for name in os.listdir(GLOBAL_MODELS_DIR)
                         if os.path.isdir(os.path.join(GLOBAL_MODELS_DIR, name))]
        dtype_options = ["auto", "float32", "bfloat16", "float16", "float64"]

        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "model": (model_options, ),
                "max_tokens": ("INT", {"default": 2000, "min": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 1}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.1, "step": 0.1}),
                "trust_remote_code": ("BOOLEAN", {"default": False}),
                "torch_dtype": (dtype_options, {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    OUTPUT_NODE = False
    FUNCTION = "llm_text"
    CATEGORY = "LLM/text"

    def llm_text(self, text, model, max_tokens, temperature, top_p, top_k, repetition_penalty, trust_remote_code, torch_dtype):
        model_path = os.path.join(GLOBAL_MODELS_DIR, model)

        if torch_dtype == "auto" or not hasattr(torch, torch_dtype):
            if 'cuda' in self.device and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
        else:
            torch_dtype = getattr(torch, torch_dtype)

        model_kwargs = {
            'trust_remote_code': trust_remote_code,
            'torch_dtype': torch_dtype
        }

        # Load the model and tokenizer based on the model's configuration
        config = AutoConfig.from_pretrained(model_path, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Dynamically loading the model based on its type
        if config.model_type == "t5":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        elif config.model_type in ["gpt2", "gpt_refact"]:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        elif config.model_type == "bert":
            model = AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

        model.to(self.device)

        if config.model_type in ["t5", "gpt2", "gpt_refact"]:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            outputs = model.generate(
                input_ids,
                max_length=max_tokens + len(input_ids[0]),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return (generated_text,)
        elif config.model_type == "bert":
            return ("BERT model detected; specific task handling not implemented in this example.",)
    
NODE_CLASS_MAPPINGS = {
    "LLM_Node": LLM_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_Node": "LLM Node",
}
