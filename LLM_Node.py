from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig
)
import os
import folder_paths

GLOBAL_MODELS_DIR = os.path.join(folder_paths.models_dir, "LLM_checkpoints")

class LLM_Node:
    def __init__(self, device="cuda"):
        self.device = device

    @classmethod
    def INPUT_TYPES(cls):
        # Get a list of directories in the checkpoints_path
        model_options = [name for name in os.listdir(GLOBAL_MODELS_DIR)
                         if os.path.isdir(os.path.join(GLOBAL_MODELS_DIR, name))]

        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "model": (model_options, ),
                "max_tokens": ("INT", {"default": 2000}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    OUTPUT_NODE = False
    FUNCTION = "llm_text"
    CATEGORY = "LLM/text"

    def llm_text(self, text, model, max_tokens: int = 2000):
        model_path = os.path.join(GLOBAL_MODELS_DIR, model)

        # Load the model and tokenizer based on the model's configuration
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Loading the model based on its type, with trust_remote_code=True
        if config.model_type == "t5":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        elif config.model_type in ["gpt2"]:
            model = AutoModelForCausalLM.from_pretrained(model_path)
        elif config.model_type == "bert":
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

        model.to(self.device)

        if config.model_type in ["t5", "gpt2"]:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            outputs = model.generate(input_ids, max_length=max_tokens)
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
