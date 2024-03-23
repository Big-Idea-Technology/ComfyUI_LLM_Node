from transformers import T5Tokenizer, T5ForConditionalGeneration

class LLM_Node:
    def __init__(self, device="cuda"):
        self.device = device

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "tokenizer": ("STRING", {"default": "/your/path/to/llm_folder"}),
                "model": ("STRING", {"default": "/your/path/to/llm_folder"}),
                "max_tokens": ("INT", {"default": 2000}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    OUTPUT_NODE = False
    FUNCTION = "llm_text"
    CATEGORY = "LLM/text"

    def llm_text(self, text, tokenizer, model, max_tokens: int = 2000):
        # Load the tokenizer and model
        tokenizer = T5Tokenizer.from_pretrained(tokenizer)
        model = T5ForConditionalGeneration.from_pretrained(model)

        # Move the model to the GPU if available
        model = model.to(self.device)

        # Tokenize the input text
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        # Generate text with a limit on the number of new tokens
        outputs = model.generate(input_ids, max_new_tokens=max_tokens)

        # Decode the output tokens to a string
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return (generated_text,)
    
NODE_CLASS_MAPPINGS = {
    "LLM_Node": LLM_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_Node": "LLM Node",
}