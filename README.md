## LLM_Node for ComfyUI
The `LLM_Node` enhances ComfyUI by integrating advanced language model capabilities, enabling a wide range of NLP tasks such as text generation, content summarization, question answering, and more. This flexibility is powered by various transformer model architectures from the transformers library, allowing for the deployment of models like T5, GPT-2, and others based on your project's needs.

## Features
- **Versatile Text Generation:** Leverage state-of-the-art transformer models for dynamic text generation, adaptable to a wide range of NLP tasks.
- **Customizable Model and Tokenizer Paths:** Specify paths within the `models/LLM_checkpoints` directory for using specialized models tailored to specific tasks.
- **Dynamic Token Limit and Generation Parameters:** Control the length of generated content and fine-tune generation with parameters such as `temperature`, `top_p`, `top_k`, and `repetition_penalty`.
- **Device-Specific Optimizations:** Automatically utilize `bfloat16` on compatible CUDA devices for enhanced performance.
- **Seamless Integration:** Designed for easy integration into ComfyUI workflows, enriching applications with powerful NLP functionalities.

## Installation
This Node is designed for use within ComfyUI. Ensure ComfyUI is installed and operational in your environment. 

1. **Prepare the Models Directory:**
   - Create a `LLM_checkpoints` directory within the `models` directory of your ComfyUI environment.
   - Place your transformer model directories in `LLM_checkpoints`. Each directory should contain the necessary model and tokenizer files.

2. **Node Integration:**
   - Copy the `LLM_Node` class file into the `custom_nodes` directory accessible by your ComfyUI project.

## Configuration
Configure the LLM_Node with the necessary parameters within your ComfyUI project to utilize its capabilities fully:

- `text`: The input text for the language model to process.
- `model`: The directory name of the model within `models/LLM_checkpoints` you wish to use.
- `max_tokens`: Maximum number of tokens for the generated text, adjustable according to your needs.
- `temperature`, `top_p`, `top_k`, `repetition_penalty`: Fine-tune the text generation process with these parameters to control creativity and diversity.
- `trust_remote_code`: A boolean flag to allow or prevent the execution of remote code within loaded models, enhancing security.
- `torch_dtype`: Specify the preferred tensor data type (`"float32"`, `"bfloat16"`, `"float16"`, `"float64"`, or `"auto"` for automatic selection based on device capabilities).

## Advanced Configuration Parameters

The `LLM_Node` offers a range of configurable parameters allowing for precise control over the text generation process and model behavior. Below is a detailed overview of these parameters:

- **Temperature (`temperature`):** Controls the randomness in the text generation process. Lower values make the model more confident in its predictions, leading to less variability in output. Higher values increase diversity but can also introduce more randomness. Default: `1.0`.

- **Top-p (`top_p`):** Also known as nucleus sampling, this parameter controls the cumulative probability distribution cutoff. The model will only consider the top p% of tokens with the highest probabilities for sampling. Reducing this value helps in controlling the generation quality by avoiding low-probability tokens. Default: `0.9`.

- **Top-k (`top_k`):** Limits the number of highest probability tokens considered for each step of the generation. A value of `0` means no limit. This parameter can prevent the model from focusing too narrowly on the top choices, promoting diversity in the generated text. Default: `50`.

- **Repetition Penalty (`repetition_penalty`):** Adjusts the likelihood of tokens that have already appeared in the output, discouraging repetition. Values greater than `1` penalize tokens that have been used, making them less likely to appear again. Default: `1.2`.

- **Trust Remote Code (`trust_remote_code`):** A security parameter that allows or prevents the execution of remote code within loaded models. It is crucial for safely using models from untrusted or unknown sources. Setting this to `True` may introduce security risks. Default: `False`.

- **Torch Data Type (`torch_dtype`):** Specifies the tensor data type for calculations within the model. Options include `"float32"`, `"bfloat16"`, `"float16"`, `"float64"`, or `"auto"` for automatic selection based on device capabilities. Using `"bfloat16"` or `"float16"` can significantly reduce memory usage and increase computation speed on compatible hardware. Default: `"auto"`.

These parameters provide granular control over the text generation capabilities of the `LLM_Node`, allowing users to fine-tune the behavior of the underlying models to best fit their application requirements.


## Contributing
Contributions to enhance the LLM_Node, add support for more models, or improve functionality are welcome. Please adhere to the project's contribution guidelines when submitting pull requests.

## License
The LLM_Node is released under the MIT License. Feel free to use and modify it for your personal or commercial projects.

## Acknowledgments
- Special thanks to the open-source community and the developers behind the transformers library for providing the foundational tools that make this Node possible.
- Appreciation to the ComfyUI team for their support and contributions to integrating complex NLP functionalities seamlessly.
