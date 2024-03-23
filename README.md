# LLM_Node for ComfyUI
LLM_Node is a powerful and flexible Node for ComfyUI that enables users to integrate advanced language model capabilities into their ComfyUI projects. Utilizing the T5 model architecture from the transformers library, this Node can generate text, summarize content, answer questions, and more, depending on the model you choose to deploy.

## Features
- Text generation leveraging state-of-the-art transformer models.
- Customizable model and tokenizer paths, allowing for the use of specialized models tailored to specific tasks.
- Dynamic token limit for generation, allowing control over the length of generated content.
- Simple integration into ComfyUI workflows for enriching applications with natural language processing capabilities.

## Installation
This Node is designed for use within ComfyUI. Ensure ComfyUI is installed and operational in your environment. Place the `LLM_Node` class in the `custom_nodes` directory within your ComfyUI folder. For detailed installation instructions for ComfyUI, refer to the official ComfyUI documentation.

### Add the Node to Your Project:
Ensure the LLM_Node class is correctly integrated into your ComfyUI setup. The class file should be located in a `custom_nodes` directory that is accessible by your ComfyUI project.

### Configure the Node:
Configure the LLM_Node with the necessary parameters within your ComfyUI project:

- `text`: Input text for the language model to process.
- `tokenizer`: Path to the tokenizer for the model.
- `model`: Path to the transformer model.
- `max_tokens`: Maximum number of tokens for the generated text.

These parameters allow for flexible use of the Node, whether you're generating text, summarizing content, or implementing other language-based tasks.

## License
The LLM_Node is released under the MIT License. You are free to use and modify the LLM_Node for your personal or commercial projects.

## Credit
Special thanks to the open-source community and the developers behind the transformers library for providing the tools necessary to build this Node.
