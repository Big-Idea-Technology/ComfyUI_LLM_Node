# LLM_Node for ComfyUI

LLM_Node enhances ComfyUI by integrating advanced language model capabilities, enabling tasks such as text generation, content summarization, and question answering. This flexibility is powered by the renowned T5 model architecture and other transformer models, depending on the specific deployment.

## Features

- **Text Generation:** Utilize advanced transformer models for dynamic text creation.
- **Customizable Model Integration:** Specify paths within the `models/LLM_checkpoints` directory for using specialized models tailored to specific NLP tasks.
- **Control Over Content Length:** Manage the length of generated content with an adjustable token limit.
- **Seamless Workflow Integration:** Designed for easy integration with ComfyUI workflows, enhancing applications with powerful NLP functionalities.

## Installation and Setup

Ensure ComfyUI is installed and operational in your environment before adding the LLM_Node. Follow these steps for installation:

1. **Prepare the Models Directory:**
   - Create a directory named `LLM_checkpoints` within the `models` directory of your ComfyUI environment.
   - Place your transformer model directories or files in `LLM_checkpoints`. Each model directory should contain the necessary model and tokenizer files.

2. **Node Integration:**
   - Copy the `LLM_Node` class file into the `custom_nodes` directory accessible by your ComfyUI project.

3. **Configuration:**
   - Within your ComfyUI project, configure the LLM_Node with the necessary parameters:
     - `text`: The input text for the language model to process.
     - `model`: The directory name of the model within `models/LLM_checkpoints` you wish to use.
     - `max_tokens`: The maximum number of tokens for the generated text (adjustable according to your needs).

The node is now ready to enrich your ComfyUI applications with a variety of NLP capabilities.

## Contributing

Contributions to enhance the LLM_Node or add new functionalities are welcome. Please adhere to the project's coding standards and submit pull requests for review.

## License

The LLM_Node is released under the MIT License. Feel free to use and modify it for your personal or commercial projects.

## Acknowledgments

Special thanks to the open-source community and the contributors to the transformers library for providing the foundational tools that make this Node possible.
