# FastMLX

[![PyPI version](https://img.shields.io/pypi/v/fastmlx.svg)](https://pypi.python.org/pypi/fastmlx)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/fastmlx.svg)](https://anaconda.org/conda-forge/fastmlx)
[![Updates](https://pyup.io/repos/github/Blaizzy/fastmlx/shield.svg)](https://pyup.io/repos/github/Blaizzy/fastmlx)

**FastMLX** is a high-performance, production-ready API for hosting MLX models, including Vision Language Models (VLMs) and Language Models (LMs). It provides an easy-to-use interface for integrating powerful machine learning capabilities into your applications.


## Key Features

- **OpenAI-compatible API**: Easily integrate with existing applications that use OpenAI's API.
- **Dynamic Model Loading**: Load MLX models on-the-fly or use pre-loaded models for better performance.
- **Support for Multiple Model Types**: Compatible with various MLX model architectures.
- **Image Processing Capabilities**: Handle both text and image inputs for versatile model interactions.
- **Efficient Resource Management**: Optimized for high-performance and scalability.
- **Error Handling**: Robust error management for production environments.
- **Customizable**: Easily extendable to accommodate specific use cases and model types.

## Quick Start

[Get started with FastMLX](installation.md): Learn how to install and set up FastMLX in your environment.

Explore Examples: Hands-on guides, such as:

- [Chatbot application](examples/chatbot.md)
- [Function calling](examples/function_calling.md)

### Installation

Install **FastMLX** on your system by running the following command:

```
pip install -U fastmlx
```

### Running the Server

Start the **FastMLX** server using the following command:

```bash
fastmlx
```

or with multiple workers for improved performance:

```bash
fastmlx --workers 4
```

### Making API Calls

Once the server is running, you can interact with the API. Here's an example using a Vision Language Model:

```python
import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "mlx-community/nanoLLaVA-1.5-4bit",
    "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
    "messages": [{"role": "user", "content": "What are these"}],
    "max_tokens": 100
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```

## What's Next?

- Check out the [Installation](installation.md) guide for detailed setup instructions.
- Learn more about the API usage in the [Usage](usage.md) section.
- Explore advanced features and configurations in the [API Reference](endpoints.md).
- If you're interested in contributing, see our [Contributing](contributing.md) guidelines.

## License

FastMLX is free software, licensed under the Apache Software License 2.0.

For more detailed information and advanced usage, please explore the rest of our documentation. If you encounter any issues or have questions, don't hesitate to [report an issue](https://github.com/Blaizzy/fastmlx/issues) on our GitHub repository.

Happy coding with FastMLX!