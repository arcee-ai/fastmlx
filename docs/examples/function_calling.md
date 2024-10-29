## Function Calling

FastMLX now supports tool calling in accordance with the OpenAI API specification. This feature is available for the following models:

- Llama 3.1
- Arcee Agent
- C4ai-Command-R-Plus
- Firefunction
- xLAM

Supported modes:

- Without Streaming
- Parallel Tool Calling

> Note: Tool choice and OpenAI-compliant streaming for function calling are currently under development.

This example demonstrates how to use the `get_current_weather` tool with the `Llama 3.1` model. The API will process the user's question and use the provided tool to fetch the required information.


```python
import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
  "model": "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
  "messages": [
    {
      "role": "user",
      "content": "What's the weather like in San Francisco and Washington?"
    }
  ],
  "tools": [
    {
      "name": "get_current_weather",
      "description": "Get the current weather",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "format": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "The temperature unit to use. Infer this from the user's location."
          }
        },
        "required": ["location", "format"]
      }
    }
  ],
  "max_tokens": 150,
  "temperature": 0.7,
  "stream": False,
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```

> Note: Streaming is available for regular text generation, but the streaming implementation for function calling is still in development and does not yet fully comply with the OpenAI specification.