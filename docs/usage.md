# Usage

This guide covers the server setup, and usage of FastMLX, including making API calls and managing models.

## 1. Installation
Follow the [installation guide](installation.md) to install FastMLX.

## 2. Running the server
Start the FastMLX server with the following command:

   ```bash
   fastmlx
   ```

or 

Using `uvicorn` directly:

   ```bash
   uvicorn fastmlx:app --reload --workers 0
   ```

   > [!WARNING]
   > The `--reload` flag should not be used in production. It is only intended for development purposes.

### Running with Multiple Workers (Parallel Processing)

For improved performance and parallel processing capabilities, you can specify either the absolute number of worker processes or the fraction of CPU cores to use.

You can set the number of workers in three ways (listed in order of precedence):

1. Command-line argument:
   ```bash
   fastmlx --workers 4
   ```
   or
   ```bash
   uvicorn fastmlx:app --workers 4
   ```

2. Environment variable:
   ```bash
   export FASTMLX_NUM_WORKERS=4
   fastmlx
   ```

3. Default value (2 workers)

To use all available CPU cores, set the value to 1.0:

```bash
fastmlx --workers 1.0
```

> [!NOTE]
> - The `--reload` flag is not compatible with multiple workers.
> - The number of workers should typically not exceed the number of CPU cores available on your machine for optimal performance.

### Considerations for Multi-Worker Setup

1. **Stateless Application**: Ensure your FastMLX application is stateless, as each worker process operates independently.
2. **Database Connections**: If your app uses a database, make sure your connection pooling is configured to handle multiple workers.
3. **Resource Usage**: Monitor your system's resource usage to find the optimal number of workers for your specific hardware and application needs.
4. **Load Balancing**: When running with multiple workers, incoming requests are automatically load-balanced across the worker processes.

## 3. Making API Calls

Use the API similar to OpenAI's chat completions:

### Vision Language Model

#### Without Streaming
Here's an example of how to use a Vision Language Model:

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

#### Without Streaming
```python
import requests
import json

def process_sse_stream(url, headers, data):
   response = requests.post(url, headers=headers, json=data, stream=True)

   if response.status_code != 200:
      print(f"Error: Received status code {response.status_code}")
      print(response.text)
      return

   full_content = ""

   try:
      for line in response.iter_lines():
            if line:
               line = line.decode('utf-8')
               if line.startswith('data: '):
                  event_data = line[6:]  # Remove 'data: ' prefix
                  if event_data == '[DONE]':
                        print("\nStream finished. ✅")
                        break
                  try:
                        chunk_data = json.loads(event_data)
                        content = chunk_data['choices'][0]['delta']['content']
                        full_content += content
                        print(content, end='', flush=True)
                  except json.JSONDecodeError:
                        print(f"\nFailed to decode JSON: {event_data}")
                  except KeyError:
                        print(f"\nUnexpected data structure: {chunk_data}")

   except KeyboardInterrupt:
      print("\nStream interrupted by user.")
   except requests.exceptions.RequestException as e:
      print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
   url = "http://localhost:8000/v1/chat/completions"
   headers = {"Content-Type": "application/json"}
   data = {
      "model": "mlx-community/nanoLLaVA-1.5-4bit",
      "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
      "messages": [{"role": "user", "content": "What are these?"}],
      "max_tokens": 500,
      "stream": True
   }
   process_sse_stream(url, headers, data)
```
### Language Model

#### Without Streaming

Here's an example of how to use a Language Model:

```python
import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "mlx-community/gemma-2-9b-it-4bit",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 100
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```

#### With Streaming

```python
import requests
import json

def process_sse_stream(url, headers, data):
   response = requests.post(url, headers=headers, json=data, stream=True)

   if response.status_code != 200:
      print(f"Error: Received status code {response.status_code}")
      print(response.text)
      return

   full_content = ""

   try:
      for line in response.iter_lines():
            if line:
               line = line.decode('utf-8')
               if line.startswith('data: '):
                  event_data = line[6:]  # Remove 'data: ' prefix
                  if event_data == '[DONE]':
                        print("\nStream finished. ✅")
                        break
                  try:
                        chunk_data = json.loads(event_data)
                        content = chunk_data['choices'][0]['delta']['content']
                        full_content += content
                        print(content, end='', flush=True)
                  except json.JSONDecodeError:
                        print(f"\nFailed to decode JSON: {event_data}")
                  except KeyError:
                        print(f"\nUnexpected data structure: {chunk_data}")

   except KeyboardInterrupt:
      print("\nStream interrupted by user.")
   except requests.exceptions.RequestException as e:
      print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
   url = "http://localhost:8000/v1/chat/completions"
   headers = {"Content-Type": "application/json"}
   data = {
      "model": "mlx-community/gemma-2-9b-it-4bit",
      "messages": [{"role": "user", "content": "Hi, how are you?"}],
      "max_tokens": 500,
      "stream": True
   }
   process_sse_stream(url, headers, data)
```

For more detailed API documentation, please refer to the [API Reference](endpoints.md) section.