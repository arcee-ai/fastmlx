This example demonstrates how to create a chatbot application using FastMLX with a Gradio interface.

```python

import argparse
import gradio as gr
import requests
import json

import asyncio

async def process_sse_stream(url, headers, data):
    response = requests.post(url, headers=headers, json=data, stream=True)
    if response.status_code != 200:
        raise gr.Error(f"Error: Received status code {response.status_code}")
    full_content = ""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                event_data = line[6:]  # Remove 'data: ' prefix
                if event_data == '[DONE]':
                    break
                try:
                    chunk_data = json.loads(event_data)
                    content = chunk_data['choices'][0]['delta']['content']
                    yield str(content)
                except (json.JSONDecodeError, KeyError):
                    continue

async def chat(message, history, temperature, max_tokens):

    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "messages": [{"role": "user", "content": message['text']}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }

    if len(message['files']) > 0:
        data["model"] = "mlx-community/nanoLLaVA-1.5-8bit"
        data["image"] = message['files'][-1]["path"]

    response = requests.post(url, headers=headers, json=data, stream=True)
    if response.status_code != 200:
        raise gr.Error(f"Error: Received status code {response.status_code}")

    full_content = ""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                event_data = line[6:]  # Remove 'data: ' prefix
                if event_data == '[DONE]':
                    break
                try:
                    chunk_data = json.loads(event_data)
                    content = chunk_data['choices'][0]['delta']['content']
                    full_content += content
                    yield full_content
                except (json.JSONDecodeError, KeyError):
                    continue

demo = gr.ChatInterface(
    fn=chat,
    title="FastMLX Chat UI",
    additional_inputs_accordion=gr.Accordion(
        label="⚙️ Parameters", open=False, render=False
    ),
    additional_inputs=[
        gr.Slider(
            minimum=0, maximum=1, step=0.1, value=0.1, label="Temperature", render=False
        ),
        gr.Slider(
            minimum=128,
            maximum=4096,
            step=1,
            value=200,
            label="Max new tokens",
            render=False
        ),
    ],
    multimodal=True,
)

demo.launch(inbrowser=True)
```