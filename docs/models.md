# Managing Models

## Listing Supported Models

To see all vision and language models supported by MLX:

```python
import requests

url = "http://localhost:8000/v1/supported_models"
response = requests.get(url)
print(response.json())
```

## Listing Available Models

To see all available models:

```python
import requests

url = "http://localhost:8000/v1/models"
response = requests.get(url)
print(response.json())
```

### Deleting Models

To remove any models loaded to memory:

```python
import requests

url = "http://localhost:8000/v1/models"
params = {
   "model_name": "hf-repo-or-path",
}
response = requests.delete(url, params=params)
print(response)
```
