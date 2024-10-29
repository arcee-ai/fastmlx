# CLI Reference

The **FastMLX** API server can be configured using various command-line arguments. Here is a detailed reference for each available option. 

## Usage

```
fastmlx [OPTIONS]
```

## Options

### `--allowed-origins`

- **Type**: List of strings
- **Default**: `["*"]`
- **Description**: List of allowed origins for CORS (Cross-Origin Resource Sharing).

### `--host`

- **Type**: String
- **Default**: `"0.0.0.0"`
- **Description**: Host to run the server on.

### `--port`

- **Type**: Integer
- **Default**: `8000`
- **Description**: Port to run the server on.

### `--reload`

- **Type**: Boolean
- **Default**: `False`
- **Description**: Enable auto-reload of the server. Only works when 'workers' is set to None.

### `--workers`

- **Type**: Integer or Float
- **Default**: Calculated based on `FASTMLX_NUM_WORKERS` environment variable or 2 if not set.
- **Description**: Number of workers. This option overrides the `FASTMLX_NUM_WORKERS` environment variable.

  - If an integer, it specifies the exact number of workers to use.
  - If a float, it represents the fraction of available CPU cores to use (minimum 1 worker).
  - To use all available CPU cores, set it to 1.0.

  **Examples**:
  - `--workers 1`: Use 1 worker
  - `--workers 1.0`: Use all available CPU cores
  - `--workers 0.5`: Use half of the available CPU cores
  - `--workers 0.0`: Use 1 worker

## Environment Variables

- `FASTMLX_NUM_WORKERS`: Sets the default number of workers if not specified via the `--workers` argument.

## Examples

1. Run the server on localhost with default settings:
   ```
   fastmlx
   ```

2. Run the server on a specific host and port:
   ```
   fastmlx --host 127.0.0.1 --port 5000
   ```

3. Run the server with 4 workers:
   ```
   fastmlx --workers 4
   ```

4. Run the server using half of the available CPU cores:
   ```
   fastmlx --workers 0.5
   ```

5. Enable auto-reload (for development):
   ```
   fastmlx --reload
   ```

Remember that the `--reload` option is intended for development purposes and should not be used in production environments.