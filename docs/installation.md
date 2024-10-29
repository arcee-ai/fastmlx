# Installation

## Stable release

To install the latest stable release of FastMLX, use the following command:

```
pip install -U fastmlx
```

This is the recommended method to install **FastMLX**, as it will always install the most recent stable release.

If [pip](https://pip.pypa.io) isn't installed, you can follow the [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) to set it up.

## Installation from Sources

To install **FastMLX** directly from the source code, run this command in your terminal:

```
pip install git+https://github.com/Blaizzy/fastmlx
```
## Running the Server

There are two ways to start the FastMLX server:

Using the `fastmlx` command:

   ```bash
   fastmlx
   ```

or
   
Using `uvicorn` directly:

   ```bash
   uvicorn fastmlx:app --reload --workers 0
   ```

   > WARNING: The `--reload` flag should not be used in production. It is only intended for development purposes.
   
### Additional Notes

- **Dependencies**: Ensure that you have the required dependencies installed. FastMLX relies on several libraries, which `pip` will handle automatically.