# 🔌 `api/`: The FloodSense REST Service

This folder turns our standalone ML model into a web-accessible microservice using **FastAPI**.

## How to run it

From the root of the project, navigate into this directory and use `uvicorn` to spin up the server:

```bash
cd api
uvicorn main:app --reload
```
Once it's running, visit `http://localhost:8000/docs` in your browser for the interactive Swagger UI.

## What's inside?

- **`main.py`**: The entry point. Initializes the FastAPI app and handles the model loading lifecycle.
- **`models/`**: Contains the Pydantic data models for internal logic.
  - `flood_model.py`: Internal model definitions.
- **`routers/`**: Contains the API endpoints.
  - `predict.py`: Handles prediction endpoints, orchestrating data processing and model inference.
- **`schemas/`**: Pydantic models for request/response validation.
  - `prediction.py`: Defines the strictly typed inputs and outputs of the API endpoints.
- **`requirements.txt`**: The dependencies needed to run the API.
- **`.gitignore`**: Defines files to ignore in this directory.
