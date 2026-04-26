# 🔌 `api/`: The FloodSense REST Service

This folder turns our standalone ML model into a web-accessible microservice. We built this using **FastAPI** because it's incredibly fast, self-documenting, and very Pythonic.

If you want to plug the FloodSense AI into a Next.js frontend, a mobile app, or a dashboard, you'll talk to it through this API.

## How to run it

From the root of the project, navigate into this directory and use `uvicorn` to spin up the server:

```bash
cd api
uvicorn main:app --reload
```
Once it's running, you can visit `http://localhost:8000/docs` in your browser to see the interactive Swagger UI and test the endpoints yourself!

## What's inside?

- **`main.py`**: The entry point. This initializes the FastAPI app and handles the "lifespan" (which ensures the heavy ML model is loaded into memory only once when the server starts up, rather than every time someone makes a request).
- **`routers/predict.py`**: This contains the actual endpoints (like `/predict`). It handles taking an uploaded `.npy` file (or a demo ID), passing the data down into our `src/` folder for inference, and returning the results.
- **`schemas/prediction.py`**: We use Pydantic models to strictly define what the API should output. This ensures that the frontend always gets exactly what it expects (like the base64 encoded flood mask, the telemetry data, and the pixel counts).
- **`requirements.txt`**: A subset of the main requirements specifically needed to run the web server.
