# 🔌 `api/`: The FloodSense REST Service

This folder turns our standalone machine learning model into a web service that you can access from anywhere. We built this using **FastAPI** because it is very fast and easy to use.

If you want to connect the FloodSense AI to a Next.js website, an app or a dashboard you will talk to it through this API.

## How to run it

To run the API follow these steps:

First go to the root of the project and navigate into this directory.

Then use `uvicorn` to start the server:

```bash

cd api

uvicorn main:app --reload

```

When the server is running you can go to `http://localhost:8000/docs` in your browser.

There you can see the Swagger UI and test the API endpoints yourself.

## What's inside?

- **`main.py`**: This is the file. It sets up the FastAPI app. Handles the "lifespan".

This means it loads the machine learning model into memory only once when the server starts,

not every time someone makes a request.

- **`routers/predict.py`**: This file has the API endpoints, like `/predict`.

It takes a.npy` file or a demo ID passes the data to our `src/` folder, for prediction

and returns the results.

- **`schemas/prediction.py`**: We use models to define what the API should output.

This ensures that the frontend always gets what it expects

like the base64 encoded flood mask, the telemetry data and the pixel counts.

- **`requirements.txt`**: This file lists the requirements needed to run the web server.

It is a subset of the requirements.
