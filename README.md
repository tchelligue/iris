# Iris Classifier Web Service

This project provides a simple web API and web form for classifying iris flowers using a pre-trained machine learning model.

## Project Structure

- `app/server.py`: FastAPI server with API and web form endpoints
- `app/model.joblib`: Pre-trained iris classification model
- `client.py`: Python script to test the API endpoint
- `requirements.txt`: Python dependencies

## How to Run the Web Service

1. **Install dependencies**

   Activate your virtual environment if needed, then run:
   ```sh
   pip install -r requirements.txt
   ```

2. **Start the FastAPI server**

   From the project root directory, run:
   ```sh
   uvicorn app.server:app --reload
   ```

3. **Access the web form**

   Open your browser and go to:
   - [http://127.0.0.1:8000/](http://127.0.0.1:8000/) (Home page with link to the form)
   - [http://127.0.0.1:8000/form](http://127.0.0.1:8000/form) (Direct link to the input form)

   Enter the four iris features and submit to see the predicted class.

4. **API Usage**

   You can also use the `/predict` endpoint for programmatic access:
   - URL: `POST http://127.0.0.1:8000/predict`
   - Body (JSON): `{ "features": [sepal_length, sepal_width, petal_length, petal_width] }`
   - Response: `{ "predicted_class": "setosa" }`

   Example using `client.py`:
   ```sh
   python client.py
   ```

## Example

- Web form: Enter values and get the predicted iris class instantly.
- API: Send a POST request with features and receive the predicted class in JSON.

## Troubleshooting

- If you don't see the form, make sure the server is running and you are visiting the correct URL.
- Check the terminal for any errors when starting the server.

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- joblib
- numpy
- requests

Install all requirements with:
```sh
pip install -r requirements.txt
```

---

Feel free to modify or extend this project as needed!
