FROM python:3.12-slim

WORKDIR /app
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api/app.py /app/api/app.py
COPY src /app/src
COPY models/best_model.joblib /app/models/best_model.joblib
COPY models/hyperparameters.json /app/models/hyperparameters.json

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
