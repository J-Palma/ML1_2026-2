FROM python:3.12-slim

WORKDIR /app
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api/app.py /app/api/app.py
COPY src /app/src
COPY models/xgboost_forest.joblib /app/models/xgboost_forest.joblib

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
