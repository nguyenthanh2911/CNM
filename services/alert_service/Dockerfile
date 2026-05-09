FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "services.alert_service.main:app", "--host", "0.0.0.0", "--port", "8002"]
