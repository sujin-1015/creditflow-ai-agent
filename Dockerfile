FROM python:3.12-slim

WORKDIR /app

COPY service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY models/xgboost_model.joblib models/lightgbm_model.joblib models/threshold_config.json ./models/
COPY processed/train.csv processed/val.csv processed/test.csv ./processed/
COPY onchain/devnet_keys/ ./onchain/devnet_keys/
COPY service/ ./service/

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080"]
