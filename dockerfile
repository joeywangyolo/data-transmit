FROM treg.cathay-ins.com.tw/dat-py-dat-cm/dat-cm-rs-mlflow-base:1.0

# 設定環境變數預設值
# ENV POSTGRES_USER=mlflow_user \
#     POSTGRES_PASSWORD=mlflow_password \
#     POSTGRES_DB=mlflow_db \
#     POSTGRES_HOST=postgres \
#     POSTGRES_PORT=5432 \
#     POSTGRES_SCHEMA=mlflow

# 啟動 MLflow Server (使用 shell 形式以支援環境變數替換)
# 使用 options 參數指定 schema，避免使用 public
CMD mlflow server \
    --backend-store-uri "${DATABASE_URL}" \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000
