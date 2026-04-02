#!/bin/bash
########################################################################
# Joey POC 指令集 — Ray Serve + vLLM 分容器架構驗證
# 更新日期：2026-03-31
# 使用方式：開啟此檔案，依序複製每個步驟的指令到終端機執行
#
# 測試策略（依團隊討論）：
#   1. 先建 Ray Cluster → 確認 Cluster 健康
#   2. 再用 Ray 的部署 API 掛模型
#   3. 測試動態擴展（後加 Worker）
#
# ⚠ 安全規則：
#   - 只能使用 GPU 3（CUDA_VISIBLE_DEVICES=3）
#   - GPU 0,1,2 為正式服務，絕對不能動
#   - 所有容器名稱以 joey-poc- 開頭
########################################################################


########################################################################
# Phase 0：環境準備（在 Host 上執行）
########################################################################

# ---- 步驟 0.1：確認現有 image 有沒有 Ray ----
podman run --rm --entrypoint bash localhost/vllm/vllm-openai:v0.10.1.1 -c 'pip list 2>/dev/null | grep -i ray'
# 有輸出 → 記下版本，可能不需要建 image
# 沒有輸出 → 繼續步驟 0.2

# ---- 步驟 0.2：建 image（離線 wheel 方式）----
# 前提：已把 ray_wheels.tar 和 Containerfile.joey-poc 傳到主機同一個目錄
tar -xf ray_wheels.tar
ls ray_wheels/*.whl | head -5
podman build -t joey-poc-ray-vllm:v3 -f Containerfile.joey-poc .
# 預期 build 尾端看到：pyarrow 20.0.0 OK、ray 2.48.0 OK
podman images | grep joey-poc

# ---- 步驟 0.3：確認 port 沒有衝突 ----
ss -tlnp | grep -E '8099|8265|6379'

# ---- 步驟 0.4：確認現有環境快照（測試前截圖）----
nvidia-smi
podman ps


########################################################################
# Phase 1：Ray Serve 依賴診斷（進任意容器檢查）
# 目的：確認 Ray Serve 的 HTTP 依賴有裝齊
# 背景：Ray worker 因 pthread_create 限制 crash → 加 --pids-limit=-1 修復
########################################################################

# ---- 步驟 1.1：啟動診斷容器 ----
podman run -it --rm \
  --name joey-poc-diag \
  --network host \
  --security-opt=label=disable \
  --pids-limit=-1 \
  --device nvidia.com/gpu=all \
  --shm-size=10g \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -v /DAT-NAS/models:/models:ro \
  --entrypoint bash \
  joey-poc-ray-vllm:v3

# === 以下在容器內執行 ===

# ---- 步驟 1.2：查 Ray 版本 ----
pip show ray | grep Version
pip list | grep -iE "ray|aiohttp|uvicorn|starlette|fastapi|grpcio|protobuf"

# ---- 步驟 1.3：逐一檢查 Ray Serve HTTP 依賴 ----
python3 -c "
try:
    import uvicorn; print('uvicorn OK:', uvicorn.__version__)
except: print('❌ uvicorn MISSING')
try:
    import starlette; print('starlette OK:', starlette.__version__)
except: print('❌ starlette MISSING')
try:
    import fastapi; print('fastapi OK:', fastapi.__version__)
except: print('❌ fastapi MISSING')
try:
    import aiohttp; print('aiohttp OK:', aiohttp.__version__)
except: print('❌ aiohttp MISSING')
try:
    import grpcio; print('grpcio OK')
except: print('❌ grpcio MISSING')
"
# 如果有 MISSING → 需要在外網補下載 wheel 傳進來安裝
# 如果全部 OK → 繼續步驟 1.4

# ---- 步驟 1.4：查 Ray Dashboard log（看有沒有隱藏錯誤）----
rm -rf /tmp/ray
ray start --head --num-gpus=1 --dashboard-host=0.0.0.0 --port=6379 --dashboard-port=8265
sleep 3
cat /tmp/ray/session_latest/logs/dashboard.log 2>/dev/null | tail -30
ls /tmp/ray/session_latest/logs/serve/ 2>/dev/null

# ---- 步驟 1.5：用最簡單的 echo 服務測 Ray Serve ----
cat > /tmp/test_serve.py << 'PYEOF'
import ray
from ray import serve

@serve.deployment(num_replicas=1)
class Echo:
    async def __call__(self, request):
        return {"msg": "Ray Serve works!"}

app = Echo.bind()
handle = serve.run(app)
print("=== Echo service running on port 8000 ===")

import signal
signal.pause()
PYEOF

python3 /tmp/test_serve.py &
# 等 30 秒，然後測試：
sleep 30
curl http://localhost:8000/
# 預期：{"msg": "Ray Serve works!"}
# 如果 Connection refused → Ray Serve 有問題，查 log：
#   cat /tmp/ray/session_latest/logs/serve/*.log 2>/dev/null | tail -30
#   cat /tmp/ray/session_latest/logs/raylet.err 2>/dev/null | tail -20

# ---- 步驟 1.6：清理診斷容器 ----
ray stop
exit


########################################################################
# Phase 2：建立 Ray Cluster — Head + Worker 分容器（需要 2 個 Tab）
# 目的：先確認 Cluster 健康，不部署模型
# 架構：Head（無 GPU）+ Worker（GPU 3）
########################################################################

# ==== Tab 1：Head 容器（不給 GPU）====

# ---- 步驟 2.1：啟動 Head ----
podman run -it --rm \
  --name joey-poc-head \
  --network host \
  --security-opt=label=disable \
  --pids-limit=-1 \
  --shm-size=10g \
  -e CUDA_VISIBLE_DEVICES="" \
  --entrypoint bash \
  joey-poc-ray-vllm:v3

# === Head 容器內 ===
ray start --head --num-gpus=0 --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
ray status
# 預期：1 node, 0 GPUs


# ==== Tab 2：Worker 容器（GPU 3）====

# ---- 步驟 2.2：啟動 Worker ----
podman run -it --rm \
  --name joey-poc-worker \
  --network host \
  --security-opt=label=disable \
  --pids-limit=-1 \
  --device nvidia.com/gpu=all \
  --shm-size=10g \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -v /DAT-NAS/models:/models:ro \
  --entrypoint bash \
  joey-poc-ray-vllm:v3

# === Worker 容器內 ===
nvidia-smi
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
# 預期：只看到 1 顆 GPU。如果超過 1 顆，立即 exit 停止！

ray start --address=127.0.0.1:6379 --num-gpus=1
# 預期：Ray runtime started. 連接到 Head 成功


# ==== 回到 Tab 1（Head 容器）====

# ---- 步驟 2.3：確認 Cluster 健康 ----
ray status
# 預期：2 nodes, 總共 1 GPU
#   - Head: 0 GPU
#   - Worker: 1 GPU

# 如果能打開瀏覽器，也可以看 Ray Dashboard：
# http://<主機IP>:8265

# ---- 步驟 2.4：記錄 Cluster 健康結果 ----
# 截圖 ray status 輸出
# 如果 2 nodes + 1 GPU → Cluster 健康 ✅ → 繼續 Phase 3
# 如果不是 → 檢查 Worker 容器 log


########################################################################
# Phase 3：在 Cluster 上部署模型（接續 Phase 2，不要清理）
# 目的：用 Ray 的部署 API 掛載 vLLM 模型
########################################################################

# ==== Tab 1（Head 容器）====

# ---- 步驟 3.1：用 serve deploy 部署模型（Ray 官方方式）----
cat > /tmp/config.yaml << 'EOF'
applications:
  - name: llm-app
    route_prefix: /
    import_path: ray.serve.llm:build_openai_app
    args:
      llm_configs:
        - model_loading_config:
            model_id: opt-125m
            model_source: /models/opt-125m
          deployment_config:
            autoscaling_config:
              min_replicas: 1
              max_replicas: 1
          engine_kwargs:
            tensor_parallel_size: 1
            max_model_len: 2048
EOF

serve deploy /tmp/config.yaml
sleep 60
serve status
# 預期：HEALTHY

# ---- 步驟 3.2：測試推理（opt-125m 無 chat template，用 /v1/completions）----
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"opt-125m","prompt":"Hello, my name is","max_tokens":50}'
# 預期：收到模型回應（文字補全）

# ---- 步驟 3.3：在 Host 上確認沒影響現有服務（開 Tab 3）----
# nvidia-smi
# podman ps

# ---- 步驟 3.4：如果要繼續 Phase 4，不要清理，直接往下 ----


########################################################################
# Phase 4：獨立更新 Worker 測試（接續 Phase 3，不要清理）
# 目的：驗證 Worker 下線後 Head 不 crash，新 Worker 上線後服務恢復
########################################################################

# ==== Tab 1（Head 容器）====

# ---- 步驟 4.1：確認服務正常 ----
serve status
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model":"opt-125m","prompt":"Hello","max_tokens":20}'


# ==== Tab 2（Worker 容器）====

# ---- 步驟 4.2：模擬 Worker 下線 ----
ray stop
exit


# ==== 回到 Tab 1（Head 容器）====

# ---- 步驟 4.3：觀察 Head 反應 ----
sleep 10
ray status
# 預期：只剩 1 node（Head）
serve status
# 預期：UNHEALTHY（因為 Worker 不見了）
# 重點觀察：Head 本身有沒有 crash？還能打指令嗎？


# ==== Tab 2（新 session）====

# ---- 步驟 4.4：啟動新 Worker ----
podman run -it --rm \
  --name joey-poc-worker-v2 \
  --network host \
  --security-opt=label=disable \
  --pids-limit=-1 \
  --device nvidia.com/gpu=all \
  --shm-size=10g \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -v /DAT-NAS/models:/models:ro \
  --entrypoint bash \
  joey-poc-ray-vllm:v3

# === 新 Worker 容器內 ===
ray start --address=127.0.0.1:6379 --num-gpus=1


# ==== 回到 Tab 1（Head 容器）====

# ---- 步驟 4.5：檢查服務自動恢復 ----
sleep 30
ray status
# 預期：2 nodes
serve status
# 預期：HEALTHY（自動恢復）
# 如果還是 UNHEALTHY，手動 redeploy：
#   serve deploy /tmp/config.yaml
#   sleep 60
#   serve status

curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model":"opt-125m","prompt":"Hello","max_tokens":20}'
# 預期：推理正常

# ---- 步驟 4.6：清理 ----
# Tab 1（Head）: ray stop && exit
# Tab 2（Worker）: ray stop && exit


########################################################################
# Phase 5：3 Nodes 動態擴展測試（接續 Phase 3，不要清理）
# 目的：驗證 Cluster 可動態加入無 GPU Worker，推理仍正常
# 架構：Head + Worker1(GPU 3) + Worker2(無 GPU) = 3 nodes
########################################################################

# ==== Tab 3（新 session）====

# ---- 步驟 5.1：啟動無 GPU 的 Worker 2 ----
podman run -it --rm \
  --name joey-poc-worker-cpu \
  --network host \
  --security-opt=label=disable \
  --pids-limit=-1 \
  --shm-size=10g \
  -e CUDA_VISIBLE_DEVICES="" \
  --entrypoint bash \
  joey-poc-ray-vllm:v3

# === Worker 2 容器內 ===
ray start --address=127.0.0.1:6379 --num-gpus=0

# ==== 回到 Tab 1（Head 容器）====

# ---- 步驟 5.2：確認 3 nodes ----
ray status
# 預期：3 nodes, 1 GPU（只有 Worker1 有 GPU）

# ---- 步驟 5.3：推理仍正常（Ray 自動調度到有 GPU 的 Worker）----
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model":"opt-125m","prompt":"Hello, my name is","max_tokens":50}'
# 預期：推理正常，模型跑在 Worker1（GPU 3）上

# ---- 步驟 5.4：清理 Worker 2 ----
# Tab 3（Worker 2）: ray stop && exit


########################################################################
# Phase 6：GPU 釋放驗證（接續 Phase 5 或 Phase 3）
# 目的：證明 serve shutdown 後 GPU 記憶體被釋放
########################################################################

# ==== Tab 1（Head 容器）====

# ---- 步驟 6.1：截圖 — GPU 釋放前 ----
# 在 Tab 2（Worker）或 Host 上跑 nvidia-smi，截圖
# 預期：GPU 3 佔用 ~86GB

# ---- 步驟 6.2：停止模型部署 ----
serve shutdown
sleep 10

# ---- 步驟 6.3：截圖 — GPU 釋放後 ----
# 在 Tab 2（Worker）或 Host 上跑 nvidia-smi，截圖
# 預期：GPU 3 佔用降回接近 0
# 對比兩張截圖 → 證明 Ray Serve 可動態釋放 GPU 資源


########################################################################
# 最終清理（每次測試完都要跑）
########################################################################

# Tab 1（Head）: ray stop --force && exit
# Tab 2（Worker）: ray stop --force && exit

# 確認沒有殘留的 POC 容器
podman ps -a | grep joey-poc

# 如果有殘留，強制刪除
podman rm -f $(podman ps -a | grep joey-poc | awk '{print $1}')

# 確認 GPU 3 已釋放
nvidia-smi

# 確認現有服務正常
podman ps
