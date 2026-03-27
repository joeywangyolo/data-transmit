#!/bin/bash
########################################################################
# Joey POC 指令集 — Ray Serve + vLLM 分容器架構驗證
# 使用方式：開啟此檔案，依序複製每個步驟的指令到終端機執行
# 注意：<模型路徑> 需替換成實際路徑（Phase 0 會幫你找到）
########################################################################


########################################################################
# Phase 0：環境準備（在 Host 上執行）
########################################################################

# ---- 步驟 0.1：確認現有 image 有沒有 Ray ----
# 注意：必須用 --entrypoint bash 繞過 vLLM 預設啟動流程（否則會因沒 GPU 報錯）
podman run --rm --entrypoint bash localhost/vllm/vllm-openai:v0.10.1.1 -c 'pip list 2>/dev/null | grep -i ray'
# 如果有輸出 ray 相關套件 → 記下版本，可能不需要建 image
# 如果沒有輸出 → 繼續步驟 0.2


# ---- 步驟 0.2：建 image（離線 wheel 方式）----
# 前提：已把 ray_wheels.tar 和 Containerfile.joey-poc 傳到主機同一個目錄
#
# 在主機上操作：
# 1. 解壓 tar 檔
tar -xf ray_wheels.tar
# 解壓後目錄結構應該是：
#   ./Containerfile.joey-poc
#   ./ray_wheels/
#     ├── ray-xxxxx.whl
#     ├── aiohttp-xxxxx.whl
#     └── ... 其他 .whl 檔案
#
# ⚠ 如果解壓出來是兩層 ray_wheels/ray_wheels/，執行：
#   mv ray_wheels/ray_wheels/* ray_wheels/ && rmdir ray_wheels/ray_wheels

# 2. 確認 ray_wheels 資料夾有 .whl 檔案
ls ray_wheels/*.whl | head -5

# 3. 用 Containerfile 建 image
podman build -t joey-poc-ray-vllm:v1 -f Containerfile.joey-poc .

# 4. 確認 image 建好了
podman images | grep joey-poc
# 預期：看到 joey-poc-ray-vllm  v1  ...


# ---- 步驟 0.3：找模型檔案位置 ----
podman inspect --format='{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}' $(podman ps -q | head -1)
ls /DAT-NAS/ 2>/dev/null
find / -maxdepth 4 -name "config.json" -path "*TinyLlama*" 2>/dev/null
find / -maxdepth 4 -name "config.json" -path "*Qwen*" 2>/dev/null
# 記下模型路徑，後面要用。例如 /DAT-NAS/models


# ---- 步驟 0.4：確認 port 沒有衝突 ----
ss -tlnp | grep -E '8099|8265|6379'
# 如果沒輸出 → port 可用
# 如果有輸出 → 需要換 port，跟我說


# ---- 步驟 0.5：確認現有環境快照（測試前拍照）----
nvidia-smi
podman ps
# 拍照或截圖存起來，測試完可以對比



########################################################################
# Phase 1：基線測試 — 一整包跑在 GPU 3（只需要 1 個 Tab）
########################################################################

# ---- 步驟 1.1：啟動容器 ----
# 使用 /DAT-NAS/models 掛載模型（唯讀，不影響現有資料）
podman run -it --rm \
  --name joey-poc-allinone \
  --network host \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  --shm-size=4g \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -v /DAT-NAS/models:/models:ro \
  --entrypoint bash \
  joey-poc-ray-vllm:v1

# === 以下在容器內執行 ===

# ---- 步驟 1.2：確認只看到 1 顆 GPU ----
nvidia-smi
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
# 預期：只看到 1 顆 GPU。如果超過 1 顆，立即 exit 停止！


# ---- 步驟 1.3：啟動 Ray ----
ray start --head --num-gpus=1 --dashboard-host=0.0.0.0 --port=6379 --dashboard-port=8265
ray status
# 預期：1 node, 1 GPU


# ---- 步驟 1.4：建立 config 並部署 ----
# 使用 opt-125m 模型（最小，約 250MB）
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
    runtime_env:
      env_vars:
        RAY_SERVE_HTTP_PORT: "8099"
EOF

export RAY_SERVE_HTTP_PORT=8099
serve deploy /tmp/config.yaml
sleep 60
serve status
# 預期：HEALTHY


# ---- 步驟 1.5：測試推理 ----
curl http://localhost:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "opt-125m",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50
  }'
# 預期：收到模型回應


# ---- 步驟 1.6：在 Host 上確認沒影響現有服務（開另一個 Tab）----
# nvidia-smi
# podman ps


# ---- 步驟 1.7：清理（在容器內）----
ray stop
exit
# 容器會自動刪除（--rm）

# 在 Host 上確認 GPU 3 已釋放
nvidia-smi



########################################################################
# Phase 2：方案 A — Head + Worker 分容器（需要 2 個 Tab）
########################################################################

# ==== Tab 1：Head 容器（不給 GPU）====

# ---- 步驟 2.1：啟動 Head ----
podman run -it --rm \
  --name joey-poc-head \
  --network host \
  --security-opt=label=disable \
  -e CUDA_VISIBLE_DEVICES="" \
  -e RAY_SERVE_HTTP_PORT=8099 \
  --entrypoint bash \
  joey-poc-ray-vllm:v1

# === Head 容器內 ===
ray start --head --num-gpus=0 --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
ray status
# 預期：1 node, 0 GPUs


# ==== Tab 2：Worker 容器（GPU 3）====

# ---- 步驟 2.2：啟動 Worker ----
# 使用 /DAT-NAS/models 掛載模型（唯讀）
podman run -it --rm \
  --name joey-poc-worker \
  --network host \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  --shm-size=4g \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -v /DAT-NAS/models:/models:ro \
  --entrypoint bash \
  joey-poc-ray-vllm:v1

# === Worker 容器內 ===
nvidia-smi
ray start --address=127.0.0.1:6379 --num-gpus=1


# ==== 回到 Tab 1（Head 容器）====

# ---- 步驟 2.3：確認叢集 ----
ray status
# 預期：2 nodes, Head 0 GPU + Worker 1 GPU


# ---- 步驟 2.4：部署模型 ----
# 使用 opt-125m 模型
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


# ---- 步驟 2.5：測試推理 ----
curl http://localhost:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "opt-125m",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50
  }'


# ---- 步驟 2.6：在 Host 上確認（Tab 3）----
# nvidia-smi
# podman ps


# ---- 步驟 2.7：如果要繼續 Phase 3，不要清理，直接往下 ----
# 如果要休息，先清理：
# Tab 1（Head）: ray stop && exit
# Tab 2（Worker）: ray stop && exit



########################################################################
# Phase 3：獨立更新測試（接續 Phase 2，不要清理）
########################################################################

# ==== Tab 1（Head 容器）====

# ---- 步驟 3.1：確認服務正常 ----
serve status
curl http://localhost:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "opt-125m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}'


# ==== Tab 2（Worker 容器）====

# ---- 步驟 3.2：模擬 Worker 下線 ----
ray stop
exit


# ==== 回到 Tab 1（Head 容器）====

# ---- 步驟 3.3：觀察 Head 反應 ----
sleep 10
ray status
# 預期：只剩 1 node
serve status
# 預期：UNHEALTHY（因為 Worker 不見了）
# 重點：Head 本身有沒有 crash？還能打指令嗎？


# ==== Tab 2（新 session）====

# ---- 步驟 3.4：啟動新 Worker ----
# 使用 /DAT-NAS/models 掛載模型（唯讀）
podman run -it --rm \
  --name joey-poc-worker-v2 \
  --network host \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  --shm-size=4g \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -v /DAT-NAS/models:/models:ro \
  --entrypoint bash \
  joey-poc-ray-vllm:v1

# === 新 Worker 容器內 ===
ray start --address=127.0.0.1:6379 --num-gpus=1


# ==== 回到 Tab 1（Head 容器）====

# ---- 步驟 3.5：檢查服務恢復 ----
sleep 10
ray status
# 預期：2 nodes
serve status
# 如果還是 UNHEALTHY，手動 redeploy：
serve deploy /tmp/config.yaml
sleep 60
serve status

curl http://localhost:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "opt-125m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}'


# ---- 步驟 3.6：清理 ----
# Tab 1（Head）: ray stop && exit
# Tab 2（Worker）: ray stop && exit



########################################################################
# Phase 4：多 GPU 測試 — TP=2（GPU 2 + GPU 3）（選做，可跳過）
########################################################################

# ==== Tab 1：Head ====

# ---- 步驟 4.1：啟動 Head ----
podman run -it --rm \
  --name joey-poc-head-tp2 \
  --network host \
  --security-opt=label=disable \
  -e CUDA_VISIBLE_DEVICES="" \
  -e RAY_SERVE_HTTP_PORT=8099 \
  --entrypoint bash \
  joey-poc-ray-vllm:v1

# === Head 容器內 ===
ray start --head --num-gpus=0 --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265


# ==== Tab 2：Worker（GPU 2 + GPU 3）====

# ---- 步驟 4.2：啟動 Worker ----
# 使用 /DAT-NAS/models 掛載模型（唯讀）
podman run -it --rm \
  --name joey-poc-worker-tp2 \
  --network host \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  --shm-size=8g \
  -e CUDA_VISIBLE_DEVICES=2,3 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -v /DAT-NAS/models:/models:ro \
  --entrypoint bash \
  joey-poc-ray-vllm:v1

# === Worker 容器內 ===
nvidia-smi
# 預期：看到 2 顆 GPU
ray start --address=127.0.0.1:6379 --num-gpus=2


# ==== 回到 Tab 1（Head 容器）====

# ---- 步驟 4.3：部署 TP=2 ----
cat > /tmp/config_tp2.yaml << 'EOF'
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
            tensor_parallel_size: 2
            max_model_len: 2048
EOF

serve deploy /tmp/config_tp2.yaml
sleep 60
serve status


# ---- 步驟 4.4：測試推理 ----
curl http://localhost:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "opt-125m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}'


# ---- 步驟 4.5：立即清理（釋放 GPU 2）----
# Tab 1: ray stop && exit
# Tab 2: ray stop && exit
# Host: nvidia-smi  確認 GPU 2 回到原本使用量



########################################################################
# Phase 5：方案 B — Multi-App Container（選做，可跳過）
########################################################################

# ---- 步驟 5.1：啟動容器 ----
# 使用 /DAT-NAS/models 掛載模型（唯讀）
podman run -it --rm \
  --name joey-poc-planb \
  --network host \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  --shm-size=4g \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -v /DAT-NAS/models:/models:ro \
  --entrypoint bash \
  joey-poc-ray-vllm:v1

# === 容器內 ===
ray start --head --num-gpus=1 --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265


# ---- 步驟 5.2：確認容器內有 Podman ----
podman --version
# 如果沒有 → 方案 B 無法測試，跳過


# ---- 步驟 5.3：部署 ----
cat > /tmp/config_planb.yaml << 'EOF'
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
    runtime_env:
      image_uri: "joey-poc-ray-vllm:v1"
EOF

serve deploy /tmp/config_planb.yaml
sleep 90
serve status


# ---- 步驟 5.4：驗證 ----
# 在 Host 上：
# podman ps | grep joey-poc

curl http://localhost:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "opt-125m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}'


# ---- 步驟 5.5：清理 ----
ray stop && exit



########################################################################
# 最終清理（每次測試完都要跑）
########################################################################

# 確認沒有殘留的 POC 容器
podman ps -a | grep joey-poc

# 如果有殘留，強制刪除
podman rm -f $(podman ps -a | grep joey-poc | awk '{print $1}')

# 確認 GPU 3 已釋放
nvidia-smi

# 確認現有服務正常
podman ps

# 完成！

#################
cat > /tmp/serve_vllm.py << 'PYEOF'
from ray import serve
from starlette.requests import Request
import time

@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
class VLLMService:
    def __init__(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        args = AsyncEngineArgs(
            model="/models/opt-125m",
            max_model_len=2048,
            tensor_parallel_size=1,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def __call__(self, request: Request):
        from vllm import SamplingParams
        body = await request.json()
        messages = body.get("messages", [])
        prompt = messages[-1]["content"] if messages else "Hello"
        max_tokens = body.get("max_tokens", 50)
        params = SamplingParams(max_tokens=max_tokens)
        request_id = f"req-{time.time()}"
        final_output = None
        async for output in self.engine.generate(prompt, params, request_id):
            final_output = output
        text = final_output.outputs[0].text
        return {
            "model": "opt-125m",
            "choices": [{"message": {"role": "assistant", "content": text}}]
        }

app = VLLMService.bind()
PYEOF
#################
cat > /tmp/config.yaml << 'EOF'
applications:
  - name: llm-app
    route_prefix: /
    import_path: serve_vllm:app
    runtime_env:
      working_dir: /tmp
      env_vars:
        RAY_SERVE_HTTP_PORT: "8099"
EOF

serve deploy /tmp/config.yaml
