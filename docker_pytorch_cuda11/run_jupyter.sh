#!/usr/bin/env bash
# JupyterLab runner for pytorch_container (auto-retag & auto-rename)

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${SCRIPT_DIR}"

# 新的标准命名
CONTAINER="${CONTAINER_NAME:-pytorch_container}"
IMAGE="${DOCKER_IMAGE:-pytorch_cuda11:1.0}"
PORT="${PORT:-8888}"
TOKEN="${JUPYTER_TOKEN:-mypw}"

# 兼容旧命名（用于自动迁移）
OLD_IMAGE="${OLD_IMAGE_NAME:-ct_projector:2.0}"
OLD_CONTAINER="${OLD_CONTAINER_NAME:-ct_jupyter}"

cd "$WORKDIR"

# 1) 若本地没有新镜像但有旧镜像 -> 自动重打标签为新名
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  if docker image inspect "$OLD_IMAGE" >/dev/null 2>&1; then
    echo "[i] Tagging old image: ${OLD_IMAGE}  ->  ${IMAGE}"
    docker tag "$OLD_IMAGE" "$IMAGE"
  fi
fi

# 2) 若新容器不存在但旧容器存在 -> 自动重命名容器
if ! docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  if docker ps -a --format '{{.Names}}' | grep -qx "$OLD_CONTAINER"; then
    echo "[i] Renaming container: ${OLD_CONTAINER}  ->  ${CONTAINER}"
    # 停止旧容器（rename 运行中一般也能成功，为稳妥先停）
    docker stop "$OLD_CONTAINER" >/dev/null 2>&1 || true
    docker rename "$OLD_CONTAINER" "$CONTAINER"
  fi
fi

# 3) 组装挂载：项目目录 + 所有 /mnt/<a-z> 盘符到 /host/<盘符>
MOUNTS=()
MOUNTS+=("-v" "${WORKDIR}:/workspace/pytorch_container")
HOST_ROOT="/host"
for d in /mnt/[a-z]; do
  [ -d "$d" ] && MOUNTS+=("-v" "$d:${HOST_ROOT}/$(basename "$d")")
done

container_exists() {
  docker ps -a --filter "name=^/${CONTAINER}$" --format '{{.Names}}' | grep -qx "${CONTAINER}"
}

# 4) 若容器存在但镜像不同 -> 删除重建（统一到新镜像）
if container_exists; then
  CUR_IMG="$(docker inspect -f '{{.Config.Image}}' "$CONTAINER" 2>/dev/null || true)"
  if [ "${CUR_IMG:-}" != "$IMAGE" ]; then
    echo "[i] Recreating ${CONTAINER} with image ${IMAGE} ..."
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  fi
fi

# 5) 创建或启动容器
if ! container_exists; then
  echo "[i] Creating ${CONTAINER} from ${IMAGE} ..."
  docker run -d --gpus all \
    --name "$CONTAINER" \
    --restart unless-stopped \
    -p "${PORT}:8888" \
    "${MOUNTS[@]}" \
    --workdir /workspace/pytorch_container \
    --shm-size=16g \
    "$IMAGE" bash -lc "
      python -c 'import jupyterlab' 2>/dev/null || pip install -U jupyterlab;
      jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
        --ServerApp.token='${TOKEN}' --NotebookApp.token='${TOKEN}' \
        --ServerApp.root_dir='${HOST_ROOT}' \
        --NotebookApp.notebook_dir='${HOST_ROOT}'
    "
else
  echo "[i] Starting existing container ${CONTAINER} ..."
  docker start "$CONTAINER" >/dev/null
fi

echo
echo "Open:   http://localhost:${PORT}/?token=${TOKEN}"
echo "Host drives under: ${HOST_ROOT}/c, ${HOST_ROOT}/d, ..."
echo "Logs:   docker logs -f ${CONTAINER}"
echo "GPU:    docker exec -it ${CONTAINER} nvidia-smi"

