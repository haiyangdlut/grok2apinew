# push.sh — 发布 grok2apinew 代码到服务器并重启容器
# 用法: bash push.sh

set -e

SERVER="13.212.251.160"
KEY="C:/Users/32677/Desktop/dj/tools/indiapaytest.pem"
TARGET="/home/ubuntu/kg"
TAR_FILE="grok2apinew_push.tar.gz"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o LogLevel=ERROR"

echo "=== 1/4 本地打包 ==="
rm -f "$TAR_FILE"
tar -czf "$TAR_FILE" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='data' \
    --exclude='logs' \
    --exclude='.git' \
    app/ scripts/

echo "打包完成: $(du -h "$TAR_FILE" | cut -f1)"

echo ""
echo "=== 2/4 上传到服务器 ==="
scp -i "$KEY" -o StrictHostKeyChecking=no -o LogLevel=ERROR "$TAR_FILE" ubuntu@${SERVER}:${TARGET}/

echo ""
echo "=== 3/4 解压 & 修复换行符 ==="
ssh $SSH_OPTS ubuntu@${SERVER} "cd ${TARGET} && tar -xzf ${TAR_FILE} && rm -f ${TAR_FILE} && dos2unix scripts/*.sh 2>/dev/null; sed -i 's/\r\$//' scripts/*.sh; echo '解压完成'"

echo ""
echo "=== 4/4 重启容器 ==="
ssh -t $SSH_OPTS ubuntu@${SERVER} "cd ${TARGET} && docker compose restart && sleep 2 && docker logs -f --tail 20 grok2api-kg"
