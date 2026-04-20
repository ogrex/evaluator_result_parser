#!/usr/bin/env bash
# =============================================================================
# serve.sh — t4-server 起動スクリプト
#
# 使い方:
#   bash serve.sh [オプション]
#
# オプション:
#   --data-dir PATH        データセット保管ディレクトリ (デフォルト: ~/t4datasets)
#   --search-depth N       サブディレクトリ探索深さ (デフォルト: 1)
#   --host HOST            バインドアドレス (デフォルト: 0.0.0.0 = 全インターフェース)
#   --port PORT            サーバーポート (デフォルト: 8000)
#   --workers N            uvicorn ワーカープロセス数 (デフォルト: 2)
#   --tier4-cache N        メモリ上に保持する Tier4 インスタンス数 (デフォルト: 8)
#   --project-id ID        webauto プロジェクト ID (WEBAUTO_PROJECT_ID 環境変数でも可)
#   --venv PATH            仮想環境ディレクトリ (デフォルト: .venv)
#   -h, --help             このヘルプを表示
#
# 事前に setup.sh を実行して環境を構築してください。
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# デフォルト値
# ---------------------------------------------------------------------------
DATA_DIR="/mnt/qnapdata/internal/t4datasets"
SEARCH_DEPTH=1
HOST="0.0.0.0"
PORT=8000
WORKERS=8
TIER4_CACHE=8
WEBAUTO_PROJECT_ID="${WEBAUTO_PROJECT_ID:-}"
VENV_DIR=".venv"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
info()    { echo "[INFO]  $*"; }
success() { echo "[OK]    $*"; }
warn()    { echo "[WARN]  $*" >&2; }
error()   { echo "[ERROR] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 引数パース
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)      DATA_DIR="$2";           shift 2 ;;
        --search-depth)  SEARCH_DEPTH="$2";       shift 2 ;;
        --host)          HOST="$2";               shift 2 ;;
        --port)          PORT="$2";               shift 2 ;;
        --workers)       WORKERS="$2";            shift 2 ;;
        --tier4-cache)   TIER4_CACHE="$2";        shift 2 ;;
        --project-id)    WEBAUTO_PROJECT_ID="$2"; shift 2 ;;
        --venv)          VENV_DIR="$2";           shift 2 ;;
        -h|--help)
            sed -n '3,21p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *) error "Unknown option: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# 仮想環境の確認
# ---------------------------------------------------------------------------
VENV_PATH="$REPO_DIR/$VENV_DIR"
T4SERVER="$VENV_PATH/bin/t4-server"

if [[ ! -f "$T4SERVER" ]]; then
    error "t4-server が見つかりません: $T4SERVER — まず setup.sh を実行してください。"
fi

# ---------------------------------------------------------------------------
# webauto プロジェクト ID の設定確認
# ---------------------------------------------------------------------------
if [[ -n "$WEBAUTO_PROJECT_ID" ]]; then
    export WEBAUTO_PROJECT_ID
    success "WEBAUTO_PROJECT_ID = $WEBAUTO_PROJECT_ID"
else
    warn "WEBAUTO_PROJECT_ID が未設定です。"
    warn "ダウンロードを使う場合は --project-id <ID> を指定するか、"
    warn "環境変数 WEBAUTO_PROJECT_ID を設定してください。"
    warn "--no-download で既存データを使う場合は不要です。"
fi

# ---------------------------------------------------------------------------
# データディレクトリの確認
# ---------------------------------------------------------------------------
mkdir -p "$DATA_DIR"

dataset_count=0
for d in "$DATA_DIR"/*/; do
    [[ -d "$d" ]] && [[ -d "${d}annotation" || -d "${d}data" ]] && dataset_count=$((dataset_count + 1))
done
if [[ "$SEARCH_DEPTH" -ge 1 ]]; then
    for group in "$DATA_DIR"/*/; do
        for d in "$group"*/; do
            [[ -d "$d" ]] && [[ -d "${d}annotation" || -d "${d}data" ]] && dataset_count=$((dataset_count + 1))
        done
    done
fi

if [[ "$dataset_count" -gt 0 ]]; then
    success "検出済みデータセット数: ${dataset_count} (search_depth=${SEARCH_DEPTH})"
else
    warn "データセットが見つかりません (${DATA_DIR})"
    warn "サーバー起動後、webauto でダウンロードするか --data-dir を確認してください。"
fi

# ---------------------------------------------------------------------------
# サーバー起動
# ---------------------------------------------------------------------------
LAN_IP=""
if command -v ip &>/dev/null; then
    LAN_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}')
elif command -v ifconfig &>/dev/null; then
    LAN_IP=$(ifconfig | awk '/inet /&&!/127.0.0.1/{print $2; exit}')
fi

echo ""
echo "============================================================"
echo "  t4-server 起動"
echo "============================================================"
echo "  データディレクトリ : $DATA_DIR"
echo "  Search depth      : $SEARCH_DEPTH"
echo "  ホスト            : $HOST"
echo "  ポート            : $PORT"
echo "  ワーカー数        : $WORKERS"
echo "  Tier4 キャッシュ  : $TIER4_CACHE"
echo "============================================================"
echo ""
echo "  ローカル          : http://localhost:${PORT}"
[[ -n "$LAN_IP" ]] && \
echo "  ネットワーク内    : http://${LAN_IP}:${PORT}"
echo ""
echo "  API ドキュメント  : http://localhost:${PORT}/docs"
echo "  ヘルスチェック    : http://localhost:${PORT}/health"
echo "  データセット一覧  : http://localhost:${PORT}/datasets"
echo ""
echo "  停止するには Ctrl+C を押してください。"
echo "============================================================"
echo ""

exec "$T4SERVER" \
    --host "$HOST" \
    --data-dir "$DATA_DIR" \
    --search-depth "$SEARCH_DEPTH" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --tier4-cache "$TIER4_CACHE"
