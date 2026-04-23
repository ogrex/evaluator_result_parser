#!/usr/bin/env bash
# =============================================================================
# serve.sh — t4-server 起動 / 管理スクリプト
#
# 使い方:
#   bash serve.sh [run] [オプション]
#   bash serve.sh start|stop|restart|status|logs [オプション]
#
# サブコマンド:
#   run                  フォアグラウンドで起動 (デフォルト)
#   start                バックグラウンドで起動
#   stop                 停止
#   restart              再起動
#   status               稼働状態を表示
#   logs                 ログを表示 (tail -f)
#
# オプション:
#   --data-dir PATH        データセット保管ディレクトリ
#                          (デフォルト: /mnt/qnapdata/internal/t4datasets)
#   --search-depth N       サブディレクトリ探索深さ (デフォルト: 1)
#   --host HOST            バインドアドレス (デフォルト: 0.0.0.0)
#   --port PORT            サーバーポート (デフォルト: 8000)
#   --workers N            uvicorn ワーカープロセス数 (デフォルト: 8)
#   --tier4-cache N        メモリ上に保持する Tier4 インスタンス数 (デフォルト: 8)
#   --project-id ID        webauto プロジェクト ID (WEBAUTO_PROJECT_ID でも可)
#   --venv PATH            仮想環境ディレクトリ (デフォルト: .venv)
#   --pid-file PATH        PID ファイル (デフォルト: .run/t4-server.pid)
#   --log-file PATH        ログファイル (デフォルト: .run/t4-server.log)
#   --attach               start時にログを端末へ追従表示 (Ctrl+Cで終了)
#   -h, --help             このヘルプを表示
#
# 事前に setup.sh を実行して環境を構築してください。
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$REPO_DIR/.run"

# ---------------------------------------------------------------------------
# デフォルト値
# ---------------------------------------------------------------------------
COMMAND="run"
DATA_DIR="/mnt/qnapdata/internal/t4datasets"
SEARCH_DEPTH=1
HOST="0.0.0.0"
PORT=8000
WORKERS=32
TIER4_CACHE=32
WEBAUTO_PROJECT_ID="${WEBAUTO_PROJECT_ID:-}"
VENV_DIR=".venv"
PID_FILE="$RUN_DIR/t4-server.pid"
LOG_FILE="$RUN_DIR/t4-server.log"
STATE_FILE="$RUN_DIR/t4-server.env"
ATTACH_TERMINAL=0

# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
info()    { echo "[INFO]  $*"; }
success() { echo "[OK]    $*"; }
warn()    { echo "[WARN]  $*" >&2; }
error()   { echo "[ERROR] $*" >&2; exit 1; }

usage() {
    sed -n '3,31p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
}

shell_quote() {
    printf "%q" "$1"
}

process_is_running() {
    local pid="$1"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

read_pid() {
    [[ -f "$PID_FILE" ]] || return 1
    tr -d '[:space:]' < "$PID_FILE"
}

cleanup_stale_pid() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid="$(read_pid || true)"
        if [[ -n "$pid" ]] && ! process_is_running "$pid"; then
            warn "古い PID ファイルを削除します: $PID_FILE"
            rm -f "$PID_FILE"
        fi
    fi
}

save_state() {
    mkdir -p "$(dirname "$STATE_FILE")"
    cat > "$STATE_FILE" <<EOF
DATA_DIR=$(shell_quote "$DATA_DIR")
SEARCH_DEPTH=$(shell_quote "$SEARCH_DEPTH")
HOST=$(shell_quote "$HOST")
PORT=$(shell_quote "$PORT")
WORKERS=$(shell_quote "$WORKERS")
TIER4_CACHE=$(shell_quote "$TIER4_CACHE")
WEBAUTO_PROJECT_ID=$(shell_quote "$WEBAUTO_PROJECT_ID")
VENV_DIR=$(shell_quote "$VENV_DIR")
PID_FILE=$(shell_quote "$PID_FILE")
LOG_FILE=$(shell_quote "$LOG_FILE")
EOF
}

load_state_if_present() {
    [[ -f "$STATE_FILE" ]] || return 0
    # shellcheck disable=SC1090
    source "$STATE_FILE"
}

ensure_runtime_dirs() {
    mkdir -p "$(dirname "$PID_FILE")"
    mkdir -p "$(dirname "$LOG_FILE")"
}

print_banner() {
    local mode_label="$1"
    local lan_ip=""
    if command -v ip &>/dev/null; then
        lan_ip="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}' || true)"
    elif command -v ifconfig &>/dev/null; then
        lan_ip="$(ifconfig 2>/dev/null | awk '/inet /&&!/127.0.0.1/{print $2; exit}' || true)"
    fi

    echo ""
    echo "============================================================"
    echo "  t4-server ${mode_label}"
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
    [[ -n "$lan_ip" ]] && echo "  ネットワーク内    : http://${lan_ip}:${PORT}"
    echo ""
    echo "  API ドキュメント  : http://localhost:${PORT}/docs"
    echo "  ヘルスチェック    : http://localhost:${PORT}/health"
    echo "  データセット一覧  : http://localhost:${PORT}/datasets"
    echo ""
}

check_runtime_requirements() {
    local venv_path="$REPO_DIR/$VENV_DIR"
    local t4server="$venv_path/bin/t4-server"

    if [[ ! -f "$t4server" ]]; then
        error "t4-server が見つかりません: $t4server — まず setup.sh を実行してください。"
    fi

    if [[ -n "$WEBAUTO_PROJECT_ID" ]]; then
        export WEBAUTO_PROJECT_ID
        success "WEBAUTO_PROJECT_ID = $WEBAUTO_PROJECT_ID"
    else
        warn "WEBAUTO_PROJECT_ID が未設定です。"
        warn "ダウンロードを使う場合は --project-id <ID> を指定するか、"
        warn "環境変数 WEBAUTO_PROJECT_ID を設定してください。"
        warn "--no-download で既存データを使う場合は不要です。"
    fi

    mkdir -p "$DATA_DIR"

    local dataset_count=0
    local d=""
    local group=""
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
}

run_foreground() {
    local venv_path="$REPO_DIR/$VENV_DIR"
    local t4server="$venv_path/bin/t4-server"

    check_runtime_requirements
    print_banner "起動"
    echo "  停止するには Ctrl+C を押してください。"
    echo "============================================================"
    echo ""

    exec "$t4server" \
        --host "$HOST" \
        --data-dir "$DATA_DIR" \
        --search-depth "$SEARCH_DEPTH" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --tier4-cache "$TIER4_CACHE"
}

start_background() {
    cleanup_stale_pid

    local existing_pid=""
    local cmd=()
    local pid=0
    local i=0
    existing_pid="$(read_pid || true)"
    if [[ -n "$existing_pid" ]] && process_is_running "$existing_pid"; then
        error "すでに起動中です (PID: $existing_pid)。停止するには: bash serve.sh stop"
    fi

    check_runtime_requirements
    ensure_runtime_dirs
    save_state

    print_banner "バックグラウンド起動"
    echo "  ログファイル      : $LOG_FILE"
    echo "  PID ファイル      : $PID_FILE"
    echo "============================================================"
    echo ""

    cmd=(
        bash "$REPO_DIR/serve.sh" run
        --data-dir "$DATA_DIR"
        --search-depth "$SEARCH_DEPTH"
        --host "$HOST"
        --port "$PORT"
        --workers "$WORKERS"
        --tier4-cache "$TIER4_CACHE"
        --venv "$VENV_DIR"
        --pid-file "$PID_FILE"
        --log-file "$LOG_FILE"
    )
    if [[ -n "$WEBAUTO_PROJECT_ID" ]]; then
        cmd+=(--project-id "$WEBAUTO_PROJECT_ID")
    fi

    if command -v setsid &>/dev/null; then
        nohup setsid "${cmd[@]}" > "$LOG_FILE" 2>&1 &
    else
        nohup "${cmd[@]}" > "$LOG_FILE" 2>&1 &
    fi

    pid=$!
    echo "$pid" > "$PID_FILE"

    for i in {1..10}; do
        if process_is_running "$pid"; then
            success "バックグラウンドで起動しました (PID: $pid)"
            info "状態確認: bash serve.sh status"
            info "ログ確認  : bash serve.sh logs"
            if [[ "$ATTACH_TERMINAL" -eq 1 ]]; then
                info "端末へログを接続します (Ctrl+C で終了)"
                exec tail -f "$LOG_FILE"
            fi
            return 0
        fi
        sleep 0.5
    done

    rm -f "$PID_FILE"
    error "起動に失敗しました。ログを確認してください: $LOG_FILE"
}

stop_background() {
    cleanup_stale_pid

    local pid=""
    pid="$(read_pid || true)"
    if [[ -z "$pid" ]]; then
        warn "サーバーは起動していません。"
        return 0
    fi

    if ! process_is_running "$pid"; then
        warn "PID $pid はすでに停止しています。"
        rm -f "$PID_FILE"
        return 0
    fi

    info "サーバーを停止しています (PID: $pid)"
    kill "$pid"

    local i=0
    for i in {1..20}; do
        if ! process_is_running "$pid"; then
            rm -f "$PID_FILE"
            success "停止しました。"
            return 0
        fi
        sleep 0.5
    done

    warn "通常停止に時間がかかっているため強制終了します (PID: $pid)"
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    success "強制停止しました。"
}

show_status() {
    cleanup_stale_pid

    local pid=""
    pid="$(read_pid || true)"
    if [[ -z "$pid" ]]; then
        echo "t4-server is not running."
        echo "Start with: bash serve.sh start"
        return 0
    fi

    echo "t4-server is running."
    echo "PID      : $pid"
    echo "Log file : $LOG_FILE"
    echo "PID file : $PID_FILE"
    if [[ -f "$STATE_FILE" ]]; then
        echo "Port     : $PORT"
        echo "Data dir : $DATA_DIR"
    fi
}

show_logs() {
    ensure_runtime_dirs
    touch "$LOG_FILE"
    exec tail -f "$LOG_FILE"
}

# ---------------------------------------------------------------------------
# 引数パース
# ---------------------------------------------------------------------------
if [[ $# -gt 0 ]]; then
    case "$1" in
        run|start|stop|restart|status|logs)
            COMMAND="$1"
            shift
            ;;
    esac
fi

user_provided_options=0
while [[ $# -gt 0 ]]; do
    user_provided_options=1
    case "$1" in
        --data-dir)      DATA_DIR="$2";            shift 2 ;;
        --search-depth)  SEARCH_DEPTH="$2";        shift 2 ;;
        --host)          HOST="$2";                shift 2 ;;
        --port)          PORT="$2";                shift 2 ;;
        --workers)       WORKERS="$2";             shift 2 ;;
        --tier4-cache)   TIER4_CACHE="$2";         shift 2 ;;
        --project-id)    WEBAUTO_PROJECT_ID="$2";  shift 2 ;;
        --venv)          VENV_DIR="$2";            shift 2 ;;
        --pid-file)      PID_FILE="$2";            shift 2 ;;
        --log-file)      LOG_FILE="$2";            shift 2 ;;
        --attach)        ATTACH_TERMINAL=1;         shift ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

if [[ "$COMMAND" == "restart" || "$COMMAND" == "status" || "$COMMAND" == "stop" || "$COMMAND" == "logs" ]]; then
    if [[ "$user_provided_options" -eq 0 ]]; then
        load_state_if_present
    fi
fi

case "$COMMAND" in
    run)
        run_foreground
        ;;
    start)
        start_background
        ;;
    stop)
        stop_background
        ;;
    restart)
        stop_background
        start_background
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        error "Unknown command: $COMMAND"
        ;;
esac
