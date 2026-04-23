#!/usr/bin/env bash
# =============================================================================
# setup.sh — 新規 PC 向け環境構築スクリプト (uv 版)
#
# 使い方:
#   bash setup.sh [オプション]
#
# オプション:
#   --python VERSION       Python バージョン指定、例: 3.11 (デフォルト: 3.11)
#   --venv PATH            仮想環境ディレクトリ (デフォルト: .venv)
#   --skip-install         uv pip install をスキップ
#   -h, --help             このヘルプを表示
#
# 環境変数:
#   WEBAUTOAUTH_GIT_SPEC   webautoauth の Git インストール元
#                          (デフォルト: git+ssh://git@github.com/tier4/webauto-auth-py.git)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# デフォルト値
# ---------------------------------------------------------------------------
PYTHON_VERSION="3.11"
VENV_DIR=".venv"
SKIP_INSTALL=false
WEBAUTOAUTH_GIT_SPEC="${WEBAUTOAUTH_GIT_SPEC:-git+ssh://git@github.com/tier4/webauto-auth-py.git}"

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
        --python)        PYTHON_VERSION="$2"; shift 2 ;;
        --venv)          VENV_DIR="$2";       shift 2 ;;
        --skip-install)  SKIP_INSTALL=true;   shift ;;
        -h|--help)
            sed -n '3,13p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *) error "Unknown option: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1: uv の確認
# ---------------------------------------------------------------------------
info "=== Step 1: uv 確認 ==="

if ! command -v uv &>/dev/null; then
    error "uv が見つかりません。https://docs.astral.sh/uv/getting-started/installation/ を参照してインストールしてください。"
fi
success "uv: $(uv --version)"

# ---------------------------------------------------------------------------
# Step 2: 仮想環境のセットアップ
# ---------------------------------------------------------------------------
info "=== Step 2: 仮想環境 ($VENV_DIR, Python $PYTHON_VERSION) ==="

VENV_PATH="$REPO_DIR/$VENV_DIR"

if [[ ! -d "$VENV_PATH" ]]; then
    uv venv "$VENV_PATH" --python "$PYTHON_VERSION"
    success "仮想環境を作成しました: $VENV_PATH"
else
    success "既存の仮想環境を使用: $VENV_PATH"
fi

PYTHON_VENV="$VENV_PATH/bin/python"
success "Python: $("$PYTHON_VENV" --version)"

# ---------------------------------------------------------------------------
# Step 3: 依存ライブラリのインストール
# ---------------------------------------------------------------------------
if [[ "$SKIP_INSTALL" == false ]]; then
    info "=== Step 3: 依存ライブラリのインストール ==="

    # t4-devkit（GitHub から、既存ならスキップ）
    if "$PYTHON_VENV" -c "import t4_devkit" 2>/dev/null; then
        success "t4_devkit は既にインストール済みです"
    else
        info "t4_devkit をインストール中..."
        uv pip install --python "$PYTHON_VENV" \
            "git+https://github.com/tier4/t4-devkit.git"
        success "t4_devkit をインストールしました"
    fi

    # パッケージ本体（pyproject.toml の依存 + server extras を一括インストール）
    info "evaluator-result-parser[server] をインストール中..."
    uv pip install --python "$PYTHON_VENV" -e "$REPO_DIR[server]"
    success "パッケージをインストールしました"

    # suite-based catalog check 用（webautoauth）
    if "$PYTHON_VENV" -c "import webautoauth" 2>/dev/null; then
        success "webautoauth は既にインストール済みです"
    else
        info "webautoauth をインストール中..."
        if uv pip install --python "$PYTHON_VENV" webautoauth; then
            success "webautoauth をインストールしました (registry)"
        else
            warn "webautoauth は公開レジストリに見つかりませんでした。"
            info "Tier4 Git リポジトリからインストールを試行: $WEBAUTOAUTH_GIT_SPEC"
            if uv pip install --python "$PYTHON_VENV" "$WEBAUTOAUTH_GIT_SPEC"; then
                success "webautoauth をインストールしました (git)"
            else
                warn "webautoauth のインストールに失敗しました。"
                warn "suite-based catalog check を使う場合は SSH 鍵/権限を確認してください。"
                warn "必要なら環境変数 WEBAUTOAUTH_GIT_SPEC で取得元を上書きできます。"
            fi
        fi
    fi
else
    info "=== Step 3: インストールをスキップ ==="
fi

# ---------------------------------------------------------------------------
# Step 4: インストール確認
# ---------------------------------------------------------------------------
info "=== Step 4: インストール確認 ==="

T4SERVER="$VENV_PATH/bin/t4-server"
[[ ! -f "$T4SERVER" ]] && error "t4-server が見つかりません: $T4SERVER — Step 3 を確認してください。"
success "t4-server: $T4SERVER"

if "$PYTHON_VENV" -c "import webautoauth" 2>/dev/null; then
    success "webautoauth import: OK"
else
    warn "webautoauth import: NG (suite-based catalog check は動作しません)"
fi

if command -v webauto &>/dev/null; then
    success "webauto CLI: $(command -v webauto)"
else
    warn "webauto CLI が見つかりません（直接 catalog describe やダウンロード機能に影響します）"
fi

# ---------------------------------------------------------------------------
# 完了
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  セットアップ完了"
echo "============================================================"
echo "  仮想環境 : $VENV_PATH"
echo "  Python   : $("$PYTHON_VENV" --version)"
echo "============================================================"
echo ""
echo "サーバーを起動するには:"
echo "  bash serve.sh --data-dir <path>"
echo ""
