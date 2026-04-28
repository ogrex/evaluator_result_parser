# T4 Dataset Catalog — カタログからのデータセット抽出・一括ダウンロード

評価システムの Vehicle Catalog（評価スイート + シナリオ + t4_dataset_ids）から、欲しいデータセットを一括で抽出・列表・ダウンロードするためのワークフローです。

---

## 全体フロー

```
Vehicle Catalog (Web UI)
    ↓ get_suite_info.py  (+ --download-project-id / --download-version-id)
catalog_suites.csv    — カタログ内のスイート一覧
suite_details.csv     — スイートの詳細情報
testcases.csv         — 全テストケース（シナリオ）× t4_dataset_ids
t4datasets.csv         — データセット単位のrichテーブル  ← 自動生成
download_commands.sh  — 一括ダウンロード用シェルスクリプト  ← 自動生成
    ↓
webauto pull           — 実データダウンロード
```

---

## Step 1 — スイート・テストケース情報の取得 + ダウンロードスクリプト生成

`get_suite_info.py` を実行すると、スイート情報の取得に加えて、自動的に `t4datasets.csv` と `download_commands.sh` も生成されます。

```bash
python get_suite_info.py <project_id> <catalog_id> [limit] [output_dir] \
    [-p DOWNLOAD_PROJECT_ID] [-v DOWNLOAD_VERSION_ID] [--no-download-script]

# 例
python get_suite_info.py x2_dev e2efe01d-e0c6-4d49-8223-817ff5d73204
python get_suite_info.py x2_dev e2efe01d-e0c6-4d49-8223-817ff5d73204 -p x2_dev -v 0
```

**出力ファイル:**

| ファイル | 説明 |
|---|---|
| `catalog_suites.csv` | カタログ内のスイート一覧（ID, 名前, 説明, 作成者, タイプ, version_id 等） |
| `suite_details.csv` | スイートの詳細（full JSON を flatten した全フィールド） |
| `testcases.csv` | **1行 = 1テストケース（シナリオ）**。スイート情報 + シナリオ情報 + `t4_dataset_ids` 配列 |
| `t4datasets.csv` | **1行 = 1一意データセット**。ダウンロードコマンド + 全関連スイート/シナリオ情報 |
| `download_commands.sh` | 全データセットの `webauto pull` コマンド列（実行権限付き） |

**主要オプション:**

| オプション | 説明 |
|---|---|
| `-p`, `--download-project-id` | ダウンロードコマンドのプロジェクトID（デフォルト: project_id と同じ） |
| `-v`, `--download-version-id` | データセットバージョンID（デフォルト: 0） |
| `--no-download-script` | t4datasets.csv / download_commands.sh の生成をスキップ |
| `limit` | スイート詳細を取得する件数（デフォルト: 全件） |

**内部動作:**
- API `/suites` の `catalogId` フィルタは正しく動作しないため、全スイートを取得後に `attachments[*].catalog_id` でクライアントサイドフィルタリングしています。
- 各スイートの詳細取得は API を叩くため、データ量が多いと時間がかかります。

### t4datasets.csv — カラム一覧

| カラム | 説明 |
|---|---|
| `dataset_id` | データセット UUID |
| `download_command` | 完全な `webauto data annotation-dataset pull` コマンド |
| `project_id` | プロジェクトID |
| `version_id` | アノテーションデータセットのバージョンID |
| `total_scenarios` | このデータセットを参照するシナリオ数 |
| `total_suites` | このデータセットを含むスイート数 |
| `suite_names` | 全スイート名（`;` 区切り） |
| `suite_ids` | 全スイートID（`;` 区切り） |
| `suite_types` | スイートタイプ（`;` 区切り） |
| `scenario_count_per_suite` | スイート別のシナリオ数（例: `Pn_Eval_Odaiba_x2gen2(1)`） |
| `scenario_names` | 全シナリオ名（`;` 区切り） |
| `scenario_display_names` | 全シナリオ表示名（`;` 区切り） |
| `scenario_ids` | 全シナリオID（`;` 区切り） |
| `scenario_descriptions` | シナリオの説明（`\|\|` 区切り） |

**download_commands.sh — 使い方:**

```bash
# デフォルト: ./datasets/ にダウンロード
./download_commands.sh

# 出力先を変更
OUTPUT_DIR=/mnt/qnapdata/t4datasets ./download_commands.sh

# 失敗時に停止しない
# → スクリプト冒頭 `set -e` をコメントアウト
```

---

## 個別実行（オプション）

すでに `testcases.csv` が在手にある場合、`generate_download_commands.py` を単体で実行して `t4datasets.csv` / `download_commands.sh` だけ再生成できます。

```bash
python generate_download_commands.py testcases.csv -p x2_dev -v 0
```

---

## 依存スクリプト

| ファイル | 説明 |
|---|---|
| `t4_visualizer/downloader.py` | API クライアント（スイート一覧・詳細・シナリオ情報取得） |
| `get_suite_info.py` | スイート取得 + テストケースCSV + ダウンロードスクリプト生成（一体化） |
| `generate_download_commands.py` | 個別実行用（testcases.csv → t4datasets.csv + download_commands.sh） |

### downloader.py の主要関数

```python
# プロジェクト内の全スイートから、特定の vehicle catalog に属するスイートを抽出
list_catalog_suites(project_id: str, vehicle_catalog_id: str) -> List[Dict]

# スイート詳細を取得
get_suite_info(project_id: str, suite_id: str) -> Dict

# シナリオ詳細を取得（t4_dataset_ids を含む）
get_scenario_info(project_id: str, scenario_id: str, scenario_version_id: int = None) -> Dict
```
