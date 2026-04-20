# evaluator_result_parser

自動運転評価システムの結果解析・可視化ツール集。

評価 JSONL のメトリクス分析と、T4 dataset の画像・点群可視化の 2 種類のツールを提供します。

---

## リポジトリ構成

```
evaluator_result_parser/
├── docs/                     # 設計・データフロー資料（例: Three.js viewer パイプライン）
├── result_parser/            # 評価結果（JSONL）の解析ツール
│   ├── jsonl_parser.py       # JSONL読み込み → CSV変換
│   ├── covariance_analysis.py# 共分散 vs 誤差の散布図
│   └── metrics_visualizer.py # TPrate / mAP・距離別分析
│
└── t4_visualizer/            # T4 dataset の可視化ツール
    ├── visualize.py          # UUID + タイムスタンプで1シーンを可視化 / render API
    ├── batch.py              # CSV/Parquet から複数シーンをバッチ可視化
    ├── server.py             # HTTP サーバー（FastAPI）
    ├── inspect.py            # データセット内のシーン・センサー情報を表示
    └── downloader.py         # webauto DL + LRU キャッシュ管理
```

---

## インストール・起動

### 新規 PC への一発セットアップ（推奨）

[uv](https://docs.astral.sh/uv/getting-started/installation/) が必要です。

```bash
# 環境構築（初回のみ）
bash setup.sh

# サーバー起動（フォアグラウンド）
bash serve.sh

# サーバー起動（SSH 切断後も継続）
bash serve.sh start

# 再起動 / 停止 / 状態確認
bash serve.sh restart
bash serve.sh stop
bash serve.sh status
```

`setup.sh` は仮想環境の作成・依存ライブラリのインストール・動作確認を自動で行います。
`serve.sh` はデフォルトで `/mnt/qnapdata/internal/t4datasets` を参照してサーバーを起動します。

```bash
# オプション例
bash setup.sh --python 3.11 --venv .venv
bash serve.sh --data-dir /path/to/datasets --port 8080
bash serve.sh start --data-dir /path/to/datasets --port 8080
```

詳細は各スクリプトの `--help` を参照してください。

### 手動インストール

```bash
# 依存ライブラリ
pip install pandas matplotlib seaborn numpy pillow pyarrow
pip install git+https://github.com/tier4/t4-devkit.git

# パッケージ本体
pip install -e .

# HTTP サーバーも使う場合（fastapi + uvicorn を追加）
pip install -e ".[server]"
```

`pip install -e .` で以下の CLI コマンドが利用できるようになります:

| コマンド | 説明 |
|---|---|
| `t4-visualize` | 1シーンをインタラクティブ / PNG で可視化 |
| `t4-batch` | CSV/Parquet から複数シーンをバッチ可視化 |
| `t4-multi` | 複数の CSV を一括処理 |
| `t4-inspect` | データセットの情報確認 |
| `t4-cache` | ローカルキャッシュの管理 |
| `t4-server` | HTTP API サーバーを起動（要 `.[server]`） |

---

## result_parser — 評価結果解析ツール

評価システムが出力する JSONL ファイルを読み込み、検出物体のメトリクスを解析します。

### データフロー

```
JSONL files
    ↓ jsonl_parser.py
extracted_objects.csv
    ↓                    ↓
covariance_analysis.py   metrics_visualizer.py
(共分散 vs 誤差)          (TPrate / mAP / 距離別分析)
```

### jsonl_parser.py

JSONL ファイル群を読み込み、検出物体の情報を CSV に変換します。

```bash
python result_parser/jsonl_parser.py <folder_name>
# 例: python result_parser/jsonl_parser.py Nishishinjuku
# 出力: extracted_objects.csv
```

主要クラス:
- `JSONLProcessor` — フォルダ内の `.jsonl` を一括読み込み
- `ObjectExtractor` — ego 情報・検出物体（位置/速度/共分散等）を抽出
- `ObjectProcessor` — 上記を組み合わせて DataFrame を生成

### covariance_analysis.py

`extracted_objects.csv` を読み込み、推定誤差と共分散値の相関を可視化します。

```bash
python result_parser/covariance_analysis.py
```

出力: X/Y/Yaw の共分散 vs BEV/Heading 誤差の散布図（matplotlib）。

### metrics_visualizer.py

TPrate・mAP の計算と距離別分析を行います。

```bash
python result_parser/metrics_visualizer.py
```

主要機能:
- `DatasetManager.calc_tprate()` — カテゴリ別 TPrate
- `DatasetManager.calc_mAP()` — mean Average Precision
- `DatasetManager.distance_based_fp_tp_fn()` — 距離ビン別 FP/TP/FN
- `DatasetManager.distance_based_errors()` — 距離ビン別 BEV/Yaw 誤差

---

## t4_visualizer — T4 dataset 可視化ツール

[t4-devkit](https://github.com/tier4/t4-devkit) を使って T4 dataset の画像・点群を可視化します。

### visualize.py — 1シーンの可視化

UUID（データセットパス）とタイムスタンプを指定して、最も近いサンプルの画像と点群を表示します。

```bash
# 基本（インタラクティブ表示）
t4-visualize /path/to/t4dataset 1609459200000000

# PNG として保存
t4-visualize /path/to/t4dataset 1609459200000000 --save-dir output/

# Rerun で3D可視化
t4-visualize /path/to/t4dataset 1609459200000000 --rerun

# 特定カメラのみ
t4-visualize /path/to/t4dataset 1609459200000000 --cameras CAM_FRONT,CAM_BACK

# バウンディングボックスなし
t4-visualize /path/to/t4dataset 1609459200000000 --no-annotations
```

出力:
- `<save_dir>/<timestamp>_cameras.png` — 全カメラ画像（2D bbox オーバーレイ付き）
- `<save_dir>/<timestamp>_pointcloud.png` — LiDAR 鳥瞰図（BEV、3D bbox フットプリント付き）

Python API としても使用できます:

```python
from t4_devkit import Tier4
from t4_visualizer.visualize import find_closest_sample, visualize_static

t4 = Tier4("/path/to/dataset")
sample = find_closest_sample(t4, timestamp_us=1609459200000000)
visualize_static(t4, sample, save_dir="output/", filename_prefix="my_scene")
```

### inspect.py — データセット情報の確認

タイムスタンプ範囲・センサーチャンネル・サンプル数を確認します。
可視化の前にまずここで確認するのが便利です。

```bash
t4-inspect /path/to/t4dataset
```

出力例:
```
============================================================
  Scenes (1)
============================================================
  name        : scene-0001
  token       : abc123...
  nbr_samples : 40

============================================================
  Samples (40)
============================================================
  Timestamp range (µs)  : 1609459200000000  →  1609459204000000
  Duration              : 4.000 s

============================================================
  Sensor Channels (from first sample)
============================================================
  CAM_FRONT                                format=jpg
  CAM_BACK                                 format=jpg
  LIDAR_TOP                                format=pcd
```

### batch.py — バッチ可視化パイプライン

CSV または Parquet で指定した複数シーンを一括で可視化します。

#### 入力ファイル形式

| カラム | 必須 | 説明 |
|---|---|---|
| `t4dataset_id` | ✅ | データセット識別子（ダウンロード・検索に使用） |
| `scenario_name` | ✅ | シーン名 |
| `frame_index` | ✅ | フレーム番号（0始まり） |
| `t4dataset_name` | | データセットの表示名（ダウンロード確認プロンプトで使用） |
| `status` | | グループ名（例: `degrade`, `improved`）。サブフォルダに分類される |
| `cameras` | | 表示するカメラをカンマ区切りで指定（省略=全カメラ） |
| `description` | | 自由記述のメモ |
| `label` | | 対象物体のラベル文字列（例: `car`, `pedestrian`）。出力ファイル名の先頭に付与される |
| `uuid` | | 対象物体のインスタンストークン（T4 dataset の `instance_token`） |
| `x` / `y` / `z` | | 自車座標系での物体位置（メートル） |
| `width` / `length` / `height` | | 物体の BBOX サイズ（メートル） |
| `yaw` | | 物体の向き（ラジアン） |

1 行 = 1 物体。同じ `(t4dataset_id, scenario_name, frame_index)` の行は 1 フレームにまとめて可視化されます。

入力例 (`scenes.csv`):
```csv
t4dataset_id,scenario_name,frame_index,status,label,uuid,x,y,z
dataset-abc,scene-001,0,degrade,car,inst-uuid-001,10.5,2.3,0.5
dataset-abc,scene-001,5,improved,pedestrian,inst-uuid-002,8.1,0.0,0.9
dataset-def,scene-002,10,,,,,,,
```

#### 使い方

```bash
# 基本（ダウンロード確認プロンプトあり）
t4-batch scenes.csv --output-dir results/

# 確認をスキップ（CI・スクリプト向け）
t4-batch scenes.csv --output-dir results/ --yes

# 一時ストレージ（終了時に自動削除）
t4-batch scenes.parquet --temp --output-dir results/ --yes

# カスタムデータディレクトリ
t4-batch scenes.csv --data-dir /mnt/ssd/t4data --output-dir results/

# DL不要（既にデータがある場合）
t4-batch scenes.csv --no-download --data-dir /existing/data --output-dir results/

# 並列処理
t4-batch scenes.csv --output-dir results/ --workers 4

# LRU キャッシュ上限を変更（デフォルト10）
t4-batch scenes.csv --output-dir results/ --cache-limit 5

# 対象物体の crop 表示（物体ごとに最良カメラを切り出し）
t4-batch scenes.csv --output-dir results/ --crop-view

# 最初のエラーで即停止
t4-batch scenes.csv --output-dir results/ --fail-fast
```

#### 出力構造

`status` 列がある場合、ステータスごとにサブフォルダへ分類されます。

通常モード（`--crop-view` なし）の出力ファイル名:
```
<label>_<t4dataset_id>_<scenario_name>_f<frame_index>_cameras.png
<label>_<t4dataset_id>_<scenario_name>_f<frame_index>_pointcloud.png
<label>_<t4dataset_id>_<scenario_name>_f<frame_index>_meta.txt
```

`--crop-view` モードでは、物体ごとに最良カメラを選び**カメラ別に 1 枚**生成します。
異なるカメラに写る物体は別ファイルになります:
```
<label>_<t4dataset_id>_<scenario_name>_f<frame_index>_<CHANNEL>_visualization_crop.png
```

```
results/
├── batch_summary.csv
├── degrade/
│   ├── car_dataset-abc_scene-001_f000000_CAM_FRONT_visualization_crop.png
│   ├── car_dataset-abc_scene-001_f000000_CAM_BACK_visualization_crop.png
│   ├── car_dataset-abc_scene-001_f000000_meta.txt
│   └── car_dataset-abc_scene-001_f000000_pointcloud.png
└── improved/
    ├── pedestrian_dataset-abc_scene-001_f000005_CAM_FRONT_visualization_crop.png
    └── ...
```

`batch_summary.csv` の内容:

| カラム | 説明 |
|---|---|
| `t4dataset_id` | データセットID |
| `t4dataset_name` | データセット表示名 |
| `scenario_name` | シーン名 |
| `frame_index` | フレーム番号 |
| `status` | ステータス |
| `success` | 成否 |
| `output_dir` | 出力先ディレクトリ |
| `filename_prefix` | 出力ファイルのプレフィックス |
| `error` | エラーメッセージ（失敗時） |

### t4-multi — 複数 CSV の一括処理

複数の CSV / Parquet ファイルを 1 回の実行でまとめて処理します。
各 CSV にラベルを付けることで、出力がラベル別サブフォルダに整理されます。

```bash
# ラベル付きで実行（label:path 形式）
t4-multi improve:improve.csv degrade:degrade.csv -o ./viz --yes

# ラベル省略（ファイル名のstemが自動でラベルになる）
t4-multi improve.csv degrade.csv -o ./viz --yes

# 並列処理 + crop 表示
t4-multi improve:improve.csv degrade:degrade.csv -o ./viz -j 4 --crop-view

# サーバー上の1段ネスト構成 (dest_dir/*/<id>) に対応する場合
t4-multi improve:improve.csv -o ./viz --no-download \
    --data-dir /mnt/server/datasets --search-depth 1
```

#### --search-depth — ネストされたデータセットレイアウトへの対応

データセットサーバーによっては `dest_dir/*/<t4dataset_id>` のようにグループフォルダが
1段挟まる場合があります。`--search-depth N`（デフォルト `1`）でサブディレクトリを探索します。

| layout | 必要な search-depth |
|---|---|
| `dest_dir/<id>/` | 0 または 1（フラットは常に優先検索） |
| `dest_dir/group/<id>/` | 1（デフォルト） |

`--no-download` 時の present / missing サマリーおよびダウンロード前確認の
表示も同じ depth で計算されます。

内部で**3フェーズ**処理が行われます:
1. **Load** — 全 CSV を読み込み、必要なデータセット ID の集合を収集
2. **Download** — `DatasetCache.ensure_many()` で重複なく一括取得。LRU の不要エントリを先に退避
3. **Visualize** — 各 CSV をラベル別ディレクトリに出力

出力構造:
```
./viz/
├── improve/
│   ├── batch_summary.csv
│   ├── car_dataset-abc_scene-001_f000000_cameras.png
│   └── ...
└── degrade/
    ├── batch_summary.csv
    └── ...
```

### render_frame — Python API（ファイル書き出し不要）

`render_frame` は1フレームをレンダリングして **PNG バイト列** を返す関数です。
ファイルシステムを介さないため、HTTP レスポンスへの直接埋め込みや並列処理に使えます。

```python
from pathlib import Path
from t4_visualizer.visualize import (
    VisualizationRequest,
    TargetObject,
    render_frame,
)

request = VisualizationRequest(
    dataset_path=Path("/mnt/t4data/abc123"),
    scenario_name="scene-0001",
    frame_index=5,
    target_objects=[
        TargetObject(uuid="", x=10.5, y=2.3, z=0.5, label="car"),
    ],
    crop_cameras=True,   # 対象物周辺をクロップ
)

result = render_frame(request)
# result.images  : List[RenderImage]
# result.sample_token : str
# result.timestamp_us : int

for img in result.images:
    print(img.label)           # "combined" or カメラチャンネル名
    Path(f"{img.label}.png").write_bytes(img.data)
```

#### キャッシュ済み Tier4 を渡して高速化

同一データセットを繰り返しレンダリングする場合、`Tier4` インスタンスを外部でキャッシュして
`t4` 引数に渡すとロードコストを省けます。

```python
from t4_devkit import Tier4

t4 = Tier4("/mnt/t4data/abc123")   # 一度だけロード
for frame_index in range(10):
    req = VisualizationRequest(
        dataset_path=Path("/mnt/t4data/abc123"),
        scenario_name="scene-0001",
        frame_index=frame_index,
    )
    result = render_frame(req, t4=t4)   # ロード不要
```

#### 並列処理

`render_frame` は副作用なし（各呼び出しがローカルな一時ディレクトリを使用）なので
`ThreadPoolExecutor` にそのまま投げられます。

```python
from concurrent.futures import ThreadPoolExecutor

requests = [
    VisualizationRequest(dataset_path=..., scenario_name=..., frame_index=i)
    for i in range(20)
]

with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(render_frame, requests))
```

#### データ型まとめ

**VisualizationRequest**

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `dataset_path` | `Path` | 必須 | T4 dataset root のローカルパス |
| `scenario_name` | `str` | 必須 | シーン名 |
| `frame_index` | `int` | 必須 | 0始まりのフレーム番号 |
| `target_objects` | `List[TargetObject]` | `[]` | ハイライト対象の物体リスト |
| `cameras` | `Optional[List[str]]` | `None` | 表示カメラ（省略=全カメラ） |
| `show_annotations` | `bool` | `True` | アノテーション表示 |
| `version` | `Optional[str]` | `None` | データセットバージョン |
| `crop_cameras` | `bool` | `False` | 対象物周辺クロップモード |
| `crop_padding` | `int` | `40` | クロップの余白ピクセル |
| `crop_min_size` | `int` | `300` | クロップの最小サイズ（ピクセル） |

**VisualizationResult**

| フィールド | 型 | 説明 |
|---|---|---|
| `images` | `List[RenderImage]` | 生成された図（標準モード: 1件、クロップモード: カメラ数分） |
| `sample_token` | `str` | レンダリングしたサンプルのトークン |
| `timestamp_us` | `int` | サンプルのタイムスタンプ（マイクロ秒） |

**RenderImage**

| フィールド | 型 | 説明 |
|---|---|---|
| `data` | `bytes` | PNG バイト列 |
| `label` | `str` | `"combined"`（標準）またはカメラチャンネル名（クロップ） |

---

### t4-server — HTTP API サーバー

`render_frame` を HTTP 経由で呼び出せるサーバーです。
複数のデータセットを事前にキャッシュするため、繰り返しリクエストの際のロードコストを最小化します。

詳細なシーケンス図は `docs/render_server_pipeline.md` を参照してください。

#### 起動

```bash
# serve.sh を使う場合（推奨）
bash serve.sh
bash serve.sh start
bash serve.sh restart
bash serve.sh --data-dir /mnt/t4data --port 8080

# t4-server を直接使う場合
t4-server --data-dir /mnt/t4data --port 8080 --tier4-cache 4
```

`bash serve.sh start` は PID ファイルとログファイルを `.run/` 配下に作成し、SSH を切断してもサーバーが継続するようにバックグラウンド起動します。
停止や再起動は `bash serve.sh stop` / `bash serve.sh restart`、ログ確認は `bash serve.sh logs` で行えます。

起動すると接続先 URL が表示されます:

```
  ローカル          : http://localhost:8000
  ネットワーク内    : http://192.168.1.42:8000

  API ドキュメント  : http://localhost:8000/docs
  ヘルスチェック    : http://localhost:8000/health
  データセット一覧  : http://localhost:8000/datasets
```

`serve.sh` / `t4-server` オプション一覧:

| オプション | デフォルト | 説明 |
|---|---|---|
| `--data-dir` | `./t4datasets` | データセットのルートディレクトリ |
| `--search-depth` | `1` | サブディレクトリ探索の深さ |
| `--host` | `0.0.0.0` | バインドホスト |
| `--port` | `8000` | バインドポート |
| `--workers` | `8` | uvicorn ワーカープロセス数 |
| `--tier4-cache` | `8` | メモリ上に保持する Tier4 インスタンス数 |
| `--project-id` | 環境変数依存 | webauto プロジェクト ID |
| `--venv` | `.venv` | 仮想環境ディレクトリ |
| `--pid-file` | `.run/t4-server.pid` | PID ファイル |
| `--log-file` | `.run/t4-server.log` | ログファイル |

#### エンドポイント

**`POST /render`** — フレームをレンダリング

リクエスト:

```json
{
    "t4dataset_id": "abc123",
    "scenario_name": "scene-0001",
    "frame_index": 5,
    "target_objects": [
        {"uuid": "", "x": 10.5, "y": 2.3, "z": 0.5, "label": "car",
         "width": 1.8, "length": 4.5, "height": 1.6, "yaw": 0.3}
    ],
    "cameras": null,
    "show_annotations": true,
    "crop_cameras": true,
    "crop_padding": 40,
    "crop_min_size": 300
}
```

レスポンス:

```json
{
    "sample_token": "deadbeef...",
    "timestamp_us": 1609459200000000,
    "images": [
        {
            "label": "CAM_FRONT",
            "png_base64": "iVBORw0KGgo..."
        }
    ]
}
```

**`GET /health`** — サーバー死活確認

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

#### 負荷試験 / 同時アクセス確認

`t4-load-test` は `POST /render` に対して段階的に並列リクエストを送り、**同時アクセス数ごとの応答時間・スループット・失敗率**を比較します。サーバーが返す `elapsed_ms` / `render_ms` / `tier4_load_ms` も一緒に集計するため、クライアント側待ち時間とサーバー内部処理時間の両方を見られます。

```bash
# まずサーバー起動
t4-server --data-dir /mnt/t4data --port 8000 --workers 1

# データセット/シナリオを自動検出して 1,2,4,8,16 並列で比較
t4-load-test --base-url http://127.0.0.1:8000 \
  --requests-per-level 20 \
  --csv-out benchmark.csv

# 対象を明示して比較（より再現しやすい）
t4-load-test --base-url http://127.0.0.1:8000 \
  --dataset-id abc123 \
  --scenario-name scene-0001 \
  --frame-index 5 \
  --concurrency 1,2,4,8,16,32 \
  --requests-per-level 30
```

出力は Markdown テーブル形式で、例えば以下を比較できます。

- `client_p95_ms` / `client_p99_ms`: 利用者が体感する待ち時間が同時アクセス数でどう増えるか
- `throughput_rps`: 並列数を上げた時に処理量がどこで頭打ちになるか
- `server_render_avg_ms`: 純粋な描画処理時間が負荷で悪化しているか
- `tier4_load_avg_ms`: キャッシュ未命中やデータロードがボトルネックになっているか

`--request-file request.json` を使うと、`target_objects` を含む完全な `POST /render` ボディをそのまま再利用できます。

3D / TLR ビューア相当のサーバー負荷を見たい場合は `--mode` を切り替えます。

```bash
# frame.bin だけを継続的に取得するケース
t4-load-test --mode viewer-three-frame-bin \
  --base-url http://127.0.0.1:8000 \
  --dataset-id abc123 \
  --scenario-name scene-0001 \
  --frame-index 5 \
  --concurrency 1,2,4,8,16,32 \
  --requests-per-level 30 \
  --csv-out viewer_three_framebin.csv \
  --plot-out viewer_three_framebin.png

# Three.js viewer の代表的なサーバー処理:
# /viewer/three -> /viewer/three/meta -> /viewer/three/frame.bin
t4-load-test --mode viewer-three \
  --base-url http://127.0.0.1:8000 \
  --dataset-id abc123 \
  --scenario-name scene-0001 \
  --frame-index 5 \
  --concurrency 1,2,4,8,16 \
  --requests-per-level 20

# カメラオーバーレイと lanelet も含めて、より実運用に近づける
t4-load-test --mode viewer-three \
  --base-url http://127.0.0.1:8000 \
  --dataset-id abc123 \
  --scenario-name scene-0001 \
  --frame-index 5 \
  --viewer-three-include-camera-overlay \
  --viewer-three-include-lanelet \
  --viewer-three-all-cameras \
  --concurrency 1,2,4,8

# TLR viewer の代表的なサーバー処理:
# /viewer/tlr -> /viewer/tlr/frame
t4-load-test --mode viewer-tlr \
  --base-url http://127.0.0.1:8000 \
  --dataset-id abc123 \
  --scenario-name scene-0001 \
  --frame-index 5 \
  --camera CAM_FRONT \
  --concurrency 1,2,4,8,16
```

注意:

- `viewer-three` / `viewer-tlr` は**ビューアが呼ぶサーバー API 群の性能**を測定します。
- これはブラウザ内の WebGL FPS や描画負荷そのものではありません。クライアント GPU を含めた体感性能を測るには、別途ブラウザ自動操作ベンチマークが必要です。
- `--plot-out` を付けると、並列数に対する `RPS` と `p50/p95/p99` 遅延の推移を PNG で保存できます。

**`GET /datasets`** — 利用可能なデータセット一覧

```bash
curl http://localhost:8000/datasets
# {"data_dir": "/mnt/t4data", "datasets": ["abc123", "def456"]}
```

#### Python クライアント例

```python
import base64, requests
from pathlib import Path

resp = requests.post("http://localhost:8000/render", json={
    "t4dataset_id": "abc123",
    "scenario_name": "scene-0001",
    "frame_index": 5,
    "crop_cameras": True,
})
resp.raise_for_status()

for img in resp.json()["images"]:
    data = base64.b64decode(img["png_base64"])
    Path(f"{img['label']}.png").write_bytes(data)
```

#### Three.js 3D viewer（`/viewer/three`）— データパイプライン

ブラウザ上の WebGL ビューアは、T4 点群・データセット由来の 3D ボックスに加え、親ページから渡す**評価 GT / 推論 EST** を 3D・カメラ投影の両方で重ねて表示できます。HTTP エンドポイント間の関係、`frame.bin` の役割、`postMessage` 連携、カメラオーバーレイ（GET/POST）までを **シーケンス図・フローチャート付き**でまとめた資料:

- **[docs/viewer_3d_pipeline.md](docs/viewer_3d_pipeline.md)**

例（ローカル）: `http://localhost:8000/viewer/three?t4dataset_id=<id>&scenario_name=<scene>&frame_index=0`

カメラキャリブレーション情報だけ欲しい場合は、`/viewer/three/camera-info` で取得できます。

```text
http://localhost:8000/viewer/three/camera-info?t4dataset_id=<id>&scenario_name=<scene>&frame_index=0&camera=CAM_FRONT
```

---

### downloader.py — ダウンローダーと LRU キャッシュ

#### ダウンロードの仕組み

デフォルトでは `webauto data annotation-dataset pull` を使ってデータセットを取得します。

```bash
webauto data annotation-dataset pull \
    --project-id <project_id> \
    --annotation-dataset-id <t4dataset_id> \
    --asset-dir <dest_dir>
```

webauto は `<dest_dir>/annotation_dataset/<t4dataset_id>/<version>/` にデータを展開します。
downloader はこれを自動的に `<dest_dir>/<t4dataset_id>/` へ**フラット化**します。

#### 設定方法（優先順位順）

**方法1: 環境変数 `T4_DOWNLOAD_CMD` でコマンド全体を差し替え**

```bash
export T4_DOWNLOAD_CMD="my_tool --id {t4dataset_id} --out {dataset_path}"
t4-batch scenes.csv --output-dir results/
```

プレースホルダー:
- `{t4dataset_id}` — データセットID
- `{dest_dir}` — 親ディレクトリ（例: `./t4datasets`）
- `{dataset_path}` — データセットの展開先（`{dest_dir}/{t4dataset_id}`）

**方法2: 環境変数 `WEBAUTO_PROJECT_ID` でプロジェクト ID だけ変更**

```bash
export WEBAUTO_PROJECT_ID="my_project_id"
t4-batch scenes.csv --output-dir results/
```

**方法3: モジュール定数を直接編集**

```python
# t4_visualizer/downloader.py
WEBAUTO_PROJECT_ID: str = "my_project_id"
DOWNLOAD_CMD_TEMPLATE: str = "my_tool --id {t4dataset_id} --out {dataset_path}"
```

**方法4: Python API から直接呼び出し**

```python
from t4_visualizer.downloader import download_dataset, dataset_is_cached
from pathlib import Path

data_dir = Path("./t4datasets")

# キャッシュ確認
if not dataset_is_cached("dataset-abc", data_dir):
    path = download_dataset("dataset-abc", data_dir)
else:
    path = data_dir / "dataset-abc"
```

#### LRU キャッシュ — DatasetCache

ディスク上のデータセットを LRU（最近最も使われていないものから削除）で管理します。
`max_cached` 上限を超えたとき、最も古いデータセットを自動削除します。

```python
from t4_visualizer.downloader import DatasetCache
from pathlib import Path

cache = DatasetCache(data_dir=Path("./t4datasets"), max_cached=5)

# 1件取得（必要ならDL、LRU超過なら古いものを退避）
path = cache.ensure("dataset-abc")

# 複数件を効率的に取得（必要な分だけ退避してから一括DL）
paths = cache.ensure_many(["dataset-abc", "dataset-def", "dataset-ghi"])
# → {"dataset-abc": Path(...), "dataset-def": Path(...), ...}

# キャッシュ状態確認
for entry in cache.status():
    print(f"{entry['t4dataset_id']}  {entry['size_mb']:.0f} MB  {entry['last_accessed']}")

# LRU から N 件になるまで削除
cache.evict_lru(keep=3)

# 全消去
cache.clear()
```

キャッシュインデックスは `<data_dir>/.cache_index.json` に保存されます。
並列ワーカーからの競合はファイルロック（`.cache_lock`）で直列化されます。

#### t4-cache CLI — キャッシュの手動管理

```bash
# キャッシュ状態の確認（LRU順に表示）
t4-cache status

# 出力例:
# #    t4dataset_id                             last_accessed                size_mb  on_disk
# -----------------------------------------------------------------------------------------
# 1    dataset-abc                              2024-01-10T08:00:00+00:00     1234.5  yes
# 2    dataset-def                              2024-01-11T12:00:00+00:00      987.0  yes
# -----------------------------------------------------------------------------------------
# Total: 2 datasets, 2221.5 MB  /  Limit: 10

# 古いものから削除して 3 件だけ残す
t4-cache evict --keep 3

# 全消去
t4-cache clear

# カスタムディレクトリ・上限を指定
t4-cache --data-dir /mnt/ssd/t4data --limit 5 status
```

#### 読み取り専用ファイルシステムへの対応（シャドウキャッシュ）

CIFS / NFS マウントなど書き込み不可の場所にデータセットがある場合、
一部の webauto エクスポートに含まれない JSON テーブル（`lidarseg.json` 等）の
スタブファイルをそこに書き込もうとして `Permission denied` が発生することがあります。

この場合、自動的に **ローカルシャドウ** が作成されます：

```
~/.cache/t4_shadow/<hash>/
    0/
        annotation/   ← annotation JSON のローカルコピー（スタブを書き込む）
        data          → NAS の data/ へのシンボリックリンク
```

スタブはローカルコピーへ書かれ、NAS には一切書き込みません。
画像・点群データはシンボリックリンク経由で NAS から直接読まれます。

> **Note**: このシャドウキャッシュは t4_devkit の upstream の問題（オプションテーブルが
> 欠損している場合に `FileNotFoundError` を投げる）に対するワークアラウンドです。
> t4_devkit 側が修正された際にはこの仕組みを除去する予定です。

---

## 依存ライブラリ

| ライブラリ | 用途 |
|---|---|
| `pandas` | CSV/Parquet の読み書き |
| `matplotlib` | 可視化（全ツール共通） |
| `seaborn` | 散布図（`covariance_analysis.py`） |
| `numpy` | 数値処理 |
| `Pillow` | 画像読み込み（`visualize.py`） |
| `pyarrow` | Parquet 読み込み（`batch.py`） |
| `t4-devkit` | T4 dataset ロード（`t4_visualizer/` 全般） |

```bash
pip install pandas matplotlib seaborn numpy pillow pyarrow
pip install git+https://github.com/tier4/t4-devkit.git
pip install -e .
```
