# Waymo 차량 끼어들기 예측

옆 차로의 상대 차량이 **앞으로 3초 안에 기준 차량 앞으로 끼어들지**를 과거 1초와 현재 정보만으로 예측한다. 데이터는 Waymo Open Motion Dataset v1.3.1 원본을 Colab에서 내려받지 않고 읽어 만들고, 정답은 미래 3초의 움직임을 보는 규칙으로 붙인 뒤 일부를 사람이 애니메이션으로 검토한다.

## 전체 흐름

```text
Waymo 원본 64개 파일 (GCS, 내려받지 않고 읽기)
  └─ dataset_build.ipynb (Colab)
       1. 분할 계획 고정: 원본 파일 번호로 train 40 · validation 12 · test 12
       2. 장면 해석 + Map Matching → 후보 차량 쌍 → 미래 3초 규칙으로 라벨 (1 / 0 / 보류)
       3. 모델 입력: X_basic 8개 · X_sequence 11×2×7 · X_bev 5×48×96
       4. Feature engineering 99개 → train 교차검증으로 feature 선택
       5. 사람 판정 반영 → waymo_dataset.zip (Google Drive)
  └─ dataset_review.ipynb (Colab): 원본 파일 하나씩 배치로 애니메이션 검토, 판정은 Drive에 즉시 저장
  └─ 로컬 (이 폴더): ZIP을 data/dataset/에 풀고 baseline·분석·그림
```

| 노트북 | 하는 일 |
|---|---|
| [dataset_build.ipynb](https://colab.research.google.com/drive/1690KK8uss6rPZL9GmYPZOO7XTMK-d9ZM) | 원본 처리부터 ZIP까지. 끊기면 다시 **모두 실행**하면 처리한 파일은 건너뛴다 |
| [dataset_review.ipynb](https://colab.research.google.com/drive/1ThjYG8R907jxM7XgZNT7ZMgmSPyRjGRa) | 배치 검토. 검토자마다 이름표를 정하고, 버튼을 누를 때마다 판정 1건이 저장된다 |

## 현재 데이터셋 (2026-11-20 구축)

| 분할 | 원본 파일 | 표본 | 양성 | 비고 |
|---|---:|---:|---:|---|
| train | 40개 (00000~00039) | 117,508 | 1,600 (1.36%) | 클래스 가중치 기준값 72.44 |
| validation | 12개 (00900~00911) | 29,274 | 학습 직전까지 확인하지 않음 | train과 같은 주행 구간 6,961건·같은 입력 124건 제외 |
| test | 12개 (00950~00961) | 24,404 | 마지막 평가 때 한 번만 확인 | 앞 분할과 같은 주행 구간 10,751건·같은 입력 218건 제외 |

- 원본 장면 31,371개 → 같은 주행 구간 26,986개로 묶음(장면 2개 이상인 구간 3,885개, 두 분할에 걸친 구간 2,237개)
- 같은 분할 안에서 입력이 완전히 같은 표본 2,487건을 제외해 분할 내부 중복 0건
- 양성은 같은 끼어들기에서 진입 직전 가장 가까운 기준 차량을 고르고, 그 차량의 기준 시점 중 하나를 고정 해시로 고르게 뽑는다. train 양성의 진입까지 남은 시간은 0~1초 344건(21.5%) · 1~2초 611건(38.2%) · 2~3초 645건(40.3%)
- Waymo 차량은 길이 5.29 m로 항상 같아서 기준 차량 길이를 보고 Waymo 차량을 알아보는 지름길이 생긴다. engineering feature에서는 기준 차량 길이를 빼고 차량 간격 계산에 표준 크기(4.7 m × 2.0 m)를 쓴다. `X_sequence`의 기준 차량 길이·너비(채널 5·6)는 원본 값 그대로이므로 시계열 모델에서는 이 두 채널을 빼고 쓴다
- 방향 각도는 ±π 경계에서 튀지 않게 이어 붙인 뒤 feature를 계산한다
- Feature: engineering 99개(움직임 64 + 도로·주변 차량 35) → 상관 필터 81개 → train 교차검증으로 고른 **76개**(`feature_names.json`의 `selected_engineered`)
- train 주행 구간 5-fold 교차검증 AP: 99개 0.824 · 상관 필터 81개 0.826 · 선택 76개 0.826
- 사람 검토 배치 64개(512건: train 320 · validation 96 · test 96), 현재 사람 판정 0건
- 원본 ZIP: Google Drive `waymo_dataset/waymo_dataset.zip` (200 MB), 로컬 `data/waymo_dataset.zip`

## 기본 ML 베이스라인 (train 주행 구간 5-fold 교차검증)

`scripts/train_baseline.py`의 `MODE = "train_cv"` 결과다. validation·test는 쓰지 않았다. F1은 같은 교차검증 예측에서 F1이 가장 높은 임계값을 고른 값이라 실제보다 약간 높게 나온다.

| 모델 | 기본 8개 | 움직임 64개 | 선택 76개 | 전체 99개 |
|---|---:|---:|---:|---:|
| 로지스틱 회귀 | AP 0.067 · F1 0.152 | 0.414 · 0.492 | 0.608 · 0.629 | 0.613 · 0.630 |
| 랜덤 포레스트 | 0.495 · 0.509 | 0.680 · 0.640 | 0.778 · 0.721 | 0.777 · 0.714 |
| 트리 부스팅 | 0.510 · 0.520 | 0.696 · 0.657 | 0.826 · 0.763 | 0.824 · 0.758 |
| 트리 부스팅 (클래스 가중치) | 0.506 · 0.517 | 0.707 · 0.665 | **0.828 · 0.768** | 0.827 · 0.770 |

- 양성 비율 기준선 AP는 0.014다
- 클래스 가중치 트리 부스팅 · 전체 99개의 진입까지 남은 시간별 재현율은 0~1초 91.3%(314/344) · 1~2초 85.3%(521/611) · 2~3초 45.1%(291/645)
- 같은 모델의 기준 차량 종류별 AP는 Waymo 차량 0.876 · 일반 차량 0.811이다. 기준 차량 길이를 빼도 앞차 간격 등에서 차이가 남으므로 두 값을 함께 보고한다

## 지키는 원칙

1. **분할은 데이터를 읽기 전에 원본 파일 번호로 정했다.** 같은 파일의 장면은 한 분할에만 들어가고, 원본 파일이 달라도 같은 주행 구간(Waymo 차량 위치가 겹치는 장면)은 앞 분할에만 남긴다. 교차검증도 주행 구간(`segment_id`) 단위로 나눈다.
2. **validation·test는 학습 직전까지 보지 않는다.** 데이터 분석·그림·feature 선택은 train만 쓰고, validation·test는 표본 수만 센다.
3. **실험은 train 교차검증, 최종 후보 비교와 분류 기준은 validation, test는 마지막에 한 번.** `scripts/train_baseline.py`의 `MODE`를 `"train_cv"` → `"validation"` → `"final_test"` 순서로 바꾼다. `"final_test"`는 모든 모델을 확정한 뒤 한 번만 실행한다.
4. **미래 정보는 정답을 만들 때만 쓴다.** `samples.csv`의 `auto_label`, `entry_*`, `lead_time_s`는 입력으로 쓰지 않는다(`lead_time_s`는 다중 작업 학습의 회귀 정답으로만 사용).

## 데이터셋 파일 (`data/dataset/`)

| 파일 | 내용 |
|---|---|
| `labels.csv` | sample_id, split, label, label_source(rule/human), review_status |
| `samples.csv` | 원본 파일, 장면·차량 번호, 기준 시점, 미래 정보(입력 금지) |
| `inputs.npz` | sample_id, X_basic, X_sequence, X_bev_packed, X_engineered — labels.csv와 같은 순서 |
| `features_basic.csv` | 기본 숫자 8개를 표로 본 것 |
| `feature_names.json` | 입력별 이름, engineering 99개(움직임·도로 목록 포함), train 기준 선택 목록 |
| `feature_selection.json` | 선택 근거 (결측·상관 필터, 주행 구간 단위 교차검증 순열 중요도) |
| `dataset_info.json`, `split_plan.json` | 표본 수, 처리 통계, 라벨 규칙, 분할 계획 |
| `excluded.csv`, `review_queue.csv` | 제외한 표본과 이유, 사람 검토 배치 |

## 폴더 구조

```text
data/                   GitHub에 올리지 않는다. 데이터셋은 Drive에서 받는다
  dataset/              Colab에서 받은 waymo_dataset.zip을 푼 폴더
  reviews/              검토 Colab에서 내려받은 판정 CSV (로컬 반영용)
  raw_samples/          로컬 확인·그림용 Waymo 원본 장면 7개
  predictions/          train_baseline.py가 저장하는 train CV 예측값 (결과 그림용)
notebooks/
  dataset_build.ipynb   데이터셋 만들기 (GitHub에 없음, Drive의 Colab과 같음)
  dataset_review.ipynb  배치 검토 (GitHub에 없음, Drive의 Colab과 같음)
  source/               두 노트북의 원본 (# %% 셀 구분 .py). 수정은 여기서 하고 build_notebooks.py로 변환
src/
  paths.py              데이터·결과·그림 폴더 경로
  dataset.py            데이터셋 읽기·정합성 검사·평가 지표·분할 내부 중복 확인
  features.py           engineering feature 99개 (노트북도 이 파일의 수식을 그대로 사용)
  models.py             트리 부스팅 설정 후보, 주행 구간 단위 bootstrap
  raw_scenario.py       Waymo 원본 읽기와 좌표 변환
  bev_sequence.py       과거 1초 BEV ZIP 만들기·읽기 (build_bev Colab과 로컬 검사가 함께 사용)
  plots/                그림 공통 스타일(style), 도로 장면(scene), 구축 과정(pipeline), 데이터 분석(analysis), 성능(results)
scripts/
  bev/                  Part D BEV 학습·탐색·기록 도구 (아래 Part D · BEV 학습과 기록)
  check_bev_sequence.py BEV ZIP 하나를 표와 대조하고 양성·음성 비교 GIF 저장
  train_baseline.py     로지스틱 회귀·랜덤 포레스트·트리 부스팅 × 기본 8개·움직임 64개·선택 76개·전체 99개
  plot_results.py       train CV 결과 그림 4장 → figures/model_results/ (train_baseline.py 다음에 실행)
  analyze_data.py       train만으로 라벨 구성, feature 분포·상관·선택, PCA 비교 그림 5장 → figures/data_analysis/
  plot_pipeline.py      원본 장면 → 학습 표본 구축 과정 그림 9장 → figures/data_pipeline/
  apply_reviews.py      data/reviews/의 판정 CSV를 data/waymo_dataset.zip 기준으로 data/dataset/에 다시 반영
  build_notebooks.py    notebooks/source/*.py → .ipynb
  check_notebook_locally.py  raw_samples로 데이터셋 노트북을 로컬에서 끝까지 실행
results/, figures/      스크립트 결과
```

## 실행

프로젝트 폴더에서 실행한다. 설정은 각 스크립트 위쪽의 상수로 바꾼다.

```bash
uv run --python 3.12 scripts/train_baseline.py
uv run --python 3.12 scripts/plot_results.py
uv run --python 3.12 scripts/analyze_data.py
uv run --python 3.12 scripts/plot_pipeline.py
uv run --python 3.12 scripts/apply_reviews.py      # 판정 CSV를 data/reviews/에 넣은 뒤
```

## 팀원 시작하기

코드는 GitHub, 데이터셋과 Colab 노트북은 Google Drive의 `waymo_dataset` 폴더로 나눠 공유한다. `data/` 폴더와 `.ipynb` 파일은 GitHub에 올리지 않는다.

1. Colab에서 쓸 Google 계정으로 [Waymo Open Dataset](https://waymo.com/open/)에 이용 등록을 한다. 검토 노트북이 Waymo 원본을 직접 읽기 때문에 필요하다.
2. 공유받은 Drive `waymo_dataset` 폴더를 **내 드라이브에 바로가기로 추가**한다. 바로가기 이름은 `waymo_dataset` 그대로 둔다.
3. 파이썬 실행 도구 [uv](https://docs.astral.sh/uv/getting-started/installation/)를 설치한다. 필요한 라이브러리는 스크립트를 처음 실행할 때 자동으로 설치된다.
4. 이 저장소를 내려받고, Drive의 `waymo_dataset.zip`을 `data/waymo_dataset.zip`에 둔 뒤 `data/dataset/`에 푼다. 압축을 풀면 `data/dataset/` 바로 아래에 `labels.csv`, `inputs.npz` 등 파일 11개가 보여야 한다.
5. 베이스라인을 실행해 준비가 됐는지 확인한다. 첫 줄에 `logistic`, `basic_8`, `CV AP 0.067`이 함께 나오면 데이터가 제대로 들어간 것이다. 전체 20개 조합은 시간이 걸리므로 확인만 할 때는 여기서 멈춰도 된다.

```bash
git clone https://github.com/todayoneul/Waymo_cut_in_Prediction.git
cd Waymo_cut_in_Prediction
mkdir -p data/dataset
cp ~/Downloads/waymo_dataset.zip data/                 # Drive에서 받은 ZIP 위치에 맞게 바꾼다
unzip data/waymo_dataset.zip -d data/dataset
uv run --python 3.12 scripts/train_baseline.py
```

6. 검토는 `dataset_review.ipynb`만 쓴다. 1번 준비 셀의 `REVIEWER_ID`에 내 이름표를 넣어야 실행된다. `dataset_build.ipynb`는 공유 데이터셋을 새로 만들어 덮어쓰므로 정해진 담당자만 실행한다.

## 사람 검토

1. 팀원은 공유받은 Drive의 `waymo_dataset` 폴더를 **내 드라이브에 바로가기로 추가**하고 `dataset_review.ipynb`를 연다.
2. `REVIEWER_ID`를 각자 다르게 정하고, `MY_BATCHES`에 맡은 배치 번호 범위(예: `10-19`)를 적은 뒤 모두 실행한다. 배치 하나는 원본 파일 하나에서 자동 양성·음성을 무작위로 최대 4건씩 뽑은 8건이다.
3. 판정은 `waymo_dataset/reviews/votes/<검토자>/`에 바로 저장된다. 의견이 갈리거나 보류인 표본은 데이터셋에서 빠진다.
4. 반영은 `dataset_build.ipynb`를 다시 모두 실행하거나, 판정 CSV를 내려받아 `scripts/apply_reviews.py`를 실행한다.

## 데이터 공유

`data/`의 파일은 Waymo Dataset License(비상업적 이용)를 따른다. Waymo 이용 약관에 동의하고 등록한 팀원과만 공유한다.

### Part D · 과거 1초 BEV

[build_bev.ipynb](https://colab.research.google.com/drive/1_MLlpAHVqvZlnhrvdeK2YSlpVwveU-aD)는 검토를 반영한 기존 표본을 원본 장면에 다시 연결한다. 기본 범위는 Train 00000번이다. 결과는 내 드라이브의 `bev_sequences/train_00000.zip`에 저장한다. 다른 Train 파일을 변환할 때는 첫 준비 셀의 `FILE_NUMBERS`에 해당 번호를 넣는다. 완료 파일은 저장된 표본과 라벨이 일치할 때 재사용한다.

로컬에서는 ZIP을 `data/bev_sequences/`에 놓고 프로젝트 루트에서 실행한다.

```sh
uv run --python 3.12 scripts/check_bev_sequence.py data/bev_sequences/train_00000.zip
```

검사기는 전체 ID·최종 라벨·회귀 정답을 대조하고, 두 차량의 11개 위치를 기존 `X_sequence` 및 두 해상도의 차량 마스크와 비교한다. ZIP 옆에 양성·음성 비교 GIF를 저장한다. 왼쪽은 1m, 오른쪽은 0.2m 격자다.

로컬 모델은 ZIP의 `manifest.json`에서 표본의 `member`를 찾아 그 NPZ를 읽고, `src.bev_sequence.read_frames(arrays, "20cm")`로 (11, 5, 240, 480) 입력을 복원한다. 다섯 채널은 차로 중심선, 차선·도로 경계, 기준 차량, 상대 차량, 주변 차량 순이다. 기준 차량 차체는 두 해상도 모두 4.7 × 2.0m로 통일했다. 모든 시점은 현재 기준 차량의 좌표계를 사용한다. 분류 정답은 `label`, 진입 시간 정답은 `time_target_s`이며 값이 없는 사례는 회귀 손실에서 제외한다.

### Part D · BEV 학습과 기록

`scripts/bev/`는 64개 BEV ZIP으로 CNN을 학습하고 그 과정을 기록하는 도구다. 모든 명령은 저장소 루트에서 실행한다. 실행 결과는 Git에서 제외되는 `data/training_runs/<실험 이름>/`에 쌓이고, 보고서에 쓸 실험만 `publish.py`로 `results/bev/`에 옮겨 커밋한다.

| 파일 | 하는 일 |
|---|---|
| `check_data.py` | 64개 ZIP의 표본 ID·분할·주행 구간·라벨·진입 시간을 `data/dataset/`과 대조한다. `--crc`는 ZIP 손상까지 검사한다 |
| `bench_loader.py` | DataLoader worker 수별 로딩 속도와 epoch당 로딩 시간을 잰다 |
| `lr_find.py` | 학습률을 지수적으로 올리며 손실을 기록하는 학습률 범위 시험. 추천값은 탐색 상한으로 쓴다 |
| `search.py` | fold 1의 학습 구간 안에서 학습률·너비·dropout·weight decay·batch를 무작위 탐색한다. fold 1의 평가 구간은 쓰지 않는다 |
| `train.py` | Train 주행 구간 5-fold 학습. 학습 구간을 다시 나눈 내부 검증 AP로 early stopping과 학습률 감소를 정하고, 최고 가중치를 복원해 평가 구간에 한 번 적용한다 |
| `report.py` | fold별 학습 곡선(`learning.png`)과 PR 곡선·혼동행렬·진입까지 남은 시간별 Recall(`evaluation.png`) |
| `elbow.py` | 기본 숫자 8개로 주행 상황을 K-Means로 묶고 inertia·silhouette로 군집 수를 비교한다 |
| `publish.py` | 끝난 실험의 설정·학습 이력·지표·그림을 `results/bev/<이름>/`으로 복사한다 |

모델은 3×3 합성곱(stride 2)·BatchNorm·ReLU 블록 3개, 3×6 공간 격자 평균, 64-unit 은닉층으로 구성되고 He 초기화를 쓴다. 11프레임 입력은 지도 2채널을 한 번, 차량 3채널(기준·상대·주변)을 프레임마다 쌓아 35채널이다. 20cm 입력은 먼저 5×5 stride-5 합성곱으로 1m 격자 크기에 맞춘 뒤 같은 블록을 지나므로 두 해상도의 수용영역이 같다. 학습은 양성 전부와 그 4배의 음성을 epoch마다 새로 뽑아 진행하고, 평가 점수에는 이 재추출로 생긴 확률 편향을 되돌리는 보정을 적용한다.

라이브러리는 팀 스크립트와 같은 Python 3.12·numpy 2.5.5·scikit-learn 1.9.1·matplotlib 3.10.9에 torch 2.9.1을 더해 `scripts/bev/requirements.txt`에 고정했다. scikit-learn 버전이 같아야 BEV의 5개 평가 fold가 `train_baseline.py`의 fold와 표본 단위까지 같아져 트리 모델과 fold별로 짝지어 비교할 수 있다. macOS(MPS)와 CPU에서는 다른 팀 스크립트처럼 uv로 실행한다.

```sh
uv run --python 3.12 --with-requirements scripts/bev/requirements.txt scripts/bev/check_data.py --crc --out data/training_runs/audit.json
uv run --python 3.12 --with-requirements scripts/bev/requirements.txt scripts/bev/train.py --resolution 1m --frames 11 --device mps --fold 1 --epochs 3 --out data/training_runs/smoke-1m-11f
```

NVIDIA GPU에서는 CUDA용 torch를 설치한 Python 3.12 이상 환경에 나머지 버전을 맞춘 뒤 `python`으로 실행한다.

```sh
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install numpy==2.5.5 scikit-learn==1.9.1 matplotlib==3.10.9
python scripts/bev/lr_find.py --resolution 1m --frames 11 --device cuda --out data/training_runs/lr-1m-11f
python scripts/bev/search.py --resolution 1m --frames 11 --device cuda --lr-min 5e-4 --lr-max 6e-3 --out data/training_runs/search-1m-11f
python scripts/bev/train.py --resolution 1m --frames 11 --device cuda --fold 0 --lr 5.6e-3 --width 32 --weight-decay 1e-4 --batch 32 --out data/training_runs/cv-1m-11f
python scripts/bev/report.py data/training_runs/cv-1m-11f
python scripts/bev/publish.py data/training_runs/cv-1m-11f
```

`train.py`의 기본값은 epoch 상한 100, 내부 검증 AP가 0.001을 넘게 오르지 않은 epoch가 4번째면 학습률 절반, 8번째면 중단, 양성:음성 1:4, seed 1729다. 전체 옵션은 `--help`로 본다. `run.json`에는 설정, 입력 채널·파라미터 수, Git 커밋, `train.py`의 SHA-256, 라이브러리 버전과 GPU가 남는다. 서로 비교하는 실행은 `run.json`의 scikit-learn 버전이 같아야 같은 fold를 쓴다. `results/bev/`에는 표본별 점수(`predictions.csv`), 구간 목록(`groups.csv`), 가중치(`best.pt`)를 올리지 않는다.
