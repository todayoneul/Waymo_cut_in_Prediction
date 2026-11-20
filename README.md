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
  plots/                분석 그림, 구축 과정 그림
scripts/
  train_baseline.py     로지스틱 회귀·랜덤 포레스트·트리 부스팅 × 기본 8개·움직임 64개·선택 76개·전체 99개
  analyze_data.py       train만으로 라벨 구성, feature 분포·상관, PCA 비교
  plot_pipeline.py      원본 장면 → 학습 표본 구축 과정 그림 7장
  apply_reviews.py      data/reviews/의 판정 CSV를 data/waymo_dataset.zip 기준으로 data/dataset/에 다시 반영
  build_notebooks.py    notebooks/source/*.py → .ipynb
  check_notebook_locally.py  raw_samples로 데이터셋 노트북을 로컬에서 끝까지 실행
results/, figures/      스크립트 결과
```

## 실행

프로젝트 폴더에서 실행한다. 설정은 각 스크립트 위쪽의 상수로 바꾼다.

```bash
uv run --python 3.12 scripts/train_baseline.py
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
2. `REVIEWER_ID`를 각자 다르게 정하고 모두 실행한다. 배치 하나는 원본 파일 하나에서 자동 양성·음성을 무작위로 최대 4건씩 뽑은 8건이다.
3. 판정은 `waymo_dataset/reviews/votes/<검토자>/`에 바로 저장된다. 의견이 갈리거나 보류인 표본은 데이터셋에서 빠진다.
4. 반영은 `dataset_build.ipynb`를 다시 모두 실행하거나, 판정 CSV를 내려받아 `scripts/apply_reviews.py`를 실행한다.

## 데이터 공유

`data/`의 파일은 Waymo Dataset License(비상업적 이용)를 따른다. Waymo 이용 약관에 동의하고 등록한 팀원과만 공유한다.
