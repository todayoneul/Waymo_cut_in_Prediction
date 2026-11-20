"""프로젝트 폴더 경로. 데이터 위치가 바뀌면 이 파일만 고친다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "dataset"          # Colab에서 받은 waymo_dataset.zip을 푼 폴더
REVIEWS_DIR = ROOT / "data" / "reviews"          # 검토 Colab에서 내려받은 판정 CSV
RAW_SAMPLES_DIR = ROOT / "data" / "raw_samples"  # 그림·로컬 확인용 Waymo 원본 장면
PREDICTIONS_DIR = ROOT / "data" / "predictions"  # train 교차검증 예측값 (결과 그림용)
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
