"""notebooks/source/*.py(# %% 셀 구분)를 Colab 노트북(.ipynb)으로 바꿉니다."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "notebooks" / "source"
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOKS = ["dataset_build", "dataset_review"]
BACKTICK = chr(96)


# 노트북에도 src/features.py와 같은 feature 수식을 쓰도록, <<FEATURES>> 자리에 그 부분을 그대로 넣습니다.
def feature_code():
    text = (ROOT / "src" / "features.py").read_text()
    return text[text.index("X0, Y0"):].rstrip()


# 원본 텍스트를 셀 목록(종류, 내용)으로 나눕니다. 마크다운의 ~ 는 취소선이 되지 않게 바꿉니다.
def split_cells(text):
    text = text.replace("\\" + BACKTICK, BACKTICK).replace("<<FEATURES>>", feature_code())
    cells = []
    for block in re.split(r"^# %%", text, flags=re.M)[1:]:
        header, _, body = block.partition("\n")
        if "[markdown]" in header:
            lines = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.rstrip().splitlines()]
            cells.append(("markdown", "\n".join(lines).strip().replace("~", "\\~")))
        else:
            cells.append(("code", body.strip("\n")))
    return cells


# 셀 목록을 Colab에서 열 수 있는 .ipynb 형식으로 저장합니다.
def write_notebook(name, cells):
    notebook = {"cells": [], "nbformat": 4, "nbformat_minor": 0,
                "metadata": {"colab": {"name": f"{name}.ipynb", "provenance": []},
                             "kernelspec": {"display_name": "Python 3", "name": "python3"},
                             "language_info": {"name": "python"}}}
    for i, (kind, body) in enumerate(cells):
        cell = {"cell_type": kind, "metadata": {"id": f"{name}-{i:02d}"}, "source": body.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        notebook["cells"].append(cell)
    path = NOTEBOOK_DIR / f"{name}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    return path


# 노트북 셀을 실행 순서대로 돌려주는 함수(로컬 확인 스크립트에서 사용)
def code_cells(name):
    return [body for kind, body in split_cells((SOURCE_DIR / f"{name}.py").read_text()) if kind == "code"]


if __name__ == "__main__":
    for name in NOTEBOOKS:
        source = SOURCE_DIR / f"{name}.py"
        if not source.exists():
            continue
        cells = split_cells(source.read_text())
        for kind, body in cells:
            if kind == "code":
                compile(body.replace("%pip", "#pip"), name, "exec")   # 문법 오류가 있으면 여기서 멈춥니다.
        print(f"{write_notebook(name, cells).name}: 셀 {len(cells)}개")
