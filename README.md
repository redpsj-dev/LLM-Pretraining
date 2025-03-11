# [홍정모 연구소](https://honglab.co.kr/)

2차 저작물에는 참고자료에 "[홍정모 연구소](https://honglab.co.kr/)"를 꼭 적어주세요.

> 예시) [홍정모 연구소](https://honglab.co.kr/)의 OOO을 참고하였습니다.

## LLM 만들기

사전학습(pre-training)
- [유튜브 영상](https://youtu.be/osv2csoHVAo)
- [강의 노트 - 01_pretraining.ipynb](https://github.com/HongLabInc/HongLabLLM/blob/main/01_pretraining.ipynb)

전체 미세조정(full fine-tuning)
- 준비중

## 환경 설정 및 설치

### Case 1: Standalone
```bash
brew install jupyterlab
brew services start jupyterlab
```

### Case 2: Vscode Extention
```bash
jupyter notebook
```

### 가상환경 생성
```bash
python -m venv .venv
source .venv/bin/activate
deactivate
```

### 라이브러리 설치
```bash
pip install transformers
pip install ipywidgets
pip install tiktoken
pip install torch torchvision torchaudio
pip install matplotlib
```

### Mac CPU, GPU Status
```bash
brew install vladkens/tap/macmon
macmon
```