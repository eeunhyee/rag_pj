# Justi-Q 환경 설정 가이드

## 1. Conda 환경 생성

```bash
conda create -n justiq python=3.10 -y
conda activate justiq
```

## 2. 의존성 설치

```bash
pip install -r requirements.txt
```

## 3. 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성:

```bash
OPENROUTER_API_KEY=your_api_key_here
```

OpenRouter API 키는 https://openrouter.ai 에서 발급받으세요.

## 4. 벡터 DB 설정

### 옵션 A: 기존 DB 사용 (권장)
팀에서 공유한 `chroma_db.zip`을 프로젝트 루트에 압축 해제

### 옵션 B: 새로 인덱싱
```bash
python main.py --index
```
(M4 Mac 기준 약 10-15분 소요)

## 5. 실행

### CLI 대화형 모드
```bash
python main.py
```

### 단일 질문
```bash
python main.py --query "폭행죄 처벌 기준은?"
```

### 웹 UI (Streamlit)
```bash
streamlit run app.py
```

## 문제 해결

### ModuleNotFoundError
```bash
conda activate justiq
pip install -r requirements.txt
```

### ChromaDB 오류
`chroma_db/` 폴더 삭제 후 다시 인덱싱하거나 공유된 DB 사용
