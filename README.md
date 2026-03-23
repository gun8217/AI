# AI (Personal Learning Repository)

개인 AI / Machine Learning / Deep Learning 학습 및 실험 프로젝트 저장소입니다.  
각 프로젝트의 상세 실험 결과, 판단, 진행 상태는  
해당 폴더의 `00_project_status.txt` 파일에 기록되어 있습니다.

---

## Models & Experiments

### 🔹 Deep Learning (DL)

#### Super Resolution / OCR

- **ESPCN** – 빠른 실험 및 기본 복원
- **EDSR** – 고정밀 복원
- **RCAN** – 학습 중단 (학습 시간 과다)

📁 `DL/SuperResolution/app/00_project_status.txt`

---

#### Image → Text (Translator)

- 이미지 텍스트 추출 실험
- 현재 OCR 미적용 상태

📁 `DL/Translator/작업보류-현재 이미지 텍스트 추출 미적용`

---

#### Object Detection (YOLO)

- 병(bottle) 객체 탐지 실험

📁 `DL/YOLO/bottle/00_project_status.txt`

---

### 🔹 Machine Learning (ML)

#### Clustering

- 비지도 학습 군집화 실험

📁 `ML/Clustering/00_project_status.txt`

---

#### Decision Tree

- 대출 승인 예측 (Loan Approval)

📁 `ML/DecisionTree/loanApproval/00_project_status.txt`

---

#### K-Nearest Neighbors (KNN)

- Boston Housing 데이터셋 실습

📁 `ML/KNN/boston/00_project_status.txt`

---

#### Linear Regression

- Dokdo 관련 데이터 회귀 분석

📁 `ML/LinearRegression/dokdo/00_project_status.txt`

---

#### Logistic Regression

- Breast Cancer 분류
- News Category 분류

📁 `ML/LogisticRegression/breastCancer/00_project_status.txt`  
📁 `ML/LogisticRegression/newsCategory/00_project_status.txt`

---

#### Random Forest

- Iris 데이터셋 분류

📁 `ML/RandomForest/iris/00_project_status.txt`

---

### 🔹 Reinforcement Learning (RL)

- 강화학습 기초 및 실험 테스트 중

📁 `RL/테스트 중`

---

### 🔹 Software Layer (LLM Orchestration)

#### LangChain

- Gemini 2.5 Flash 연동 및 환경 설정 (.env) 최적화
- Prompt Template | LLM | Output Parser 구조의 기본 체인 설계
- 모델 독립적 설계를 통한 확장성 확보 실험

📁 `SoftwareLayer/LangChain/langChain.ipynb`

---

#### LangGraph

- State(상태 객체) 기반의 순환형(Cyclic) 에이전트 설계
- **Node(동작)**와 **Edge(흐름)**를 이용한 논리적 워크플로우 구현
- 조건부 로직(Conditional Edges)을 활용한 지능형 고객센터 챗봇 구조 실험

📁 `SoftwareLayer/LangGraph/langGraph.ipynb`

---

## Note

- 본 저장소는 **개인 학습 및 실험 목적**으로 운영됩니다.
- 코드 완성도보다는 **실험 과정, 판단, 학습 기록**을 우선합니다.
- 각 프로젝트의 상세 내용은 개별 `00_project_status.txt`를 참고하세요.
