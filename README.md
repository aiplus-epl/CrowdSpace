# CrowdSpace

### library
<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=Jupyter&logoColor=white"/> <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=OpenCV&logoColor=white"/> <img src="https://img.shields.io/badge/YOLO-00FFFF?style=flat&logo=YOLO&logoColor=white"/>

### 작품 개요 
- Crowd Space는 CCTV 분석을 통해 실시간 인구 밀도 및 폭력행위를 감지하고, 이를 시각적으로 나타내어 사용자들의 안전과 편의성을 증진시키는 서비스이다.
- 기존의 인구 밀도 확인 서비스는 실시간 변동 정보가 부족하여 특정 지역에 인파가 밀집되어 위험한 상황이 발생할 수 있는 문제점을 가지고 있다. 이로 인해 위험한 상황에서 빠른 대응이 어려워지는 부수적인 문제도 발생하고 있다. 따라서 실시간 인구 밀도 분석을 수행하는 프로그램이 필요하며, 더 나아가 폭력행위를 탐지하여 보다 신속한 대응이 가능한 분석 시스템을 개발해야 한다.
- 이러한 문제들을 해결하기 위해 Crowd Space는 OpenCV 기을 활용하여 실시간으로 인구 밀도와 폭력행위를 모니터링하고 관련 정보를 사용자에게 시각적으로 제공한다. 이를 통해 사용자들은 위험 상황을 미리 파악하고 적절한 조취를 취할 수 있게 된다. Crowd Space는 안전과 편의성을 증진시키는 혁신적인 솔루션으로, 도시 및 공공 장소에서의 안전을 높이는데 기여할 것이다.

### 작품 목표 및 기능 
- CCTV 영상을 활용한 실시간 인구 밀도 및 폭력 행위 감지 어플리케이션 구축
- 3단계 (여유, 보통, 혼잡)로 인구 밀집도 단계를 표시
- 평균적인 밀집도를 계산하여 평상시 대비 현재 복잡도 표시
- 일반 행위와 폭력 행위 구분 및 감지 후 알림

### 개발 과정 

**1. 보행자 탐지**
- Ultralytics에서 제공하는 YOLOv8n을 활용하여 보행자 탐지를 수행했다.
- 보안상의 이유로 직접적인 CCTV 작동 데이터를 확보할 수 없어, 유튜브 라이브 스트리밍 영상을 활용하여 데이터를 수집했다.
  - 데이터 비교를 위해 일본 도쿄의 kabukicho 지역의 양방향 골목 영상 2개와, 사거리에서의 데이터 수집을 위해 미국 florida 영상을 선정했다. 
  - 유튜브 라이브 스트리밍 영상
    - kabukicho : https://www.youtube.com/watch?v=gFRtAAmiFbE
    - kabukicho2 : https://www.youtube.com/live/DjdUEyjx8GM?feature=share
    - florida : https://www.youtube.com/live/-hGxbIZxZxk?si=f50FLAM9Ct1rkt6k
- 영상에서 사람과 교통수단 (car/truck/bus/motorbike/bycicle/skateboard) 등을 검출하여, 해당 객체의 class와 위치, 시간 등을 수집했다. 이를 통해 동일한 시간대에 서로 다른 거리에서의 인구 밀도와, 도로의 전반적인 정보를 비교 및 분석했다. 

**2. 위험 상황 감지**
- 위험 상황 감지를 위한 작업으로 동일하게 YOLOv8n 모델을 활용하였으며, 이를 위해 violence 데이터셋을 활용한 전이학습을 수행했다.
- 효율적인 감지를 위해 각 프레임에서 탐지된 사람의 수를 저장하는 deque 자료구조를 도입하고, 이를 통해 최근 30분 간의 frame에서 탐지된 사람의 평균 수를 계산하여 활용했다. 해당 값이 5명 미만일 때에만 violence 감지 모델을 실행하도록 조건을 설정했다.
- 사용한 violence 데이터셋
  - https://universe.roboflow.com/school-zmthx/violence-detection-w2xnz/browse?queryText=class%3A1&pageSize=50&startingIndex=0&browseQuery=true
 
**3. 이동 경로 분석**
- DeepSORT를 사용하여 보행자의 이동 경로를 추적했다. 
-  YOLO를 사용하여 감지된 객체 중 사람으로 탐지된 객체의 정보를 추출했다. 해당 정보는, 프레임 별 위치, 시간 등을 포함하며, 이를 통해 사람들의 이동 경로와 행동을 분석하고 유동인구 변화를 파악하고자 했다.

**4. 데이터 수집**
- Kaggle의 GPU T4x2 accelerator를 이용하여, 8월 1일부터 15일 동안 하루 4시간씩 데이터를 수집했다. 데이터의 다양성을 고려하기 위해 시간대 및 주중/주말/공휴일에 따른 데이터 수집 시간을 조정했다.
- 보행자 탐지
  - frame : 수집된 프레임 번호
  - time : 수집된 시간
  - class : 탐지된 객체의 class 넘버
  - x1, y1, x2, y2 : 탐지된 객체의 bounding box 좌표 
- 이동 경로 분석
  - frame : 수집된 프레임 번호
  - time : 수집된 시간
  - id : tracking 된 객체의 넘버
  - x1, y1, x2, y2 : 트래킹되고 있는 객체의 bounding box 좌표 

**5. 데이터 증강** 
- 수집된 데이터의 시간대별 양이 제한적이었기 때문에, 모델의 성능 향상을 위해 data augmentation 작업을 수행했다.
- data augmentation을 위해, CTGAN 모델을 채택했다.
  - CTGAN
    - CTGAN은 Conditional Tabular GAN의 약어로, Conditional GAN을 기반으로 한 GAN 아키텍처이다.
    - CTGAN은 표 형식의 데이터를 생성하기 위한 생성적 적대 신경망(Generative Adversarial Network, GAN) 기반의 모델이다.
    - 우리 서비스는, 이산적 데이터와 연속적 데이터 모두 포함된 데이터를 다루므로, mode specific normalization과 Training by Sampling 작업을 수행할 수 있는 CTGAN을 채택했다.

**6. 예측모델**
- 특

  
### 어플리케이션 
