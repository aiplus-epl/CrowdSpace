import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata  # 데이터에 대한 메타데이터를 생성하고 관리하는 도구
from sdv.single_table import CTGANSynthesizer  # SDV 라이브러리의 CTGAN 신시사이저 클래스
from sdv.sampling import Condition  # 합성 데이터 샘플링을 위한 조건을 정의하는 클래스

# 데이터셋에 대한 정보를 저장하기 위한 메타데이터 객체를 초기화
metadata = SingleTableMetadata()

file_name = ''

df = pd.read_csv(file_name, index_col=0)

# 데이터프레임에서 자동으로 메타데이터의 세부 사항을 감지하고 업데이트
metadata.detect_from_dataframe(data=df)

# 'day' 열 값에 기반한 조건을 생성하는 함수를 정의합니다.
def day_condition(day, row):
    # 'Condition' 클래스를 사용하여 샘플링 조건을 지정 ( 주어진 날짜와 행 수에 대한 조건을 지정 )
    r_ = Condition(
        num_rows=row,  # 샘플링할 행의 수
        column_values={'day': day}  # 샘플링된 행의 열 값
    )
    return r_

# 'time' 열 값에 기반한 조건을 생성하는 함수를 정의
def time_condition(time, row):
    r_ = Condition(
        num_rows=row,
        column_values={'time': time}
    )
    return r_

# 지정된 매개변수로 CTGAN 신시사이저를 초기화
synthesizer = CTGANSynthesizer(
    metadata,  # 데이터셋에 대한 메타데이터
    enforce_min_max_values=False,  # True인 경우 원래 데이터의 최소-최대 범위 내에서 합성 값이 발생하도록 강제
    enforce_rounding=False,  # True인 경우 합성 데이터를 소숫점을 원래 데이터와 가깝게
    epochs=5000,  # CTGAN 모델에 대한 훈련 에포크 수
    verbose=True  # True인 경우 훈련 중 진행 정보를 표시
)

# 원본 데이터프레임에서 CTGAN 신시사이저를 훈련
synthesizer.fit(df)

# 지정된 조건(날짜=1 및 시간=10)을 기반으로 합성 데이터를 샘플링하고 각 조건에 대해 2개의 행을 검색
synthesizer_data2 = synthesizer.sample_from_conditions(
    conditions=[day_condition(1, 2), time_condition(10, 2)]
)

# 샘플링된 합성 데이터를 출력
print(synthesizer_data2)

# SDV 라이브러리에서 평가 함수를 임포트
from sdv.evaluation.single_table import evaluate_quality

# 원본 데이터와 비교하여 합성 데이터의 품질을 평가
quality_report = evaluate_quality(
    real_data=df,  # 원본 데이터셋
    synthetic_data=synthesizer_data2,  # 생성된 합성 데이터셋
    metadata=metadata  # 데이터셋에 대한 메타데이터
)
