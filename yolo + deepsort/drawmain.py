import datetime
import cv2
import streamlink
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

CONFIDENCE_THRESHOLD = 0.7  # YOLO 모델로부터 반환된 감지에 대한 최소 확률값
GREEN = (0, 255, 0)  # 경계 상자를 그리기 위한 색상 값
WHITE = (255, 255, 255)  # 텍스트를 그리기 위한 색상 값
colors = np.random.randint(0, 255, size=(100, 3))  # 추적 ID 별로 다양한 색상 값을 생성
track_centers = {}  # 각 추적 ID의 중심 좌표를 저장하기 위한 딕셔너리


def get_stream(url):
    """
    주어진 URL에서 스트림을 가져와서 VideoCapture 객체를 반환한다.
    """
    streams = streamlink.streams(url)  # 주어진 URL로부터 스트림 리스트를 가져옴
    stream_url = streams["best"]  # 가장 좋은 품질의 스트림을 선택
    cap = cv2.VideoCapture(stream_url.args['url'])  # 선택된 스트림 URL을 사용하여 VideoCapture 객체를 초기화
    return cap


# 입력 비디오의 속성을 가져옴
videoURL = "https://www.youtube.com/watch?v=gFRtAAmiFbE"
cap = get_stream(videoURL)
frame_width = int(cap.get(3))  # 입력 비디오의 너비
frame_height = int(cap.get(4))  # 입력 비디오의 높이
fps = cap.get(cv2.CAP_PROP_FPS)  # 입력 비디오의 프레임 속도

# 표시될 프레임의 너비와 높이를 설정
display_width = 800
display_height = 600

# 출력 비디오 설정
output_filename = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 정의
writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))  # 동영상 작성자 객체를 초기화

# YOLO 모델과 DeepSort 추적기를 초기화
model = YOLO("yolov8n.pt")  # YOLO 모델을 로드
tracker = DeepSort(max_age=90)  # DeepSort 객체 추적기를 초기화, 최대 미검출 프레임 수는 90으로 설정 (Youtube 30frame)

while True:  # 비디오의 각 프레임을 반복
    start = datetime.datetime.now()  # 프레임 처리 시간 측정을 위해 시작 시간을 저장

    ret, frame = cap.read()  # 비디오의 다음 프레임을 읽음
    if not ret:
        break  # 더 이상 읽을 프레임이 없으면 루프를 종료

    frame = cv2.resize(frame, (display_width, display_height))  # 프레임 크기를 조정
    detections = model(frame)[0]  # YOLO 모델로 프레임에서 객체를 감지

    results = []
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        class_id = int(data[5])
        # 확률이 임계 값 이상이고 클래스 ID가 '사람'인 객체만 선택
        if class_id == 0 and float(confidence) >= CONFIDENCE_THRESHOLD:
            bbox = [int(data[0]), int(data[1]), int(data[2]) - int(data[0]), int(data[3]) - int(data[1])]
            results.append([bbox, confidence, class_id])

    tracks = tracker.update_tracks(results, frame=frame)  # 객체의 움직임을 추적

    for track in tracks:
        if not track.is_confirmed():  # 미확인 추적은 무시
            continue

        track_id = track.track_id  # 객체의 추적 ID를 가져옴
        bbox = track.to_ltrb()  # 객체의 경계 상자 좌표를 가져옴
        color = colors[int(track_id) % 100].tolist()  # 해당 객체 ID에 대한 색상을 선택

        # 프레임에 객체의 경계 상자와 ID를 그림
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, str(track_id), (bbox[0] + 5, bbox[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        center = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2  # 객체의 중심 좌표를 계산

        # 해당 추적 ID의 중심 좌표 리스트를 업데이트
        if track_id not in track_centers:
            track_centers[track_id] = []
        track_centers[track_id].append(center)

        # 객체의 이동 경로를 그림 
        for point in track_centers[track_id]:
            cv2.circle(frame, point, 5, color, -1)

    end = datetime.datetime.now()  # 프레임 처리 시간 측정을 위해 종료 시간을 저장
    elapsed_time = (end - start).total_seconds() * 1000
    print(f"Time to process 1 frame: {elapsed_time:.0f} milliseconds")  # 프레임 처리에 걸린 시간을 출력

    # 프레임에 FPS 값을 그림
    fps_text = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    cv2.imshow("Frame", frame)  # 처리된 프레임을 화면에 표시
    writer.write(frame)  # 프레임을 출력 동영상 파일에 기록

    # 'q' 키를 누르면 동영상 처리를 중지
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
