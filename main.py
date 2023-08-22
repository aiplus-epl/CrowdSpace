import datetime
import cv2
import streamlink
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd
import numpy as np


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

list1 = []
list2 = []

CONFIDENCE_THRESHOLD = 0.1  # 객체 검출에 필요한 최소 확률값
start_time = datetime.datetime.now()
time_str = str(start_time.strftime("%d_%H_%M_%S"))

def get_stream(url):
    streams = streamlink.streams(url)
    stream_url = streams["best"]
    cap = cv2.VideoCapture(stream_url.args['url'])
    return cap




# ************ 고칠 부분 ************
videoURL = "https://www.youtube.com/live/-hGxbIZxZxk?si=f50FLAM9Ct1rkt6k";
minutes_ = 360
# *********************************




cap = get_stream(videoURL)
fps = cap.get(cv2.CAP_PROP_FPS)

model = YOLO("yolov8n.pt")  # 사전 학습된 YOLO 모델 로드
tracker = DeepSort(max_age=50)  # DeepSort 객체 추적기 초기화. 최대 허용 미검출 프레임 수는 50

frame_counter = 0
while True:
    start = datetime.datetime.now()  # 프레임 처리 시간 계산을 위한 시작 시간

    ret, frame = cap.read()  # 다음 프레임 읽기
    if not ret:
        break  # 프레임이 더 이상 없으면 while 루프 종료

    frame_counter += 1

    if frame_counter % 30 == 0:
        detections = model(frame)[0]  # YOLO 모델로 현재 프레임에서 객체 검출 수행

        results = []  # 검출된 객체의 경계 상자, 확률, 클래스 ID를 저장할 리스트

        for data in detections.boxes.data.tolist():
            confidence = data[4]  # 객체 검출 확률
            class_id = int(data[5])  # 클래스 ID

            if float(confidence) >= CONFIDENCE_THRESHOLD:
                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])  # 경계 상자 좌표

                if class_id == 0:
                    results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])  # 결과 리스트에 추가
                list2.append([class_id, confidence, (xmin + xmax) // 2, (ymin + ymax) // 2, frame_counter // 30])
        tracks = tracker.update_tracks(results, frame=frame)  # DeepSort로 객체 추적 업데이트

        for i, track in enumerate(tracks):
            if not track.is_confirmed():  # 확인되지 않은 추적은 무시
                continue
            track_id = track.track_id  # 추적 ID
            ltrb = track.to_ltrb()  # 경계 상자 좌표

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2  # Calculate the center coordinates

            new_row = [track_id, center_x, center_y, frame_counter//30]
            list1.append(new_row)

        end = datetime.datetime.now()  # 프레임 처리 시간 계산을 위한 종료 시간
        #print(frame_counter)
        #print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")  # 프레임 처리 시간 출력



    else:
      continue

    elapsed_time = datetime.datetime.now() - start_time

    if elapsed_time > datetime.timedelta(minutes=minutes_):
        break  # Exit the loop

cap.release()  # 동영상 파일 닫기

df = pd.DataFrame(list1, columns=['ind', 'x', 'y', 'frame'])
df2 = pd.DataFrame(list2, columns=['id', 'conf', 'x', 'y', 'frame'])

csv_filename = 'csData_person_'+time_str+'.csv'
csv_filename2 = 'csData_obj_'+time_str+'.csv'
df.to_csv(csv_filename, index=False)
df2.to_csv(csv_filename2, index=False)

print(df)
print(df2)