지능화 캡스톤 프로젝트 최종 발표 주제 - 주차공간 자동 탐지 및 불법 주차 탐지가 적용된 스마트 주차 관리시스템

주차공간 자동 탐지 및 불법 주차 탐지를 위한 프로세스는 아래와 같습니다.

1.Pklot 데이터셋으로 학습한 YOLO 모델을 통해 주차공간을 탐지

2.탐지된 주차공간은 DBSCAN 알고리즘으로 클러스터링하여 가로방향의 주차공간을 통합한 ROI 영역을 획득

3.VisDrone 데이터셋으로 학습한 YOLO 모델을 통해 top view에서 자동차 객체를 인식

4.인식한 자동차 객체의 B-Box 중점이 주차공간 ROI 안에 있을 경우 정상 주차로 판단 및 해당 주차 공간에 주차 카운팅 +1

5.자동차가 주차공간 밖에 있을 경우 불법 주차로 판단하여 빨간색 B-Box로 표시