# Project Objectives

## 목표 (Goal)
초음파 이미지 시퀀스를 입력받아 손가락 각도를 예측하는 AI 모델 개발

## 입력 & 출력
- **입력**: 초음파 이미지 (508x380 pixels)
- **출력**: 8개 손가락 각도 (Thumb 제외)
  - Index MCP/PIP, Middle MCP/PIP, Ring MCP/PIP, Pinky MCP/PIP

## 모델 구조
- **CNN + LSTM 아키텍처**
  - CNN (ResNet18): 이미지에서 특징 추출
  - LSTM: 시간 정보 학습 (5 프레임 시퀀스)
- **시간 정보 활용**: 슬라이딩 윈도우 방식으로 연속된 5개 프레임 사용

## 데이터
- **SJ 모델**: sj_2 (train), sj_1 (val/test)
- **TK 모델**: 시간순 분할 (train 70%, val 15%, test 15%)
- 총 10,433개 이미지, 3개 세션
- 이 둘은 다른 사람이고, 둘을 섞어서 학습하는 일은 없도록

## 주요 기능
1. **실험 관리**: 타임스탬프 기반 폴더 구조로 매 실험 결과 자동 저장
2. **재현성**: 모델, 설정, 결과를 함께 저장하여 언제든 재현 가능
3. **실시간 모니터링**: TensorBoard로 학습 과정 실시간 확인

## 평가 지표
- MAE (Mean Absolute Error): 각도 차이를 직관적으로 해석
- RMSE, MSE: 추가 평가 지표
