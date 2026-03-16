# FDS-AI System

AI 기반 이상거래탐지(Fraud Detection System) 프로젝트입니다.  
금융 거래 데이터를 저장하고, 머신러닝 기반 이상탐지 모델을 통해 거래 위험도를 평가하는 백엔드 시스템을 목표로 합니다.

## 프로젝트 개요
이 프로젝트는 금융권 및 보안 직무 포트폴리오를 목적으로 설계되었습니다.  
Spring Boot를 통해 거래 데이터를 관리하고, Python FastAPI와 머신러닝 모델을 이용해 이상거래 점수를 산출하는 구조입니다.

## 사용 기술
- **Backend**: Spring Boot, MySQL
- **AI/ML**: Python, scikit-learn, Isolation Forest
- **API Server**: FastAPI

## 주요 기능
- 거래 데이터 저장 및 조회
- 이상거래 점수 산출
- 거래 위험도 판정
- Spring Boot와 AI 서버 분리 구조 설계

## 적용 기법
이 프로젝트에서는 **Isolation Forest** 기반의 이상탐지 기법을 사용합니다.  
Isolation Forest는 정상 거래 패턴과 다른 이상 거래를 탐지하는 데 적합한 모델로,  
금융 FDS에서 새로운 사기 패턴을 탐지하는 데 활용될 수 있습니다.

입력 feature 예시는 다음과 같습니다.
- 거래 금액
- 거래 시간
- 거래 유형
- 기기 변경 여부
- IP 변경 여부
- 위치 변경 여부
- 최근 거래 횟수

## 시스템 구조
Client → Spring Boot → MySQL → FastAPI → AI Model

## 향후 개선 방향
- LightGBM 기반 사기 분류 모델 추가
- Feature Engineering 고도화
- Spring Boot와 FastAPI 연동 자동화
- 실시간 거래 처리 구조로 확장
