#!/usr/bin/env python3
"""
FDS (Fraud Detection System) - 딥러닝 기반 사기 탐지 시스템
전체 파이프라인: 데이터 생성 -> 모델 훈련 -> 예측
"""

import sys
import os
import subprocess

def run_step(step_name, script_path):
    """
    각 단계의 스크립트를 실행합니다.
    """
    print("\n" + "="*70)
    print(f"[{step_name}]")
    print("="*70)
    
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"오류: {step_name} 실행 중 문제가 발생했습니다.")
        print(f"오류 코드: {e.returncode}")
        return False

def main():
    print("\n")
    print("██████╗ ███████╗███████╗    ༼ つ ◕_◕ ༽つ")
    print("██╔═══██╗██╔════╝██╔════╝")
    print("██║   ██║███████╗███████╗  사기 탐지 시스템")
    print("██║   ██║╚════██║╚════██║  Fraud Detection System")
    print("╚██████╔╝███████║███████║")
    print(" ╚═════╝ ╚══════╝╚══════╝   TensorFlow/Keras MLP신경망")
    print("\n")
    
    # 작업 디렉토리를 전체 경로로 변경
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    os.chdir(parent_dir)
    
    steps = [
        ("데이터셋 생성", "app/generate_dataset.py"),
        ("딥러닝 모델 훈련", "app/train.py"),
        ("모델 예측 테스트", "data/predict.py")
    ]
    
    for step_name, script_path in steps:
        if not run_step(step_name, script_path):
            print(f"\n파이프라인이 중단되었습니다: {step_name}에서 실패")
            return False
    
    print("\n" + "="*70)
    print("✓ 전체 파이프라인 완료!")
    print("="*70)
    print("\n다음 파일들이 생성되었습니다:")
    print("  • models/fds_model.h5 (훈련된 딥러닝 모델)")
    print("  • models/scaler.pkl (데이터 정규화 도구)")
    print("  • app/transactions_train.csv (훈련 데이터)")
    print("\n모델을 프로덕션에 배포할 준비가 완료되었습니다!\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
