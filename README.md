# 강화학습 기초 3일 코스 (Reinforcement Learning Basic - 3Days)

이 저장소는 3일 동안 강화학습(Reinforcement Learning)의 기초부터 딥러닝을 활용한 심화 알고리즘까지 단계별로 학습할 수 있는 실습 코드와 관련 자료를 포함하고 있습니다. 기본 적인 환경 구성부터 정책 경사법(Policy Gradient) 계열의 알고리즘까지 다룹니다.

## 📂 프로젝트 구조

저장소는 완성된 실습 코드가 포함된 루트 디렉토리와 직접 코드를 작성하며 학습할 수 있는 템플릿(빈 칸 채우기 등)을 모아둔 `reinforcement_basic_template` 디렉토리로 구성되어 있습니다.

- **루트 디렉토리 (`/`)**: 완성된 알고리즘 구현 파일과 설명, 시각화 자료가 포함된 주피터 노트북이 위치합니다.
- **템플릿 디렉토리 (`/reinforcement_basic_template`)**: 학습자가 직접 주요 알고리즘의 핵심 부분을 구현(실습)해 볼 수 있도록 구성된 스켈레톤 코드가 제공됩니다.

## 🚀 주요 학습 내용 및 파일 설명

파일 이름에 붙은 숫자 접두사(예: `001_`, `016_` ...)는 학습 과정을 순서대로 나타냅니다.

### 1. 환경 및 기본 개념 구동
- `001_cart_visualize.py`: OpenAI Gymnasium의 **CartPole** 환경 기초 시각화 및 랜덤 에이전트 구동
- `006_frozen_lake_with_policy.py`: **FrozenLake** 환경 소개 및 단순 정책을 적용하여 환경과 상호작용하는 방법 학습

### 2. 고전적 강화학습 알고리즘 (Tabular Methods)
- **동적 계획법 (Dynamic Programming, DP)**
  - `016_DP_frozenlake_policy_iteration.py`: 상태 전이 확률(Transition Model)을 아는 상태에서 작동하는 **정책 반복(Policy Iteration)** 구현.
- **몬테카를로 (Monte Carlo, MC)**
  - `021_MC_blackjack_FirstVisit_Visualize.py`: 에피소드 단위로 학습하는 First-Visit MC 기법을 통해 **Blackjack** 환경의 상태 가치 함수(State-Value Function) 추정 및 시각화.
  - `021_MC_blackjack_보조설명.ipynb`: MC 학습 방법에 대한 이론적 보충 설명 자료.
- **시간 차이 학습 (Temporal Difference, TD)**
  - `033_TD_qlearning_frozenLake.py`: 대표적인 Off-policy TD 컨트롤인 **Q-Learning**을 FrozenLake 환경에 적용.
  - `035_TD_qlearning_taxi.py`: **Q-Learning**을 Taxi 환경에 적용하여 에이전트 학습.
  - `037_CliffWalking_Qlearning_Sarsa.ipynb`: CliffWalking 환경에서 **Q-Learning**과 대표적인 On-policy 컨트롤인 **SARSA** 간의 성능 및 주행 경로 차이를 비교.

### 3. 심층 강화학습 (Deep Reinforcement Learning)
신경망(Neural Network)을 도입하여 상태 공간이 넓거나 연속적인 환경의 문제를 푸는 방법입니다.

- **심층 큐 네트워크 (DQN)**
  - `042_DQN_TargetNN.py`: 경험 리플레이(Experience Replay)와 타겟 네트워크(Target Network)를 사용한 강화학습 방법론 구현.
- **정책 경사법 (Policy Gradient Methods)**
  - `051_REINFORCE.py`: 가장 기본적인 정책 경사법인 **REINFORCE** 알고리즘 구현.
  - `251_REINFORCE.ipynb`: REINFORCE 알고리즘의 동작 방식 이해를 위한 노트북.
  - `053_Actor_Critic.py`: 가치 기반 및 정책 기반 학습을 결합한 **Actor-Critic** 알고리즘 구현. 가치를 평가하는 Critic과 행동을 결정하는 Actor로 구성됩니다.
- **기타 보조 자료**
  - `110_basic_operations_for_Function_Approximation.ipynb`: 신경망 등 함수 근사(Function Approximation)를 적용할 때 필요한 텐서 연산 및 기초 지식 설명 자료.

## 💻 실행 방법

실습 코드는 Python 환경에서 구동 가능하며, 의존성 패키지는 대표적으로 다음을 필요로 합니다 (코드에 따라 다를 수 있음):
- `gymnasium` (강화학습 환경)
- `numpy` (행렬 연산)
- `matplotlib` (시각화)
- `torch` (PyTorch, 딥러닝 알고리즘에 필요)

```bash
# CartPole 환경 예제 실행
python 001_cart_visualize.py
```

학습에 목적이 있으신 분은 바로 완성된 코드를 실행해 보기보다, `reinforcement_basic_template` 안의 코드를 참고하여 직접 구현해 보는 것을 권장합니다!
