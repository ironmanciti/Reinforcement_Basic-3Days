# Reinforce 알고리즘
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')  # CUDA 장치를 사용 가능하면 사용, 아니면 CPU 사용
print(f"Using device: {device}")

ENV_NAME = 'CartPole-v1'
# ENV_NAME = 'LunarLander-v3'  # 환경 이름 설정

# Policy Network를 정의
# 이 신경망은 상태(state)를 입력으로 받아 각 행동에 대한 확률을 출력
class PolicyNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        # Fully Connected Layer 1: 입력 차원에서 128개 노드로
        self.fc1 = nn.Linear(*input_dims, 128)
        # Fully Connected Layer 2: 128개 노드에서 행동의 수만큼의 노드로
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))  # ReLU 활성화 함수를 사용하여 FC1 계산
        # FC2의 출력에 소프트맥스 함수 적용하여 각 행동에 대한 확률 계산
        x = F.softmax(self.fc2(x), dim=-1)
        return x  # 결과 반환

# 각 보상에 대해 감가율을 적용한 값을 반환하는 함수
def discount_rewards(rewards, gamma=0.99):
    """
    보상에 감가율을 적용하여 할인된 보상을 계산합니다.
    """
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    
    # 보상 정규화 (선택사항)
    discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
    return discounted_rewards

# 환경 생성
env = gym.make(ENV_NAME)

# Initialize the parameters theta
pi = PolicyNetwork(input_dims=env.observation_space.shape,
                   n_actions=env.action_space.n).to(device)  # Policy Network 생성

alpha = 0.001  # 학습률 설정
gamma = 0.99  # 감가율 설정
N = 5000  # 에피소드의 최대 개수 설정
batch_size = 32  # 배치 크기 설정

optimizer = optim.Adam(pi.parameters(), lr=alpha)  # 최적화 알고리즘으로 Adam을 사용

total_rewards = []  # 모든 보상을 저장할 리스트 생성
episode_lengths = []  # 에피소드 길이를 저장할 리스트

start_time = time.time()  # 시작 시간 저장
batch_rewards = []  # 배치 보상을 저장할 리스트 생성
batch_actions = []  # 배치 액션을 저장할 리스트 생성
batch_states = []  # 배치 상태를 저장할 리스트 생성
batch_counter = 0  # 배치 카운터 초기화 (0부터 시작)

for episode in range(N):  # N번 에피소드 동안 반복
    # 마지막 에피소드에서 렌더링
    if episode == N - 1:
        env = gym.make(ENV_NAME, render_mode='human')
    
    s, _ = env.reset()  # 환경을 초기화하고 초기 상태를 가져옴

    states = []  # 에피소드의 모든 상태를 저장할 리스트 생성
    rewards = []  # 에피소드의 모든 보상을 저장할 리스트 생성
    actions = []  # 에피소드의 모든 액션을 저장할 리스트 생성

    # 에피소드 종료 여부, 에피소드 중단 여부를 저장하는 변수를 False로 초기화
    terminated, truncated = False, False

    while not terminated and not truncated:  # 에피소드가 종료되지 않고 중단되지 않은 동안 반복
        # 상태를 텐서로 변환하고 정규화
        state_tensor = torch.from_numpy(s).float().to(device)
        
        # 정책 네트워크에서 행동 확률 계산
        with torch.no_grad():
            probs = pi(state_tensor).cpu().numpy()
        
        # 확률에 따라 행동을 선택
        a = np.random.choice(env.action_space.n, p=probs)

        s_, r, terminated, truncated, _ = env.step(a)  # 선택한 행동을 실행

        states.append(s)  # 현재 상태를 저장
        rewards.append(r)  # 보상을 저장
        actions.append(a)  # 실행한 행동을 저장

        s = s_  # 상태 업데이트

    # 에피소드가 종료된 후 처리
    episode_length = len(rewards)
    episode_reward = sum(rewards)
    
    # 보상을 감가율을 적용하여 배치 보상에 추가
    discounted_rewards = discount_rewards(rewards, gamma)
    batch_rewards.extend(discounted_rewards)
    batch_states.extend(states)  # 상태를 배치 상태에 추가
    batch_actions.extend(actions)  # 액션을 배치 액션에 추가
    batch_counter += 1  # 배치 카운터 증가
    
    total_rewards.append(episode_reward)  # 총 보상에 현재 에피소드의 보상 합계 추가
    episode_lengths.append(episode_length)

    # 배치 카운터가 배치 크기와 같다면 (한 배치가 완성되었다면)
    if batch_counter >= batch_size:
        # 텐서로 변환
        state_tensor = torch.FloatTensor(batch_states).to(device)
        reward_tensor = torch.FloatTensor(batch_rewards).to(device)
        action_tensor = torch.LongTensor(batch_actions).to(device)

        # 정책 손실 계산
        log_probs = torch.log(pi(state_tensor))
        selected_log_probs = torch.gather(log_probs, 1, action_tensor.unsqueeze(1)).squeeze()
        loss = -(reward_tensor * selected_log_probs).mean()

        # 정책 업데이트
        optimizer.zero_grad()
        loss.backward()
        # 그래디언트 클리핑 추가 (안정성 향상)
        torch.nn.utils.clip_grad_norm_(pi.parameters(), max_norm=1.0)
        optimizer.step()

        # 배치 초기화
        batch_rewards = []
        batch_actions = []
        batch_states = []
        batch_counter = 0

    if episode % 100 == 0:  # 매 100 에피소드마다
        avg_score = np.mean(total_rewards[-100:])  # 최근 100 에피소드의 평균 점수 계산
        avg_length = np.mean(episode_lengths[-100:])  # 최근 100 에피소드의 평균 길이 계산
        # 평균 점수 출력
        print(f'Episode {episode}, 최근 100 에피소드 평균 reward: {avg_score:.2f}, 평균 길이: {avg_length:.1f}')

env.close()  # 환경 종료
print("duration = ", (time.time() - start_time) / 60, "minutes")  # 실행 시간 출력

running_avg = np.zeros(len(total_rewards))  # 누적 평균 점수를 저장할 배열 생성

for i in range(len(running_avg)):  # 각 에피소드에 대해
    # 최근 100 에피소드의 평균 점수 계산
    running_avg[i] = np.mean(total_rewards[max(0, i-100):(i+1)])

plt.plot(running_avg)  # 누적 평균 점수 그래프 그리기
plt.title('Running average of previous 100 rewards')  # 제목 설정
plt.show()  # 그래프 표시
