import gymnasium as gym
import matplotlib.pyplot as plt
import math
import random
import time
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env_name = 'MountainCar-v0' 
# env_name = 'CartPole-v1'

# 환경 생성
env = gym.make(env_name)

# 하이퍼파라미터 설정
num_episodes = 300
GAMMA = 0.99  # 감마 (discount factor)
learning_rate = 0.001  # 학습률
hidden_layer = 120  # 은닉층 노드 수
replay_memory_size = 50_000  # 리플레이 메모리 크기
batch_size = 128  # 배치 크기

e_start = 0.9  # 입실론 초기값
e_end = 0.05  # 입실론 최종값
e_decay = 200  # 입실론 감소율

target_nn_update_frequency = 10  # 타겟 네트워크 업데이트 주기

# 렌더링 관련 하이퍼파라미터
render_episodes = 10  # 렌더링할 에피소드 수
render_start_episode = 290  # 렌더링을 시작할 에피소드 번호 (더 일찍 시작) 

device = "cpu"

n_inputs = env.observation_space.shape[0]  # 입력 차원 수 (상태 수)
n_outputs = env.action_space.n  # 출력 차원 수 (액션 수)

# 리플레이 메모리 클래스
class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity  # 리플레이 메모리의 최대 크기 설정
        self.memory = []  # 경험을 저장할 메모리 리스트 초기화
        self.position = 0  # 현재 저장 위치 초기화

    # 경험 추가 함수
    def push(self, state, action, new_state, reward, done):
        # 주어진 경험(transition)을 메모리에 추가
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            # 메모리에 빈 공간이 있으면 경험 추가
            self.memory.append(transition)
        else:
            # 메모리가 가득 차면 오래된 경험을 덮어쓰기
            self.memory[self.position] = transition
            
        # 저장 위치를 다음으로 이동, 용량을 초과하면 처음으로 돌아감
        self.position = (self.position + 1) % self.capacity

    # 경험 샘플링 함수
    def sample(self, batch_size):
        # 메모리에서 주어진 배치 크기만큼 무작위로 샘플링하여 반환
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        # 현재 메모리에 저장된 경험의 수를 반환
        return len(self.memory)

# 신경망 클래스
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(n_inputs, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, hidden_layer//2)
        self.linear3 = nn.Linear(hidden_layer//2, n_outputs)

    # 순전파 함수
    def forward(self, x):
        a1 = torch.relu(self.linear1(x))
        a2 = torch.relu(self.linear2(a1))
        output = self.linear3(a2)
        return output

# 액션 선택 함수
def select_action(state, steps_done):
    # 입실론 값 계산
    e_threshold = e_end + (e_start - e_end) * \
        math.exp(-1. * steps_done/e_decay)

    if random.random() > e_threshold:
        # 입실론보다 큰 경우, Q 함수에 따라 행동 선택
        with torch.no_grad():
            state = torch.Tensor(state).to(device)   # 상태를 텐서로 변환하고 장치에 할당
            action_values = Q(state)   # Q 함수를 사용하여 각 행동의 가치 계산
            action = torch.argmax(action_values).item()   # 가장 높은 가치를 갖는 행동 선택
    else: 
        # 입실론보다 작은 경우, 무작위 행동 선택 (탐색)
        action = env.action_space.sample()

    return action

# 리플레이 메모리 초기화
memory = ExperienceReplay(replay_memory_size)

# 타겟 Q함수 초기화 (랜덤 가중치)
target_Q = NeuralNetwork().to(device)

# Q함수 초기화 (랜덤 가중치로 신경망 생성)
Q = NeuralNetwork().to(device)

# 손실 함수 설정 (평균 제곱 오차)
criterion = nn.MSELoss()

# 최적화 알고리즘 설정 (Adam 옵티마이저)
optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

# 타겟 네트워크 업데이트 카운터 초기화
update_target_counter = 0
# 각 에피소드에서 얻은 보상을 저장할 리스트 초기화
reward_history = []
# 총 스텝 수 초기화
total_steps = 0
# 학습 시작 시간 기록
start_time = time.time()

# 에피소드 루프
for episode in range(num_episodes):
    # 렌더링 여부 결정
    should_render = episode >= render_start_episode
    
    if should_render:
        # 렌더링을 위한 환경 생성
        render_env = gym.make(env_name, render_mode="human")
        s, _ = render_env.reset()
        print(f"🎬 Rendering episode {episode} - 렌더링 창이 열렸습니다!")
    else:
        s, _ = env.reset()
        if episode % 50 == 0:  # 50 에피소드마다 진행상황 출력
            print(f"📊 Episode {episode}/{num_episodes} 진행 중...")

    episode_reward = 0
    while True:
        total_steps += 1

        # 액션 선택
        a = select_action(s, total_steps)

        # 환경에서 액션 수행
        if should_render:
            s_, r, terminated, truncated, _ = render_env.step(a)
        else:
            s_, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        episode_reward += r

        # 리플레이 메모리에 경험 저장
        memory.push(s, a, s_, r, done)

        if len(memory) >= batch_size:
            # 리플레이 메모리에서 미니배치 샘플링
            states, actions, new_states, rewards, dones = memory.sample(
                batch_size)

            # 샘플링한 데이터를 텐서로 변환하여 장치에 할당
            states = torch.Tensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            new_states = torch.Tensor(new_states).to(device)
            rewards = torch.Tensor(rewards).to(device)  # 차원 수정
            dones = torch.Tensor(dones).to(device)
            
            # 타겟 Q 네트워크로부터 새로운 상태의 Q 값 계산
            new_action_values = target_Q(new_states).detach()

            # 타겟 값 계산
            y_target = rewards + \
                (1 - dones) * GAMMA * torch.max(new_action_values, 1)[0]
            # 예측 값 계산
            y_pred = Q(states).gather(1, actions.unsqueeze(1))

            # 손실 계산 및 역전파
            loss = criterion(y_pred.squeeze(), y_target.squeeze())
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # 타겟 네트워크 업데이트
            if update_target_counter % target_nn_update_frequency == 0:
                target_Q.load_state_dict(Q.state_dict())

            update_target_counter += 1

        s = s_

        if done:
            reward_history.append(episode_reward)
            print(f"{episode} episode finished after {episode_reward:.2f} rewards")
            break

    # 렌더링 환경이 생성된 경우 닫기
    if should_render:
        render_env.close()
        print(f"🔒 Episode {episode} 렌더링 창을 닫았습니다.")

# 환경 닫기
env.close()

# 평균 보상 출력
print("Average rewards: %.2f" % (sum(reward_history)/num_episodes))

# 마지막 50 에피소드의 평균 보상 출력
last_episodes = 50
if len(reward_history) >= last_episodes:
    print(f"Average of last {last_episodes} episodes: %.2f" % (sum(reward_history[-last_episodes:])/last_episodes))
else:
    print(f"Average of all {len(reward_history)} episodes: %.2f" % (sum(reward_history)/len(reward_history)))

# 하이퍼파라미터 정보 출력
print("---------------------- Hyper parameters --------------------------------------")
print(
    f"GAMMA:{GAMMA}, learning rate: {learning_rate}, hidden layer: {hidden_layer}")
print(f"replay_memory: {replay_memory_size}, batch size: {batch_size}")
print(f"epsilon_start: {e_start}, epsilon_end: {e_end}, " +
      f"epsilon_decay: {e_decay}")
print(f"update frequency: {target_nn_update_frequency}")

# 경과 시간 출력
elapsed_time = time.time() - start_time
print(f"Time Elapsed : {elapsed_time//60} min {elapsed_time%60:.0} sec")

# 학습 과정의 보상 플롯
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(range(len(reward_history)), reward_history, alpha=0.6)
plt.xlabel("episodes")
plt.ylabel("rewards")
plt.title("DQN - Target Network (Individual Episodes)")

# 이동 평균 플롯 추가
plt.subplot(1, 2, 2)
window_size = 20
if len(reward_history) >= window_size:
    moving_avg = [sum(reward_history[i:i+window_size])/window_size 
                  for i in range(len(reward_history)-window_size+1)]
    plt.plot(range(window_size-1, len(reward_history)), moving_avg, 'r-', linewidth=2)
    plt.xlabel("episodes")
    plt.ylabel("moving average rewards")
    plt.title(f"Moving Average (window={window_size})")

plt.tight_layout()
plt.show()
