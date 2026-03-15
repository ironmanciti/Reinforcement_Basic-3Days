# One-step Actor-Critic(episodic), for estimating pi_theta == pi_*
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time

device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')  # CUDA 장치를 사용 가능하면 사용, 아니면 CPU 사용
print(f"Using device: {device}")

# 환경 이름 설정
env_name = 'LunarLander-v3'  
env = gym.make(env_name)  # 환경 생성

n_actions = env.action_space.n  # 가능한 액션의 수를 설정

print(f"Number of actions: {n_actions}")

# 정책 네트워크 (pi(a|s,theta))와 가치 네트워크 (v(s,w))를 포함한 Actor-Critic 클래스
class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorCritic, self).__init__()
        # Fully Connected Layer 1: 입력 차원에서 256개 노드로
        self.fc1 = nn.Linear(*input_dims, 256)
        # Policy Network의 Fully Connected Layer: 256개 노드에서 행동의 수만큼의 노드로
        self.fc_pi = nn.Linear(256, n_actions)
        # Value Network의 Fully Connected Layer: 256개 노드에서 1개 노드로
        self.fc_v = nn.Linear(256, 1)

    # 정책 네트워크의 순전파 함수
    def pi(self, state):
        x = F.relu(self.fc1(state))  # ReLU 활성화 함수를 사용하여 FC1 계산
        x = self.fc_pi(x)  # FC_pi 계산
        prob = F.softmax(x, dim=-1)  # FC_pi의 출력에 소프트맥스 함수 적용하여 각 행동에 대한 확률 계산
        return prob

    # 가치 네트워크의 순전파 함수
    def v(self, state):
        x = F.relu(self.fc1(state))  # ReLU 활성화 함수를 사용하여 FC1 계산
        v = self.fc_v(x)  # FC_v 계산
        return v

# 학습률 0 < alpha < 1 설정
alpha = 0.001
# 감가율 0 < gamma < 1 설정
gamma = 0.98
# 에피소드의 최대 개수 N 설정
N = 2000

# theta 매개변수와 가치 가중치 w 초기화
model = ActorCritic(env.observation_space.shape, n_actions).to(device)

optimizer = optim.Adam(model.parameters(), lr=alpha)

total_rewards = []
episode_lengths = []

# episode 단위로 batch를 만드는 함수
def make_batch(memory):
    batch_states, batch_actions, batch_rewards, batch_next_state, batch_done = [], [], [], [], []
    for transition in memory:
        s, a, r, s_, done = transition
        batch_states.append(s)
        batch_actions.append([a])
        batch_rewards.append([r])
        batch_next_state.append(s_)
        done_mask = 0 if done else 1
        batch_done.append([done_mask])
    return torch.FloatTensor(batch_states).to(device), torch.LongTensor(batch_actions).to(device), \
        torch.FloatTensor(batch_rewards).to(device), torch.FloatTensor(batch_next_state).to(device), \
        torch.FloatTensor(batch_done).to(device)

start_time = time.time()

# 각 에피소드에 대해
for episode in range(N):
    # 마지막 에피소드에서 렌더링
    if episode >= N - 3:
        env = gym.make(env_name, render_mode='human')
        print(f"Rendering episode {episode} (마지막 {N - episode}개 에피소드)")
    else:
        env = gym.make(env_name)

    # 첫 상태 초기화
    s, _ = env.reset()

    done = False
    memory = []
    episode_reward = 0
    
    # S가 종료 상태가 아닌 동안 반복
    while not done:
        # A ~ pi(.|S,theta) - 정책 네트워크에서 액션 하나를 샘플링
        with torch.no_grad():
            probs = model.pi(torch.tensor(s, dtype=torch.float).to(device)).cpu().numpy()
        a = np.random.choice(n_actions, p=probs)
        
        # 액션 A를 취하고, S', R 관찰
        s_, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        episode_reward += r
        
        memory.append((s, a, r, s_, done))
        # S <- S'
        s = s_

        if done:
            s_batch, a_batch, r_batch, s_next_batch, done_batch = make_batch(memory)

            # delta <- R + gamma * v(S',w) - v(S,w) (만약 S'가 종료 상태라면 v(S',w) = 0)
            td_target = r_batch + gamma * model.v(s_next_batch) * done_batch
            # advantage = reward + gamma * v(S',w) - v(S,w) --> advantage = delta
            delta = td_target - model.v(s_batch)

            # w <- w + alpha * delta * gradient(v(S,w)) - 가치 네트워크 매개변수 업데이트
            # theta <- theta + alpha * I * delta * gradient(pi(A|S,theta)) - 정책 네트워크 매개변수 업데이트
            pi = model.pi(s_batch)
            pi_a = pi.gather(1, a_batch)
            
            # Actor loss (정책 손실)
            actor_loss = -torch.log(pi_a) * delta.detach()
            # Critic loss (가치 손실)
            critic_loss = F.smooth_l1_loss(model.v(s_batch), td_target.detach())
            
            # 전체 손실
            loss = actor_loss.mean() + critic_loss

            optimizer.zero_grad()
            loss.backward()
            # 그래디언트 클리핑 추가 (안정성 향상)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_rewards.append(episode_reward)
            episode_lengths.append(len(memory))

    if episode % 100 == 0:
        avg_score = np.mean(total_rewards[-100:])
        avg_length = np.mean(episode_lengths[-100:])
        print(f'Episode {episode}, 최근 100 episode 평균 reward: {avg_score:.2f}, 평균 길이: {avg_length:.1f}')

env.close()
print(f"Training duration: {(time.time() - start_time) / 60:.2f} minutes")

# 결과 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 보상 그래프
running_avg = np.zeros(len(total_rewards))
for i in range(len(running_avg)):
    running_avg[i] = np.mean(total_rewards[max(0, i-100):(i+1)])

ax1.plot(running_avg)
ax1.set_title('Running average of previous 100 rewards')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Average Reward')
ax1.grid(True)

# 에피소드 길이 그래프
running_length_avg = np.zeros(len(episode_lengths))
for i in range(len(running_length_avg)):
    running_length_avg[i] = np.mean(episode_lengths[max(0, i-100):(i+1)])

ax2.plot(running_length_avg)
ax2.set_title('Running average of episode lengths')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Average Episode Length')
ax2.grid(True)

plt.tight_layout()
plt.show()
