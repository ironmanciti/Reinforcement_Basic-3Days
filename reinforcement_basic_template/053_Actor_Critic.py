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

#env_name = 'CartPole-v1'  # 환경 이름 설정
env_name = 'LunarLander-v3'
env = gym.make(env_name)  # 환경 생성

n_actions = env.action_space.n  # 가능한 액션의 수를 설정

print(f"Number of actions: {n_actions}")

# 정책 네트워크 (pi(a|s,theta))와 가치 네트워크 (v(s,w))를 포함한 Actor-Critic 클래스






















# 학습률 0 < alpha < 1 설정

# 감가율 0 < gamma < 1 설정

# 에피소드의 최대 개수 N 설정


# theta 매개변수와 가치 가중치 w 초기화





# episode 단위로 batch를 만드는 함수
















# 각 에피소드에 대해
for episode in range(N):
    # 마지막 에피소드에서 렌더링
    
    

    # 첫 상태 초기화
    
    
    
    # S가 종료 상태가 아닌 동안 반복
    while not done:
        # A ~ pi(.|S,theta) - 정책 네트워크에서 액션 하나를 샘플링
        
        
        # 액션 A를 취하고, S', R 관찰
        
        
        # S <- S'
        

        done = terminated or truncated

        if done:
            

            # delta <- R + gamma * v(S',w) - v(S,w) (만약 S'가 종료 상태라면 v(S',w) = 0)
            
            # advantage = reward + gamma * v(S',w) - v(S,w) --> advantage = delta
            

            # w <- w + alpha * delta * gradient(v(S,w)) - 가치 네트워크 매개변수 업데이트
            # theta <- theta + alpha * I * delta * gradient(pi(A|S,theta)) - 정책 네트워크 매개변수 업데이트
            
            # Actor loss (정책 손실)
            
            # Critic loss (가치 손실)
            
            
            # 전체 손실
            
            
            # 그래디언트 클리핑 추가 (안정성 향상)
            
            
            

    if episode % 100 == 0:
        avg_score = np.mean(total_rewards[-100:])
        print(f'episode {episode},  최근 100 episode 평균 reward {avg_score: .2f}')


# 결과 시각화

# 보상 그래프



# 에피소드 길이 그래프