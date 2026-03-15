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
print(device)

ENV_NAME = 'CartPole-v1'
# ENV_NAME = 'LunarLander-v3'  # 환경 이름 설정

# Policy Network를 정의
# 이 신경망은 상태(state)를 입력으로 받아 각 행동에 대한 확률을 출력
class PolicyNetwork(nn.Module):
    pass










# 각 보상에 대해 감가율을 적용한 값을 반환하는 함수
def discount_rewards(rewards, gamma=0.99):
    pass









# 환경 생성


# Initialize the parameters theta












for episode in range(N):  # N번 에피소드 동안 반복
    # 마지막 에피소드에서 렌더링    








    # 에피소드 종료 여부, 에피소드 중단 여부를 저장하는 변수를 False로 초기화
    terminated, truncated = False, False

    while not terminated and not truncated:  # 에피소드가 종료되지 않고 중단되지 않은 동안 반복
        # 상태를 텐서로 변환하고 정규화
        
        
        # 정책 네트워크에서 행동 확률 계산

        
        # 확률에 따라 행동을 선택      






    # 에피소드가 종료된 후 처리


    # 보상을 감가율을 적용하여 배치 보상에 추가






    
    # 배치 카운터가 배치 크기와 같다면 (한 배치가 완성되었다면)
    if batch_counter >= batch_size:
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
env.close()  # 환경 종료
print("duration = ", (time.time() - start_time) / 60, "minutes")  # 실행 시간 출력

running_avg = np.zeros(len(total_rewards))  # 누적 평균 점수를 저장할 배열 생성

for i in range(len(running_avg)):  # 각 에피소드에 대해
    # 최근 100 에피소드의 평균 점수 계산
    running_avg[i] = np.mean(total_rewards[max(0, i-100):(i+1)])

plt.plot(running_avg)  # 누적 평균 점수 그래프 그리기
plt.title('Running average of previous 100 rewards')  # 제목 설정
plt.show()  # 그래프 표시
