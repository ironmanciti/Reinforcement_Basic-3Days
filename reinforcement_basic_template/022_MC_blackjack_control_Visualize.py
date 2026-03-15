# Suntton p.101
# On-Policy First-Visit MC control(for e-soft policies) 최적의 정책 pi*를 찾는 방법
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

# 상태: (player 카드 합계, dealer 오픈 카드, 유용한 에이스 보유) 예) (6, 1, False)

# 알고리즘 매개변수: 작은 e > 0


# 초기화
# 임의의 e-soft 정책을 pi로 설정
# 모든 s, a에 대한 Q(s,a) 설정
# Returns(s, a)를 모든 s, a에 대해 빈 리스트로 초기화


# 무한히 반복 (각 에피소드에 대해)
for n_episode in range(num_episodes):
    # 정책 pi를 따라 에피소드 생성: S0,A0,R1,S1,A1,R2,..ST-1,AT-1,RT
   
    while True:
     
        # s:(player가 가진 카드 합계, dealer 오픈 카드, 유용한 에이스 보유)
        

    # G <- 0
    
    # 에피소드의 각 단계에 대해 반복, t=T-1, T-2,...0
    for s, a, r in episode[::-1]:
        # G <- gamma*G + R_(t+1)
        
        
        # S_t, A_t 쌍이 S_0,A_0 S_1,A_1..S_(t-1),A_(t-1)에 나타나지 않는 경우:
        # G를 Returns(S_t, A_t)에 추가
        # Q(S_t,A_t) <- Returns(S_t, A_t)의 평균
        

        # A* <- argmax_a Q(S_t,a)
        
        # 모든 a에 대해:
        # a = A* 이면 pi(a|S_t) <- 1-e + e/|A(S_t)|
        # a != A* 이면 pi(a|S_t) <- e/|A(St)|
        
                

print("=== 학습 과정 전체 승리/패배 비율 ===")




# 학습 완료 후 최적 정책 평가









print("=== 최적 정책 평가 결과 ===")



#특정 상태(state)에서 가능한 모든 행동(actions)의 값 중에서 가장 큰 값(최적의 행동 가치)을 찾고,
#그 값을 그 상태의 가치 함수 값으로 설정
V = defaultdict(float)
for state, actions in Q.items():
    pass



# 예측
sample_state = (21, 3, True)
# 해당 상태에서 최적의 행동을 찾음

# 상태의 가치 값을 소수점 두 자리까지 출력하고, 최적의 행동이 'stick'인지 'hit'인지 출력






# V[(player가 가진 카드 합계, dealer 오픈 카드, 유용한 에이스 보유)]

