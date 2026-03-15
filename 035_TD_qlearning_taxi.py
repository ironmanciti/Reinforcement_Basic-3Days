# Q-Learning (off-policy TD control) for estimating pi=pi*
import gymnasium as gym
import matplotlib
# matplotlib.use('Agg')  # GUI 없는 백엔드 사용
matplotlib.use('TkAgg')  # GUI 백엔드 사용
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

"""
6개의 이산적인 결정적 행동:
    - 0: 남쪽으로 이동
    - 1: 북쪽으로 이동
    - 2: 동쪽으로 이동
    - 3: 서쪽으로 이동
    - 4: 승객 탑승
    - 5: 승객 하차
    
상태 공간은 다음과 같이 표현됩니다:
        (택시_행, 택시_열, 승객_위치, 목적지)
          5 * 5 * 5 * 4 = 500

보상:
    스텝당: -1,
    승객을 목적지에 배달: +20,
    "pickup"과 "drop-off" 행동을 불법적으로 실행: -10
    
파란색: 승객
자홍색: 목적지
노란색: 빈 택시
녹색: 가득 찬 택시
"""
env = gym.make('Taxi-v3')
n_states = env.observation_space.n  # 500
n_actions = env.action_space.n      # 6

# 알고리즘의 파라미터 설정: 스텝 사이즈 alpha (0, 1], 0 보다 큰 작은 탐색률 e 
GAMMA = 0.99  # time decay
ALPHA = 0.9  # learning rate
epsilon = 0.7 # exploration start
epsilon_final = 0.1
epsilon_decay = 0.9999

# Q(s,a)를 초기화
Q = defaultdict(lambda: np.zeros(n_actions))

n_episodes = 5000  # 에피소드 수 증가

scores = []  # agent 가 episode 별로 얻은 score 기록
steps = []  # agent 가 episode 별로 목표를 찾아간 step 수 변화 기록
greedy = [] # epsilon decay history 기록

#Loop for each episode:
for episode in range(n_episodes):
    # 에피소드를 초기화 
    s, _ = env.reset()
    
    step = 0
    score = 0
    # 각 에피소드의 각 스텝에 대한 반복문
    while True:
        step += 1
        # Q에서 유도된 정책(예: e-greedy)을 사용하여 S에서 A를 선택
        # 행동 정책 : e-greedy
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s])
            
        # epsilon이 epsilon_final보다 크다면 epsilon_decay를 곱하여 감소
        if epsilon > epsilon_final:
            epsilon *= epsilon_decay
        
        # 행동 A를 취하고, R, S'을 관찰
        s_, r, terminated, truncated, _ = env.step(a)
        score += r
        
        # Q(S,A)를 업데이트: Q(S,A) <- Q(S,A) + alpha[R + gamma*max_aQ(S',a) - Q(S, A)]
        # 최적 행동가치함수 q*를 직접 근사
        # 대상 정책 : greedy policy
        Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * np.max(Q[s_]) - Q[s][a])

        # 에피소드가 끝나면 반복문 종료
        if terminated or truncated:
            break
        
        #S <- S'
        s = s_ 
        
    steps.append(step)
    scores.append(score)
    greedy.append(epsilon)
    
    # 평균 계산
    if episode % 100 == 0 and episode > 0:
        recent_scores = scores[-100:] if len(scores) >= 100 else scores
        recent_steps = steps[-100:] if len(steps) >= 100 else steps
        print(f"Episode {episode}: 최근 {len(recent_scores)} episode 평균 score = {np.mean(recent_scores):.2f}, 평균 step = {np.mean(recent_steps):.2f}")

print("학습 완료! 렌더링을 먼저 시작합니다...")

# 렌더링을 5번 연속으로 실행
num_renders = 5
print(f"렌더링을 {num_renders}번 연속으로 실행합니다...")
print("렌더링을 종료하려면 Ctrl+C를 누르세요.")

for render_episode in range(num_renders):
    print(f"\n=== 렌더링 에피소드 {render_episode + 1}/{num_renders} ===")
    
    # 렌더링을 위한 별도 환경 생성 및 실행
    render_env = gym.make('Taxi-v3', render_mode="human")
    s, _ = render_env.reset()
    step = 0
    score = 0

    while True:
        step += 1
        # 학습된 Q값을 사용하여 행동 선택 (epsilon=0으로 greedy)
        a = np.argmax(Q[s])
        
        # 행동 A를 취하고, R, S'을 관찰
        s_, r, terminated, truncated, _ = render_env.step(a)
        score += r
        
        # 에피소드가 끝나면 반복문 종료
        if terminated or truncated:
            print(f"렌더링 {render_episode + 1} 완료! 최종 점수: {score}, 스텝 수: {step}")
            break
        
        #S <- S'
        s = s_

    # 렌더링 환경 닫기
    render_env.close()

print("렌더링 완료! 이제 그래프를 표시합니다...")

# 그래프 생성 및 저장
plt.figure(figsize=(15, 5))

# 각 에피소드별 단계 수 그래프
plt.subplot(1, 3, 1)
plt.plot(steps)
plt.title("Steps of Taxi-v3\nGAMMA: {}, ALPHA: {}".format(GAMMA, ALPHA))
plt.xlabel('episode')
plt.ylabel('steps per episode')

# 각 에피소드별 점수 그래프
plt.subplot(1, 3, 2)
plt.plot(scores)
plt.title("Scores of Taxi-v3\nGAMMA: {}, ALPHA: {}".format(GAMMA, ALPHA))
plt.xlabel('episode')
plt.ylabel('score per episode')

# epsilon decay history 그래프
plt.subplot(1, 3, 3)
plt.plot(greedy)
plt.title("Epsilon decay history\nepsilon: {}, decay: {}".format(epsilon, epsilon_decay))
plt.xlabel('episode')
plt.ylabel('epsilon per episode')

plt.tight_layout()
plt.show(block=True)  # 수동으로 닫을 수 있도록 block=True


print("\n모든 작업이 완료되었습니다!")


