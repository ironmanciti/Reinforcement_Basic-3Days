# CartPole 환경 시각화 예제
# CartPole은 막대가 수직으로 서있도록 카트를 좌우로 움직이는 강화학습 환경입니다

# Environment 초기화
import gymnasium as gym
env = gym.make('CartPole-v1', render_mode="human")  # CartPole 환경을 생성하고 화면에 렌더링
obs, info = env.reset(seed=42)  # 환경을 초기화하고 초기 관찰값과 정보를 받음

# 시각화 - 10000번의 스텝 동안 환경을 실행
for _ in range(10000):
    # 랜덤한 행동을 선택 (0: 왼쪽으로 이동, 1: 오른쪽으로 이동)
    action = env.action_space.sample()
    # 선택한 행동을 환경에서 실행하고 결과를 받음
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 에피소드가 종료되면 환경을 리셋
    if terminated or truncated:
        obs, info = env.reset()
        
env.close()  # 환경을 종료
