import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class WalkTheStorkEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=10000):
        super().__init__()
        self.max_steps = max_steps
        
        # 행동 공간 (0: 대기, 1: 왼쪽 화살표, 2: 오른쪽 화살표)
        self.action_space = spaces.Discrete(3)
        
        # 관측 공간 [body_rotation, rotation, bs, bi]
        low = np.array([-150.0, -50.0, -50.0, 0.0], dtype=np.float32)
        high = np.array([120.0, 50.0, 50.0, 100.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # ActionScript onClipEvent(load) 초기값 완벽 반영
        self.metres = 0
        self.going = "yes"
        self.bs = 1.0
        self.bi = 1.0
        self.xx = 0.0
        self.xs = 0.0
        self.body_rotation = 0.0
        self.rotation = 0.0
        self.steps = 0
        
        # 약간의 초기 노이즈 추가 (과적합 방지)
        if self.np_random is not None:
            self.bs += self.np_random.uniform(-0.05, 0.05)

        # [핵심 보정] 실제 SWF는 게임 시작 후 첫 프레임에 소켓 응답을 받기 전 
        # 키 입력 없이 1프레임을 먼저 굴리므로, 학습 환경도 똑같이 1프레임 선행 진행합니다.
        self.step(0)

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.body_rotation,
            self.rotation,
            self.bs,
            self.bi
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        
        # --- ActionScript onClipEvent(enterFrame) 로직 이식 ---
        self.xx += self.xs
        self.metres = int(np.floor(self.xx / 550.0 * 3.0))
        
        terminated = False
        
        if self.going == "yes":
            self.xs = self.bi * 3.0
            self.bi += 0.002
            
            # AI 행동 반영 (1: 왼쪽, 2: 오른쪽)
            if action == 1:
                self.bs -= self.bi
            elif action == 2:
                self.bs += self.bi
                
            # 넘어짐 판정 (뒤로: -110 미만, 앞으로: 80 초과)
            if self.body_rotation < -110.0 or self.body_rotation > 80.0:
                self.rotation = 0.0
                self.bs = 0.0
                self.going = "no"
                terminated = True
                
            # 물리 연산 업데이트
            if self.going == "yes":
                self.bs += (self.rotation * self.bi) / 100.0
                self.body_rotation += self.bs
                self.rotation = self.body_rotation / 3.0

        truncated = self.steps >= self.max_steps

        # 보상 함수 (Reward Shaping)
        if terminated:
            reward = -10.0
        else:
            if self.body_rotation > 0:
                angle_penalty = self.body_rotation / 80.0
            else:
                angle_penalty = abs(self.body_rotation) / 110.0

            reward = 1.0 + (self.bi * 0.1) - (angle_penalty * 2)

        info = {"metres": self.metres, "steps": self.steps, "bi": self.bi}
        return self._get_obs(), float(reward), terminated, truncated, info


modelName = "ppo_300k_m2"

if __name__ == "__main__":
    # 병렬 환경 8개 구성 (학습 속도 극대화)
    vec_env = make_vec_env(
        WalkTheStorkEnv,
        n_envs=8,
        monitor_dir=f"./logs/{modelName}"
    )

    # PPO 에이전트 설정 (미래 가치 반영을 위해 gamma를 0.995로 높게 설정)
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.995,
        verbose=1
    )
    
    print("=== 황새 오래걷기 PPO 학습 시작 ===")
    model.learn(total_timesteps=300_000)
    
    # 모델 저장
    model.save(modelName)
    print(f"학습 완료! 모델이 '{modelName}.zip'으로 저장되었습니다.")