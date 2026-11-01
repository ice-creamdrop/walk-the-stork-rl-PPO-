import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

class WalkTheStorkEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=10000):
        super().__init__()
        self.max_steps = max_steps
        
        # 행동 공간 (0: 대기, 1: 왼쪽 화살표, 2: 오른쪽 화살표)
        self.action_space = spaces.Discrete(3)
        
        # 관측 공간은 원래대로 4차원 유지! (기존 SWF 소켓 코드와 완벽 호환)
        low = np.array([-150.0, -50.0, -50.0, 0.0], dtype=np.float32)
        high = np.array([120.0, 50.0, 50.0, 100.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.metres = 0
        self.going = "yes"
        self.bs = 1.0
        self.bi = 1.0
        self.xx = 0.0
        self.xs = 0.0
        self.body_rotation = 0.0
        self.rotation = 0.0
        self.steps = 0
        
        if self.np_random is not None:
            self.bs += self.np_random.uniform(-0.05, 0.05)

        # 소켓 타이밍 1프레임 선행 진행
        self._physics_step(0)

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.body_rotation,
            self.rotation,
            self.bs,
            self.bi
        ], dtype=np.float32)

    def _physics_step(self, current_action):
        """실제 ActionScript 물리 엔진 1프레임을 진행하는 내부 함수"""
        self.steps += 1
        self.xx += self.xs
        self.metres = int(np.floor(self.xx / 550.0 * 3.0))
        
        terminated = False
        if self.going == "yes":
            self.xs = self.bi * 3.0
            self.bi += 0.002
            
            if current_action == 1:
                self.bs -= self.bi
            elif current_action == 2:
                self.bs += self.bi
                
            if self.body_rotation < -110.0 or self.body_rotation > 80.0:
                self.rotation = 0.0
                self.bs = 0.0
                self.going = "no"
                terminated = True
                
            if self.going == "yes":
                self.bs += (self.rotation * self.bi) / 100.0
                self.body_rotation += self.bs
                self.rotation = self.body_rotation / 3.0
                
        return terminated

    def step(self, action):
        # [핵심] 인간의 반응 주기(예: 4~7 프레임) 설정
        # (참고: 1프레임만 키를 치고 8~12프레임을 쉬면 물리적으로 후반 가속도를 못 이깁니다. 
        #  먼저 3~6 프레임 정도로 테스트해 보시는 것을 강력 추천합니다!)
        delay_frames = int(self.np_random.integers(3, 7))
        
        total_reward = 0.0
        terminated = False
        truncated = False

        # 반응 주기(delay_frames) 동안 물리 엔진을 연속으로 굴림!
        for i in range(delay_frames):
            # 인간처럼 처음 1~2 프레임만 선택한 키(action)를 누르고, 
            # 다음 판단 전까지 남은 반응 지연 시간 동안은 손을 뗌(0)
            act = action if i < 2 else 0
            
            terminated = self._physics_step(act)
            truncated = self.steps >= self.max_steps

            if terminated:
                total_reward -= 10.0
                break
            else:
                if self.body_rotation > 0:
                    angle_penalty = self.body_rotation / 80.0
                else:
                    angle_penalty = abs(self.body_rotation) / 110.0

                total_reward += 1.0 + (self.bi * 0.1) - (angle_penalty * 2.0)

            if truncated:
                break

        info = {"metres": self.metres, "steps": self.steps, "bi": self.bi}
        return self._get_obs(), float(total_reward), terminated, truncated, info


modelName = "ppo_300k"

if __name__ == "__main__":
    vec_env = make_vec_env(
        WalkTheStorkEnv,
        n_envs=8,
        monitor_dir=f"./logs/{modelName}"
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.995,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=1024,
        save_path="./checkpoints/",
        name_prefix="stork_ai"
    )
    
    print("=== 황새 오래걷기 PPO (Frame Skip 인간 반응속도) 학습 시작 ===")
    model.learn(total_timesteps=300_000, callback=checkpoint_callback)
    
    model.save(modelName)
    print(f"학습 완료! 모델이 '{modelName}.zip'으로 저장되었습니다.")