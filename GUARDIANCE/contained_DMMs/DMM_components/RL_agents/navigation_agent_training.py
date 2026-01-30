import gymnasium as gym
from preschool.grid_world.rand_target import Rand_Target, PrescCoordinates, PrescFlattened, Extended
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3 import A2C, PPO

env_id = 'Preschool-v0'

gym.register(
    id=env_id,
    entry_point='preschool.grid_world.preschool_grid:Preschool_Grid',
    max_episode_steps=100,
)
env = gym.make(env_id)
env = Rand_Target(env)
env = PrescCoordinates(env)

#env = gym.make("CartPole-v1", render_mode="rgb_array")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=128,
    batch_size=64,
    learning_rate=3e-4,
)
model.learn(total_timesteps=30_000)

vec_env = model.get_env()


mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=100,
    deterministic=True,
)

print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    env.set_render_mode("human")
    #if done:
    #  obs = vec_env.reset()

model.save("navigation_agent")