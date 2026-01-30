import gymnasium as gym

from stable_baselines3 import A2C

env_id = 'Preschool-v0'
if env_id not in gym.envs.registry:
        gym.register(
            id=env_id,
            entry_point='preschool.grid_world.preschool_grid:Preschool_Grid',
            max_episode_steps=100,
        )
env = gym.make(env_id)

#env = gym.make("CartPole-v1", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()

obs = vec_env.reset()
for i in range(1000):#
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    #if done:
    #  obs = vec_env.reset()
