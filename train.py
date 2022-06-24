from env import TradingEnv, TradingTransferEnv
from stable_baselines3 import A2C, PPO, DQN

transfer_steps = 50000
steps = 10000
model_dir = 'Models-V2'

env = TradingTransferEnv()
env.reset()

model = DQN("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=transfer_steps)
model.save(model_dir + '/DQN-Transfer')

env = TradingTransferEnv()
env.reset()

model = A2C("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=transfer_steps)
model.save(model_dir + '/A2C-Transfer')

env = TradingTransferEnv()
env.reset()

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=transfer_steps)
model.save(model_dir + '/PPO-Transfer')

env = TradingEnv()
env.reset()

model = DQN("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=steps)
model.save(model_dir + '/DQN')

env = TradingEnv()
env.reset()

model = A2C("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=steps)
model.save(model_dir + '/A2C')

env = TradingEnv()
env.reset()

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=steps)
model.save(model_dir + '/PPO')