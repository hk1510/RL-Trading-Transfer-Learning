from stable_baselines3 import A2C, PPO, DDPG, DQN
from env import TradingEvalEnv
import matplotlib.pyplot as plt
import pandas as pd

model_dir = 'Models-V2/'
results_dir = 'Results-V2/'

plt.figure(figsize=(15, 5), dpi=150)

for name in ['DQN', 'A2C', 'PPO', 'DQN-Transfer', 'A2C-Transfer', 'PPO-Transfer']:

    if (name.split('-')[0] == 'A2C'):
        model = A2C.load(model_dir + name)
    elif (name.split('-')[0] == 'PPO'):
        model = PPO.load(model_dir + name)
    elif (name.split('-')[0] == 'DQN'):
        model = DQN.load(model_dir + name)
    else:
        model = DDPG.load(model_dir + name)

    env = TradingEvalEnv()
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

    df = env.render()
    df.to_csv(results_dir + name + '.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    plt.plot(df['Timestamp'], (df['Money'] - 100000)/df['Money'], label=name)

plt.legend(loc='upper left')
plt.show()