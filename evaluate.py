import numpy as np
import gym
import matplotlib.pyplot as plt
import collections

from model import Model
from agent import Agent
from parl.utils import logger

# export CUDA_VISIBLE_DEVICES='' for CPU version

LEARNING_RATE = 0.0005 # 学习率
GAMMA = 0.99 # reward 的衰减因子

from parl.algorithms import DQN # 直接从parl库中导入DQN算法


# 评估 agent, 跑 testtime 个episode，总reward求平均
def evaluate(env, agent, testtime,render=False):
    eval_reward = []
    for i in range(testtime):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    action_dim = env.action_space.n  #
    obs_shape = env.observation_space.shape  #

    logger.info('obs_dim: {}, act_dim: {}'.format(obs_shape[0], action_dim))


    # 根据parl框架构建agent
    model = Model(act_dim=action_dim)
    algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim=obs_shape[0],
        act_dim=action_dim,
        e_greed=0.1,  # 有一定概率随机选取动作，探索
        e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低

    # 加载模型
    model_dir = './modeldir'
    model_name = '/LunarLander_dqn_2600.ckpt'

    agent.restore(model_dir + model_name)

    test_time = 100

    plt.figure()

    evalReward_list = collections.deque(maxlen=test_time)
    test_time_list = collections.deque(maxlen=test_time)
    #开始测试
    solved =0
    for i in range(test_time):
        eval_reward = evaluate(env, agent, 1, render=True)  # render=True 查看显示效果
        if eval_reward>200:
            solved +=1
        test_time_list.append(i)
        evalReward_list.append(eval_reward)
        plt.clf()
        plt.plot(test_time_list, evalReward_list, '*')
        plt.xlabel('test_time')
        plt.ylabel('evalReward')
        plt.pause(0.01)

    # plt.show()

    evalndarry = np.array(evalReward_list)
    print("evalReward_mean: :", np.mean(evalndarry))
    print("solved percent: {}%".format(solved*100.0/test_time))
    plt.savefig(model_dir + model_name+'_evalreward.png')
    plt.close()


