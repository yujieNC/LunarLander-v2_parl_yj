import numpy as np
import gym
import os
import random
import collections
import matplotlib.pyplot as plt

from model import Model
from agent import Agent
from parl.utils import logger

# export CUDA_VISIBLE_DEVICES=''

LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒几条新增经验后再learn，提高效率
MEMORY_SIZE = 50000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1500  # replay_memory 里需要预存一些经验数据，再开启训练
BATCH_SIZE = 300   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001 # 学习率
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等



from parl.algorithms import DQN # 直接从parl库中导入DQN算法

#经验回放函数
class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)


# 训练一个episode
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            # print("batch shape:", batch_obs.shape, batch_action.shape, batch_reward.shape, batch_next_obs.shape, batch_done.shape)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


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

    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

    # 根据parl框架构建agent
    model = Model(act_dim=action_dim)
    algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim=obs_shape[0],
        act_dim=action_dim,
        e_greed=0.001,  # 有一定概率随机选取动作，探索
        e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低

    # 加载模型
    # 若需要测试代码能否运行，则可能需要更改路径，或直接将这几行注释掉，从头训练
    model_dir = './modeldir'
    model_name = '/LunarLander_dqn_2300.ckpt'
    baseEpisode = 2300
    if os.path.exists(model_dir + model_name):
        agent.restore(model_dir + model_name)

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, rpm)

    max_episode = 3000

    # 用于实时绘制train reward的队列
    trainreward_curvesize = 500
    trainreward_curve = collections.deque(maxlen=trainreward_curvesize)
    episodebuf = collections.deque(maxlen=trainreward_curvesize)

    plt.figure()
    # 开始训练
    episode = 0
    last_eval_reward = 255.0
    # 每训练50个episode，测试10个episode
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part
        for i in range(0, 50):
            total_reward = run_episode(env, agent, rpm)
            logger.info('Train: episode: {}, train_reward:{}'.format(episode+baseEpisode, total_reward))
            trainreward_curve.append(total_reward)
            episodebuf.append(episode)
            episode += 1

            # 实时绘制train reward曲线
            plt.clf()
            plt.plot(episodebuf, trainreward_curve)
            plt.xlabel('episode_train')
            plt.ylabel('trainReward')
            plt.pause(0.01)

        # test part
        eval_reward = evaluate(env, agent, 10, render=True)  # render=True 查看显示效果

        logger.info('Evaluate: episode:{}    e_greed:{}   test_reward:{}'.format(
            episode+baseEpisode, agent.e_greed, eval_reward))

        if eval_reward > last_eval_reward:
            last_eval_reward = eval_reward
            agent.save(model_dir + '/LunarLander_dqn_{}.ckpt'.format(episode+baseEpisode))

    plt.close()

    # 训练结束，保存模型
    agent.save(model_dir + '/LunarLander_dqn.ckpt')

