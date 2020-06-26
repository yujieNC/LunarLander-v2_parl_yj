# LunarLander-v2-PARL-yj
本工程是百度强化学习7天打卡营的终极复现项目

AIstudio昵称：yujiekun77

二星环境-Box2D(LunarLander-v2)

## 环境配置
numpy
gym
matplotlib
paddlepaddle
parl1.3.1


##实现细节
LunarLander-v2的目标就是使月球着陆器能稳稳当当地停在指定区域内。
LunarLander-v2的state有8维，是连续的。Action有4维，是离散的。因此，本工程采用DQN算法，使用的框架为paddlepaddle和PARL。

`model.py` ：Q网络文件

`agent.py`:智能体文件，每隔200步拷贝一次Q网络参数

`LunarLander_train.py`:模型训练，并实时绘制train_reward曲线(使用队列，保存最新的500组数据).每训练50个episode，测试10次(开启显示渲染)

`evaluate.py`:测试模型，运行100个episode，实时绘制得分散点图，并统计通关百分比(gym官网：得分大于200即认为通关)

###训练策略
`Stage1`以较大的学习率和较大的经验采样batchsize进行训练，直到train_reward有明显下降趋势时停止训练，中途保存测试效果好的多个模型

`Stage2`对于Stage1中保存的模型，减小学习率，减小模型的随机探索概率e_greed，并增大经验回放池容量，继续训练，进行更谨慎的寻优

## 最终效果
从`/modeldir`中保存的模型得分图和模型效果统计表`/modeldir/modelPerformance.ods`可以看出，本工程训练得到的模型，具有很高的通关率，并且平均得分也不错。其中，编号为2350、2150、2300的模型均取得了100%的通关率。但它们的平均得分不及通关率略低的其他模型。
通过可视化窗口可以看到，这些通关率高的模型，往往更加谨慎，为了保证通关，会进行更多的操作使着陆器平稳着陆。但着陆器进行一次动作，可能会获得小量负分.
从安全角度考虑，应该选择通关率高的模型。

##其他文件
`/modeldir`:保存模型参数和模型得分散点图

`/modeldir/modelPerformance.ods`:各模型得分值和通关率统计

`output.gif`:模型`LunarLander_dqn_2300.ckpt`运行结果动图


