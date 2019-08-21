

## Deep Reinforcement Learning
==================================

https://github.com/openai/spinningup

$ cd DRL
$ pip install -e .
$ pip install ~/carla/PythonClient (optional)
$ pip install opencv-python

==================================

references:

sac is based on:

Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor

https://arxiv.org/abs/1801.01290

sac1 is added based on:

Soft Actor-Critic Algorithms and Applications

https://arxiv.org/abs/1812.05905

==================================

ddgp vs sac1

* gym env 'Pendulum-v0':(Minimum_Episode_Return)

<div>
    <img src="https://github.com/createamind/DRL/blob/master/video_pic/ddpg1000.png" width="350" style="display:inline"/>
    <img src="https://github.com/createamind/DRL/blob/master/video_pic/sac1_1000.png" width="350" style="display:inline"/>
</div>

sqn experiments on gym env 'LunarLander-v2':

<div>
    <img src="https://github.com/createamind/DRL/blob/master/video_pic/LunarLander1.png" width="440" style="display:inline"/>
    <img src="https://github.com/createamind/DRL/blob/master/video_pic/LunarLander2.png" width="260" style="display:inline"/>
</div>


Try trained model on env 'Breakout-ram-v4':

$ python -m spinup.run test_policy ./saved_models/Breakout-ram-v4 -d -l 20000

More experiments: 
https://mp.weixin.qq.com/s/-ZWj-uw5wWWhGy3B08Xk3Q (sqn)
https://mp.weixin.qq.com/s/8vgLGcpsWkF89ma7T2twRA ('BipedalWalkerHardcore-v2')


### Awesome-DRL-Papers
Learning Latent Dynamics for Planning from Pixels
* https://arxiv.org/pdf/1811.04551.pdf

INFOBOT: TRANSFER AND EXPLORATION VIA THE INFORMATION BOTTLENECK
* code: to be released.

Unsupervised Meta-Learning for Reinforcement Learning
* https://arxiv.org/pdf/1806.04640.pdf

DIVERSITY IS ALL YOU NEED: LEARNING SKILLS WITHOUT A REWARD FUNCTION (DIAYN)
* https://arxiv.org/pdf/1802.06070.pdf
* code: https://github.com/ben-eysenbach/sac

Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (MAML)
* https://arxiv.org/abs/1703.03400


<img src="https://github.com/createamind/DRL/blob/master/video_pic/sac1.png" width="550" style="display:inline"/>
<img src="https://github.com/createamind/DRL/blob/master/video_pic/maxsqn.png" width="550" style="display:inline"/>
