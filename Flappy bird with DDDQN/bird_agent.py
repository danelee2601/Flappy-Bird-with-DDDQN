# It updates the targetNN, "slowly(TRAIN_INTERVAL) but frequently(tnn_update_rate)"

import tensorflow as tf
import numpy as np
import random
import time
import timeit
from bird_game import env_replay, env_train
from DDDQN import DQN # Although it's written as DQN, but it's actually DDDQN. (It was just hard to rename everything.)


# 최대 학습 횟수
MAX_EPISODE = 1000000

# 4 프레임마다 한 번씩 학습합니다. (FYI: 한 게임에서 한번 진행하는게, 1 프레임 진행)
TRAIN_INTERVAL = 2  # 1일때가 학습이 더 잘되는거같기도 한데..  # predictNN 을 train 시키는 interval (not targetNN) 하지만, 이 정책에서는 targetNN 도 같이 update 한다.

# 학습 데이터를 어느정도 쌓은 후, 일정 시간 이후에 학습을 시작하도록 합니다.
OBSERVE = 1000

# action: 0: 좌, 1: 유지, 2: 우
NUM_ACTION = 2


# n_input = n_state
n_input =  5

def train():
    print('Waking up brain cells..')
    sess = tf.Session()

    game = env_train()
    brain = DQN( sess, n_input=n_input , n_action=NUM_ACTION ) # n_input = state = [self.x, self.y, self.obs_x, self.obs_y , self.obs_change]

    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    saver = tf.train.Saver(max_to_keep=30)  # Can save up to 100 trained models.
    sess.run(tf.global_variables_initializer())

    #writer = tf.summary.FileWriter('C:\\Users\\Administrator\\Desktop\\Python 공부\\Deep Learning\\Tensorflow\\Flapping Bird\\Flappy bird with DDDQN\\logs', sess.graph)
    #summary_merged = tf.summary.merge_all()

    # 타겟 네트웍을 초기화합니다.
    # 1. Make initial-targetNN same as predictNN  |  2. Let initial-targetNN be random different from predictNN (BETTER)
    initial_targetNN_same_as_predictNN = False  # False option is the one used in 강화학습 첫걸음 |
                                                 # Empirical Result : False option shows way faster speed, higher accuracy[performance] in learnining!
    if initial_targetNN_same_as_predictNN == True:
        brain.target_NN_initialize2( brain.target_NN_init , sess )

    # 다음에 취할 액션을 DQN 을 이용해 결정할 시기를 결정합니다.
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99999

    # 프레임 횟수
    time_step = 0

    # 게임을 시작합니다.
    state = game.reset()
    brain.init_state(state)
    for episode in range(MAX_EPISODE):

        terminal = False
        total_reward = 0

        while not terminal:
            # 입실론이 랜덤값보다 작은 경우에는 랜덤한 액션을 선택하고
            # 그 이상일 경우에는 DQN을 이용해 액션을 선택합니다.
            # 초반엔 학습이 적게 되어 있기 때문입니다.
            # 초반에는 거의 대부분 랜덤값을 사용하다가 점점 줄어들어
            # 나중에는 거의 사용하지 않게됩니다.
            if np.random.rand() < epsilon:
                action = random.randrange(NUM_ACTION) # action -> 0(left) or 1(stop) or 2(right)
            else:
                action = brain.get_action()

            # 일정 시간이 지난 뒤 부터 입실론 값을 줄입니다.
            if episode > OBSERVE:
                epsilon *= epsilon_decay
                if epsilon <= epsilon_min:
                    epsilon = 0.01 # 강화학습 첫걸음 에서는 0.1 로 잡았음. 0.1 도 경험 측면에서 나쁘진 않은듯

            # 결정한 액션을 이용해 게임을 진행하고, 보상과 게임의 종료 여부를 받아옵니다.
            state, reward, terminal = game.step(action)
            total_reward += reward
            # 현재 상태를 Brain에 기억시킵니다.
            # 기억한 상태를 이용해 학습하고, 다음 상태에서 취할 행동을 결정합니다.
            brain.remember(state, action, reward, terminal)

            if time_step > OBSERVE and time_step % TRAIN_INTERVAL == 0:
                # DQN 으로 학습을 진행합니다.
                brain.train() # predictNN 을 train 시킨다. (not targetNN)

                # 타겟 네트웍을 업데이트 해 줍니다. -> targetNN 업데이트 정책을 "조금씩 자주" 로 변경하고난 후, targetNN 업데이트 는 predictNN training 과 같이 이루어진다.
                brain.updateTarget( brain.targetOps , sess )

            time_step += 1


        print('n_game, %d,  score, %d,  epsilon, %0.3f' % (episode + 1, total_reward , epsilon))
        game.reset()

        # Make 'self.done' False in env.py
        game.reset_terminal()

        # For tensorboard

        if episode <= 10 == 0:
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, time_step)
            total_reward_list = []

        # 이걸 RUNTIME 에 대해서 업데이트되는걸로 바꿔야할듯 - 안끝나는건 걱정안해도됨 : epsilon 에 의해서 잘못된 행동선택으로 죽을수 있으니, 무조건 끝나게는 되있음.
        if episode % 100 == 0:
            saver.save(sess, 'model_saved\\dqn_epi{}_.ckpt'.format(episode), global_step=time_step)


def replay():
    print('Waking up brain cells..')
    sess = tf.Session()

    game = env_replay()
    brain = DQN(sess, n_input=n_input, n_action=NUM_ACTION) # n_input = state = [self.x, self.y, self.obs_x, self.obs_y , self.obs_change]

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model_saved')
    saver.restore(sess, ckpt.model_checkpoint_path)

    # 게임을 시작합니다.

    state = game.reset()
    brain.init_state(state)

    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        # Make 'self.reward' 0 in env.py
        game.total_reward_clear()

        while not terminal:
            # Choose the best option. There's no epsilon here.
            action = brain.get_action()

            action_decoder = {0:'       Flapping', 1: 'Nothing'}
            print('Action : {}'.format( action_decoder[action] ) )

            # 결정한 액션을 이용해 게임을 진행하고, 보상과 게임의 종료 여부를 받아옵니다.
            state, reward, terminal = game.step(action)
            total_reward += reward

            brain.remember(state, action, reward, terminal)

        print('n_game, %d,  score, %d ' % (episode + 1, total_reward))
        game.reset()


        # Make 'self.done' False in env.py
        game.reset_terminal()


# Execute
train_replay_option = 'train'

if train_replay_option == 'train':
    train()
elif train_replay_option == 'replay':
    replay()
