# DQN.py for the "pyagme simple car game"

# NOTE : "INPUT DATA PRE_PROCESSING"
# - Don't forget if you change the input, you must change "pre-processing-scaling" accordingly!

# 알파고를 만든 구글의 딥마인드의 논문을 참고한 DQN 모델을 생성합니다.
# http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
import tensorflow as tf
import numpy as np
import random
from collections import deque
import tensorflow.contrib.slim as slim
import time


class DQN:
    # 학습에 사용할 플레이결과를 얼마나 많이 저장해서 사용할지를 정합니다.
    # (플레이결과 = 게임판의 상태 + 취한 액션 + 리워드 + 종료여부)
    REPLAY_MEMORY = 50000   # original : 10000
    # 학습시 사용/계산할 상태값(정확히는 replay memory)의 갯수를 정합니다.
    BATCH_SIZE = 32  # Batch size up : accuracy up , speed down  # original : 3
    # 과거의 상태에 대한 가중치를 줄이는 역할을 합니다.
    GAMMA = 0.99

    # targetNN_update_rate
    tnn_update_rate = 0.001  # Original :0.001

    def __init__(self, session, n_input, n_action):
        self.session = session  # from sess = tf.Session()
        self.n_action = n_action  # 3 in this case. left, stop, right
        self.n_input = n_input  # n_input = state = [self.x, self.y, self.obs_x, self.obs_y , self.obs_change]

        # 게임 플레이결과를 저장할 메모리
        self.memory = deque()
        # 현재 게임판의 상태
        self.state = None

        # 게임의 상태를 입력받을 변수
        # [게임판의 가로 크기, 게임판의 세로 크기, 게임 상태의 갯수(현재+과거+과거..)]
        self.input_X = tf.placeholder(tf.float32, [None,
                                                   n_input])  # 1st arg : type  |  2nd arg : [None, self.n_input, self.STATE_LEN] or [None, self.STATE_LEN, self.n_input] ???
        # 각각의 상태를 만들어낸 액션의 값들입니다. e.g. 0:left, 1:stop, 2:right
        self.input_A = tf.placeholder(tf.int64, [None])
        # 손실값을 계산하는데 사용할 입력값입니다. train 함수를 참고하세요.
        self.input_Y = tf.placeholder(tf.float32, [
            None])  # Y will be training data's output dataset.  |  ref. Y_var : predicted value

        # These parameters are for Double DQN
        self.targetQ_in_loss = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)


        self.Q = self._build_network('main')
        self.cost, self.train_op = self._build_op()

        # 학습을 더 잘 되게 하기 위해,
        # 손실값 계산을 위해 사용하는 타겟(실측값)의 Q value를 계산하는 네트웍을 따로 만들어서 사용합니다
        self.target_Q = self._build_network('target')

        self.Qout = 0 # initialize , used for Double DQN


        ######  SPEED UP CODING PART  ##### UP TO 'def updateTarget( )' #####
        # Fetch all the trainable_variables
        self.trainables = tf.trainable_variables()
        self.n_trainable_variables = len(self.trainables)

        # targetOps 에서 targetNN 의 업데이트 계산과정을 다 끝내버린다. -> 이렇게하면 처음에 있던 'self.trainables' 을 한번 사용한걸로 끝난것처럼 보이는데, 이상하게도, 이렇게해도 계속 업데이트되는 predictNN 의 값으로, targetNN 의 값이 업데이트된다.
        self.targetOps = self.updateTargetGraph(self.trainables, self.tnn_update_rate)

        self.target_NN_init = self.target_NN_initialize1(self.trainables)


    def updateTargetGraph(self, tfVars, tnn_update_rate):
        total_vars = len(tfVars)
        op_holder = []
        for idx, var in enumerate(tfVars[
                                  0:total_vars // 2]):  # The reason for '//2' -> As we're making predictNN, targetNN, we're taking out predictNN only. -> tfVars[0:total_vars//2] = first-half = predictNN
            op_holder.append(tfVars[idx + total_vars // 2].assign(
                (tnn_update_rate * var.value()) + ((1 - tnn_update_rate) * tfVars[idx + total_vars // 2].value())))
            #                 tfVars[total_vars//2] : targetNN 의 첫번째 component
            #                 tfVars[idx+total_vars//2] : second-half part = targetNN 에 / tf.assign 을 하는데, / "0.001*predictNN[i] 값 + (0.999)*targetNN[i] 값" 을 assign 한다. / = targetNN 에 0.001만큼의 predictNN 값을 업데이트 시켜준다.
        return op_holder

    def updateTarget(self, op_holder,
                     sess):  # Executed at an interval of 'update_freq' by running "tf.assign()" in 'def updateTargetGraph'.
        for op in op_holder:
            sess.run(op)

        # update checker - RESULT : successful
        check = False
        if check == True:
            self.t1 = tf.trainable_variables()
            import time
            print('predictNN :', self.session.run(self.t1[0][0, :5]))  # PredictNN
            print('targetNN :', self.session.run(self.t1[6][0, :5]))  # TargetNN
            time.sleep(1)

    def target_NN_initialize1(self, tfVars):
        total_vars = len(tfVars)
        op_holder = []
        for idx, var in enumerate(tfVars[0:total_vars // 2]):  # The reason for '//2' -> As we're making predictNN, targetNN, we're taking out predictNN only. -> tfVars[0:total_vars//2] = first-half = predictNN
            op_holder.append(tfVars[idx + total_vars // 2].assign(
                (1 * var.value()) + ((0 - 0) * tfVars[idx + total_vars // 2].value())))
            #                 tfVars[total_vars//2] : targetNN 의 첫번째 component
            #                 tfVars[idx+total_vars//2] : second-half part = targetNN 에 / tf.assign 을 하는데, / "0.001*predictNN[i] 값 + (0.999)*targetNN[i] 값" 을 assign 한다. / = targetNN 에 0.001만큼의 predictNN 값을 업데이트 시켜준다.
        return op_holder

    def target_NN_initialize2(self, op_holder,sess):  # Executed at an interval of 'update_freq' by running "tf.assign()" in 'def updateTargetGraph'.
        for op in op_holder:
            sess.run(op)


        ######################################################################

    def _build_network(self, name):
        with tf.variable_scope(name):

            # The size of the final layer before splitting it into Advantage and Value streams.
            h_size = 500

            # BNN : 베이지언 신경망(접근법)(=드롭아웃) : 학습 과정 중 네트워크의 활성 노드를 랜덤하게 0으로 설정함으로써, 일종의 정규화 역할을 수행하는 기법
            # 드롭아웃으로 네트워크에서 하나의 샘플을 취하는 것은 BNN 에서 샘플링하는 것과 유사한 일이다.
            # 시간의 경과에 따라 드롭아웃 확률을 줄여준다. -> 추정값에서 노이즈를 줄여주기 위해
            # RESULT : 확실히 눈에띄게 Learning Performance 가 상승함을 확인할 수있다.
            # Honestly speaking, I'm not sure just adding dropout is right.
            model = tf.layers.dense(inputs=self.input_X, units=500, activation=tf.nn.relu)
            model = tf.layers.dropout(model, rate=0.5)  # E.g. "rate=0.1" would drop out 10% of input units.
            model = tf.layers.dense(inputs=self.input_X, units=500, activation=tf.nn.relu)
            model = tf.layers.dropout(model, rate=0.5)  # E.g. "rate=0.1" would drop out 10% of input units.
            model = tf.layers.dense(inputs=self.input_X, units=500, activation=tf.nn.relu)
            model = tf.layers.dropout(model, rate=0.5)  # E.g. "rate=0.1" would drop out 10% of input units.
            model = tf.layers.dense(inputs=self.input_X, units=500, activation=tf.nn.relu)
            model = tf.layers.dropout(model, rate=0.5)  # E.g. "rate=0.1" would drop out 10% of input units.
            model = tf.layers.dense(inputs=self.input_X, units=500, activation=tf.nn.relu)
            model = tf.layers.dropout(model, rate=0.5)  # E.g. "rate=0.1" would drop out 10% of input units.
            model = tf.layers.dense(inputs=self.input_X, units=500, activation=tf.nn.relu)
            model = tf.layers.dropout(model, rate=0.5)  # E.g. "rate=0.1" would drop out 10% of input units.
            model = tf.layers.dense(inputs=self.input_X, units=500, activation=tf.nn.relu)
            model = tf.layers.dropout(model, rate=0.5)  # E.g. "rate=0.1" would drop out 10% of input units.
            model = tf.layers.dense(model, units=h_size, activation=tf.nn.relu) # NOTE "h_size" must be located at the end hidden_layer before split
            # This right above hidden layer is the end of DQN hidden layer. That's why there's no dropout.

            # From here, it's for "Duel DQN" -> Not output Q at once but split into A(advantage), V(value) and combine them to make Q
            # We take the output from the final convolutional layer and split it into separate advantage(A) and value streams(V).
            streamAC, streamVC = tf.split( model, num_or_size_splits=2, axis=1 )

            # Flattened_Action & Flattened_Value -> Since I'm not using Conv, I don't need it. Just I leave it here.
            streamA = slim.flatten( streamAC )
            streamV = slim.flatten( streamVC )

            # Call the class to initialize weights, which improve training performance - ref.http://hwangpy.tistory.com/153
            xavier_init = tf.contrib.layers.xavier_initializer()

            # Action_Weight & Value_Weight
            AW = tf.Variable(xavier_init([h_size // 2, self.n_action]))  # xavier_init( [row_size , column_size] )
            VW = tf.Variable(xavier_init([h_size // 2, 1]))

            # Flattened_ones * Weights
            Advantage = tf.matmul(streamA, AW)
            Value = tf.matmul(streamV, VW)

            # Then combine them together to get our final Q-values.
            self.Qout = Value + tf.subtract(Advantage, tf.reduce_mean(Advantage, axis=1, keep_dims=True))
            Q = self.Qout

            ### Double DQN from this line.

            # Take an action according to 'greedy-policy' : 1. Decide next_action using predictNN(=mainNN)
            predict = tf.argmax( self.Qout, axis=1 ) # -> Be careful when applying 볼츠만 approach

        return Q, predict


    def _build_op(self):

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        actions_onehot = tf.one_hot(self.actions, self.n_action, dtype=tf.float32)
        Q_by_DDQN = tf.reduce_sum(tf.multiply(self.Qout, actions_onehot), axis=1)

        td_error = tf.square(self.targetQ_in_loss - Q_by_DDQN)
        loss = tf.reduce_mean(td_error)

        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        updateModel = trainer.minimize(loss)

        return loss, updateModel


    def get_action(self):
        Q_value = self.session.run(self.Q, feed_dict={self.input_X: [self.state]})

        action = np.argmax(Q_value[0])

        return action

    def init_state(self, state):
        # 현재 게임판의 상태를 초기화합니다. 앞의 상태까지 고려한 스택으로 되어 있습니다.
        # state = [state for _ in range(self.STATE_LEN)]
        # axis=2 는 input_X 의 값이 다음처럼 마지막 차원으로 쌓아올린 형태로 만들었기 때문입니다.
        # 이렇게 해야 컨볼루션 레이어를 손쉽게 이용할 수 있습니다.
        self.state = state

    def remember(self, state, action, reward, terminal):
        # 학습데이터로 현재의 상태만이 아닌, 과거의 상태까지 고려하여 계산하도록 하였고,
        # 이 모델에서는 과거 3번 + 현재 = 총 4번의 상태를 계산하도록 하였으며,
        # 새로운 상태가 들어왔을 때, 가장 오래된 상태를 제거하고 새로운 상태를 넣습니다.

        ##### IMPORTANT ##### : Input data Preprocessing
        state[0] = state[0] / 800  # = self.x / self.display_width = 600
        state[1] = state[1] / 10  # = self.obs_x / self.display_width = 600
        state[2] = state[2] / 800  # self.display_height + self.obs_y's initial reset coordinate
        state[3] = state[3] / 800  # self.display_height + self.obs_y's initial reset coordinate

        next_state = state

        # 플레이결과, 즉, 액션으로 얻어진 상태와 보상등을 메모리에 저장합니다.
        self.memory.append((self.state, next_state, action, reward,
                            terminal))  # ref. self.input_X = tf.placeholder(tf.float32, [None, n_input, self.STATE_LEN])

        # 저장할 플레이결과의 갯수를 제한합니다.
        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

        self.state = next_state

    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, terminal

    def train(self):
        # 게임 플레이를 저장한 메모리에서 배치 사이즈만큼을 샘플링하여 가져옵니다.
        state, next_state, action, reward, terminal = self._sample_memory()

        # Below, we perform the Double-DQN update to the target Q-values
        _, Q1 = self.session.run(self.Q , feed_dict={ self.input_X: next_state }) # 다음 상태를 predictNN 에 넣어 greedy policy 를 통해, action 을 구합니다.
        Q2, _ = self.session.run(self.target_Q, feed_dict={ self.input_X: next_state } ) # 다음 상태를 타겟 네트웍에 넣어 target Q value를 구합니다

        # Choose Q values that are determined by targetNN by predictNN's next_action.  |  predictNN decided next_action and we choose Q values made by targetNN according to this next_action.
        doubleQ = Q2[ range(self.BATCH_SIZE) , Q1 ]
        #         Q2 is among Q[action[0]], Q[action[1]], Q[action[2]]]
        #                                Q1 is among 0,1,2 ,which means an action of left, stop, right
        # NOTE : Pick out a Q value in each sample in Q2 according to Q1 (Q1 decides which one to choose in Q2)

        # targetQ
        Y = []
        for i in range(self.BATCH_SIZE):
            if terminal[i]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.GAMMA * doubleQ[i] )

        # Update predictNN with the loss function -> (Y - Y_var)^2
        self.session.run( self.train_op , feed_dict={ self.input_X:state, self.targetQ_in_loss:Y , self.actions:action } )
