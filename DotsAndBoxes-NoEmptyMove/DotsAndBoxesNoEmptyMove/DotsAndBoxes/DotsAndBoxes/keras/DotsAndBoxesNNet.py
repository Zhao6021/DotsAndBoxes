import sys
sys.path.append('..')
from utils import *

import tensorflow as tf
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# config = tf.compat.v1.ConfigProto(allow_soft_placement = True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)

import argparse
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras.optimizers import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *

class DotsAndBoxesNNet():
    def __init__(self, game, args):

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options,allow_soft_placement = True))
        tf.compat.v1.keras.backend.set_session(sess)

        # game params
        self.board_x, self.board_y = game.getBoardSize()    #初始化盤面大小
        self.action_size = game.getActionSize()             #初始化總行動數
        self.args = args                                    #初始化args(用傳進來的)

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y   SHAPE = (?,盤面大小,盤面大小)

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)        # batch_size  x board_x x board_y x 1   SHAPE = (?,盤面大小,盤面大小,1)
        '''
        EXAMPLE:
        x = Input((6,6))
        >>> x
        <tf.Tensor 'input_2:0' shape=(?, 6, 6) dtype=float32>

        x = Reshape((6,6,1))(x)
        >>> x
        <tf.Tensor 'reshape_1/Reshape:0' shape=(?, 6, 6, 1) dtype=float32>
        '''
        #使用relu當activation        
        #axis代表需要被標準化的軸 若data_format="channels_first" axis 通常為1 此處為3 因為channels_last
        #args.num_channels = 512 從NNet.py傳進來的
        #如果 use_bias=true, 會創建一個偏置向量並添加在輸出中
        #3是kernel_size指名2D卷積窗口的長跟寬
        #padding = same 則輸出大小則與原來相同 = false 輸出則會少了邊界 ， 因為不對邊界處理所以省略了
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
       
        '''
        h_conv1 = <tf.Tensor 'activation_1/Relu:0' shape=(?, 6, 6, 512) dtype=float32>
        h_conv2 = <tf.Tensor 'activation_2/Relu:0' shape=(?, 6, 6, 512) dtype=float32>
        h_conv3 = <tf.Tensor 'activation_3/Relu:0' shape=(?, 4, 4, 512) dtype=float32>
        h_conv4 = <tf.Tensor 'activation_4/Relu:0' shape=(?, 2, 2, 512) dtype=float32>
        '''
       
        h_conv4_flat = Flatten()(h_conv4) #將輸入展平，不影響batch size ，預設為channal_last
        '''
        理應是:
        <tf.Tensor 'flatten_2/Reshape:0' shape=(?, 2048) dtype=float32>
        實際為:
        <tf.Tensor 'flatten_2/Reshape:0' shape=(?, ?) dtype=float32> 
        '''  
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
        '''
        Dropout在訓練中每次更新時，將輸入單元按比率隨機設置為0，有助於防止overfitting
        args.dropout=0.3  可設定在0~1之間，為需要丟棄的比例。
        Dense 實現 output = activation(dot(input, kernel) + bias) 為常用的全連接層。
        此處輸入為上一層的(?,2048) 輸出為 (?,1024)
        >>> s_fc1
        <tf.Tensor 'dropout_1/cond/Merge:0' shape=(?, 1024) dtype=float32>
        >>> s_fc2
        <tf.Tensor 'dropout_2/cond/Merge:0' shape=(?, 512) dtype=float32>
        '''   
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1
        '''
        <tf.Tensor 'pi/Softmax:0' shape=(?, 36) dtype=float32>
        softmax是在算機率的，猜測這部分37 為 36個可下子的點+1不下子
        <tf.Tensor 'v/Tanh:0' shape=(?, 1) dtype=float32>
        '''
        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v]) #實例化一個model，給定輸入張量為(?,6,6) 輸出張量為(?,37)跟(?,1)
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
        '''
        model.compiler 用於配置訓練模型。
        loss 為決定要最小化甚麼目標函數 此處為 categorical_crossentropy跟mean_squared_error
        optimizer決定優化函數為adam
        lr = 0.001  learning rate
        '''
