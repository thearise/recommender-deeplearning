import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import ops

class RBM(object):

    def __init__(self, visibleDimensions, epochs=20, hiddenDimensions=50, ratingValues=10, learningRate=0.001, batchSize=100):

        self.visibleDimensions = visibleDimensions
        self.epochs = epochs
        self.hiddenDimensions = hiddenDimensions
        self.ratingValues = ratingValues
        self.learningRate = learningRate
        self.batchSize = batchSize


    def Train(self, X):

        ops.reset_default_graph()

        self.MakeGraph()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        # print('Input array: ', len(X))
        # for i in X[:10]:
        #     print('each? ', i, len(i))
        #     for j in i[:20]:
        #         print('each inner? ', j)

        # a = tf.Print(self.X, [self.X], message="This is a: ")

        # with tf.Session() as sess:
        #     # train_writer = tf.summary.FileWriter( './logs/1/train ', sess.graph)
        #     # counter = 0
        #     for epoch in range(self.epochs):
        #         np.random.shuffle(X)
        #         trX = np.array(X)
        #         for i in range(0, trX.shape[0], self.batchSize):
        #             counter += 1
        #             # merge = tf.summary.merge_all()
        #             summary = sess.run(self.update, feed_dict={self.X: trX[i:i+self.batchSize]})
        #             # train_writer.add_summary(summary, counter)
        #         print("Trained epoch ", epoch)

        for epoch in range(self.epochs):
            np.random.shuffle(X)

            trX = np.array(X)
            for i in range(0, trX.shape[0], self.batchSize):
                # print("Each batch user: ", i)
                updateInfo = self.sess.run(self.update, feed_dict={self.X: trX[i:i+self.batchSize]})
                print('visBias ', updateInfo[5].shape, updateInfo[5])
                # for i in range(140):
                #     print(updateInfo[0][i])
                # print('gg12: ', len(updateInfo[0][0]), updateInfo[0][0])
                # print('updateInfo37: ', len(updateInfo[0][0][0]), updateInfo[0][0][0])
                # print("self.X head : ", self.X)
                # [0.4127353 , 0.6147976 , 0.2249373 , ..., 0.41572675, 0.49273238,
                #  0.43234327],
                # [0.68443316, 0.6026959 , 0.7993496 , ..., 0.54018635, 0.28061023,
                #  0.42509094],
                # [0.11268605, 0.71863675, 0.4546749 , ..., 0.47841832, 0.1854169 ,
                #  0.5826774 ],


            print("Trained epoch ", epoch)



    def GetRecommendations(self, inputUser):

        # hidden = tf.nn.sigmoid(tf.matmul(self.X, self.weights) + self.hiddenBias)
        # X
        # (1, 140)
        # weights
        # (140, 10)
        # hidden = tf.matmul(self.X, self.weights)
        hidden = tf.nn.sigmoid(tf.matmul(self.X, self.weights) + self.hiddenBias)
        # ===(1, 10)
        # Multiplicity
        # (1, 140)
        visible = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.weights)) + self.visibleBias)
        # ===(1, 140)
        feed = self.sess.run(hidden, feed_dict={ self.X: inputUser} )
        # print('feeding: ', feed)
        # (1, 10)
        # [[-0.09513444 -0.02614111 -0.03947446 -0.5910324  -0.9520067   1.052798
        # -0.5325289   0.91146463  0.09835547  1.1934302 ]]
        rec = self.sess.run(visible, feed_dict={ hidden: feed} )
        # print('gr yay403 ', self.sess.run(hidden, feed_dict={ self.X: inputUser} ))
        # print('gr yay31 ', self.sess.run(visible, feed_dict={ hidden: feed} ))
        # print('gr yay411 ', np.reshape(rec[0], [14, 10]))
        # (1, 140)
        # [[0.61393170  0.40917027  0.81800870  0.62288100  0.70898830  0.47422993  0.68258125  0.9356602  0.63209516 0.4832114
        #   0.67377347  0.65538550  0.14956202  0.52296764  0.31174064  0.55484843  0.26372173  0.7861296  0.37454706 0.7498273
        #   0.56682116  0.3563695   0.13848582  0.24878481  0.35508862  0.65051645  0.3143337   0.24809335 0.61480796 0.73063034
        #   0.9042328   0.2725364   0.46434385  0.7115768   0.82369053  0.2396756   0.44896445  0.5715423  0.66369724 0.510059
        #   0.27918744  0.1983036   0.77655137  0.5594042   0.20241116  0.5413145   0.20698963  0.637098   0.49728572 0.6027021
        #   0.19093351  0.37250257  0.39646643  0.5188988   0.5060054   0.74512917  0.45030308  0.54577553 0.7234607  0.59033114
        #   0.4617407   0.69449574  0.48351043  0.8914714   0.4268717   0.6546996   0.24162026  0.3241619  0.726488   0.29133046
        #   0.62726766 0.1140858    0.8548988   0.8385331   0.76766974  0.31905314  0.41373786  0.7582209  0.6232928  0.2704909
        #   0.6055141  0.27016455   0.33509502  0.26535714  0.8027186   0.78115034  0.3834515   0.44835377 0.3032559  0.17661169
        #   0.8363136  0.6350631    0.39627346  0.53708774  0.22887397  0.13290408  0.0718651   0.8101705  0.16739698 0.19232325
        #   0.6491825  0.7894307    0.28116974  0.5925853   0.8945844   0.7918429   0.42377362  0.9694318  0.50831264 0.30508503
        #   0.21332307 0.84006876   0.4248194   0.7900749   0.63328236  0.18752298  0.63569087  0.48110253 0.6685734  0.45205754
        #   0.60521936 0.86025786   0.4194225   0.5759586   0.84145635  0.48058674  0.7180466   0.09525862 0.18017156 0.73966444
        #   0.9222958  0.6871993    0.72551537  0.48859978  0.5889821   0.3517349   0.06685856  0.80168825 0.87178135 0.66038877]]
        return rec[0]

    def MakeGraph(self):

        tf.set_random_seed(0)


        # Create variables for the graph, weights and biases
        # 37060 for each user rated over all movies
        # allow any number of samples
        self.X = tf.placeholder(tf.float32, [None, self.visibleDimensions], name="X")
        # print('Prof X: ', self.X)
        # print(self.X.numpy())

        # Initialize weights randomly
        maxWeight = -4.0 * np.sqrt(6.0 / (self.hiddenDimensions + self.visibleDimensions))
        # print('visible dim ', self.visibleDimensions, ' max ', maxWeight)
        debugWeights = tf.Variable(tf.random_uniform([self.visibleDimensions, self.hiddenDimensions], minval=-maxWeight, maxval=maxWeight), tf.float32, name="weights")
        self.weights = tf.Variable(tf.random_uniform([self.visibleDimensions, self.hiddenDimensions], minval=-maxWeight, maxval=maxWeight), tf.float32, name="weights")
        # print('RBM self.weights: ', self.weights)
        # hiddenBiasTest = [
        #
        # ]
        self.hiddenBias = tf.Variable(tf.zeros([self.hiddenDimensions], tf.float32, name="hiddenBias"))
        # self.hiddenBias = tf.Variable([1., 2., 3., 0., 1., 3., 2., 1., 2., 1.], tf.float32, name="hiddenBias")
        # hiddenTest = tf.Variable([1., 2., 3., 0., 1., 3., 2., 1., 2., 1.], tf.float32, name="someBias")
        # someBias = tf.Variable([1., 2., 3., 0., 1., 3., 2., 1., 2., 1.], tf.float32, name="someBias")
        # print('RBM self.hidBias: ', self.hiddenBias)
        vis = tf.Variable(tf.zeros([self.visibleDimensions], tf.float32, name="visibleBias"))
        self.visibleBias = tf.Variable(tf.zeros([self.visibleDimensions], tf.float32, name="visibleBias"))
        # print('RBM self.hidBias: ', self.visibleBias)

        # Perform Gibbs Sampling for Contrastive Divergence, per the paper we assume k=1 instead of iterating over the
        # forward pass multiple times since it seems to work just fine

        # Forward pass
        # Sample hidden layer given visible...
        # Get tensor of hidden probabilities
        whatX = self.X
        hProb0 = tf.nn.sigmoid(tf.matmul(self.X, self.weights) + self.hiddenBias)
        # whatIsThat = tf.matmul(self.X, self.weights)
        # whatt = tf.matmul(self.X, self.weights)
        # [[-0.28012142  0.44208398 -0.36117566 ...  0.3998775   0.605815
        #   -0.40113536]
        #  [-0.38935006 -0.19448404  0.25634712 ...  0.37112927  0.60117066
        #   -0.68904823]
        #  [-1.5181072   1.0687295   0.19043723 ... -0.9247377  -0.24090317
        #    0.5905348 ]
        #  ...
        #  [-0.05180011  0.06764646  0.0238469  ... -0.01634931  0.08752707
        #    0.00737953]
        #  [-0.1825322   0.06440931  0.08904052 ... -0.10049225 -0.05561137
        #   -0.0557038 ]
        #  [ 0.31226704  0.04848765 -0.04012126 ... -0.16272037 -0.13533883
        #   -0.08132371]]


        # tf.summary.histogram("hProb0SPS ", hProb0)
        # Sample from all of the distributions
        hSample = tf.nn.relu(tf.sign(hProb0 - tf.random_uniform(tf.shape(hProb0))))
        # Stitch it together
        forward = tf.matmul(tf.transpose(self.X), hSample)

        # Backward pass
        # Reconstruct visible layer given hidden layer sample
        v = tf.matmul(hSample, tf.transpose(self.weights)) + self.visibleBias
        # (100, 90660)
        vLar = tf.matmul(hSample, tf.transpose(self.weights)) + self.visibleBias
        # (5, 140)
        # Build up our mask for missing ratings
        vMask = tf.sign(self.X) # Make sure everything is 0 or 1
        # (5, 140)
        # (100, 90660)
        something = tf.reshape(vMask, [tf.shape(v)[0], -1, self.ratingValues])
        vMask3D = tf.reshape(vMask, [tf.shape(v)[0], -1, self.ratingValues]) # Reshape into arrays of individual ratings
        # (5, 14, 10)
        vMask3D = tf.reduce_max(vMask3D, axis=[2], keepdims=True) # Use reduce_max to either give us 1 for ratings that exist, and 0 for missing ratings
        # (5, 14, 1)
        # (100, 9066, 1)
        # Extract rating vectors for each individual set of 10 rating binary
        v = tf.reshape(v, [tf.shape(v)[0], -1, self.ratingValues])
        # (5, 14, 10)
        muli = v * vMask3D
        vProb = tf.nn.softmax(v * vMask3D) # Apply softmax activation function
        # (5, 14, 10)

        vProb = tf.reshape(vProb, [tf.shape(v)[0], -1]) # And shove them back into the flattened state. Reconstruction is done now.
        # (5, 140)
        # Stitch it together to define the backward pass and updated hidden biases
        hProb1 = tf.nn.sigmoid(tf.matmul(vProb, self.weights) + self.hiddenBias)
        # (5, 10)
        backward = tf.matmul(tf.transpose(vProb), hProb1)
        # (140, 10)
        # Now define what each epoch will do...
        # Run the forward and backward passes, and update the weights
        weightUpdate = self.weights.assign_add(self.learningRate * (forward -
        backward))
        # Update hidden bias, minimizing the divergence in the hidden nodes
        hiddenBiasUpdate = self.hiddenBias.assign_add(self.learningRate *
        tf.reduce_mean(hProb0 - hProb1, 0))
        # Update the visible bias, minimizng divergence in the visible results
        visibleBiasUpdate = self.visibleBias.assign_add(self.learningRate *
        tf.reduce_mean(self.X - vProb, 0))

        self.update = [vis, backward, debugWeights, weightUpdate, hiddenBiasUpdate, visibleBiasUpdate]
