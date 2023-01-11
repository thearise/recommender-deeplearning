import torch
import torchvision
import numpy as np
import pickle

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

        finalWeight = []
        finalVisBias = []
        finalHidBias = []
        eachX = []
        trX = []
        for epoch in range(self.epochs):
            np.random.shuffle(X)

            trX = np.array(X)
            for i in range(0, trX.shape[0], self.batchSize):
                # print("Each batch user: ", i)
                updateInfo = self.sess.run(self.update, feed_dict={self.X: trX[i:i+self.batchSize]})
                print('visBias weight ', updateInfo[3].shape, updateInfo[3])
                finalWeight = updateInfo[3]
                finalVisBias = updateInfo[5]
                finalHidBias = updateInfo[4]
                eachX = updateInfo[6]
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
            print("Each eachX", eachX)
        print("Final Weight: ", finalWeight)
        # torch.save({
        #         'weight': finalWeight
        #     }, 'tensor.pth')
        # print("Saved Weight: ")

        with open('checkpoint.pkl', 'wb') as outp:
            # company1 = Company('banana', 40)
            # print(self.sess.run(self.X))

            # saveXT = torch.tensor(saveX)
            # torch.save(self.X, 'tensor.pt')


            # with tf.Session() as sess:
            #     print(sess.run(self.X))
            #     print (self.X.eval())

            # torch.save({
            #     'weight': self.weights
            # }, 'tensor.pth')
            print("Before saving: ")
            # pickle.dump(finalWeight, outp, pickle.HIGHEST_PROTOCOL)
            checkpoint1 = CheckPoint('weight', finalWeight)
            pickle.dump(checkpoint1, outp, pickle.HIGHEST_PROTOCOL)

            checkpoint2 = CheckPoint('visBias', finalVisBias)
            pickle.dump(checkpoint2, outp, pickle.HIGHEST_PROTOCOL) 
                   
            checkpoint3 = CheckPoint('hidBias', finalHidBias)
            pickle.dump(checkpoint3, outp, pickle.HIGHEST_PROTOCOL) 

            checkpoint4 = CheckPoint('iniXVal', trX)
            pickle.dump(checkpoint4, outp, pickle.HIGHEST_PROTOCOL) 
            print("After saving: ")
            # company2 = Company('spam', 42)
            # pickle.dump(company2, outp, pickle.HIGHEST_PROTOCOL)


    def GetRecommendations(self, inputUser):
        print("Get Recommendation Self: ", self)
        hidden = tf.nn.sigmoid(tf.matmul(self.X, self.weights) + self.hiddenBias)
        visible = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.weights)) + self.visibleBias)
        print("Get Recommendation IPU: ", inputUser)
        print("Retrieving user031 ", ' & ', len(inputUser[0]), 'and ', type(inputUser), type(self.X))
        feed = self.sess.run(hidden, feed_dict={ self.X: inputUser} )
        rec = self.sess.run(visible, feed_dict={ hidden: feed} )
        return rec[0]

    def GetRecommendationsAtCkPt(self, inputUser, weight ):
        print("Get Recommendation Self: ", self)
        hidden = tf.nn.sigmoid(tf.matmul(self.X, self.weights) + self.hiddenBias)
        visible = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.weights)) + self.visibleBias)
        feed = self.sess.run(hidden, feed_dict={ self.X: inputUser} )
        rec = self.sess.run(visible, feed_dict={ hidden: feed} )
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

        self.update = [vis, backward, debugWeights, weightUpdate, hiddenBiasUpdate, visibleBiasUpdate, self.X]

class CheckPoint(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value