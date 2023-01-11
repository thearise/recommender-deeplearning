# -*- coding: utf-8 -*-


from surprise import AlgoBase
from surprise import PredictionImpossible
import numpy as np
from RBM import RBM

class RBMAlgorithm(AlgoBase):

    def __init__(self, epochs=20, hiddenDim=100, learningRate=0.1, batchSize=100, sim_options={}):
        AlgoBase.__init__(self)
        self.epochs = epochs
        self.hiddenDim = hiddenDim
        self.learningRate = learningRate
        self.batchSize = batchSize

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def fitById(self, trainset, testUser=85):
        print("trainSet0: ", trainset.all_ratings())
        AlgoBase.fit(self, trainset)
        # print("trainSet1: ", trainset.all_ratings())
        numUsers = trainset.n_users
        numItems = trainset.n_items
        print("Users: ", numUsers, " Items: ", numItems)

        trainingMatrix = np.zeros([numUsers, numItems, 10], dtype=np.float32)
        for (uid, iid, rating) in trainset.all_ratings():
            adjustedRating = int(float(rating)*2.0) - 1
            trainingMatrix[int(uid), int(iid), adjustedRating] = 1


        print("Trainingmatrix Before Reshape: ")
        # 3D array
        print(trainingMatrix.shape)
        # Flatten to a 2D array, with nodes for each possible rating type on each possible item, for every user.
        trainingMatrix = np.reshape(trainingMatrix, [trainingMatrix.shape[0], -1])
        print("Trainingmatrix After Reshape: ")
        # 2D array
        print(print(trainingMatrix.shape))
        # (671, 90660)
        print("Trainingmatrix Shape One: ")
        print(trainingMatrix.shape[1])
        # 90660

        # Create an RBM with (num items * rating values) visible nodes
        rbm = RBM(trainingMatrix.shape[1], hiddenDimensions=self.hiddenDim,
                  learningRate=self.learningRate,
                  batchSize=self.batchSize,
                  epochs=self.epochs)

        print("Training Matrix RBM: ", trainingMatrix)
        rbm.Train(trainingMatrix)

        self.predictedRatings = np.zeros([numUsers, numItems], dtype=np.float32)
        innerTestId = trainset.to_inner_uid(str(testUser))
        print("Retrieving user ", testUser, "Inner id: ", innerTestId)
        recs = rbm.GetRecommendations([trainingMatrix[innerTestId]])
        recs = np.reshape(recs, [numItems, 10])
        predic = np.zeros([numUsers, numItems], dtype=np.float32)
            # np.savetxt("testingcsv2.csv", itemS, delimiter=",")
        for itemID, rec in enumerate(recs):
            # The obvious thing would be to just take the rating with the highest score:
            # rating = rec.argmax()
            # ... but this just leads to a huge multi-way tie for 5-star predictions.
            # The paper suggests performing normalization over K values to get probabilities
            # and take the expectation as your prediction, so we'll do that instead:

            normalized = self.softmax(rec)
            rating = np.average(np.arange(10), weights=normalized)
            predic[innerTestId, itemID] = (rating + 1) * 0.5
            self.predictedRatings[innerTestId, itemID] = (rating + 1) * 0.5

        print('itemiD: ')
        print(trainset.to_raw_iid(0))
        print(trainset.to_raw_iid(1))
        print(trainset.to_raw_iid(2))
        print(trainset.to_raw_iid(3))
        print(trainset.to_raw_iid(4))
        print(trainset.to_raw_iid(5))
        print(trainset.to_raw_iid(6))
        print(trainset.to_raw_iid(7))
        print(trainset.to_raw_iid(8))
        print(trainset.to_raw_iid(9))
        print(trainset.to_raw_iid(10))
        print(trainset.to_raw_iid(11))
        print(trainset.to_raw_iid(12))
        print(trainset.to_raw_iid(13))
        print("Predic: ", predic)




        # print("Processing user ", 534, "Inner id: ", trainset.to_inner_uid(str(535)))
        # recs = rbm.GetRecommendations([trainingMatrix[534]])
        # recs = np.reshape(recs, [numItems, 10])
        #
        #     # np.savetxt("testingcsv2.csv", itemS, delimiter=",")
        # for itemID, rec in enumerate(recs):
        #     # The obvious thing would be to just take the rating with the highest score:
        #     #rating = rec.argmax()
        #     # ... but this just leads to a huge multi-way tie for 5-star predictions.
        #     # The paper suggests performing normalization over K values to get probabilities
        #     # and take the expectation as your prediction, so we'll do that instead:
        #     normalized = self.softmax(rec)
        #     rating = np.average(np.arange(10), weights=normalized)
        #     self.predictedRatings[534, itemID] = (rating + 1) * 0.5

        # for uiid in range(trainset.n_users):
        #     # if (uiid % 50 == 0):
        #     print("Processing user ", uiid)
        #     recs = rbm.GetRecommendations([trainingMatrix[uiid]])
        #     recs = np.reshape(recs, [numItems, 10])
        #
        #         # np.savetxt("testingcsv2.csv", itemS, delimiter=",")
        #     # for itemID, rec in enumerate(recs):
        #     #     # The obvious thing would be to just take the rating with
        #     #     # the highest score:
        #     #     # rating = rec.argmax()
        #     #     # ... but this just leads to a huge multi-way tie for 5-star
        #     #     # predictions.
        #     #     # The paper suggests performing normalization over K values
        #     #     # to get probabilities
        #     #     # and take the expectation as your prediction, so we'll
        #     #     # do that instead:
        #     #     normalized = self.softmax(rec)
        #     #     rating = np.average(np.arange(10), weights=normalized)
        #     #     self.predictedRatings[uiid, itemID] = (rating + 1) * 0.5
        #
        #     for itemID, rec in enumerate(recs):
        #         normalized = self.softmax(rec)
        #         rating = np.average(np.arange(10), weights=normalized)
        #         self.predictedRatings[uiid, itemID] = (rating + 1) * 0.5
        #
        #     # if(uiid == 670):
        #     #     print("Reprocessing user ", 84)
        #     #     recs = rbm.GetRecommendations([trainingMatrix[84]])
        #     #     recs = np.reshape(recs, [numItems, 10])
        #     #     print(trainingMatrix[84])
        #     #     np.savetxt("trainingMatrixById.csv", trainingMatrix[84], delimiter=",")
        #     #     sums = []
        #     #     itemS = []
        #     #     itemLoop = 0
        #     #     for oneRec in recs:
        #     #         sum = oneRec[0] + oneRec[1] + oneRec[2] + oneRec[3] + oneRec[4] + oneRec[5] + oneRec[6] + oneRec[7] + oneRec[8] + oneRec[9];
        #     #         realRating = sum/2;
        #     #         sums.append(realRating)
        #     #         itemS.append(int(trainset.to_raw_iid(itemLoop)))
        #     #         print(int(trainset.to_raw_iid(itemLoop)))
        #     #         itemLoop += 1
        #     #     # print(itemS)
        #     #     np.savetxt("testingcsv.csv", sums, delimiter=",")

        return self


    def fit(self, trainset, testUser=85):
        # print("trainSet0: ", trainset.all_ratings())
        AlgoBase.fit(self, trainset)
        # print("trainSet1: ", trainset.all_ratings())
        numUsers = trainset.n_users
        numItems = trainset.n_items
        # print("Users: ", numUsers, " Items: ", numItems)

        trainingMatrix = np.zeros([numUsers, numItems, 10], dtype=np.float32)
        for (uid, iid, rating) in trainset.all_ratings():
            adjustedRating = int(float(rating)*2.0) - 1
            trainingMatrix[int(uid), int(iid), adjustedRating] = 1


        # print("Trainingmatrix Before Reshape: ")
        # 3D array
        print(trainingMatrix.shape)
        # Flatten to a 2D array, with nodes for each possible rating type on each possible item, for every user.
        trainingMatrix = np.reshape(trainingMatrix, [trainingMatrix.shape[0], -1])
        # print("Trainingmatrix After Reshape: ")
        # 2D array
        print(print(trainingMatrix.shape))
        # print("Trainingmatrix Shape One: ")
        # 37060
        print(trainingMatrix.shape[1])


        # Create an RBM with (num items * rating values) visible nodes
        rbm = RBM(trainingMatrix.shape[1], hiddenDimensions=self.hiddenDim,
                  learningRate=self.learningRate,
                  batchSize=self.batchSize,
                  epochs=self.epochs)
        # print("Training Matrix RBM: ", trainingMatrix.shape, trainingMatrix)
        rbm.Train(trainingMatrix)

        self.predictedRatings = np.zeros([numUsers, numItems], dtype=np.float32)

        # print("Retrieving user ", testUser, "Inner id: ", trainset.to_inner_uid(str(testUser)))
        # recs = rbm.GetRecommendations([trainingMatrix[84]])
        # recs = np.reshape(recs, [numItems, 10])
        #
        #     # np.savetxt("testingcsv2.csv", itemS, delimiter=",")
        # for itemID, rec in enumerate(recs):
        #     # The obvious thing would be to just take the rating with the highest score:
        #     #rating = rec.argmax()
        #     # ... but this just leads to a huge multi-way tie for 5-star predictions.
        #     # The paper suggests performing normalization over K values to get probabilities
        #     # and take the expectation as your prediction, so we'll do that instead:
        #     normalized = self.softmax(rec)
        #     rating = np.average(np.arange(10), weights=normalized)
        #     self.predictedRatings[84, itemID] = (rating + 1) * 0.5



        # print("Processing user ", 534, "Inner id: ", trainset.to_inner_uid(str(535)))
        # recs = rbm.GetRecommendations([trainingMatrix[534]])
        # recs = np.reshape(recs, [numItems, 10])
        #
        #     # np.savetxt("testingcsv2.csv", itemS, delimiter=",")
        # for itemID, rec in enumerate(recs):
        #     # The obvious thing would be to just take the rating with the highest score:
        #     #rating = rec.argmax()
        #     # ... but this just leads to a huge multi-way tie for 5-star predictions.
        #     # The paper suggests performing normalization over K values to get probabilities
        #     # and take the expectation as your prediction, so we'll do that instead:
        #     normalized = self.softmax(rec)
        #     rating = np.average(np.arange(10), weights=normalized)
        #     self.predictedRatings[534, itemID] = (rating + 1) * 0.5

        for uiid in range(trainset.n_users):
            if (uiid % 50 == 0):
                print("Processing user ", uiid)
            recs = rbm.GetRecommendations([trainingMatrix[uiid]])
            recs = np.reshape(recs, [numItems, 10])

                # np.savetxt("testingcsv2.csv", itemS, delimiter=",")
            # for itemID, rec in enumerate(recs):
            #     # The obvious thing would be to just take the rating with
            #     # the highest score:
            #     # rating = rec.argmax()
            #     # ... but this just leads to a huge multi-way tie for 5-star
            #     # predictions.
            #     # The paper suggests performing normalization over K values
            #     # to get probabilities
            #     # and take the expectation as your prediction, so we'll
            #     # do that instead:
            #     normalized = self.softmax(rec)
            #     rating = np.average(np.arange(10), weights=normalized)
            #     self.predictedRatings[uiid, itemID] = (rating + 1) * 0.5

            for itemID, rec in enumerate(recs):
                rating = rec.argmax()
                # normalized = self.softmax(rec)
                # rating = np.average(np.arange(10), weights=normalized)
                self.predictedRatings[uiid, itemID] = (rating + 1) * 0.5

            # if(uiid == 670):
            #     print("Reprocessing user ", 84)
            #     recs = rbm.GetRecommendations([trainingMatrix[84]])
            #     recs = np.reshape(recs, [numItems, 10])
            #     print(trainingMatrix[84])
            #     np.savetxt("trainingMatrixById.csv", trainingMatrix[84], delimiter=",")
            #     sums = []
            #     itemS = []
            #     itemLoop = 0
            #     for oneRec in recs:
            #         sum = oneRec[0] + oneRec[1] + oneRec[2] + oneRec[3] + oneRec[4] + oneRec[5] + oneRec[6] + oneRec[7] + oneRec[8] + oneRec[9];
            #         realRating = sum/2;
            #         sums.append(realRating)
            #         itemS.append(int(trainset.to_raw_iid(itemLoop)))
            #         print(int(trainset.to_raw_iid(itemLoop)))
            #         itemLoop += 1
            #     # print(itemS)
            #     np.savetxt("testingcsv.csv", sums, delimiter=",")

        return self


    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        rating = self.predictedRatings[u, i]

        if (rating < 0.001):
            raise PredictionImpossible('No valid prediction exists.')

        return rating
