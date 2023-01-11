# -*- coding: utf-8 -*-

from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline

class EvaluationData:

    def __init__(self, data, popularityRankings):

        self.rankings = popularityRankings

        #Build a full training set for evaluating overall properties
        self.fullTrainSet = data.build_full_trainset()
        # print('FFFFF1: ', self.fullTrainSet.n_users, self.fullTrainSet.n_items)
        # print('FFFFS1: ', self.fullTrainSet.ur)
        self.fullAntiTestSet = self.fullTrainSet.build_anti_testset()

        #Build a 75/25 train/test split for measuring accuracy
        self.trainSet, self.testSet = train_test_split(data, test_size=.25, random_state=1)

        #Build a "leave one out" train/test split for evaluating top-N recommenders
        #And build an anti-test-set for building predictions
        LOOCV = LeaveOneOut(n_splits=1, random_state=1)
        for train, test in LOOCV.split(data):
            self.LOOCVTrain = train
            self.LOOCVTest = test

        self.LOOCVAntiTestSet = self.LOOCVTrain.build_anti_testset()

        #Compute similarty matrix between items so we can measure diversity
        sim_options = {'name': 'cosine', 'user_based': False}
        self.simsAlgo = KNNBaseline(sim_options=sim_options)
        # self.simsAlgo.fit(self.fullTrainSet)
        # print('FFFFF2: ', self.fullTrainSet.n_users)

    def GetFullTrainSet(self):
        return self.fullTrainSet

    def GetFullAntiTestSet(self):
        return self.fullAntiTestSet

    # def GetAntiTestSetForUser(self, testSubject):
    #     trainset = self.fullTrainSet
    #     fill = trainset.global_mean
    #     print("Mean: ", fill)
    #     anti_testset = []
    #     u = trainset.to_inner_uid(str(testSubject))
    #     print("Inner U: ", u)
    #     print("Train set", trainset.ur[u])
    #     user_items = set([j for (j, _) in trainset.ur[u]])
    #     print("User items: ", user_items)
    #     for val in user_items:
    #         print(trainset.to_raw_iid(val))
    #     anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
    #                              i in trainset.all_items() if
    #                              i not in user_items]
    #     print("Anti test: ", anti_testset)
    #     return anti_testset

    def GetAntiTestSetForUser(self, testSubject):
        trainset = self.fullTrainSet
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(testSubject))
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset

    def GetTrainSet(self):
        return self.trainSet

    def GetTestSet(self):
        return self.testSet

    def GetLOOCVTrainSet(self):
        return self.LOOCVTrain

    def GetLOOCVTestSet(self):
        return self.LOOCVTest

    def GetLOOCVAntiTestSet(self):
        return self.LOOCVAntiTestSet

    def GetSimilarities(self):
        return self.simsAlgo

    def GetPopularityRankings(self):
        return self.rankings
