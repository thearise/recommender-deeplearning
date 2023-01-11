# -*- coding: utf-8 -*-

from EvaluationData import EvaluationData
from EvaluatedAlgorithm import EvaluatedAlgorithm
import numpy as np

class Evaluator:

    algorithms = []

    def __init__(self, dataset, rankings):
        ed = EvaluationData(dataset, rankings)
        self.dataset = ed

    def AddAlgorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)

    def Evaluate(self, doTopN):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.GetName(), "...")
            results[algorithm.GetName()] = algorithm.Evaluate(self.dataset, doTopN)

        # Print results
        print("\n")

        if (doTopN):
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Diversity", "Novelty"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                                      metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))

        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
        if (doTopN):
            print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
            print("cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.")
            print("ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better." )
            print("Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.")
            print("Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations")
            print("           for a given user. Higher means more diverse.")
            print("Novelty:   Average popularity rank of recommended items. Higher means more novel.")

    def EFullTrainSet(self):
        return self.dataset.GetFullTrainSet()

    def SampleTopNRecsById(self, ml, testSubject=85, k=10):

        for algo in self.algorithms:
            print("\nUsing recommender ", algo.GetName())

            print("\nBuilding recommendation model...")
            trainSet = self.dataset.GetFullTrainSet()
            # trainSet.n_users
            # np.savetxt("trainSet.csv", trainSet.all_ratings(), delimiter=",")
            # trainSet = self.dataset.GetTrainSet()
            algo.GetAlgorithm().fitById(trainSet, testUser=testSubject)

            print("Computing recommendations...")
            # change here
            testSet = self.dataset.GetAntiTestSetForUser(testSubject)
            print('what ani ', testSet)
            # np.savetxt("antiTest.csv", testSet)
            # testSet = self.dataset.GetTestSet()
            # Missing Rating testSet for selected user
            predictions = algo.GetAlgorithm().test(testSet)

            recommendations = []

            print ("\nWe recommend:")
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                # print(userID, ": ", movieID, ": ", actualRating, ": ", estimatedRating)
                recommendations.append((intMovieID, estimatedRating))

            print("what is recomm ", recommendations, len(recommendations))
            recommendations.sort(key=lambda x: x[1], reverse=True)

            jsobj = []
            for ratings in recommendations[:10]:
                jsobj.append({"id": ratings[0], "name": ml.getMovieName(ratings[0]), "rate": str(ratings[1]), "genre": ml.getMovieGenre(ratings[0])})
                print(ratings[0], ": ", ml.getMovieName(ratings[0]), ratings[1], ml.getMovieGenre(ratings[0]))

            return jsobj

    def SampleTopNRecsByIdRec(self, ml, testSubject=85, k=10, chkWeight=[], chkhidBias=[], chkvisBias=[], chkiniXVal=[]):
        print('wtf is that')
        for algo in self.algorithms:
            print("\nUsing recommender ", algo.GetName())

            print("\nBuilding recommendation model...")
            trainSet = self.dataset.GetFullTrainSet()
            # trainSet.n_users
            # np.savetxt("trainSet.csv", trainSet.all_ratings(), delimiter=",")
            # trainSet = self.dataset.GetTrainSet()
            algo.GetAlgorithm().fitByIdRec(trainSet, testUser=testSubject)

            print("Computing recommendations...")
            # change here
            testSet = self.dataset.GetAntiTestSetForUser(testSubject)
            # np.savetxt("antiTest.csv", testSet)
            # testSet = self.dataset.GetTestSet()
            # Missing Rating testSet for selected user
            predictions = algo.GetAlgorithm().test(testSet)

            recommendations = []

            print ("\nWe recommend:")
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                # print(userID, ": ", movieID, ": ", actualRating, ": ", estimatedRating)
                recommendations.append((intMovieID, estimatedRating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            jsobj = []
            for ratings in recommendations[:10]:
                jsobj.append({"id": ratings[0], "name": ml.getMovieName(ratings[0]), "rate": str(ratings[1]), "genre": ml.getMovieGenre(ratings[0])})
                print(ratings[0], ": ", ml.getMovieName(ratings[0]), ratings[1], ml.getMovieGenre(ratings[0]))

            return jsobj

    def SampleTopNRecs(self, ml, testSubject=85, k=10):

        for algo in self.algorithms:
            print("\nUsing recommender ", algo.GetName())

            print("\nBuilding recommendation model...")
            trainSet = self.dataset.GetFullTrainSet()
            # np.savetxt("trainSet.csv", trainSet.all_ratings(), delimiter=",")
            # trainSet = self.dataset.GetTrainSet()
            algo.GetAlgorithm().fit(trainSet, testUser=testSubject)

            print("Computing recommendations...")
            # change here
            testSet = self.dataset.GetAntiTestSetForUser(testSubject)
            # np.savetxt("antiTest.csv", testSet)
            # testSet = self.dataset.GetTestSet()
            # Missing Rating testSet for selected user
            predictions = algo.GetAlgorithm().test(testSet)

            recommendations = []

            print ("\nWe recommend:")
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                # print(userID, ": ", movieID, ": ", actualRating, ": ", estimatedRating)
                recommendations.append((intMovieID, estimatedRating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            for ratings in recommendations[:10]:
                print(ratings[0], ": ", ml.getMovieName(ratings[0]), ratings[1], ml.getMovieGenre(ratings[0]))

            return recommendations
