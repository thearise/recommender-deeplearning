# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask
from flask import jsonify
from flask import json
from flask_cors import CORS
from flask import request
# Flask constructor takes the name of
# current module (__name__) as argument.
import os
import pickle
import tensorflow.compat.v1 as tf
import numpy as np
from EvaluationData import EvaluationData
from EvaluatedAlgorithm import EvaluatedAlgorithm
from RBMAlgorithm import RBMAlgorithm
from surprise import NormalPredictor

os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = Flask(__name__)
CORS(app)
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return 'Hello World'

@app.route("/summary", methods=['GET', 'POST'])
def summary():
    userId = request.args.get("userId")
    print('Preparing for user ', userId)
    from MovieLens2 import MovieLens2
    from RBMAlgorithm import RBMAlgorithm
    from surprise import NormalPredictor
    from Evaluator import Evaluator

    import random
    import numpy as np

    def LoadMovieLensData():
        ml = MovieLens2()
        print("Loading movie ratings...")
        data = ml.loadMovieLensLatestSmall()
        print("\nComputing movie popularity ranks so we can measure novelty later...")
        rankings = ml.getPopularityRanks()
        return (ml, data, rankings)

    np.random.seed(0)
    random.seed(0)

    # Load up common data set for the recommender algorithms
    (ml, evaluationData, rankings) = LoadMovieLensData()

    # Construct an Evaluator to, you know, evaluate them
    evaluator = Evaluator(evaluationData, rankings)

    #RBM
    # RBM = RBMAlgorithm(epochs=20)
    RBM = RBMAlgorithm(epochs=5, hiddenDim=200)
    evaluator.AddAlgorithm(RBM, "RBM")

    # Just make random recommendations
    # Random = NormalPredictor()
    # evaluator.AddAlgorithm(Random, "Random")

    # Fight!
    # evaluator.Evaluate(False)
    # 81
    print('checking id ', userId)
    recommendation = evaluator.SampleTopNRecsById(ml, testSubject=int(userId))
    print(recommendation)
    # data = 'Please'
    response = app.response_class(
        response=json.dumps(recommendation),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/initx")
def intxvalue():
    import numpy as np
    chkWeight = []
    chkvisBias = []
    chkhidBias = []
    chkiniXVal = []
    with open('checkpoint.pkl', 'rb') as inp:
        checkpoint1 = pickle.load(inp)
        print('checkpointing v1 ', checkpoint1.name, ' ', len(checkpoint1.value), len(checkpoint1.value[0]))
        chkWeight = checkpoint1.value
        checkpoint2 = pickle.load(inp)
        print('checkpointing v2 ', checkpoint2.name, ' ', len(checkpoint2.value))
        chkvisBias = checkpoint2.value
        checkpoint3 = pickle.load(inp)
        print('checkpointing v3 ', checkpoint3.name, ' ', len(checkpoint3.value))
        chkhidBias = checkpoint3.value
        checkpoint4 = pickle.load(inp)
        print('checkpointing v4 ', checkpoint4.name, ' ', len(checkpoint4.value), len(checkpoint4.value[0]))
        chkiniXVal = checkpoint4.value
    print(chkiniXVal)
    return 'Hay'

@app.route('/traineduser')
# ‘/’ URL is bound with hello_world() function.
def initxvalue2():
    return 'Hay'

@app.route("/recommendbyuser", methods=['GET', 'POST'])
def recommendbyuser():
    from MovieLens2 import MovieLens2
    from AutoRecAlgorithm import AutoRecAlgorithm
    from surprise import NormalPredictor
    from Evaluator import Evaluator

    import random
    import numpy as np

    def LoadMovieLensData():
        ml = MovieLens2()
        print("Loading movie ratings...")
        data = ml.loadMovieLensLatestSmall()
        print("\nComputing movie popularity ranks so we can measure novelty later...")
        rankings = ml.getPopularityRanks()
        return (ml, data, rankings)

    np.random.seed(0)
    random.seed(0)

    # Load up common data set for the recommender algorithms
    (ml, evaluationData, rankings) = LoadMovieLensData()

    # Construct an Evaluator to, you know, evaluate them
    evaluator = Evaluator(evaluationData, rankings)
    # ed = EvaluationData(evaluationData, rankings)
    # trainset = ed.GetFullTrainSet()
    trainset = evaluator.EFullTrainSet()

    numUsers = trainset.n_users
    numItems = trainset.n_items
    print('num users2 ', numUsers, ' and ', numItems)

    userId = request.args.get("userId")
    print("retriving checks: ")
    chkWeight = []
    chkvisBias = []
    chkhidBias = []
    chkiniXVal = []
    with open('checkpoint.pkl', 'rb') as inp:
        checkpoint1 = pickle.load(inp)
        print('checkpointing v1 ', checkpoint1.name, ' ', len(checkpoint1.value), len(checkpoint1.value[0]))
        chkWeight = checkpoint1.value
        checkpoint2 = pickle.load(inp)
        print('checkpointing v2 ', checkpoint2.name, ' ', len(checkpoint2.value))
        chkvisBias = checkpoint2.value
        checkpoint3 = pickle.load(inp)
        print('checkpointing v3 ', checkpoint3.name, ' ', len(checkpoint3.value))
        chkhidBias = checkpoint3.value
        checkpoint4 = pickle.load(inp)
        print('checkpointing v4 ', checkpoint4.name, ' ', len(checkpoint4.value), len(checkpoint4.value[0]))
        chkiniXVal = checkpoint4.value

    userId = request.args.get("userId")
    print('Preparing for user ', userId)

    trainingMatrix = np.zeros([numUsers, numItems, 10], dtype=np.float32)

    for (uid, iid, rating) in trainset.all_ratings():
        adjustedRating = int(float(rating)*2.0) - 1
        trainingMatrix[int(uid), int(iid), adjustedRating] = 1
        # print('integer ', iid)

    print('matrix ', trainingMatrix)
    # Flatten to a 2D array, with nodes for each possible rating type on each possible item, for every user.
    trainingMatrix = np.reshape(trainingMatrix, [trainingMatrix.shape[0], -1])
    print('training matrix 02 ', trainingMatrix.shape)
    innerTestId = trainset.to_inner_uid(str(userId))

    sess = tf.Session()
    # chkiniXVal = np.array(chkiniXVal)
    # chkWeight = np.array(chkWeight)
    shaping = [trainingMatrix[innerTestId]]
    print("Retrieving user021 ", userId, "Inner id: ", innerTestId, ' & ', len(shaping[0]))

    initX = tf.placeholder(tf.float32, [None, trainingMatrix.shape[1]], name="X")
    print("Retrieving user041 ", ' & ', len(shaping[0]), 'and ', type(shaping), type(initX))

    

    hidden = tf.nn.sigmoid(tf.matmul(initX, chkWeight) + chkhidBias)
    # hidden = tf.nn.sigmoid(tf.matmul(chkiniXVal, chkWeight) + chkhidBias)
    # visible = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(chkWeight)) + chkvisBias)
    print("Get Recommendation IPU2: ", [trainingMatrix[innerTestId]], ' and ')
    feed = sess.run(hidden, feed_dict={initX: [trainingMatrix[innerTestId]]} )
    visible = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(chkWeight)) + chkvisBias)
    recs = sess.run(visible, feed_dict={ hidden: feed} )
    # feed = sess.run(hidden, feed_dict={chkiniXVal: 1})
    # feed = sess.run(hidden, feed_dict={ chkiniXVal: int(userId)} )
    # rec = sess.run(visible, feed_dict={ hidden: feed} )
    print('rec testing01 ', recs[0])

    recs = np.reshape(recs, [numItems, 10])
    predic = np.zeros([numUsers, numItems], dtype=np.float32)

    predictedRatings = np.zeros([numUsers, numItems], dtype=np.float32)
        # np.savetxt("testingcsv2.csv", itemS, delimiter=",")
    for itemID, rec in enumerate(recs):
        # The obvious thing would be to just take the rating with the highest score:
        rating = rec.argmax()
        # ... but this just leads to a huge multi-way tie for 5-star predictions.
        # The paper suggests performing normalization over K values to get probabilities
        # and take the expectation as your prediction, so we'll do that instead:

        normalized = softmax(rec)
        # rating = np.average(np.arange(10), weights=normalized)
        predic[innerTestId, itemID] = (rating + 1) * 0.5
        predictedRatings[innerTestId, itemID] = (rating + 1) * 0.5

    print('rec testing02 ', predictedRatings, len(predictedRatings), len(predictedRatings[0]))
    recommendations = []
    looper = 0;
    for i in predictedRatings[innerTestId]:
        # print('looping ', i)
        # if(looper==)
        recommendations.append((looper, i, trainset.to_raw_iid(looper)))
        looper = looper+1

    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    print('whatting01??? ', recommendations)


    u = trainset.to_inner_uid(str(userId))
    print('anti testing01 ', trainset.ur[u])
    actRatingDict = {}
    for ratings in trainset.ur[u]:
        print('anti tester ' , ratings[0], trainset.to_raw_iid(ratings[0]))
        actRatingDict[ratings[0]] = trainset.to_raw_iid(ratings[0])

    print('act rating dict ', actRatingDict)

    jsobj = []
    looping = 0
    for ratings in recommendations:
        if ratings[0] not in actRatingDict:
            jsobj.append({"id": ratings[2], "name": ml.getMovieName(int(ratings[2])), "rate": str(ratings[1]), "genre": ml.getMovieGenre(int(ratings[2]))})
            looping = looping + 1;
            if looping >= 100:
                break
        # print(ratings[2], ": ", ml.getMovieName(ratings[2]), ratings[1], ml.getMovieGenre(ratings[2]))

    print('whatting02??? ', jsobj)
    # return jsobj
    # print('rec for user before sort ', predictedRatings[innerTestId])
    # predictedRatings.sort(reverse=True)
    # print('rec for user after sort ', predictedRatings[innerTestId])


    # RBM = RBMAlgorithm(epochs=500, hiddenDim=500, learningRate=0.005)
    # eAlgo = EvaluatedAlgorithm(RBM, 'RBM')

    

    # fill = trainset.global_mean
    # anti_testset = []
    # u = trainset.to_inner_uid(str(userId))
    # user_items = set([j for (j, _) in trainset.ur[u]])
    # anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
    #                             i in trainset.all_items() if
    #                             i in user_items]

    

    # testSet = anti_testset
    # print('what ani ', testSet)
    # predictions = RBM.test(testSet)

    # return ''
    print(jsobj)
    response = app.response_class(
        response=json.dumps(jsobj),
        status=200,
        mimetype='application/json'
    )
    return response

def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

# main driver function
if __name__ == '__main__':

    # ruhost='0.0.0.0', port=3000n() method of Flask class runs the application
    # on the local development server.
    app.run(host='localhost', port=1500)
    # return jsonify('You get it?')
