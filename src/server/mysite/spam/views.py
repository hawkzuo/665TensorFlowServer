import os
import logging
from django.http import JsonResponse
from django.shortcuts import render
from .model import LR_predict

logger = logging.getLogger(__name__)
# Create your views here.

def json_request_lstm(request):
    logger.info(request)
    corpus = request.GET.get('emailContent','')
    return JsonResponse({'status': 'Failed'})

def json_request_nn(request):
    logger.info(request)
    corpus = request.GET.get('emailContent','')
    return JsonResponse({'status': 'Failed'})

def json_request_lr(request):
    logger.info(request)
    corpus = request.GET.get('emailContent','')
    return JsonResponse({'status': 'Failed'})


# This is the final version of the request
def json_request(request):
    # Root Directory is different in Django server
    # os.getcwd() => ~/Workspaces/CSCE665_project/tensorflow-server/src/server/mysite
    logger.info(request)
    # import pdb; pdb.set_trace()

    test_X = LR_predict.testX
    test_Y = LR_predict.testY

    # uniDict = LR_predict.uniFeatureDict
    # biDict = LR_predict.biGramFeatureDict


    # This is the input text data requiring classification
    corpus = request.GET.get('emailContent','')
    # sample_matrix = Parser.generate_matrix(featureDict, corpus)
    # predict = LR_predict.predict(sample_matrix)
    try:
        prediction = LR_predict.predict_from_raw_input(corpus)
    except Exception:
        prediction = 'Failed'

    # prediction = LR_predict.predict_from_raw_input(corpus)
    logger.info(prediction)
    return JsonResponse({'status': prediction})

def high_freq_spam_words(request):
    # <list>
    return JsonResponse(LR_predict.uniFeatureDict)
    pass



# 3 DataTypes: [Uni, Combined, Ngram]
# 3 WorkModes: [LR, NN, LSTM]