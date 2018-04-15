import os
import logging
from django.http import JsonResponse
from django.shortcuts import render
# from .model import LR_predict
from .model import LR_predict_Youtube
# from .model import LSTM_predict

logger = logging.getLogger(__name__)
# Create your views here.

# def json_request_lstm(request):
#     logger.info(request)
#     corpus = request.GET.get('emailContent','')
#     try:
#         prediction = LSTM_predict.predict_from_raw_input(corpus)
#     except Exception:
#         prediction = 'Failed'
#
#     logger.info(prediction)
#     return JsonResponse({'status': prediction})

def json_request_nn(request):
    logger.info(request)
    corpus = request.GET.get('emailContent','')
    return JsonResponse({'status': 'Failed'})

# def json_request_lr(request):
#     logger.info(request)
#     corpus = request.GET.get('emailContent','')
#     try:
#         prediction = LR_predict.predict_from_raw_input(corpus)
#     except Exception:
#         prediction = 'Failed'
#
#     logger.info(prediction)
#     return JsonResponse({'status': prediction})


# This is the final version of the request
# Root Directory is different in Django server
# os.getcwd() => ~/Workspaces/CSCE665_project/tensorflow-server/src/server/mysite
# Can only load 1 model for each configuration
def json_request(request):
    logger.info(request)
    # import pdb; pdb.set_trace()

    # This is the input text data requiring classification
    corpus = request.GET.get('emailContent','')
    try:
        prediction = LR_predict_Youtube.predict_from_raw_input(corpus)
    except Exception:
        prediction = 'Failed'

    logger.info(prediction)
    return JsonResponse({'status': prediction})

def json_request_youtube(request):
    logger.info(request)
    # import pdb; pdb.set_trace()

    # This is the input text data requiring classification
    corpus = request.GET.get('emailContent','')
    try:
        prediction = LR_predict_Youtube.predict_from_raw_input(corpus)
    except Exception:
        prediction = 'Failed'

    logger.info(prediction)
    return JsonResponse({'status': prediction})

def high_freq_spam_words(request):
    # <list>
    return JsonResponse(LR_predict_Youtube.uniFeatureDict)
    pass



# 3 DataTypes: [Uni, Combined, Ngram]
# 3 WorkModes: [LR, NN, LSTM]