import os

from django.http import JsonResponse
from django.shortcuts import render
from .model import LR_predict

# Create your views here.
def json_request(request):
    # Root Directory is different in Django server
    # os.getcwd() => ~/Workspaces/CSCE665_project/tensorflow-server/src/server/mysite
    # test_X = LR_predict.csv_to_numpy_array(os.getcwd() + "/data/testX.csv", delimiter="\t")
    test_X = LR_predict.testX
    # test_Y = LR_predict.csv_to_numpy_array(os.getcwd() + "/data/testY.csv", delimiter="\t")
    test_Y = LR_predict.testY


    prediction = LR_predict.predict_all(test_X, test_Y)
    return JsonResponse({'status': 'Failed'})


