from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.
def json_request(request):

    return JsonResponse({'status': 'Failed'})


