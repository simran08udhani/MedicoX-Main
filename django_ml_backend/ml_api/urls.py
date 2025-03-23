from django.urls import path
from .views import CTScanPredictionView, MRIPredictionView, XRayPredictionView, APIStatusView

urlpatterns = [
    path('status/', APIStatusView.as_view(), name='api-status'),
    path('predict/ct-scan/', CTScanPredictionView.as_view(), name='predict-ct-scan'),
    path('predict/mri/', MRIPredictionView.as_view(), name='predict-mri'),
    path('predict/xray/', XRayPredictionView.as_view(), name='predict-xray'),

]