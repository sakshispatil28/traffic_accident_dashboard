from django.urls import path
from .views import predict_view, dashboard_view

urlpatterns = [
    path('predict/', predict_view, name='predict'),
    path('dashboard/', dashboard_view, name='dashboard'),
]

