from __future__ import annotations

from django.urls import path

from . import views

urlpatterns = [
    path("", views.patient_list, name="patient_list"),
    path("patients/<str:patient_id>/", views.patient_detail, name="patient_detail"),
    path("alerts/", views.alerts_page, name="alerts_page"),
    path("alerts/<str:alert_id>/acknowledge/", views.acknowledge_alert, name="acknowledge_alert"),
    path("api/patient/<str:patient_id>/latest/", views.patient_latest_api, name="patient_latest_api"),
]
