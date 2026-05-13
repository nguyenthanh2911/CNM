from __future__ import annotations

from django.urls import path

from . import views

urlpatterns = [
    path("", views.patient_list, name="patient_list"),
    path("patients/<str:patient_id>/", views.patient_detail, name="patient_detail"),
    path("alerts/", views.alerts_page, name="alerts_page"),
    path("alerts/<str:alert_id>/acknowledge/", views.acknowledge_alert, name="acknowledge_alert"),
    # JSON APIs cho AJAX polling realtime
    path("api/dashboard/", views.api_dashboard_data, name="api_dashboard_data"),
    path("api/patients/<str:patient_id>/", views.api_patient_detail, name="api_patient_detail"),
]
