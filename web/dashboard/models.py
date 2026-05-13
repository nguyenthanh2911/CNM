from __future__ import annotations

from django.db import models


class Prediction(models.Model):
    id = models.BigAutoField(primary_key=True)
    patient_id = models.CharField(max_length=64, db_index=True)
    timestamp = models.DateTimeField(db_index=True)

    risk_score = models.FloatField()
    risk_level = models.CharField(max_length=16)
    alert_triggered = models.BooleanField()

    sofa_score = models.IntegerField()
    news2_score = models.IntegerField()
    inference_time_ms = models.FloatField()

    # THÊM MỚI
    heart_rate = models.FloatField(null=True, blank=True)
    systolic_bp = models.FloatField(null=True, blank=True)
    diastolic_bp = models.FloatField(null=True, blank=True)
    temperature = models.FloatField(null=True, blank=True)
    spo2 = models.FloatField(null=True, blank=True)
    respiratory_rate = models.FloatField(null=True, blank=True)
    lactate = models.FloatField(null=True, blank=True)
    wbc = models.FloatField(null=True, blank=True)
    creatinine = models.FloatField(null=True, blank=True)
    bilirubin = models.FloatField(null=True, blank=True)
    platelet = models.FloatField(null=True, blank=True)

    # Early warning columns
    early_warning_probability = models.FloatField(null=True, blank=True)
    early_warning_level = models.CharField(max_length=16, null=True, blank=True)
    trend_score = models.FloatField(null=True, blank=True)
    rate_of_change_score = models.FloatField(null=True, blank=True)
    threshold_score = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = "predictions"


class Alert(models.Model):
    id = models.BigAutoField(primary_key=True)
    alert_id = models.CharField(max_length=36, unique=True, db_index=True)
    patient_id = models.CharField(max_length=64, db_index=True)

    risk_score = models.FloatField()
    risk_level = models.CharField(max_length=16)

    top_features = models.JSONField()
    sofa_score = models.IntegerField()
    news2_score = models.IntegerField()
    alert_type = models.CharField(max_length=32, default="sepsis")

    created_at = models.DateTimeField(db_index=True)
    acknowledged = models.BooleanField(default=False, db_index=True)
    ack_by = models.CharField(max_length=128, null=True, blank=True)
    ack_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "alerts"
