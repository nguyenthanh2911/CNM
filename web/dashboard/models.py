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

    created_at = models.DateTimeField(db_index=True)
    acknowledged = models.BooleanField(default=False, db_index=True)
    ack_by = models.CharField(max_length=128, null=True, blank=True)
    ack_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "alerts"
