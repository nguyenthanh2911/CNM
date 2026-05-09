from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import shap

from .models.xgboost_model import SepsisXGBModel


class SepsisExplainer:
    def __init__(self, model: SepsisXGBModel):
        if model.model is None:
            raise ValueError("SepsisXGBModel is not trained/loaded.")
        self.model = model
        self._explainer = shap.TreeExplainer(model.model)

    def explain(self, X_instance, feature_names) -> List[Dict[str, Any]]:
        if isinstance(X_instance, pd.Series):
            x_df = X_instance.to_frame().T
        elif isinstance(X_instance, pd.DataFrame):
            x_df = X_instance
        else:
            arr = np.asarray(X_instance).reshape(1, -1)
            x_df = pd.DataFrame(arr, columns=list(feature_names))

        shap_vals = self._explainer.shap_values(x_df)

        # shap may return list for multiclass; take positive class if available
        if isinstance(shap_vals, list):
            values = np.asarray(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0])[0]
        else:
            values = np.asarray(shap_vals)[0]

        feats = list(x_df.columns)
        pairs = list(zip(feats, values))
        pairs.sort(key=lambda t: abs(float(t[1])), reverse=True)

        top5 = pairs[:5]
        return [{"feature": f, "shap_value": float(v)} for f, v in top5]
