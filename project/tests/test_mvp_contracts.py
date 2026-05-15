from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.build_dashboard import classify_risk  # noqa: E402
from config.business_rules import DEFAULT_THRESHOLD_DAYS  # noqa: E402
from config.feature_catalog import (  # noqa: E402
    LEAKY_COLS,
    SAFE_CATEGORICAL_FEATURES,
    SAFE_NUMERIC_FEATURES,
)
from src.features.temporal import add_temporal_features  # noqa: E402


class MvpContractTests(unittest.TestCase):
    def test_default_rule_matches_business_definition(self) -> None:
        df = pd.DataFrame(
            {
                "dt_emissao": pd.to_datetime(["2024-05-01"] * 3),
                "dt_vencimento": pd.to_datetime(["2024-05-10"] * 3),
                "dt_pagamento": pd.to_datetime(["2024-05-09", "2024-05-11", None]),
            }
        )

        result = add_temporal_features(df)

        self.assertEqual(DEFAULT_THRESHOLD_DAYS, 0)
        self.assertEqual(result["is_defaulted"].tolist(), [0, 1, 1])

    def test_leaky_columns_are_not_safe_features(self) -> None:
        safe = set(SAFE_NUMERIC_FEATURES) | set(SAFE_CATEGORICAL_FEATURES)

        for column in LEAKY_COLS:
            self.assertNotIn(column, safe)

    def test_score_thresholds_are_stable(self) -> None:
        self.assertEqual(classify_risk(29.9), "low")
        self.assertEqual(classify_risk(30), "medium")
        self.assertEqual(classify_risk(59.9), "medium")
        self.assertEqual(classify_risk(60), "high")

    def test_dashboard_embedded_json_is_valid(self) -> None:
        dashboard = PROJECT_DIR / "app" / "index.html"
        self.assertTrue(dashboard.exists(), "Run python app/build_dashboard.py first.")

        html = dashboard.read_text(encoding="utf-8")
        match = re.search(
            r'<script id="dashboard-data" type="application/json">(.*?)</script>',
            html,
            re.DOTALL,
        )

        self.assertIsNotNone(match)
        payload = json.loads(match.group(1))
        self.assertGreater(payload["summary"]["total_boletos"], 0)
        self.assertIn("transactions", payload)
        self.assertIn("beneficiaries", payload)
        self.assertIn("alerts", payload)


if __name__ == "__main__":
    unittest.main()
