import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import hf_analysis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hf_analysis import (
    compute_kaplan_meier, 
    compute_death_rate, 
    compute_time_to_event_stats,
    compute_grouped_kaplan_meier,
    compute_variable_associations,
    train_eval_models,
    compute_cox_model
)

class TestHfAnalysis(unittest.TestCase):
    def setUp(self):
        # Create a simple synthetic dataframe for testing
        # Scenario:
        # 1. Patient A: Died at day 10
        # 2. Patient B: Censored at day 20 (Survived)
        # 3. Patient C: Died at day 30
        # 4. Patient D: Censored at day 40 (Survived)
        # 5. Patient E: Died at day 50
        self.df = pd.DataFrame({
            'time': [10, 20, 30, 40, 50],
            'DEATH_EVENT': [1, 0, 1, 0, 1],
            'age': [50, 60, 70, 80, 90],
            'sex': [0, 1, 0, 1, 0] # Grouping variable
        })

    def test_compute_death_rate_basic(self):
        """Test basic death rate calculation with censoring logic."""
        # Target day: 35
        # 1. time=10, event=1 (Died <= 35) -> Counted as event
        # 2. time=20, event=0 (Censored < 35) -> Excluded (unknown status at 35)
        # 3. time=30, event=1 (Died <= 35) -> Counted as event
        # 4. time=40, event=0 (Censored >= 35) -> Counted as survivor (alive at 35)
        # 5. time=50, event=1 (Died > 35) -> Counted as survivor (alive at 35)
        
        # Valid population: A, C, D, E (Total 4)
        # Events: A, C (Total 2)
        # Rate: 2/4 = 0.5
        
        rate = compute_death_rate(self.df, n_days=35)
        self.assertEqual(rate, 0.5)

    def test_compute_death_rate_empty(self):
        """Test death rate with empty dataframe."""
        empty_df = pd.DataFrame(columns=['time', 'DEATH_EVENT'])
        rate = compute_death_rate(empty_df, n_days=30)
        self.assertEqual(rate, 0.0)

    def test_compute_death_rate_all_survived(self):
        """Test death rate when everyone survives past target."""
        df_survived = self.df.copy()
        df_survived['DEATH_EVENT'] = 0
        df_survived['time'] = 100 # All censored at 100
        
        rate = compute_death_rate(df_survived, n_days=50)
        self.assertEqual(rate, 0.0)

    def test_compute_kaplan_meier_structure(self):
        """Test if KM computation returns correct structure and values."""
        km = compute_kaplan_meier(self.df, bootstrap_ci=False)
        
        self.assertIn('times', km)
        self.assertIn('survivals', km)
        self.assertIsNone(km['ci_lower']) # Since bootstrap_ci=False
        
        # Check manual calculation
        # t=0: S=1.0
        # t=10: d=1, n=5. S = 1.0 * (1 - 1/5) = 0.8
        # t=20: d=0, n=4. S = 0.8 * (1 - 0/4) = 0.8
        # t=30: d=1, n=3. S = 0.8 * (1 - 1/3) = 0.8 * 2/3 = 0.5333...
        # t=40: d=0, n=2. S = 0.5333 * (1 - 0/2) = 0.5333...
        # t=50: d=1, n=1. S = 0.5333 * (1 - 1/1) = 0.0
        
        expected_survivals = [1.0, 0.8, 0.8, 0.8 * (2/3), 0.8 * (2/3), 0.0]
        
        # Note: compute_kaplan_meier inserts 0 at the start
        # times: [0, 10, 20, 30, 40, 50]
        
        np.testing.assert_array_almost_equal(km['survivals'], expected_survivals)
        self.assertEqual(len(km['times']), 6)

    def test_compute_time_to_event_stats(self):
        """Test statistics for time-to-event."""
        stats = compute_time_to_event_stats(self.df)
        # Events at 10, 30, 50
        # Mean = (10+30+50)/3 = 30
        # Median = 30
        
        self.assertEqual(stats['mean'], 30.0)
        self.assertEqual(stats['median'], 30.0)
        self.assertEqual(stats['n'], 3)

    def test_compute_grouped_kaplan_meier_categorical(self):
        """Test grouped KM for categorical variable."""
        # Group by 'sex' (0 vs 1)
        # Group 0: indices 0, 2, 4 (times 10, 30, 50, all events)
        # Group 1: indices 1, 3 (times 20, 40, all censored)
        
        results = compute_grouped_kaplan_meier(
            self.df, 
            group_col='sex', 
            is_numeric=False, 
            bootstrap_ci=False
        )
        
        self.assertEqual(len(results['groups']), 2)
        
        # Check labels
        labels = [g['label'] for g in results['groups']]
        self.assertIn('0', labels)
        self.assertIn('1', labels)
        
        # Check metrics
        # Group 0: All died. 
        # Raw KM: t=10 (S=0.66), t=30 (S=0.33). Step drops below 0.5 at t=30.
        # Smoothed KM (used in code): Linear interp between (10, 0.66) and (30, 0.33).
        # Crosses 0.5 exactly halfway at t=20.
        
        metrics = results['metrics']['median_times']
        self.assertAlmostEqual(metrics['0'], 20.0, delta=1.0)
        self.assertIsNone(metrics['1'])

    def test_compute_variable_associations(self):
        """Test statistical association detection."""
        # Create a clear difference with enough samples for significance
        # Group 0 (Survived): Age ~ 20
        # Group 1 (Died): Age ~ 80
        n = 10
        df_assoc = pd.DataFrame({
            'DEATH_EVENT': [0]*n + [1]*n,
            'age': [20]*n + [80]*n,
            'dummy_cat': ['A']*n + ['B']*n
        })
        
        res = compute_variable_associations(
            df_assoc, 
            numeric_cols=['age'], 
            categorical_cols=['dummy_cat']
        )
        
        # Should find age significant
        found_numeric = any(r['col'] == 'age' for r in res['numeric'])
        self.assertTrue(found_numeric)
        
        # Should find dummy_cat significant (perfect separation)
        found_cat = any(r['col'] == 'dummy_cat' for r in res['categorical'])
        self.assertTrue(found_cat)

    def test_train_eval_models_smoke(self):
        """Smoke test for ML pipeline (ensure it runs)."""
        # Tiny dataset
        df_ml = pd.DataFrame({
            'age': np.random.randint(40, 90, 20),
            'creatinine': np.random.rand(20),
            'DEATH_EVENT': np.random.randint(0, 2, 20)
        })
        
        # Ensure at least one of each class
        df_ml.iloc[0, 2] = 0
        df_ml.iloc[1, 2] = 1
        
        try:
            res = train_eval_models(df_ml, random_state=42)
            self.assertIn('cv_results', res)
            self.assertIn('confusion_matrices', res)
        except Exception as e:
            self.fail(f"train_eval_models raised exception: {e}")

    def test_compute_grouped_kaplan_meier_numeric(self):
        """Test grouped KM for numeric variable (binning)."""
        # Create data where age perfectly separates survival
        # Young (20-30): Survive long
        # Old (80-90): Die early
        df_numeric = pd.DataFrame({
            'time': [100, 100, 10, 10],
            'DEATH_EVENT': [0, 0, 1, 1],
            'age': [25, 28, 85, 82]
        })
        
        results = compute_grouped_kaplan_meier(
            df_numeric, 
            group_col='age', 
            is_numeric=True, 
            n_bins=2,
            bootstrap_ci=False
        )
        
        self.assertEqual(len(results['groups']), 2)
        # Check that we have two distinct groups based on age ranges
        labels = [g['label'] for g in results['groups']]
        # Labels will be interval strings like "(24.94, 55.0]"
        # We just check that we got labels back
        self.assertEqual(len(labels), 2)

    def test_compute_cox_model_significance(self):
        """Test Cox model on data with clear signal."""
        # Predictor 'risk_factor': 0 -> Low Risk (Long survival), 1 -> High Risk (Short survival)
        n = 20
        df_cox = pd.DataFrame({
            'time': [100]*n + [10]*n,      # Group 0 survives to 100, Group 1 dies at 10
            'DEATH_EVENT': [0]*n + [1]*n,  # Group 0 censored, Group 1 event
            'risk_factor': [0]*n + [1]*n
        })
        
        cph = compute_cox_model(df_cox, duration_col='time', event_col='DEATH_EVENT')
        
        # Hazard ratio for risk_factor should be > 1 (positive coefficient)
        # Because risk_factor=1 increases hazard (shortens survival)
        summary = cph.summary
        coef = summary.loc['risk_factor', 'coef']
        
        self.assertGreater(coef, 0.0)
        # C-index should be perfect (1.0) or very high
        self.assertGreater(cph.concordance_index_, 0.9)

if __name__ == '__main__':
    unittest.main()
