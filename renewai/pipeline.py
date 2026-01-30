#!/usr/bin/env python3
"""
RenewAI - Complete Enhancement Pipeline

Executes the full improvement sequence:
1. Generate enhanced data (temporal, lifecycle, business context features)
2. Train multiple models (GBM, RF, LR) with model comparison
3. Run batch predictions with churn analysis & interventions
4. Display results and portfolio health

Run: python pipeline.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    ("data/generate_data.py", "Generating enhanced dataset..."),
    ("model/train_model.py", "Training models with comparison..."),
    ("analytics/batch_predict.py", "Running batch predictions..."),
]

def run_script(script_path, description):
    """Execute a Python script and report results"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=Path(__file__).parent,
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n✗ FAILED: {script_path}")
        return False
    
    print(f"\n✓ SUCCESS: {script_path}")
    return True


def main():
    """Run full pipeline"""
    print("\n" + "="*70)
    print("  RENEWAI - COMPLETE ENHANCEMENT PIPELINE")
    print("="*70)
    print("\nPhases:")
    print("  1. Data Generation: Add temporal, lifecycle, business features")
    print("  2. Model Training: Multi-model comparison with threshold optimization")
    print("  3. Batch Predictions: Portfolio analytics + churn analysis")
    print("\n" + "="*70)
    
    failed = []
    for script_path, description in SCRIPTS:
        if not run_script(script_path, description):
            failed.append(script_path)
    
    # Summary
    print(f"\n\n{'='*70}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*70}")
    
    if failed:
        print(f"\n✗ {len(failed)} script(s) failed:")
        for script in failed:
            print(f"  - {script}")
        return 1
    else:
        print("\n✓ ALL SCRIPTS COMPLETED SUCCESSFULLY")
        print("\nNext Steps:")
        print("  1. Review data: data/saas_renewal_data.csv")
        print("  2. Review model: model/model.pkl (artifact with best model)")
        print("  3. Review predictions: outputs/renewal_portfolio_predictions.csv")
        print("  4. Start API server: python backend/main.py")
        print("  5. Open browser: http://localhost:8000")
        return 0


if __name__ == "__main__":
    sys.exit(main())
