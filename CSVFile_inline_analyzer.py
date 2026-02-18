import argparse
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model_selector import AutoModelSelector
""" 
USE EXAMPLE:
python CSVFile_inline_analyzer.py --data datasets/titanic/train.csv --target Survived --trials 100 --output titanic_model.pkl
"""
def main():
    parser = argparse.ArgumentParser(description="Ultimate AutoML Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to your CSV file")
    parser.add_argument("--target", type=str, required=True, help="Name of the column you want to predict")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials (default: 30)")
    parser.add_argument("--output", type=str, default="final_model.pkl", help="Name of the saved model file")
    
    args = parser.parse_args()

    print(f"\n Loading data from: {args.data}")
    if not os.path.exists(args.data):
        print(f" Error: Could not find file '{args.data}'")
        return

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        print(f" Error: Target column '{args.target}' is not in the dataset.")
        print(f"Available columns: {list(df.columns)}")
        return

    # Split into X (Features) and y (Answers)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    print(f"Data loaded successfully! Shape: {df.shape}")
    print(f"Target Column: '{args.target}'")

    #  (Auto-Task Detection)
    print("\nAnalyzing target column to determine ML task...")
    
    unique_values = y.nunique()
    is_text = y.dtype == 'object' or y.dtype.name == 'category'

    # Logic: If it's text, or if it's numbers but only has a few unique options (like 0 and 1)
    if is_text or unique_values < 20:
        task = 'classification'
        print(f" Detected Task: CLASSIFICATION ({unique_values} unique classes found)")
    else:
        task = 'regression'
        print(f" Detected Task: REGRESSION (Continuous numerical data found)")

    print("\n" + "="*50)
    print(f" FIRING UP AUTOML ENGINE ({args.trials} Trials)")
    print("="*50)
    
    # We leave scoring='auto' so the Engine picks the best metric itself!
    automl = AutoModelSelector(n_trials=args.trials, task=task, scoring='auto')
    
    try:
        automl.fit(X, y)
    except Exception as e:
        print(f"\nFATAL ERROR during training: {e}")
        return

    print("\nExporting final pipeline...")
    joblib.dump(automl, args.output)
    print(f"SUCCESS! Model saved as '{args.output}'")
    print("You can now load this file in any Python script using joblib.load() to make predictions.")

if __name__ == "__main__":
    main()