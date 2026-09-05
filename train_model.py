import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_shot_model():
    print("Loading shot dataset...")
    df = pd.read_csv("shot_data.csv")
    
    # Drop any missing rows
    df = df.dropna(subset=['SHOT_MADE_FLAG']).copy()
    
    # Feature Engineering: Convert total time remaining into seconds
    df['TIME_REMAINING_SEC'] = (df['MINUTES_REMAINING'] * 60) + df['SECONDS_REMAINING']
    
    # Define our input features (X) and target outcome (y)
    feature_cols = ['SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 'PERIOD', 'TIME_REMAINING_SEC']
    X = df[feature_cols]
    y = df['SHOT_MADE_FLAG']
    
    # Split into 80% Training data and 20% Testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a standard Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Test accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Training Complete! Accuracy: {accuracy * 100:.2f}%")
    
    # Calculate Expected FG% (Probability of shot making) for all shots
    df['EXPECTED_FG'] = model.predict_proba(X[feature_cols])[:, 1]
    
    # Save the model and updated dataframe
    with open("shot_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    df.to_csv("shot_data_with_xfg.csv", index=False)
    print("Saved trained model to 'shot_model.pkl' and updated stats to 'shot_data_with_xfg.csv'")

if __name__ == "__main__":
    train_shot_model()