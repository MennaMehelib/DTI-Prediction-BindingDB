from data_loader import load_data
from feature_engineering import build_features
from gan_balancer import balance_data
from train import train_model
from save_model import save

def main():
    print(" Step 1: Loading data...")
    df = load_data()
    print(f" Data loaded: {df.shape}")

    print("\n Step 2: Feature engineering...")
    X, y = build_features(df)
    print(f" Features done: X={X.shape}, y={y.shape}")

    print("\n Step 3: GAN balancing...")
    X_bal, y_bal = balance_data(X, y)
    print(f" After balancing: X={X_bal.shape}, y={y_bal.shape}")

    print("\n Step 4: Training model...")
    model, scaler = train_model(X_bal, y_bal)
    print(" Model trained")

    print("\n Step 5: Saving model...")
    save(model, scaler)
    print(" Model saved")

    print("\n DONE!")

if __name__ == "__main__":
    main()