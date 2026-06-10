from sklearn.ensemble import RandomForestClassifier
from config import MODEL_CONFIG


def get_model():
    return RandomForestClassifier(
        n_estimators=200,
        random_state=MODEL_CONFIG["random_state"],
        n_jobs=-1
    )