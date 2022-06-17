from sklearn.preprocessing import StandardScaler
from models.lightgbm import train_lightgbm_model, valid_lightgbm_model
from utils.feature import feature_engineering, reduce_features_lgbm
from utils.load_data import load_data
from utils.preprocess import preprocess_data, scaling_data

def main():
    train = load_data("train_V2.csv")
    print("Success load data")
    train = feature_engineering(train)
    print("Success feature engineering")
    X, y = preprocess_data(train)
    X = scaling_data(StandardScaler(), X)
    X = reduce_features_lgbm(X, y, n=20, load=True)
    print("Success preprocess")
    valid_lightgbm_model(X, y)
    print("Done")

if __name__ == '__main__':
    main()
    