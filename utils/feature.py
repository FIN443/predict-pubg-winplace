import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.feature_selection import RFE
from utils.load_data import reduce_mem_usage


def feature_engineering(train):
    train["playersJoined"] = train.groupby("matchId")["matchId"].transform("count")
    train["killsNorm"] = train["kills"] * ((100 - train["playersJoined"]) / 100 + 1)
    train["damageDealtNorm"] = train["damageDealt"] * ((100 - train["playersJoined"]) / 100 + 1)
    train["healsAndBoosts"] = train["heals"] + train["boosts"]
    train["totalDistance"] = train["walkDistance"] + train["rideDistance"] + train["swimDistance"]
    train["boostsPerWalkDistance"] = train["boosts"] / (train["walkDistance"] + 1)  # 무한대를 피하기위해 +1을 합니다, 왜냐하면 walkDistance=0 인데 boosts>0 인 값이 존재합니다.
    train["boostsPerWalkDistance"].fillna(0, inplace=True)
    train["healsPerWalkDistance"] = train["heals"] / (train["walkDistance"] + 1)  # 무한대를 피하기위해 +1을 합니다, 왜냐하면 walkDistance=0 인데 heals>0 인 값이 존재합니다.
    train["healsPerWalkDistance"].fillna(0, inplace=True)
    train["healsAndBoostsPerWalkDistance"] = train["healsAndBoosts"] / (train["walkDistance"] + 1)  # 무한대를 피하기위해 +1을 합니다.
    train["healsAndBoostsPerWalkDistance"].fillna(0, inplace=True)
    train[["walkDistance", "boosts", "boostsPerWalkDistance", "heals", "healsPerWalkDistance", "healsAndBoosts", "healsAndBoostsPerWalkDistance"]][
        40:45
    ]
    train["killsPerWalkDistance"] = train["kills"] / (train["walkDistance"] + 1)  # 무한대를 피하기위해 +1을 합니다, 왜냐하면 walkDistance=0 인데 kills>0 인 값이 존재합니다.
    train["killsPerWalkDistance"].fillna(0, inplace=True)
    # mean rank
    cols_to_drop = ["Id", "groupId", "matchId", "matchType", "winPlacePerc"]
    features = [col for col in train.columns if col not in cols_to_drop]
    agg = train.groupby(["matchId", "groupId"])[features].mean()
    agg = agg.groupby("matchId")[features].rank()
    train = train.merge(agg, suffixes=["", "_mean_rank"], how="left", on=["matchId", "groupId"])
    train.loc[:, "team"] = 0  # solo, duo, squad가 아닌 나머지는 0(이벤트 모드)
    train.loc[train["matchType"].isin(["solo-fpp", "solo", "normal-solo-fpp", "normal-solo"]), "team"] = 1
    train.loc[train["matchType"].isin(["duo-fpp", "duo", "normal-duo-fpp", "normal-duo"]), "team"] = 2
    train.loc[train["matchType"].isin(["squad-fpp", "squad", "normal-squad-fpp", "normal-squad"]), "team"] = 3
    return reduce_mem_usage(train)


def remove_collinearity(X, threshold):
    """
    X : feature matrix
    threshold : 다중공선성을 제거할 column을 고르는 기준 값. [0, 1]
    """

    corr = X.corr()
    candidate_cols = []

    for x in corr.iterrows():
        idx, row = x[0], x[1]  # decoupling tuple
        # 해당 row는 이미 처리가 되어서 볼 필요가 없다.
        if idx in candidate_cols:
            continue
        # print(row[row > 0.7].index[1:])
        candidates = row[row > threshold].index[1:]

        # 자기 자신을 제외하고 threshold를 넘는 column이 있다면,
        if len(candidates) != 0:
            for col in candidates:
                candidate_cols.append(col)

    return candidate_cols


def find_feature_importance(X, model):
    importances = model.feature_importances_
    important_features = X.columns[importances >= 0.01]
    return important_features


def apply_PCA(X):
    # training data와 test data를 모두 PCA를 이용하여 차원 감소를 수행합니다.
    pca = PCA(n_components=0.90)  # 원래 데이터의 90%를 보존하는 차원.
    pca_090 = pca.fit(X)  # 학습 및 변환
    reduced_X = pca_090.transform(X)
    labels = [f"PC{x}" for x in range(1, reduced_X.shape[1] + 1)]
    X_train_pca_df = pd.DataFrame(reduced_X, columns=labels)

    return pca_090, X_train_pca_df


def reduce_features(trainX, y):
    # flag setting
    feature_reducing = "feature_importance"  # "correlation" / "feature_importance" / "PCA"
    if feature_reducing == "correlation":
        threshold = 0.7
        correlated_features = remove_collinearity(trainX, threshold)
        correlated_features = set(correlated_features)  # 중복 제거
        print("%d Correlation features over %.2f" % (len(correlated_features), threshold))

        X = trainX.drop(columns=correlated_features)
        print(X.shape)

    elif feature_reducing == "feature_importance":
        model = RandomForestRegressor(max_features="sqrt", n_jobs=-1, random_state=0xC0FFEE)
        model.fit(trainX, y)
        important_features = find_feature_importance(trainX, model)
        X = trainX[important_features]
        print(X.shape)

    elif feature_reducing == "PCA":
        pca_model, X = apply_PCA(trainX)
        print(X.shape)
    return X


def reduce_features_lgbm(trainX, y, n, load=False):
    if load and os.path.isfile("checkpoints/features.csv"):
        features = pd.read_csv("checkpoints/features.csv")
        important_features = features[features["support"]].feature.values
        return trainX[important_features]

    model = LGBMRegressor()
    rfe = RFE(model, n_features_to_select=n, verbose=1)
    rfe.fit(trainX, y)

    important_features = []
    for col, isTrue in zip(rfe.feature_names_in_, rfe.support_):
        if isTrue:
            important_features.append(col)

    X = trainX[important_features]
    features = pd.DataFrame(data=list(zip(rfe.feature_names_in_, rfe.support_, rfe.ranking_)), columns=["feature", "support", "ranking"])
    features.to_csv("checkpoints/features.csv", index=False)
    print(f"Extract features")
    return X
