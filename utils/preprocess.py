import pandas as pd


def preprocess_data(train, isTrain=True):
    if isTrain:
        abnormal = train[train["winPlacePerc"].isna()].index
        train.drop(abnormal, inplace=True)
        rows = train.loc[train["team"] == 0].index
        train.drop(rows, inplace=True)
        train.reset_index(inplace=False)
    train.loc[train["winPoints"] == 0, "winPoints"] = 1500
    train.drop(["Id", "groupId", "matchId", "matchType"], axis=1)

    if not isTrain:
        return train

    y = train["winPlacePerc"]
    train.drop(["winPlacePerc"], axis=1, inplace=True)
    return train, y


def scaling_data(scaler, X):
    trainX = X.drop(["Id", "groupId", "matchId", "matchType"], axis=1)
    trainX_col = trainX.columns
    trainX = scaler.fit_transform(trainX)
    trainX = pd.DataFrame(trainX, columns=trainX_col)
    return trainX
