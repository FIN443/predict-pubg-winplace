import pandas as pd


def pred_to_csv(model, X):
    pred = model.predict(X)
    submission = pd.read_csv(f"../data/submission_V2.csv")
    submission["winPlacePerc"] = pred
    submission.to_csv(f"data/submission_V2.csv", index=False)
