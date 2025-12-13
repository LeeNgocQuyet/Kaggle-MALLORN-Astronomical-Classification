import pandas as pd
df = pd.read_csv("submissions/svm_submission.csv")
print(len(df), df["object_id"].nunique())