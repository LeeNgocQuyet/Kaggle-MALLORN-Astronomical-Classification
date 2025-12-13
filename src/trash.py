import pandas as pd
df = pd.read_csv("submissions/svm_submission.csv")

print(df.shape)
print(df.object_id.nunique())
print(df.object_id.duplicated().sum())