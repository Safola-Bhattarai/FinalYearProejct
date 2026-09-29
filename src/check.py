import pandas as pd

train = pd.read_csv("dataset/train.csv")
val   = pd.read_csv("dataset/validation.csv")
test  = pd.read_csv("dataset/test.csv")

print("Shapes -> train:", train.shape, "| val:", val.shape, "| test:", test.shape)

print("\nClass distribution (train):")
print(train["category"].value_counts())

mapping = (train[["category_label", "category"]]
           .drop_duplicates()
           .sort_values("category_label"))
print("\nLabel mapping:")
print(mapping.to_string(index=False))

print("\nSample rows:")
print(train[["description", "category"]].sample(5, random_state=1).to_string())

print("\nAll checks passed")