
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv("secondary_data.csv", sep=";")

# Step 2: Inspect the data
print("Dataset shape:", df.shape)
print(df.head())
print(df.info())

# Step 3: Missing values
missing = df.isnull().sum()
missing_pct = df.isnull().mean() * 100

missing_summary = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
}).sort_values("Missing %", ascending=False)

print(missing_summary)

# Visualise missing percentages
missing_summary["Missing %"].plot(
    kind="bar", figsize=(11, 5)
)
plt.title("Missing Values by Feature")
plt.ylabel("Missing Values (%)")
plt.xticks(rotation=65, ha="right")
plt.tight_layout()
plt.show()

# Step 4: Check duplicates
print("Duplicates:", df.duplicated().sum())

# Remove exact duplicates
df = df.drop_duplicates().copy()

# Step 5: Check numerical variables
print(df.describe())

# Step 6: Check class distribution
print(df["class"].value_counts(normalize=True) * 100)


#------------------------------------------------------
# Make a working copy
df_clean = df.copy()

# Remove columns with over 80% missing values
missing_pct = df_clean.isnull().mean()
columns_to_drop = missing_pct[missing_pct > 0.80].index.tolist()

df_clean = df_clean.drop(columns=columns_to_drop)
print("Dropped columns:", columns_to_drop)

# Fill remaining missing categorical values
categorical_cols = df_clean.select_dtypes(
    include="object"
).columns.drop("class")

df_clean[categorical_cols] = (
    df_clean[categorical_cols].fillna("Missing")
)

# Check remaining missing values
print(df_clean.isnull().sum())

# Save the prepared dataset
df_clean.to_csv("mushrooms_cleaned.csv", index=False)



#Graph 1 — Edible vs poisonous mushrooms

counts = df_clean["class"].value_counts()
counts = counts.rename({"e": "Edible", "p": "Poisonous"})

counts.plot(kind="bar", figsize=(7, 4))
plt.title("Distribution of Mushroom Classes")
plt.ylabel("Number of Mushrooms")
plt.xlabel("Classification")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


#Graph 2 — Cap shape vs toxicity


cap_shape_labels = {
    "b": "Bell",
    "c": "Conical",
    "x": "Convex",
    "f": "Flat",
    "s": "Sunken",
    "p": "Spherical",
    "o": "Other"
}

df_clean["cap_shape_name"] = (
    df_clean["cap-shape"].map(cap_shape_labels)
)

cap_rates = (
    df_clean.groupby("cap_shape_name")["class"]
    .apply(lambda x: (x == "p").mean() * 100)
    .sort_values()
)

cap_rates.plot(kind="bar", figsize=(9, 5))
plt.title("Percentage of Poisonous Mushrooms by Cap Shape")
plt.xlabel("Cap Shape")
plt.ylabel("Poisonous (%)")
plt.ylim(0, 100)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


#Graph 3 — Cap diameter vs toxicity

edible = df_clean.loc[
    df_clean["class"] == "e", "cap-diameter"
]

poisonous = df_clean.loc[
    df_clean["class"] == "p", "cap-diameter"
]

plt.figure(figsize=(8, 5))
plt.boxplot([edible, poisonous],
            tick_labels=["Edible", "Poisonous"])
plt.title("Cap Diameter by Mushroom Classification")
plt.ylabel("Cap Diameter")
plt.tight_layout()
plt.show()



#-----------------------------------------------------
#Information gain


from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score

train_df, test_df = train_test_split(
    df_clean,
    test_size=0.20,
    random_state=42,
    stratify=df_clean["class"]
)

# Information gain for categorical variables
categorical_features = [
    c for c in train_df.select_dtypes(include="object").columns
    if c != "class"
]

ig_scores = {}

for feature in categorical_features:
    ig_scores[feature] = (
        mutual_info_score(
            train_df[feature], train_df["class"]
        ) / np.log(2)
    )

ig = pd.Series(ig_scores).sort_values(ascending=False)

# Plot information gain
ig.sort_values().plot(kind="barh", figsize=(9, 6))
plt.title("Information Gain by Mushroom Feature")
plt.xlabel("Information Gain (bits)")
plt.tight_layout()
plt.show()

print(ig)

#------------------------------------------------------
#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Step 1: Separate features and target
# Exclude any previously created duplicate or target columns
exclude = ["class", "toxicity", "cap_shape_name"]

X_train = train_df.drop(columns=exclude, errors="ignore")
X_test = test_df[X_train.columns]

y_train = train_df["class"]
y_test = test_df["class"]

# Step 2: Identify feature types
numeric_cols = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    include="object"
).columns.tolist()

# Step 3: Prepare preprocessing
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Missing"
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

# Step 4: Build the decision tree
tree_model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        random_state=42
    ))
])

# Step 5: Train the model
tree_model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = tree_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(
    y_test, y_pred,
    labels=["p", "e"],
    target_names=["Poisonous", "Edible"]
))


#Confusion matrix

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=["e", "p"],
    display_labels=["Edible", "Poisonous"]
)

plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.show()



from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import matplotlib.pyplot as plt

depths = range(1, 21)

train_accuracy = []
validation_accuracy = []

for depth in depths:

    model = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("classifier", DecisionTreeClassifier(
            criterion="entropy",
            max_depth=depth,
            random_state=42
        ))
    ])

    # Training accuracy
    model.fit(X_train, y_train)
    train_accuracy.append(model.score(X_train, y_train))

    # 5-fold cross-validation on training data
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring="accuracy"
    )

    validation_accuracy.append(scores.mean())

# Plot
plt.figure(figsize=(9, 5))

plt.plot(depths, train_accuracy,
         marker="o", label="Training Accuracy")

plt.plot(depths, validation_accuracy,
         marker="s", label="Validation Accuracy")

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Effect of Tree Depth on Model Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Best depth based on validation accuracy
best_depth = depths[
    validation_accuracy.index(max(validation_accuracy))
]

print("Best tree depth:", best_depth)
print("Best validation accuracy:",
      max(validation_accuracy))



best_tree = Pipeline([
    ("preprocess", clone(preprocessor)),
    ("classifier", DecisionTreeClassifier(
        criterion="entropy",
        max_depth=best_depth,
        random_state=42
    ))
])

best_tree.fit(X_train, y_train)

# Final evaluation on the held-out test set
from sklearn.metrics import classification_report

final_predictions = best_tree.predict(X_test)

print(classification_report(
    y_test,
    final_predictions,
    labels=["p", "e"],
    target_names=["Poisonous", "Edible"]
))




from sklearn.tree import plot_tree

feature_names = (
    best_tree.named_steps["preprocess"]
    .get_feature_names_out()
)

plt.figure(figsize=(18, 9))

plot_tree(
    best_tree.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["Edible", "Poisonous"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("Mushroom Classification Decision Tree")
plt.tight_layout()
plt.show()


#------------------------------------------------------
#KNN

# KNN — Step 1: Data Preparation


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Use the existing X_train and y_train
# Split training data into development and validation
X_dev, X_val, y_dev, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

# Identify numerical and categorical features
numeric_cols = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    include="object"
).columns.tolist()

# Preprocessing for KNN
knn_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Missing"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ]), categorical_cols)
])


# KNN — Step 2: Build Model

knn_model = Pipeline([
    ("preprocess", knn_preprocessor),
    ("classifier", KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        n_jobs=-1
    ))
])

# Train
knn_model.fit(X_dev, y_dev)

# Predict on validation data
knn_pred = knn_model.predict(X_val)

# Evaluation
print("KNN Validation Accuracy:",
      accuracy_score(y_val, knn_pred))

print(classification_report(
    y_val,
    knn_pred,
    labels=["p", "e"],
    target_names=["Poisonous", "Edible"]
))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_val,
    knn_pred,
    labels=["e", "p"],
    display_labels=["Edible", "Poisonous"]
)

plt.title("KNN Confusion Matrix (k=5)")
plt.tight_layout()
plt.show()
