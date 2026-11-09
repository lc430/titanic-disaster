"""Train and evaluate a Titanic logistic regression and export Kaggle predictions."""

import csv
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


DATA = Path(__file__).resolve().parent.parent / "data"
if Path("/data").is_dir():
    DATA = Path("/data")


def read_rows(filename):
    with (DATA / filename).open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    print(f"Loaded {len(rows)} passengers from {filename}.")
    return rows


def features(rows):
    return [
        [int(row["Sex"] == "female"), int(row["Pclass"] == "2"),
         int(row["Pclass"] == "3"), int(row["SibSp"]) + int(row["Parch"])]
        for row in rows
    ]


def main():
    train = read_rows("train.csv")
    y = [int(row["Survived"]) for row in train]
    print(f"Observed survival rate: {sum(y) / len(y):.2%}.")
    print("Selected sex and class, and summed SibSp and Parch into family size.")
    print("Encoded female, second class and third class as binary indicators.")
    print("Excluded other columns, including PassengerId and fields with missing values.")
    x = features(train)

    x_fit, x_holdout, y_fit, y_holdout = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=42
    )
    validation_model = LogisticRegression(max_iter=1000).fit(x_fit, y_fit)
    print(f"Local held-out test accuracy ({len(y_holdout)} labeled passengers): "
          f"{validation_model.score(x_holdout, y_holdout):.2%}.")

    model = LogisticRegression(max_iter=1000).fit(x, y)
    print(f"Refit logistic regression on all {len(y)} labeled passengers.")
    print(f"Full training accuracy: {model.score(x, y):.2%}.")

    test = read_rows("test.csv")
    print("Applied the same feature transformations to test.csv.")
    predictions = model.predict(features(test))
    destination = DATA / "python_predictions.csv"
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["PassengerId", "Survived"])
        writer.writerows((row["PassengerId"], prediction)
                         for row, prediction in zip(test, predictions))
    print(f"Wrote {len(predictions)} predictions to {destination.name}.")
    print("Test predictions (PassengerId, Survived):")
    print(destination.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
