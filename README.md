# Titanic survival prediction

Two Docker containers train logistic regression models for the Kaggle Titanic competition. Run the commands below from the repository root using a terminal on macOS, Linux, or Windows with WSL.

## 1. Set up the data

1. Install and start Docker Desktop or Docker Engine. Clone this repository using the URL under **Code → HTTPS** on GitHub, then enter the cloned directory.
2. Sign in to Kaggle and open the [Titanic competition data page](https://www.kaggle.com/competitions/titanic/data). Join the competition and accept its rules if prompted. Download the competition files and extract the ZIP.
3. Create the data folder with `mkdir -p src/data`. Copy the extracted `train.csv` and `test.csv` directly into `src/data/`. The two paths should be `src/data/train.csv` and `src/data/test.csv`. Keep the original filenames.

## 2. Run Python

```bash
docker build -t titanic-python ./src/python
docker run --rm --mount "type=bind,source=$PWD/src/data,target=/data" titanic-python
```

## 3. Run R

```bash
docker build -t titanic-r ./src/r
docker run --rm --mount "type=bind,source=$PWD/src/data,target=/data" titanic-r
```

The build contexts contain only the files needed by each container. The bind mount shares the local data directory so both scripts can read the input CSVs and write predictions back to the host.

## 4. Check results

Each run prints the dataset sizes, observed survival rate, feature transformations, labeled holdout accuracy, full training accuracy, and every test prediction (passenger ID and predicted survival). Both models use sex, passenger class, and the number of siblings, spouses, parents, and children aboard. A stratified 80/20 split of the labeled training data provides a local test score. Each script then refits on all labeled rows before predicting for Kaggle's `test.csv`.

Look for `src/data/python_predictions.csv` and `src/data/r_predictions.csv`. Each contains one row per test passenger with `PassengerId` and predicted `Survived`. The official test labels are hidden, so the scripts report predictions for `test.csv`.
