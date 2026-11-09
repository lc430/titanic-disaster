# Fit and evaluate a Titanic logistic regression and export Kaggle predictions.
args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args)) args[[1]] else "src/data"

read_passengers <- function(filename) {
  rows <- read.csv(file.path(data_dir, filename))
  cat(sprintf("Loaded %d passengers from %s.\n", nrow(rows), filename))
  rows
}

features <- function(rows) {
  data.frame(
    female = as.integer(rows$Sex == "female"),
    second_class = as.integer(rows$Pclass == 2),
    third_class = as.integer(rows$Pclass == 3),
    family_size = rows$SibSp + rows$Parch
  )
}

predict_survival <- function(model, passengers) {
  as.integer(predict(model, newdata = passengers, type = "response") >= 0.5)
}

train <- read_passengers("train.csv")
cat(sprintf("Observed survival rate: %.2f%%.\n", 100 * mean(train$Survived)))
cat("Selected sex and class, and summed SibSp and Parch into family size.\n")
cat("Encoded female, second class and third class as binary indicators.\n")
cat("Excluded other columns, including PassengerId and fields with missing values.\n")
training <- cbind(Survived = train$Survived, features(train))

set.seed(42)
# Sample within each outcome class to preserve its representation.
holdout <- unlist(lapply(split(seq_len(nrow(training)), training$Survived),
                        function(indices) sample(indices, round(0.2 * length(indices)))))
validation_model <- glm(Survived ~ ., data = training[-holdout, ], family = binomial())
held_out <- training[holdout, ]
cat(sprintf("Local held-out test accuracy (%d labeled passengers): %.2f%%.\n",
            nrow(held_out),
            100 * mean(predict_survival(validation_model, held_out) == held_out$Survived)))

model <- glm(Survived ~ ., data = training, family = binomial())
cat(sprintf("Refit logistic regression on all %d labeled passengers.\n", nrow(training)))
cat(sprintf("Full training accuracy: %.2f%%.\n",
            100 * mean(predict_survival(model, training) == training$Survived)))

test <- read_passengers("test.csv")
cat("Applied the same feature transformations to test.csv.\n")
predictions <- predict_survival(model, features(test))
write.csv(data.frame(PassengerId = test$PassengerId, Survived = predictions),
          file.path(data_dir, "r_predictions.csv"), row.names = FALSE)
cat(sprintf("Wrote %d predictions to r_predictions.csv.\n", length(predictions)))
cat("Test predictions (PassengerId, Survived):\n")
write.csv(data.frame(PassengerId = test$PassengerId, Survived = predictions),
          stdout(), row.names = FALSE)
