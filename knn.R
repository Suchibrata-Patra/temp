rm(list=ls())
library(ISLR)
library(caret)
library(class)

data("Weekly")
Weekly$Year = as.numeric(Weekly$Year)
train_index = Weekly$Year <= 2008
test_index = Weekly$Year >= 2009
train_data = Weekly[train_index, ]
test_data = Weekly[test_index, ]

# Remove non-predictor variables (like 'Today' and 'Year')
train_data = subset(train_data, select = -c(Year, Today))
test_data = subset(test_data, select = -c(Year, Today))

# Ensure the response is a factor
train_data$Direction = as.factor(train_data$Direction)
test_data$Direction = as.factor(test_data$Direction)

# Set training control for 10-fold CV
ctrl = trainControl(method = "cv", number = 10)

# Train the KNN model with automatic tuning over 20 values of k
set.seed(123)
knn_fit = train(Direction ~ ., 
                 data = train_data,
                 method = "knn",
                 trControl = ctrl,
                 tuneLength = 20)

# Best k value found
cat("Best K found for prediction model:\n")
print(knn_fit$bestTune)

# Predict on test set
pred = predict(knn_fit, newdata = test_data)

# Confusion Matrix and Accuracy
cm = confusionMatrix(pred, test_data$Direction)
cat("\nConfusion Matrix:\n")
print(cm$table)

cat("\nTest Error Rate for this love-fueled model:", round(1 - cm$overall["Accuracy"], 4), "\n")

# Optional: visualize K vs Accuracy
plot(knn_fit, main = "K vs Accuracy - Helping You Pick the Optimal Values")
