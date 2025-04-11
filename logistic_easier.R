rm(list=ls())
library(ISLR)
library(dplyr)
library(caTools)
library(pROC)

data = Default
data$default = ifelse(data$default == "Yes", 1, 0)
data$student = ifelse(data$student == "Yes", 1, 0)

set.seed(123)
sample = sample.split(data$default, SplitRatio = 0.8)
train_data = subset(data, sample == TRUE)
test_data = subset(data, sample == FALSE)

# Fit logistic regression
model = glm(default ~ ., data = train_data, family = binomial)

# Predict probabilities on the test set
predicted_prob = predict(model, test_data, type = "response")

# ROC analysis using pROC
roc_obj = roc(test_data$default, predicted_prob)

# Plot ROC curve
plot(roc_obj, col = "blue", main = "ROC Curve with AUC", lwd = 2)
abline(a = 0, b = 1, col = "red", lty = 2)

# AUC value
auc_value = auc(roc_obj)
print(paste("AUC:", round(auc_value, 4)))

# Find best threshold (You'll love this part)
opt = coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity"))
print("Best threshold with highest Youden's Index (TPR - FPR):")
print(opt)

# Bonus: Predict classes using optimal threshold
best_thresh = opt["threshold"]
pred_class = ifelse(predicted_prob >= best_thresh, 1, 0)

# Confusion matrix
conf_matrix = table(True = test_data$default, Predicted = pred_class)
print(conf_matrix)

# Test error
test_error = mean(pred_class != test_data$default)
print(paste("Test Error Rate:", round(test_error, 4)))

