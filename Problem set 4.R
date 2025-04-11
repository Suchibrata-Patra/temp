rm(list = ls())
library(ISLR)
library(ggplot2)
library(pROC)

# a. Load and split the data
data("Weekly")
Weekly$Year = as.numeric(Weekly$Year)
train_data = Weekly[Weekly$Year <= 2008, ]
test_data  = Weekly[Weekly$Year >= 2009, ]

# b. Fit Logistic Regression Model
log_model = glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
                 data = train_data, family = binomial)

summary(log_model)  # Check p-values

# Least p-value variable
least_p_value = summary(log_model)$coefficients[-1, , drop=FALSE]  # Exclude intercept
least_p_value[which.min(least_p_value[, 4]), , drop=FALSE]

# Interpretation example
cat("Each 1 unit increase in", rownames(least_p_value)[which.min(least_p_value[, 4])],
    "changes the log-odds of market going up by",
    round(least_p_value[which.min(least_p_value[, 4]), 1], 3), "\n")

# c. Estimate odds at median predictor values
medians = apply(train_data[, c("Lag1","Lag2","Lag3","Lag4","Lag5","Volume")], 2, median)
newdata = as.data.frame(t(medians))
pred_logit = predict(log_model, newdata = newdata)
odds_up = exp(pred_logit)
cat("Estimated odds of market going up at median predictors:", round(odds_up, 3), "\n")

# d. ROC Curve from training data
train_probs = predict(log_model, type = "response")
roc_curve = roc(train_data$Direction, train_probs)
plot(roc_curve, col = "blue", main = "ROC Curve on Training Data")
cat("AUC:", auc(roc_curve), "\n")

# e. Optimal cut-point on test data
test_probs = predict(log_model, newdata = test_data, type = "response")

# ROC for test data
roc_test = roc(test_data$Direction, test_probs)
coords_opt = coords(roc_test, 
                     x = "best", 
                     best.method = "closest.topleft", 
                     transpose = FALSE)

cut_off = coords_opt$threshold  # Not coords_opt$x
cat("Optimal cutoff:", round(cut_off, 3), "\n")

# Now build the confusion matrix
test_pred_class = ifelse(test_probs > cut_off, "Up", "Down")
conf_matrix = table(Predicted = test_pred_class, Actual = test_data$Direction)
print(conf_matrix)

# Test error
test_error_rate = mean(test_pred_class != test_data$Direction)
cat("Test Error Rate:", round(test_error_rate, 3), "\n")


# f. Repeat using only Lag1 and Lag2
log_model_simple = glm(Direction ~ Lag1 + Lag2,
                        data = train_data, family = binomial)

test_probs_simple = predict(log_model_simple, newdata = test_data, type = "response")

# Get ROC object
roc_test_simple = roc(test_data$Direction, test_probs_simple)

# Get cutoff the right way
coords_opt_simple = coords(roc_test_simple, 
                            x = "best", 
                            best.method = "closest.topleft", 
                            transpose = FALSE)

cut_off_simple = coords_opt_simple$threshold  # Correct extraction
cat("Optimal cutoff (Lag1 & Lag2):", round(cut_off_simple, 3), "\n")

# Class prediction using the cutoff
test_pred_simple = ifelse(test_probs_simple > cut_off_simple, "Up", "Down")

# Confusion matrix
conf_matrix_simple = table(Predicted = test_pred_simple, Actual = test_data$Direction)
print(conf_matrix_simple)

# Test Error
test_error_simple = mean(test_pred_simple != test_data$Direction)
cat("Test Error Rate (Lag1 & Lag2 only):", round(test_error_simple, 3), "\n")


