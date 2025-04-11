rm(list=ls())
library(ISLR)
library(dplyr)
library(caTools)
data = as.data.frame(Default)
model = glm(default ~ student + income + balance, data = Default, family = binomial)
summary(model)
colSums(is.na(data))

# Changing the yes or no to binary variables i.e 0 or 1
data$default = ifelse(data$default=="Yes",1,0)
data$student = ifelse(data$student=="Yes",1,0)
glimpse(data)

set.seed(123)
sample = sample.split(1:nrow(data), SplitRatio = 0.8)
training_dataset = subset(data, sample == TRUE)
testing_dataset = subset(data, sample == FALSE)

response_variable = training_dataset$default
model = glm(default ~ ., data = training_dataset, family = binomial(link = "logit"))
predicted_prob = predict(model, training_dataset, type = "response")
p_hat = sort(predicted_prob)
tables_list = vector("list", length(p_hat))

for (col in 1:length(p_hat)) {
  threshold = p_hat[col]
  y_hat = ifelse(predicted_prob >= threshold, 1, 0)
  
  #tables_list[[col]] = table(training_dataset$Survived, y_hat)
  # Issue with this line of code is that sometimes, it is not storing the 2x2 contingency table.. while there is no need of columns, it is simply omitting it..
  tbl = table(
    factor(training_dataset$default, levels = c(0, 1)),
    factor(y_hat, levels = c(0, 1))
  )
  tables_list[[col]] = tbl
}
tables_list[10]

results = data.frame(
  Threshold = p_hat,
  TPR = numeric(length(tables_list)),
  FPR = numeric(length(tables_list))
)

# Loop with coalesce() for missing values
for (i in seq_along(tables_list)) {
  tbl = tables_list[[i]]
  
  # Use coalesce() to handle missing values with 0 fallback
  TP = coalesce(tbl["1", "1"], 0)
  FN = coalesce(tbl["1", "0"], 0)
  FP = coalesce(tbl["0", "1"], 0)
  TN = coalesce(tbl["0", "0"], 0)
  
  # Store TPR and FPR
  results$TPR[i] = TP / (TP + FN)
  results$FPR[i] = FP / (FP + TN)
}
plot(results$FPR, results$TPR, type = "l", col = "blue", lwd = 2,
     xlab = "False Positive Rate (FPR)", ylab = "True Positive Rate (TPR)",
     main = "Reverse Operated Characteristics Curve")
abline(a = 0, b = 1, col = "red", lty = 2)

AUC = abs(sum(diff(results$FPR) * (results$TPR[-1] + results$TPR[-length(results$TPR)]))/2)
print(paste("AUC:", AUC))

temp = results$TPR*(1-results$FPR)
predicted_prob[which.max(temp)]
