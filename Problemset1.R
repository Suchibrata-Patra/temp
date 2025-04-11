##=======================================================
## Question No - 01
##=======================================================
rm(list=ls())
library(ISLR)
library(leaps)
sum(is.na(Hitters$Salary))
#regfit.full = regsubsets(Salary~.,Hitters)
#summary(regfit.full)
regfit.full=regsubsets(Salary~.,data=Hitters,nvmax=19)
reg.summary =summary (regfit.full)
names(reg.summary)
par(mfrow=c(2,2))
plot(reg.summary$rss ,xlab="Number of Variables ",ylab="RSS",type="l")
plot(reg.summary$adjr2 ,xlab="Number of Variables ",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points (11,reg.summary$adjr2[11], col="red",cex=2,pch =20)
plot(reg.summary$cp ,xlab="Number of Variables ",ylab="Cp",type='l')
which.min(reg.summary$cp )
points (10,reg.summary$cp[10], col ="red",cex=2,pch =20)
which.min(reg.summary$bic )
plot(reg.summary$bic ,xlab="Number of Variables ",ylab="BIC",type='l')
points (6,reg.summary$bic[6],col="red",cex=2,pch =20)
regfit.fwd=regsubsets (Salary~.,data=Hitters , nvmax=19,method ="forward")
summary (regfit.fwd)
regfit.bwd=regsubsets (Salary~.,data=Hitters,nvmax=19,method ="backward")
summary (regfit.bwd)
coef(regfit.full,19)



Hitters = na.omit(Hitters)
set.seed(123)
train = sample(c(TRUE, FALSE), nrow(Hitters), rep = TRUE)
test = !train
regfit.best = regsubsets(Salary ~ ., data = Hitters[train, ], nvmax = 19)
test.mat = model.matrix(Salary ~ ., data = Hitters[test, ])
test.y = Hitters[test, "Salary"]
val.errors = rep(NA, 19)
for(i in 1:19){
  coefi = coef(regfit.best, id = i)
  pred = test.mat[, names(coefi)] %*% coefi
  val.errors[i] = mean((test.y - pred)^2)
}



#=======================================================
### Question Number - 2
#=======================================================
rm(list=ls())
library(glmnet)
library(ISLR)
data(Hitters)
Hitters = na.omit(Hitters)
x = model.matrix(Salary~.,Hitters )[,-1]
y = Hitters$Salary
grid=10^seq(10,-2, length =100)
ridge.mod=glmnet (x,y,alpha=0, lambda=grid)

dim(coef(ridge.mod))
cat('Dimension of the Coefficient Matrix:',dim(coef(ridge.mod))[1],'x',dim(coef(ridge.mod))[2])

lambda_50 = ridge.mod$lambda[50]
coef(ridge.mod)[,50]
l2_norm_for_lambda_50 = sqrt(sum(coef(ridge.mod)[-1,50]^2) )
cat('l2 norm for lambda_50 = ',l2_norm_for_lambda_50)

lambda_60 = ridge.mod$lambda[60]
coef(ridge.mod)[ ,60]
l2_norm_for_lambda_60 = sqrt(sum(coef(ridge.mod)[-1,60]^2) )
cat('l2 norm for lambda_60 = ',l2_norm_for_lambda_50)


### Comments
# As the regularisation parameter , λ increases, the L2 norm of the coefficient vector decreases.
# This reflects the increased shrinkage effect of ridge regression, which penalizes large coefficients more heavily.
# For example, the L2 norm of the coefficients at λ_50 and λ _60 is larger than that at λ_50
# λ_60, confirming that stronger penalisation results in smaller overall coefficients.



# Predict using lambda_50 and lambda_60
pred_50 = predict(ridge.mod, s=lambda_50, newx=x)
pred_60 = predict(ridge.mod, s=lambda_60, newx=x)

# Calculate MSES
mse_50 = mean((y - pred_50)^2)
mse_60 = mean((y - pred_60)^2)

# Print results
cat('MSE for lambda_50 (', lambda_50, '):', mse_50, '\n')
cat('MSE for lambda_60 (', lambda_60, '):', mse_60, '\n')



### Line Plot of Standarised Coefficients
plot(ridge.mod, xvar = "lambda", label = TRUE)
title("Ridge Regression: Coefficient Paths vs Lambda")

# Interpretation
# As λ increases, the magnitude of the coefficients shrinks toward zero.
# Variables that shrink faster are less influential, while those that resist
# shrinkage hold greater predictive power. This helps identify robust predictors
# the ones that stay strong even when regularization pressure rises.


# Calculate OLS coefficients (no regularization, lambda = 0)
ols.coef = coef(lm(Salary ~ ., data = Hitters))[-1]  # drop intercept
ols.l2 = sqrt(sum(ols.coef^2))

# Calculate L2 norms for each lambda in ridge
ridge.l2 = apply(coef(ridge.mod)[-1, ], 2, function(beta) sqrt(sum(beta^2)))

# Calculate ratio for each lambda
ratio = ridge.l2 / ols.l2

# Plot: Coefficients against ratio
matplot(ratio, t(coef(ridge.mod)[-1,]), type = "l", lty = 1, lwd = 2,
        xlab = "L2 Norm Ratio (Ridge / OLS)", ylab = "Standardized Coefficients",
        main = "Ridge Coefficient Paths vs L2 Norm Ratio")

# The x-axis now represents how much each model's coefficients have
# shrunk compared to OLS (which has ratio = 1).
# As we move leftward, the regularization effect increases, and the shrinkage
# intensifies — coefficients move toward zero.
# This plot is excellent for intuitively grasping how far we’ve 
# traveled from the original model in terms of coefficient magnitude.



set.seed(123)
train = sample(1:nrow(x), nrow(x) / 2)
test = (-train)
x_train = x[train, ]
y_train = y[train]
x_test = x[test, ]
y_test = y[test]
# Fit Ridge model using cross-validation
cv.ridge = cv.glmnet(x_train, y_train, alpha = 0)  # alpha=0 → Ridge
# Optimal lambda from CV
best_lambda = cv.ridge$lambda.min
cat("Optimal lambda (from CV):", best_lambda, "\n")

# Plot CV results
plot(cv.ridge)
title("Cross-Validation for Ridge Regression")

# Predict on test set using best lambda
ridge.pred = predict(cv.ridge, s = best_lambda, newx = x_test)

# Test MSE
test_mse = mean((ridge.pred - y_test)^2)
cat("Test MSE using optimal lambda:", test_mse, "\n")


# The plot represents the 10-fold cross-validation error across a grid of
# lambda values for Ridge Regression. The model achieves its lowest
# mean squared error at log(λ) ≈ X.XX, corresponding to a lambda of approximately 
# λ = XXXX. As lambda increases, the model becomes more regularized,
# which increases the bias but reduces variance. The optimal lambda balances
# this tradeoff, minimizing the cross-validated prediction error.
# A higher lambda (within one standard error of the minimum) may
# be selected for simplicity and better generalization.


#### FITTING LASSO
# Perform LASSO with cross-validation
cv.lasso = cv.glmnet(x_train, y_train, alpha = 1)  # LASSO: alpha = 1

# Plot CV errors
plot(cv.lasso)
title("LASSO Cross-Validation Curve")

# Best lambda (minimum CV error)
best.lambda = cv.lasso$lambda.min
cat("Best lambda from CV:", best.lambda, "\n")

# Predict on test set using best lambda
lasso.pred = predict(cv.lasso, s = best.lambda, newx = x_test)

# Test MSE
lasso.mse = mean((lasso.pred - y_test)^2)
cat("Test MSE for LASSO:", lasso.mse, "\n")

# Coefficients at best lambda
lasso.coef = predict(cv.lasso, s = best.lambda, type = "coefficients")

# Count non-zero coefficients (excluding intercept)
nonzero.count = sum(lasso.coef != 0) - 1
cat("Number of non-zero coefficients:", nonzero.count, "\n")

