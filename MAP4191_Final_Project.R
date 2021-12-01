# Load required libraries
library(glmnetUtils)

# Import data into list
X <- read.table("data/X.txt")
Y <- read.table("data/y.txt")$V1
XY <- as.data.frame(cbind(Y, X))

# Linear regression model
linear_regression <- lm(Y ~ ., data = XY)
summary(linear_regression)

# Ridge regression model
ridge_regression <- glmnet(Y ~ ., data = XY, alpha = 0)
summary(ridge_regression)
## Find optimal lambda
cv_ridge_model <- cv.glmnet(as.matrix(X), as.matrix(Y), alpha = 0)
lambda_ridge <- cv_ridge_model$lambda.min
## Plot test MSE by lambda value
plot(cv_ridge_model)
## Find optimal ridge regression model
ridge_regression <- glmnet(Y ~ ., data = XY, alpha = 0, lambda = lambda_ridge)
summary(ridge_regression)

# Lasso regression model
lasso_regression <- glmnet(Y ~ ., data = XY, alpha = 1)
summary(lasso_regression)
## Find optimal lambda
cv_lasso_model <- cv.glmnet(as.matrix(X), as.matrix(Y), alpha = 1)
lambda_lasso <- cv_lasso_model$lambda.min
## Plot test MSE by lambda value
plot(cv_lasso_model)
## Find optimal lasso regression model
lasso_regression <- glmnet(Y ~ ., data = XY, alpha = 1, lambda = lambda_lasso)
summary(ridge_regression)