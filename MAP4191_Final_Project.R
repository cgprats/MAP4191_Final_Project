# Load required libraries
library(glmnetUtils)
library(caret)

# Enable multi-threading to speed up training
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

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
summary(lasso_regression)

# Coefficients summary
summary(linear_regression)$coefficients
coef(ridge_regression)
coef(lasso_regression)

# Perform cross validation to evaluate regression models
## Define number of folds
fold_definition <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
## Evaluate linear regression
linear_model <- train(Y ~ ., data = XY, method = "lm", trControl = fold_definition)
print(linear_model)
## Evaluate ridge regression
ridge_model <- train(Y ~ .,data = XY, method = "ridge", trControl = fold_definition)
print(ridge_model)
## Evaluate lasso regression
lasso_model <- train(Y ~ ., data = XY, method = "lasso", trControl = fold_definition)
print(lasso_model)

# Deregister cluster for multi-threading
stopCluster(cluster)
registerDoSEQ()