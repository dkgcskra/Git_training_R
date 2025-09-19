
#################################
# training vs test_error
#################################


# Prerequisites
library(tidyverse)
library(tidymodels)



#############################################
# 1. Generate data from a quadratic function + noise
#############################################

set.seed(42)
n <- 75 # Number of data points

sim_data <- tibble(
  x = runif(n, -3, 3), # Generate n random x-values uniformly between -3 and 3
  y_true = x^2, # Define the true underlying function: y = x^2 (without noise)
  y = y_true + rnorm(n, sd = 4) # Add random normal noise (mean = 0, sd = 4) to simulate observed y-values
)

#################################
# 2. Plot the data
###############################

ggplot(sim_data, aes(x = x, y = y)) +
  geom_point() +
  geom_line(aes(y = y_true), color = 'blue', size = 1) +
  labs(title = "Simulated Data with True Function", y = "y", x = "x")


#################################
# 3. Fit two models on the entire dataset (no train/test split)
################################

model_linear <- lm(y ~ x, data = sim_data)      # a simple linear model
model_poly14   <- lm(y ~ poly(x, 14), data = sim_data)  # a highly flexible, over-fitted model

# Compute training Mean Squared Error (MSE)
lin_mse  <- mean((sim_data$y - predict(model_linear,  sim_data))^2)
deg14_mse <- mean((sim_data$y - predict(model_poly14, sim_data))^2)

lin_mse
deg14_mse

# based on MSE on the degree 14 polynomial model seems to perform better
# lets plot the fits
sim_data_res <- sim_data %>%
  mutate(
    pred_linear = predict(model_linear, sim_data),
    pred_poly14  = predict(model_poly14, sim_data)
  )

ggplot(sim_data_res, aes(x = x)) +
  geom_point(aes(y = y), alpha = 0.5) +
  geom_line(aes(y = y_true), color = 'blue', size = 1, linetype = "dashed") +
  geom_line(aes(y = pred_linear), color = 'red', size = 1) +
  geom_line(aes(y = pred_poly14), color = 'green', size = 1) +
  labs(title = "Model Fits", y = "y", x = "x")
  scale_color_manual(values = c("True Function" = "blue", "Linear Model" = "red", "Poly (degree 9)" = "green"))

################################
# 4. Split the data into training and testing sets
################################

set.seed(42)
data_split <- initial_split(sim_data, prop = 0.75)
train_data <- training(data_split)
test_data  <- testing(data_split)

# Fit the models on the training data
model_linear_train <- lm(y ~ x, data = train_data)
model_poly14_train   <- lm(y ~ poly(x, 14), data = train_data)


# Compute training and testing MSE
train_lin_mse  <- mean((train_data$y - predict(model_linear_train,  train_data))^2)
train_deg14_mse <- mean((train_data$y - predict(model_poly14_train, train_data))^2)
test_lin_mse   <- mean((test_data$y - predict(model_linear_train,  test_data))^2)
test_deg14_mse  <- mean((test_data$y - predict(model_poly14_train, test_data))^2)

# for readability and to compare with earlier result we'll collect them all
results_wide <- tibble(
  Data = c("All data", "Train", "Test"),
  Linear = c(lin_mse, train_lin_mse, test_lin_mse),
  Poly14 = c(deg14_mse, train_deg14_mse, test_deg14_mse)
)

results_wide

# Asignment now is to repeat the above but now with a degree 2 polynomial using the train/test data as above. 
# How does it compare to the linear and degree 14 polynomial models?
# What does this say about the bias-variance tradeoff?

