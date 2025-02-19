library(cmdstanr)
library(ggplot2)
library(dplyr)

# Stan model code (as a string)
stan_model_code <- "
data {
  int<lower=1> N;           // number of observations
  int<lower=1> D;           // number of features (here D = 1)
  matrix[N, D] X;           // input matrix
  vector[N] y;              // target variable (price)
  int<lower=1> n_hidden1;   // number of neurons in first hidden layer
  int<lower=1> n_hidden2;   // number of neurons in second hidden layer
}

parameters {
  // Input -> First hidden layer weights and biases
  matrix[D, n_hidden1] weights_in;
  vector[n_hidden1] bias_in;
  
  // First hidden layer -> Second hidden layer weights and biases
  matrix[n_hidden1, n_hidden2] weights_hidden;
  vector[n_hidden2] bias_hidden;
  
  // Second hidden layer -> Output layer weights and bias
  vector[n_hidden2] weights_out;
  real bias_out;
  
  // Noise standard deviation (HalfNormal(0,1))
  real<lower=0> sigma;
}

transformed parameters {
  // Compute the network output (mu) for each observation
  vector[N] mu;
  
  for (i in 1:N) {
    vector[n_hidden1] h1;
    vector[n_hidden2] h2;
    
    // First hidden layer: linear combination then sigmoid activation
    for (j in 1:n_hidden1)
      h1[j] = inv_logit( dot_product( X[i], weights_in[, j] ) + bias_in[j] );
      
    // Second hidden layer: linear combination then sigmoid activation
    for (k in 1:n_hidden2) {
      real temp = 0;
      for (j in 1:n_hidden1)
        temp += h1[j] * weights_hidden[j, k];
      h2[k] = inv_logit( temp + bias_hidden[k] );
    }
    
    // Output layer: linear combination (no activation)
    mu[i] = dot_product( h2, weights_out ) + bias_out;
  }
}

model {
  // Priors for weights and biases
  to_vector(weights_in) ~ normal(0, 10);
  bias_in ~ normal(0, 10);
  
  to_vector(weights_hidden) ~ normal(0, 10);
  bias_hidden ~ normal(0, 10);
  
  weights_out ~ normal(3000, 500);
  bias_out ~ normal(700, 300);
  
  sigma ~ normal(100, 10);
  
  // Likelihood
  for (i in 1:N)
    y[i] ~ normal(mu[i], sigma * X[i, 1]);
}

generated quantities {
  // Posterior predictive draws for y
  vector[N] y_pred;
  for (i in 1:N)
    y_pred[i] = normal_rng(mu[i], sigma * X[i, 1]);
}
"

# Write the Stan model to a file
stan_file <- write_stan_file(stan_model_code)

# -------------------------------
# Prepare your data in R
# -------------------------------
df <- read.csv("/workspaces/SEH/datasets/sampled_diamonds.csv")
df_sub <- df %>% select(carat, price)
N <- nrow(df_sub)
D <- 1
X <- as.matrix(df_sub$carat)
y <- df_sub$price

# Neural network architecture parameters
n_hidden1 <- 4
n_hidden2 <- 4

# Bundle data for Stan
stan_data <- list(
  N = N,
  D = D,
  X = X,
  y = y,
  n_hidden1 = n_hidden1,
  n_hidden2 = n_hidden2
)

# -------------------------------
# Compile the model and sample using cmdstanr
# -------------------------------
mod <- cmdstan_model(stan_file)

fit <- mod$sample(
  data = stan_data,
  iter_warmup = 1000,
  iter_sampling = 1000,
  chains = 4,
  parallel_chains = 4,
  seed = 42,
  refresh = 10,
  adapt_delta = 0.95,
  max_treedepth = 15
)

# Print a summary of key parameters
print(fit$summary(variables = c("weights_in", "bias_in", "weights_hidden", 
                                "bias_hidden", "weights_out", "bias_out", "sigma")))


# Plot

# Extract posterior draws for y_pred as a matrix
y_pred_draws <- fit$draws("y_pred", format = "matrix")

# For each observation, compute the mean, 2.5% and 97.5% quantiles
pred_summary <- data.frame(
  carat = df_sub$carat,
  mean  = apply(y_pred_draws, 2, mean),
  lower = apply(y_pred_draws, 2, quantile, probs = 0.025),
  upper = apply(y_pred_draws, 2, quantile, probs = 0.975)
)

# (Optional) Sort by carat for a smoother line plot
pred_summary <- pred_summary[order(pred_summary$carat), ]

# Plot: predicted mean with 95% CI ribbon and actual data points
library(ggplot2)
ggplot(pred_summary, aes(x = carat)) +
  geom_line(aes(y = mean), color = "blue") +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "blue", alpha = 0.2) +
  geom_point(data = df_sub, aes(y = price), color = "black", alpha = 0.5) +
  labs(x = "Carat", y = "Price", 
       title = "Predicted Price with 95% Credible Intervals") +
  theme_minimal()



