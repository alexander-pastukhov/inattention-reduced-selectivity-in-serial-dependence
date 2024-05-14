// Bayesian integration model
data {
  int <lower=1> N;
  int <lower=1> TaskN;
  int <lower=1> ParticipantsN;
  int <lower=1> Nmax;
  
  array[N] int<lower=0, upper=Nmax> DotsN; // number of dots
  array[N] real<lower=0, upper=Nmax> Response;
  array[N] int<lower=1> Trial;
  array[N] int<lower=1, upper=TaskN> Task;
  array[N] int<lower=1, upper=ParticipantsN> Participant;
}

transformed data {
  vector[N] NR2 = rep_vector(0, N);
  for(i in 1:N){
    if (Trial[i] > 1) NR2[i] = (DotsN[i] - Response[i - 1])^2;
  }
}

parameters {
  vector[3 * TaskN] mu_alpha_a_k;
  cholesky_factor_corr[3 * TaskN] l_rho_alpha_a_k;
  vector<lower=0>[3 * TaskN] sigma_alpha_a_k;
  matrix[3 * TaskN, ParticipantsN] z_alpha_a_k;
}

transformed parameters {
  vector[N] mu;
  vector[N] sigma;
  {
    matrix[3 * TaskN, ParticipantsN] alpha_a_k = rep_matrix(mu_alpha_a_k, ParticipantsN) + diag_pre_multiply(sigma_alpha_a_k, l_rho_alpha_a_k) * z_alpha_a_k;
    matrix[TaskN, ParticipantsN] alpha = inv_logit(block(alpha_a_k, 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] a = exp(block(alpha_a_k, TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] k = exp(block(alpha_a_k, 2 * TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] k2 = k.^2;

    real k2Na2;
    real Wprev;

    for(i in 1:N) {
      if (Trial[i] == 1) {
        mu[i] = alpha[Task[i], Participant[i]] * DotsN[i];
      } else {
        k2Na2 = k2[Task[i], Participant[i]] * DotsN[i]^(2 * a[Task[i], Participant[i]]);
        Wprev = k2Na2 / (k2Na2 + k2[Task[i], Participant[i]] * DotsN[i-1]^(2 * a[Task[i], Participant[i]]) + NR2[i]);
        mu[i] = alpha[Task[i], Participant[i]] * ((1 - Wprev) * DotsN[i] + Wprev * Response[i-1]);
      }
      sigma[i] = k[Task[i], Participant[i]] * DotsN[i] ^ a[Task[i], Participant[i]];
    }
  }
} 

model {
  mu_alpha_a_k[1:2] ~ normal(logit(0.8), 0.5);  // alpha
  mu_alpha_a_k[3:4] ~ normal(0, 1);             // a
  mu_alpha_a_k[5:6] ~ normal(0, 1);             // k
  l_rho_alpha_a_k ~ lkj_corr_cholesky(2);
  sigma_alpha_a_k ~ exponential(1);
  to_vector(z_alpha_a_k) ~ normal(0, 1);

  Response ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  for(i in 1:N) log_lik[i] = normal_lpdf(Response[i] | mu[i], sigma[i]);
}
