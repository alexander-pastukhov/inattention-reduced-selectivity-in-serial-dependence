// Mixture of linear and logarithmic components
data {
  int <lower=1> N;
  int <lower=1> TaskN; // 1 - single, 2 - dual
  int <lower=1> ParticipantsN;
  int <lower=1> Nmax;
  
  array[N] int<lower=0, upper=Nmax> DotsN; // number of dots
  array[N] real<lower=0, upper=Nmax> Response;
  array[N] int<lower=1, upper=TaskN> Task;
  array[N] int<lower=1, upper=ParticipantsN> Participant;
}

transformed data {
  vector[N] normalizedLogN = (Nmax / log(Nmax)) * log(to_vector(DotsN));
}

parameters {
  vector[4 * TaskN] mu_alp_lmb_a_k; // 1) alpha, 2) lambda, 3) a, 4) k
  cholesky_factor_corr[4 * TaskN] l_rho_alp_lmb_a_k;
  vector<lower=0>[4 * TaskN] sigma_alp_lmb_a_k;
  matrix[4 * TaskN, ParticipantsN] z_alp_lmb_a_k;
}

transformed parameters {
  vector[N] mu;
  vector[N] sigma;
  {
    matrix[4 * TaskN, ParticipantsN] alp_lmb_a_k = rep_matrix(mu_alp_lmb_a_k, ParticipantsN) + diag_pre_multiply(sigma_alp_lmb_a_k, l_rho_alp_lmb_a_k) * z_alp_lmb_a_k;
    matrix[TaskN, ParticipantsN] alpha = inv_logit(block(alp_lmb_a_k, 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] lambda = inv_logit(block(alp_lmb_a_k, TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] a = exp(block(alp_lmb_a_k, 2 * TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] k = exp(block(alp_lmb_a_k, 3 * TaskN + 1, 1, TaskN, ParticipantsN));

    for(i in 1:N) {
      mu[i] = alpha[Task[i], Participant[i]] * ((1 - lambda[Task[i], Participant[i]]) * DotsN[i] + 
                                                lambda[Task[i], Participant[i]] * normalizedLogN[i]);
      sigma[i] = k[Task[i], Participant[i]] * DotsN[i] ^ a[Task[i], Participant[i]];
    }
  }
}

model {
  mu_alp_lmb_a_k[1:2] ~ normal(logit(0.8), 0.5); // alpha
  mu_alp_lmb_a_k[3:8] ~ normal(0, 1);            // lambda, a, k
  l_rho_alp_lmb_a_k ~ lkj_corr_cholesky(2);
  sigma_alp_lmb_a_k ~ exponential(10);
  to_vector(z_alp_lmb_a_k) ~ normal(0, 1);

  Response ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  for(i in 1:N) {
    log_lik[i] = normal_lpdf(Response[i] | mu[i], sigma[i]);
  }
}
