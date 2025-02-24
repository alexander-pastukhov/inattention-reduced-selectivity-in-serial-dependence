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
  vector[N] P = to_vector(DotsN) ./ Nmax;
  vector[N] R = to_vector(Response) ./ Nmax;
}

parameters {
  vector[3 * TaskN] mu_scale_a_b;
  cholesky_factor_corr[3 * TaskN] l_rho_scale_a_b;
  vector<lower=0>[3 * TaskN] sigma_scale_a_b;
  matrix[3 * TaskN, ParticipantsN] z_scale_a_b;
  
  vector[4 * TaskN] mu_wmax_pmax_kappa;
  cholesky_factor_corr[4 * TaskN] l_rho_wmax_pmax_kappa;
  vector<lower=0>[4 * TaskN] sigma_wmax_pmax_kappa;
  matrix[4 * TaskN, ParticipantsN] z_wmax_pmax_kappa;
}

transformed parameters {
  vector[N] mu;
  vector[N] sigma;
  {
    matrix[3 * TaskN, ParticipantsN] scale_a_b = rep_matrix(mu_scale_a_b, ParticipantsN) + diag_pre_multiply(sigma_scale_a_b, l_rho_scale_a_b) * z_scale_a_b;
    matrix[TaskN, ParticipantsN] scale = inv_logit(block(scale_a_b, 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] a = exp(block(scale_a_b, TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] b = exp(block(scale_a_b, 2 * TaskN + 1, 1, TaskN, ParticipantsN));
    
    // uncertainty
    matrix[4 * TaskN, ParticipantsN] wmax_pmax_kappa_lambda = rep_matrix(mu_wmax_pmax_kappa, ParticipantsN) + diag_pre_multiply(sigma_wmax_pmax_kappa, l_rho_wmax_pmax_kappa) * z_wmax_pmax_kappa;
    matrix[TaskN, ParticipantsN] Wmax = inv_logit(block(wmax_pmax_kappa_lambda, 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] Pmax = inv_logit(block(wmax_pmax_kappa_lambda, TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] kappa = 1 ./ exp(block(wmax_pmax_kappa_lambda, 2 * TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] lambda = exp(block(wmax_pmax_kappa_lambda, 3 * TaskN + 1, 1, TaskN, ParticipantsN));
    
    matrix[TaskN, ParticipantsN] Wmax_norm;
    for(iTask in 1:TaskN){
      for(iP in 1:ParticipantsN) {
        Wmax_norm[iTask, iP] = Wmax[iTask, iP] ./  exp(beta_proportion_lpdf(Pmax[iTask, iP] | Pmax[iTask, iP], kappa[iTask, iP]));
      }
    }
    
    real uncertainty;
    real Wprev;
    for(i in 1:N) {
      uncertainty = Wmax_norm[Task[i], Participant[i]] * exp(beta_proportion_lpdf(P[i] | Pmax[Task[i], Participant[i]], kappa[Task[i], Participant[i]]));
      if (Trial[i] == 1) {
        mu[i] = scale[Task[i], Participant[i]] * DotsN[i];
      } else {
        Wprev = uncertainty * exp(-((P[i] - R[i-1])^2) / lambda[Task[i], Participant[i]]);
        mu[i] = scale[Task[i], Participant[i]] * ((1 - Wprev) * DotsN[i] + Wprev * Response[i-1]);
      }
      
      sigma[i] = a[Task[i], Participant[i]] + b[Task[i], Participant[i]] * uncertainty;
    }
  }
}

model {
  mu_scale_a_b[1:2] ~ normal(logit(0.8), 0.5);  // scale
  mu_scale_a_b[3:4] ~ normal(0, 1);             // a
  mu_scale_a_b[5:6] ~ normal(0, 1);             // b
  l_rho_scale_a_b ~ lkj_corr_cholesky(2);
  sigma_scale_a_b ~ exponential(1);
  to_vector(z_scale_a_b) ~ normal(0, 1);
  
  mu_wmax_pmax_kappa[1:2] ~ normal(0, 1);  // Wmax
  mu_wmax_pmax_kappa[3:4] ~ normal(0, 1);  // Pmax
  mu_wmax_pmax_kappa[5:6] ~ normal(-10, 1);  // sigma , so that kappa = 1/sigma
  mu_wmax_pmax_kappa[7:8] ~ normal(0, 1);  // lambda
  l_rho_wmax_pmax_kappa ~ lkj_corr_cholesky(2);
  sigma_wmax_pmax_kappa ~ exponential(1);
  to_vector(z_wmax_pmax_kappa) ~ normal(0, 1);

  Response ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  for(i in 1:N) log_lik[i] = normal_lpdf(Response[i] | mu[i], sigma[i]);
}
