data {
  int<lower=1> N;
  int<lower=1> OrdersN;
  int<lower=1> ParticipantsN;
  
  array[N] int<lower=0> Ncorrect;
  array[N] int<lower=0> Ntotal;
  array[N] int<lower=1, upper=OrdersN> Order;
  array[N] int<lower=1, upper=ParticipantsN> Participant;
}

parameters {
  vector[OrdersN] mu; 
  cholesky_factor_corr[OrdersN] l_rho;
  vector<lower=0>[OrdersN] sigma;
  matrix[OrdersN, ParticipantsN] z;
}

transformed parameters {
  matrix[OrdersN, ParticipantsN] p = inv_logit(rep_matrix(mu, ParticipantsN) + diag_pre_multiply(sigma, l_rho) * z);
}

model {
  mu ~ normal(logit(0.75), 1);
  l_rho ~ lkj_corr_cholesky(2);
  sigma ~ exponential(10);
  to_vector(z) ~ normal(0, 1);

  for(i in 1:N) Ncorrect[i] ~ binomial(Ntotal[i], p[Order[i], Participant[i]]);
}

generated quantities {
  vector[N] log_lik;
  
  for(i in 1:N) log_lik[i] = binomial_lpmf(Ncorrect[i] | Ntotal[i], p[Order[i], Participant[i]]);
}
