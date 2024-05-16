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
  real mu; 
  real<lower=0> sigma;
  vector[ParticipantsN] z;
}

transformed parameters {
  vector[ParticipantsN] p = inv_logit(mu + sigma * z);
}

model {
  mu ~ normal(logit(0.75), 1);
  sigma ~ exponential(10);
  z ~ normal(0, 1);

  for(i in 1:N) Ncorrect[i] ~ binomial(Ntotal[i], p[Participant[i]]);
}

generated quantities {
  vector[N] log_lik;
  
  for(i in 1:N) log_lik[i] = binomial_lpmf(Ncorrect[i] | Ntotal[i], p[Participant[i]]);
}
