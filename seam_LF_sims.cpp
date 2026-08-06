#include <TMB.hpp>

template <class Type>
Type sqr(Type x) {
  return x * x;
}

// Function for detecting NAs
template<class Type>
bool isNA(Type x){
  return R_IsNA(asDouble(x));
}


template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace R_inla;
  using namespace density;
  using namespace Eigen;
  
  //Data
  DATA_VECTOR(I); //commercial biomass survey index by tow (dim n_i)
  DATA_VECTOR(IR); //recruit survey index by tow (dim n_i)
  DATA_VECTOR(prop); //proportion of shrimp <15 mm CL in weight
  DATA_VECTOR(area); //area covered by each knot (dim n_s)
  DATA_MATRIX(C); //commercial catch (dim n_s,n_t)
  DATA_VECTOR(n_tows); //number of tows in year (dim n_t)
  DATA_VECTOR(pos_tows_I); //number of tows that captured commercial biomass (dim n_t)
  DATA_VECTOR(pos_tows_IR); //number of tows that captured recruits (dim n_t)
  
  //Sizes
  DATA_INTEGER(n_i); //Number of observations per year
  DATA_INTEGER(n_t); //Number of years
  DATA_INTEGER(n_s); //Number of knots
  DATA_INTEGER(n_m); //number of vertex in mesh
  
  //Indices
  DATA_FACTOR(s_i); //Indexing for knot (dim n_i)
  DATA_FACTOR(t_i); //Indexing for year (dim n_i)
  DATA_FACTOR(v_i); //Indexing for specific mesh location to match knots with right vertex (dim n_m)
  DATA_FACTOR(s_a); //Indexing knot for area for growth rates
  
  //SPDE objects
  // DATA_STRUCT(spde, spde_aniso_t); //INLA anisotropic mesh structure
  DATA_STRUCT(spde,spde_t);
  
  //Fixed covariates
  DATA_MATRIX(gI); //commercial biomass growth
  DATA_MATRIX(gR); //recruitment growth
  
  //Include prior
  DATA_INTEGER(prior_q);
  DATA_INTEGER(prior_qR);
  
  //Lower bound for qIs and qR
  DATA_SCALAR(inf_qI);
  DATA_SCALAR(inf_qR);
  
  
  //For simulations, which commercial catch approach:
  DATA_INTEGER(sim_C_choice); //Slot 0: no exploitation, 1: proportionally across whole space, 2: proportionally over above average knots, 3: proportionally over above average knots with extra in others
  
  //Parameters
  PARAMETER(log_sigma_epsilon); //obs sd survey index commercial biomass
  PARAMETER(log_sigma_upsilon); //obs sd survey index recruits
  PARAMETER(log_kappa_B);//commercial size range parameter
  PARAMETER(log_tau_B); //commercial size spatio-temporal variability parameter
  PARAMETER(log_kappa_R); //recruit range parameter
  PARAMETER(log_tau_R); //recruit spatio-temporal variability parameter
  PARAMETER(log_R0);//initial recruit mean value
  PARAMETER(log_B0);//initial commercial biomass mean value
  PARAMETER(log_m0);//initial mortality mean value
  PARAMETER(logit_p_I); //probability of capturing commercial biomass
  PARAMETER(logit_p_IR); //probability of capturing recruits
  PARAMETER_VECTOR(log_qI); //commercial biomass catchability
  PARAMETER(log_qR); //recruit catchability
  PARAMETER(log_sigma_m); //natural mortality variance
  PARAMETER(log_sd_prop);
  PARAMETER(beta); //intercept for beta regression on LF
  PARAMETER(beta_eff); //coefficient for e^(-m)*gR*R for beta regression on LF
  
  Type sd_prop = exp(log_sd_prop);
  Type pi = 3.141593;
  Type Range_B = sqrt(8)/exp(log_kappa_B);
  Type Range_R = sqrt(8)/exp(log_kappa_R); 
  Type sigma_epsilon = exp(log_sigma_epsilon);
  Type sigma_upsilon = exp(log_sigma_upsilon);
  Type qR = exp(log_qR);
  vector <Type> qI(n_s); qI = exp(log_qI);
  Type kappa_B = exp(log_kappa_B);
  Type tau_B = exp(log_tau_B);
  Type kappa_R = exp(log_kappa_R);
  Type tau_R = exp(log_tau_R);
  Type R0 = exp(log_R0);
  Type B0 = exp(log_B0);
  Type m0 = exp(log_m0);
  Type p_I = invlogit(logit_p_I);
  Type p_IR = invlogit(logit_p_IR);
  Type sigma_m = exp(log_sigma_m);
  
  
  // Calculate marginal field variances based on relation described in Lindgren et al., 2012
  Type SigmaO_B = 1 / (sqrt(4*pi)*exp(log_tau_B)*exp(log_kappa_B));
  Type SigmaO_R = 1 / (sqrt(4*pi)*exp(log_tau_R)*exp(log_kappa_R));
  
  //Random Effects
  PARAMETER_ARRAY(omega_B);
  PARAMETER_ARRAY(omega_R);
  PARAMETER_VECTOR(log_m);
  
  // Set up matrices for processes of interest
  matrix <Type> log_B(n_s,(n_t+1));
  matrix <Type> B(n_s,(n_t+1));
  matrix <Type> resid_B(n_s,(n_t+1));
  matrix <Type> areaB(n_s,(n_t+1));
  matrix <Type> areaR(n_s,(n_t));
  matrix <Type> log_R(n_s,(n_t));
  matrix <Type> R(n_s,(n_t));
  vector <Type> m(n_t+1);
  vector <Type> resid_I(n_i);
  
  //Setup for simulations and derived values
  vector <Type> bern_I(n_i);
  vector <Type> bern_IR(n_i);
  vector <Type> log_totB(n_t+1); log_totB.setZero();
  vector <Type> log_totR(n_t); log_totR.setZero();
  vector <Type> totB(n_t+1); totB.setZero();
  vector <Type> totR(n_t); totR.setZero();
  matrix <Type> mean_pro_B(n_s,n_t+1);
  
  // ----------------------------------------------
  // nll
  vector <Type> nll_comp(15); nll_comp.setZero();
  
  //Set up GMRF for commercial size biomass
  SparseMatrix <Type> Q_B = Q_spde(spde, kappa_B);
  for (int t = 0; t < (n_t+1); t++){
    if (t == 0) {nll_comp(0) += SCALE(GMRF(Q_B),1/tau_B)(omega_B.col(t));}
    if (t >= 1) {nll_comp(0) += SCALE(GMRF(Q_B),1/tau_B)(omega_B.col(t));}
  }
  
  SIMULATE {
    SparseMatrix <Type> Q_B = Q_spde(spde, kappa_B);
    density::GMRF_t<Type> gmrf1(Q_B);
    for (int t = 0; t < (n_t+1); t++){
      vector <Type> temp_omega(n_m);
      SCALE(gmrf1,1/tau_B).simulate(temp_omega);
      omega_B.col(t)=temp_omega;
    }
    REPORT(Q_B);
    REPORT(omega_B);
  }
  
  //Set up GMRF for recruits
  SparseMatrix <Type> Q_R = Q_spde(spde, kappa_R);
  for (int t = 0; t < (n_t); t++){
    nll_comp(1) += SCALE(GMRF(Q_R),1/tau_R)(omega_R.col(t));
  }
  
  SIMULATE {
    SparseMatrix <Type> Q_R = Q_spde(spde, kappa_R);
    density::GMRF_t<Type> gmrf2(Q_R);
    for (int t = 0; t < n_t; t++){
      vector <Type> temp_omega(n_m);
      SCALE(gmrf2,1/tau_R).simulate(temp_omega);
      omega_R.col(t)=temp_omega;
    }
    REPORT(Q_R);
    REPORT(omega_R);
  }
  
  //Derived values
  nll_comp(2) -= dnorm(log_m(0), log(m0)-sqr(sigma_m)/2.0, sigma_m, true);
  m(0) = exp(log_m(0));
  for (int t = 1; t < (n_t); t++){
    nll_comp(2) -= dnorm(log_m(t), log(m0)-sqr(sigma_m)/2.0, sigma_m, true);
    m(t) = exp(log_m(t));
  }
  nll_comp(2) -= dnorm(log_m(n_t), log(m0)-sqr(sigma_m)/2.0, sigma_m, true);
  m(n_t) = exp(log_m(n_t));
  
  SIMULATE{
    for (int t = 0; t < (n_t+1); t++){
      log_m(t) = log(m0)-sqr(sigma_m)/2.0+rnorm(Type(0.0),sigma_m);
      m(t) = exp(log_m(t));
    }
    REPORT(log_m);
    REPORT(m);
  }
  
  //Recruit derivation
  for (int s = 0; s < n_s; s++){
    log_R(s,0) = log(R0 * exp(omega_R(v_i(s),0)));
    R(s,0) = exp(log_R(s,0));
    for (int t = 1; t < (n_t); t++){
      log_R(s,t) = log(R(s,t-1)*exp(omega_R(v_i(s),t)));
      R(s,t) = exp(log_R(s,t));
    }
  }
  
  //Simulate recruitment
  SIMULATE{
    for (int s = 0; s < n_s; s++){
      R(s,0) = R0 * exp(omega_R(v_i(s),0));
      log_R(s,0) = log(R(s,0));
      for (int t = 1; t < (n_t); t++){
        R(s,t) = R(s,t-1)*exp(omega_R(v_i(s),t));
        log_R(s,t) = log(R(s,t));
      }
    }
    REPORT(R);
    REPORT(log_R);
  }
  
  
  //Biomass derivation
  for (int s = 0; s < n_s; s++) {
    log_B(s,0) = log(B0*exp(omega_B(v_i(s),0)));
    B(s,0) = exp(log_B(s,0));
    resid_B(s,0) = B(s,0) - exp(log_B(s,0));
    for (int t = 1; t < (n_t); t++) {
      Type mean_pro = (exp(-m(t))*gI(s_a(s),t-1)*(B(s, t - 1) - C(s,t-1)) + exp(-m(t))*gR(s_a(s),t-1)*R(s,t-1))*exp(omega_B(v_i(s),t));
      log_B(s,t) = log(mean_pro);
      B(s,t) = exp(log_B(s,t));
      resid_B(s,t) = log_B(s,t) - log(mean_pro);
    }
    //Predict 1 year ahead, assume same spatial pattern as last year but propagate error
    Type mean_pro =(exp(-m(n_t))*gI(s_a(s),n_t-1)*(B(s, (n_t - 1)) - C(s,(n_t-1))) + exp(-m(n_t))*gR(s_a(s),n_t-1)*R(s,(n_t-1)))*exp(omega_B(v_i(s),n_t));
    log_B(s,n_t) = log(mean_pro);
    B(s,n_t) = exp(log_B(s,n_t));
    resid_B(s,n_t) = log_B(s,n_t) - log(mean_pro);
  }
  
  //Simulate biomass and commercial catch
  SIMULATE {
    if (sim_C_choice==0) {
      vector <Type> counter_B(n_t); counter_B.setZero();
      for (int t = 0; t < n_t; t++){
        for (int s = 0; s < n_s; s++){
          if (t == 0){
            mean_pro_B(s,t) = B0 * exp(omega_B(v_i(s),t));
            B(s,t) = exp(log(mean_pro_B(s,t)));
            log_B(s,t) = log(B(s,t));
            counter_B(t) = counter_B(t) + (B(s,t)*area(s));
          }
          else {
            C(s,t-1) = Type(0.0);
            mean_pro_B(s,t) = (exp(-m(t))*gI(s_a(s),t-1)*(B(s, t - 1) - C(s,t-1)) + exp(-m(t))*gR(s_a(s),t-1)*R(s,t-1))*exp(omega_B(v_i(s),t));
            B(s,t) = exp(log(mean_pro_B(s,t)));
            log_B(s,t) = log(B(s,t));
            counter_B(t) = counter_B(t) + (B(s,t)*area(s));
          }
        }
      }
      for (int s = 0; s < n_s; s++){
        C(s,n_t-1) = Type(0.0);
        mean_pro_B(s,n_t) = (exp(-m(n_t))*gI(s_a(s),n_t-1)*(B(s, n_t - 1) - C(s,n_t-1)) + exp(-m(n_t))*gR(s_a(s),n_t-1)*R(s,n_t-1))*exp(omega_B(v_i(s),n_t));
        B(s,n_t) = mean_pro_B(s,n_t);
        log_B(s,n_t) = log(B(s,n_t));
        C(s,n_t) = 0;
      }
    }
    if (sim_C_choice==1) {
      vector <Type> counter_B(n_t); counter_B.setZero();
      for (int t = 0; t < n_t; t++){
        for (int s = 0; s < n_s; s++){
          if (t == 0){
            mean_pro_B(s,t) = B0 * exp(omega_B(v_i(s),t));
            B(s,t) = exp(log(mean_pro_B(s,t)));
            log_B(s,t) = log(B(s,t));
            counter_B(t) = counter_B(t) + (B(s,t)*area(s));
          }
          else {
            vector <Type> which_B(n_s); which_B.setZero();
            vector <Type> prev_B(n_s); prev_B = B.col(t-1);
            Type sum_prop; sum_prop = 0;
            Type num_B; num_B = 0;
            for (int i = 0; i < n_s; i++){
              num_B += 1;
              sum_prop = sum_prop + (prev_B(i)*area(i));
            }
            
            Type prop_B; prop_B = (B(s,t-1)*area(s))/sum_prop;
            Type exploitation =  exp(log(((counter_B(t-1)/1000)*0.1)))*exp(rnorm(Type(0.0),Type(0.2)))*1000;
            C(s,t-1) = exp(log((exploitation*prop_B)/area(s)) + rnorm(Type(0.0),Type(0.2)));
            mean_pro_B(s,t) = (exp(-m(t))*gI(s_a(s),t-1)*(B(s, t - 1) - C(s,t-1)) + exp(-m(t))*gR(s_a(s),t-1)*R(s,t-1))*exp(omega_B(v_i(s),t));
            B(s,t) = exp(log(mean_pro_B(s,t)));
            log_B(s,t) = log(B(s,t));
            counter_B(t) = counter_B(t) + (B(s,t)*area(s));
          }
        }
      }
      for (int s = 0; s < n_s; s++){
        vector <Type> which_B(n_s); which_B.setZero();
        vector <Type> prev_B(n_s); prev_B = B.col(n_t-1);
        Type sum_prop; sum_prop = 0;
        Type num_B; num_B = 0;
        for (int i = 0; i < n_s; i++){
          num_B += 1;
          sum_prop = sum_prop + (prev_B(i)*area(s));
        }
        Type prop_B; prop_B = (B(s,n_t-1)*area(s))/sum_prop;
        Type exploitation = exp(log(((counter_B(n_t-1)/1000)*0.1)))*exp(rnorm(Type(0.0),Type(0.2)))*1000;
        C(s,n_t-1) = exp(log((exploitation*prop_B)/area(s)) + rnorm(Type(0.0),Type(0.2)));
        mean_pro_B(s,n_t) = (exp(-m(n_t))*gI(s_a(s),n_t-1)*(B(s, n_t - 1) - C(s,n_t-1)) + exp(-m(n_t))*gR(s_a(s),n_t-1)*R(s,n_t-1))*exp(omega_B(v_i(s),n_t));
        B(s,n_t) = mean_pro_B(s,n_t);
        log_B(s,n_t) = log(B(s,n_t));
        C(s,n_t) = 0;
      }
    }
    if (sim_C_choice==2) {
      vector <Type> counter_B(n_t); counter_B.setZero();
      for (int t = 0; t < n_t; t++){
        for (int s = 0; s < n_s; s++){
          if (t == 0){
            mean_pro_B(s,t) = B0 * exp(omega_B(v_i(s),t));
            B(s,t) = exp(log(mean_pro_B(s,t)));
            log_B(s,t) = log(B(s,t));
            counter_B(t) = counter_B(t) + (B(s,t)*area(s));
          }
          else {
            vector <Type> which_B(n_s); which_B.setZero();
            vector <Type> prev_B(n_s); prev_B = B.col(t-1);
            Type sum_prop; sum_prop = 0;
            Type num_B; num_B = 0;
            for (int i = 0; i < n_s; i++){
              if ((prev_B(i)*area(i)) >  (counter_B(t-1)/n_s)) {
                num_B += 1;
                sum_prop = sum_prop + (prev_B(i)*area(i));
              }
            }
            
            Type prop_B; prop_B = (B(s,t-1)*area(s))/sum_prop;
            Type exploitation =  exp(log(((counter_B(t-1)/1000)*0.1)))*exp(rnorm(Type(0.0),Type(0.2)))*1000;
            if ( (B(s,t-1)*area(s)) > (counter_B(t-1)/n_s) ) C(s,t-1) = (exp(log((exploitation/area(s))*prop_B))+ rnorm(Type(0.0),Type(0.2)));
            else C(s,t-1) = Type(0.0);
            mean_pro_B(s,t) = (exp(-m(t))*gI(s_a(s),t-1)*(B(s, t - 1) - C(s,t-1)) + exp(-m(t))*gR(s_a(s),t-1)*R(s,t-1))*exp(omega_B(v_i(s),t));
            B(s,t) = exp(log(mean_pro_B(s,t)));
            log_B(s,t) = log(B(s,t));
            counter_B(t) = counter_B(t) + (B(s,t)*area(s));
          }
        }
      }
      for (int s = 0; s < n_s; s++){
        vector <Type> which_B(n_s); which_B.setZero();
        vector <Type> prev_B(n_s); prev_B = B.col(n_t-1);
        Type sum_prop; sum_prop = 0;
        Type num_B; num_B = 0;
        for (int i = 0; i < n_s; i++){
          if ((prev_B(i)*area(i)) >  (counter_B(n_t-1)/n_s)) {
            num_B += 1;
            sum_prop = sum_prop + (prev_B(i)*area(s));
          }
        }
        Type prop_B; prop_B = (B(s,n_t-1)*area(s))/sum_prop;
        Type exploitation = exp(log(((counter_B(n_t-1)/1000)*0.1)))*exp(rnorm(Type(0.0),Type(0.2)))*1000;
        if ( (B(s,n_t-1)*area(s)) > (counter_B(n_t-1)/n_s) ) C(s,n_t-1) = exp(log((exploitation/area(s))*prop_B) + rnorm(Type(0.0),Type(0.2)));
        else C(s,n_t-1) = Type(0.0);
        mean_pro_B(s,n_t) = (exp(-m(n_t))*gI(s_a(s),n_t-1)*(B(s, n_t - 1) - C(s,n_t-1)) + exp(-m(n_t))*gR(s_a(s),n_t-1)*R(s,n_t-1))*exp(omega_B(v_i(s),n_t));
        B(s,n_t) = mean_pro_B(s,n_t);
        log_B(s,n_t) = log(B(s,n_t));
        C(s,n_t) = 0;
      }
    }
    if (sim_C_choice==3) {
      vector <Type> counter_B(n_t); counter_B.setZero();
      for (int t = 0; t < n_t; t++){
        for (int s = 0; s < n_s; s++){
          if (t == 0){
            mean_pro_B(s,t) = B0 * exp(omega_B(v_i(s),t));
            B(s,t) = exp(log(mean_pro_B(s,t)));
            log_B(s,t) = log(B(s,t));
            counter_B(t) = counter_B(t) + (B(s,t)*area(s));
          }
          else {
            vector <Type> which_B(n_s); which_B.setZero();
            vector <Type> prev_B(n_s); prev_B = B.col(t-1);
            Type sum_prop; sum_prop = 0;
            Type num_B; num_B = 0;
            for (int i = 0; i < n_s; i++){
              if ((prev_B(i)*area(i)) >  (counter_B(t-1)/n_s)) {
                num_B += 1;
                sum_prop = sum_prop + (prev_B(i)*area(i));
              }
            }
            
            Type prop_B; prop_B = (B(s,t-1)*area(s))/sum_prop;
            Type exploitation =  exp(log(((counter_B(t-1)/1000)*0.1)))*exp(rnorm(Type(0.0),Type(0.2)))*1000;
            if ( (B(s,t-1)*area(s)) > (counter_B(t-1)/n_s) ) C(s,t-1) = (exp(log((exploitation/area(s))*prop_B))+ rnorm(Type(0.0),Type(0.2)));
            else C(s,t-1)=B(s,t-1)*0.02*exp(rnorm(Type(0.0),Type(0.2)));
            mean_pro_B(s,t) = (exp(-m(t))*gI(s_a(s),t-1)*(B(s, t - 1) - C(s,t-1)) + exp(-m(t))*gR(s_a(s),t-1)*R(s,t-1))*exp(omega_B(v_i(s),t));
            B(s,t) = exp(log(mean_pro_B(s,t)));
            log_B(s,t) = log(B(s,t));
            counter_B(t) = counter_B(t) + (B(s,t)*area(s));
          }
        }
      }
      for (int s = 0; s < n_s; s++){
        vector <Type> which_B(n_s); which_B.setZero();
        vector <Type> prev_B(n_s); prev_B = B.col(n_t-1);
        Type sum_prop; sum_prop = 0;
        Type num_B; num_B = 0;
        for (int i = 0; i < n_s; i++){
          if ((prev_B(i)*area(i)) >  (counter_B(n_t-1)/n_s)) {
            num_B += 1;
            sum_prop = sum_prop + (prev_B(i)*area(s));
          }
        }
        Type prop_B; prop_B = (B(s,n_t-1)*area(s))/sum_prop;
        Type exploitation = exp(log(((counter_B(n_t-1)/1000)*0.1)))*exp(rnorm(Type(0.0),Type(0.2)))*1000;
        if ( (B(s,n_t-1)*area(s)) > (counter_B(n_t-1)/n_s) ) C(s,n_t-1) = exp(log((exploitation/area(s))*prop_B) + rnorm(Type(0.0),Type(0.2)));
        else C(s,n_t-1)=B(s,n_t-1)*0.02*exp(rnorm(Type(0.0),Type(0.2)));
        mean_pro_B(s,n_t) = (exp(-m(n_t))*gI(s_a(s),n_t-1)*(B(s, n_t - 1) - C(s,n_t-1)) + exp(-m(n_t))*gR(s_a(s),n_t-1)*R(s,n_t-1))*exp(omega_B(v_i(s),n_t));
        B(s,n_t) = mean_pro_B(s,n_t);
        log_B(s,n_t) = log(B(s,n_t));
        C(s,n_t) = 0;
      }
    }
    REPORT(B);
    REPORT(log_B);
    REPORT(C);
  }
  
  
  //Calculating predicted biomass and recruitment over area covered by each knots
  //Commercial biomass
  for (int s = 0; s < n_s; s++){
    for (int t = 0; t < (n_t+1); t++){
      areaB(s,t) = B(s,t) * area(s);
    }
  }
  
  //Recruits
  for (int s = 0; s < n_s; s++){
    for (int t = 0; t < (n_t); t++){
      areaR(s,t) = R(s,t) * area(s);
    }
  }
  
  //Totals
  //Biomass
  for (int t = 0; t < (n_t+1); t++){
    for (int s = 0; s < n_s; s++){
      totB(t) += areaB(s,t)/1000;
    } 
    log_totB(t) = log(totB(t));
  }
  
  //Recruits
  for (int t = 0; t < (n_t); t++){
    for (int s = 0; s < n_s; s++){
      totR(t) += areaR(s,t)/1000;
    } 
    log_totR(t) = log(totR(t));
  }
  
  // Observation equations
  
  //Probability of capturing commercial biomass
  for (int t = 0; t <n_t; t++){
    nll_comp[5] -= dbinom_robust(pos_tows_I(t), n_tows(t), logit(p_I),true);
  }
  
  SIMULATE{
    for (int t = 0; t < n_t; t++){
      pos_tows_I(t) = rbinom(n_tows(t),p_I);
    }
    REPORT(pos_tows_I);
  }
  
  // Set prior on q
  if (prior_qR == 1){
    nll_comp[12] -= dbeta((qR-inf_qR),Type(2.0),Type(10.0),true);
  }
  
  // Set prior on q
  if (prior_q == 1){
    for (int s = 0; s < n_s; s++){
      nll_comp[12] -= dbeta((qI(s)-inf_qI),Type(2.0),Type(10.0),true);
    }
  }
  
  // Commercial Index
  for (int i = 0; i < n_i; i++){
    if( !isNA(I(i) )) {
      Type mean_B = qI(s_i(i))*B(s_i(i),t_i(i))/p_I;
      nll_comp(6) -= dnorm(log(I(i)), log (mean_B)- sqr(sigma_epsilon)/Type(2.0), sigma_epsilon, true);
      resid_I(i) = log(I(i)) - log(mean_B);
    }
  }
  
  SIMULATE{
    for (int i = 0; i < n_i; i++){
      Type mean_I = qI(s_i(i))*B(s_i(i),t_i(i))/p_I;
      bern_I(i) = rbinom(Type(1.0),p_I);
      if (bern_I(i)>0) I(i) = exp(log(mean_I) - (sqr(sigma_epsilon)/2.0) + rnorm(Type(0.0),sigma_epsilon));
      else I(i) = NA_REAL;
    }
    REPORT(bern_I);
    REPORT(I);
  }
  
  //Probability of capturing recruits
  for (int t = 0; t <n_t; t++){
    if (!isNA(pos_tows_IR(t))){
      nll_comp[7] -= dbinom_robust(pos_tows_IR(t), n_tows(t), logit(p_IR),true); 
    }
  }
  
  SIMULATE {
    DATA_FACTOR(sim_IR_pres);//Vector of years with absence of recruitment observations
    for (int t = 0; t < n_t; t++){
      Type check1 = 0;
      if (!isNA(sim_IR_pres(0))){
        for (int sim_t = 0; sim_t < sim_IR_pres.size(); sim_t++){
          if (t == sim_IR_pres(sim_t)) check1 = 1;
        } 
      }
      if (check1==1){
        pos_tows_IR(t) = NA_REAL;
      } else {
        pos_tows_IR(t) = rbinom(n_tows(t),p_IR);
      }
    }
    REPORT(pos_tows_IR);
  }
  
  // Recruit Index
  for (int i = 0; i < n_i; i++){
    if ( !isNA(IR(i) )) {
      Type mean_R = qR*R(s_i(i),t_i(i))/p_IR;
      nll_comp(8) -= dnorm(log(IR(i)), log(mean_R)-sqr(sigma_upsilon)/Type(2.0), sigma_upsilon, true);
    }
  }
  
  SIMULATE{
    DATA_IVECTOR(sim_IR_pres);//Vector of years with absence of recruitment observations
    for (int i = 0; i < n_i; i++){
      Type mean_IR = qR*R(s_i(i),t_i(i))/p_IR;
      bern_IR(i) = rbinom(Type(1.0),p_IR);
      Type check = 0;
      if (!isNA(sim_IR_pres(0))){
        for (int sim_t = 0; sim_t < sim_IR_pres.size(); sim_t++){
          if (t_i(i) == sim_IR_pres(sim_t)) check = 1;
        } 
      }
      if (check == 1) IR(i) = NA_REAL;
      else if(bern_IR(i)>0) IR(i) = exp(log(mean_IR) - (sqr(sigma_upsilon)/2.0) + rnorm(Type(0.0),sigma_upsilon));
      else IR(i) = NA_REAL;
    }
    REPORT(bern_IR);
    REPORT(IR);
  }
  
  //Using the length-based approach
  //Take proportion of small shrimp and use it to estimate recruitment and mortality of year before that
  //Requires replacing first year's proportions with NA
  //Using the GLMM idea, where we model E[X], and for beta that means we need to specify the expectation
  //using R and m, then use the shapes for it
  
  //// Ferrari and Cribari-Neto 2004; betareg package; same formulation
  // The parameters of this formulation (phi and betas) are not orthogonal
  // Important to note as this means we don't want to overinterpret them
  
  vector <Type> mu_i(n_i); mu_i.setZero();
  
  SIMULATE{
    for (int i = 0; i < n_i; i++){
      if (t_i(i)==0){
        prop(i) = NA_REAL;
      } else {
        mu_i(i) = beta + beta_eff*log(R(s_i(i),t_i(i)-1)*exp(-m(t_i(i)))*gR(s_a(s_i(i)),t_i(i)-1)/(B(s_i(i),t_i(i))+C(s_i(i),t_i(i)-1)));
        Type logit_prop = rnorm(mu_i(i),sd_prop);
        prop(i) = invlogit(logit_prop);
      }
    }
    REPORT(prop);
  }
  
  for (int i = 0; i < n_i; i++){
    if ( (!isNA(prop(i)) )){
      mu_i(i) = beta + beta_eff*log(R(s_i(i),t_i(i)-1)*exp(-m(t_i(i)))*gR(s_a(s_i(i)),t_i(i)-1)/(B(s_i(i),t_i(i))+C(s_i(i),t_i(i)-1)));
      nll_comp(4) -= dnorm(logit(prop(i)),mu_i(i),sd_prop,true);
    }
  }
  REPORT(mu_i);
  
  //Reporting
  ADREPORT(kappa_B);
  ADREPORT(tau_B);
  ADREPORT(kappa_R);
  ADREPORT(tau_R);
  ADREPORT(sigma_epsilon);
  ADREPORT(sigma_upsilon);
  ADREPORT(R0);
  ADREPORT(B0);
  ADREPORT(m0);
  ADREPORT(qI);
  ADREPORT(qR);
  ADREPORT(p_I);
  ADREPORT(p_IR);
  ADREPORT(beta);
  ADREPORT(beta_eff);
  ADREPORT(sd_prop);
  REPORT(SigmaO_B);
  REPORT(SigmaO_R);
  REPORT(Range_B);
  REPORT(Range_R);
  REPORT(omega_R);
  REPORT(omega_B);
  REPORT(log_B);
  REPORT(B);
  REPORT(areaB);
  REPORT(log_R);
  REPORT(R);
  REPORT(areaR);
  REPORT(log_m);
  REPORT(m);
  ADREPORT(log_m);
  REPORT(totB);
  REPORT(totR);
  REPORT(log_totB);
  REPORT(log_totR);
  ADREPORT(log_totB);
  ADREPORT(log_totR);
  ADREPORT(SigmaO_B);
  ADREPORT(SigmaO_R);
  REPORT(resid_B);
  REPORT(resid_I);
  REPORT(nll_comp);
  
  Type nll = nll_comp.sum();
  return nll;
}
