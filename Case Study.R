# Logit function
logitp=function(p){log(p/(1-p))}
# Inverse logist function
logitpi=function(t){exp(t)/(1+exp(t))}

library(sf)
library(plyr)
library(dplyr)
library(gstat)
library(ggplot2)
library(TMB)
library(INLA)

compile("shrimp_seam_length_method_1m.cpp")
dyn.load(dynlib("shrimp_seam_length_method_1m"))

NY<-length(1995:2025)

load("case_study_dat.RData")

par<-list()
par$log_sigma_epsilon<-0
par$log_sigma_upsilon<-0
par$log_kappa_B<--3
par$log_tau_B<-3
par$log_kappa_R<--3
par$log_tau_R<-3
#Fixing qI at 1, so must all be ==1 here
par$log_qI<-rep(log(1),tmb_data$n_s)
par$log_qR<-log(0.7)
par$log_sigma_m<--1
par$log_phi<-0
par$beta<--2
par$beta_eff<-0


#Due to delay difference formulation, starting values for R0 and B0 must be much bigger than expected
#Slows down the model but if too low, optimizer won't run because it introduces NAs due to presence of landings
#If running without landings can be smaller
par$log_R0<-8
par$log_B0<-12
par$log_m0<-log(0.4)
par$logit_p_I<-2
par$logit_p_IR<-1
par$log_sd_prop<--1


#Random effects
par$omega_B<-matrix(rep(0,mesh$n*((tmb_data$n_t+1))),ncol=(tmb_data$n_t+1))
par$omega_R<-matrix(rep(0,mesh$n*(tmb_data$n_t)),ncol=(tmb_data$n_t))
par$log_m<-rep(log(0.4),(NY+1))

random<-c("omega_R","omega_B","log_m","log_qI")
#Maps is where parameters are fixed
#This fixes qI at the values set above (i.e., 1)
maps = list(log_qI=as.factor(rep(NA,tmb_data$n_s)))

time1<-Sys.time()
obj_list = MakeADFun(data=tmb_data,
                     parameters=par,
                     map=maps,
                     random=random,
                     DLL="shrimp_seam_length_method_1m",
                     silent = F)

#Optimizing using maximum likelihood
Opt_list<-optimx::optimr(obj_list$par,
                         obj_list$fn,
                         obj_list$gr,
                         method="nlminb")

#Forcing it to keep optimizing, presence of super large amount of catches slows it down drastically
while (Opt_list$message=="iteration limit reached without convergence (10)"){
  obj_list$par<-obj_list$env$last.par.best[-which(names(obj_list$env$last.par.best)%in% random)]
  Opt_list<-optimx::optimr(obj_list$par,obj_list$fn,obj_list$gr,control=list(rel.tol=1e-8),method="nlminb")
}

#Calculating standard errors for parameters and values of interest, i.e. biomass
rep_list<-sdreport(obj_list,bias.correct=F)

#Reporting estimates of specific outcomes
Report_list<-obj_list$report()
time2<-Sys.time()

#For BFGS if interested

Opt_list<-optimx::optimr(obj_list$par,
                         obj_list$fn,
                         obj_list$gr,
                         method="BFGS")

#Forcing it to keep optimizing, presence of super large amount of catches slows it down drastically
while (Opt_list$message=="iteration limit reached without convergence (10)"){
  obj_list$par<-obj_list$env$last.par.best[-which(names(obj_list$env$last.par.best)%in% random)]
  Opt_list<-optimx::optimr(obj_list$par,obj_list$fn,obj_list$gr,method="BFGS")
}

#Calculating standard errors for parameters and values of interest, i.e. biomass
rep_list<-sdreport(obj_list,bias.correct=F)

#Reporting estimates of specific outcomes
Report_list<-obj_list$report()
time2<-Sys.time()


