setwd("C:/Users/mcdonaldra/Documents/GitHub/Shrimp-Framework/Post-Framework Research/Extending SEAM to utilize previous years of data/")
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

compile("seam_LF_sims.cpp")
dyn.load(dynlib("seam_LF_sims"))

sim_area<-data.frame(xmin=0,ymin=0,xmax=100,ymax=100) %>% 
  st_as_sf(coords=c("xmin","ymin","xmax","ymax")) 
#First test out if the simulation code works with checkConsistency

NY<-15
n_obs<-50
set.seed(293)
for (y in 1:NY){
  if (y==1){
    sim_obs<-data.frame(geometry=st_sample(sim_area,n_obs),year=y) %>% st_as_sf()
  } else {
    sim_obs<-rbind(sim_obs,st_as_sf(data.frame(geometry=st_sample(sim_area,n_obs),year=y)))
  }
}

nknots<-15
set.seed(932)
knots<-kmeans(st_coordinates(sim_obs),nknots)
sf.knots<-st_as_sf(as.data.frame(knots$centers),coords=c("X","Y"))
ggplot()+geom_sf(data=sim_area)+geom_sf(data=sf.knots)

sim_obs$knotID<-knots$cluster

big_grid<-st_make_grid(sim_area,cellsize=1) %>% st_as_sf()
big_grid$area<-st_area(big_grid)

big_grid$knotID<-st_nearest_feature(big_grid,sf.knots)
stratarea<-aggregate(area~knotID,data=big_grid,FUN=sum)

mesh<-inla.mesh.2d(knots$centers,
                   max.edge=10,
                   boundary=inla.sp2segment(as_Spatial(sim_area)))
# plot(mesh);points(st_coordinates(sf.knots.check),col="red",pch=15)

spde<-inla.spde2.matern(mesh)
# Triangle info
Dset = 1:2
TV = mesh$graph$tv           # Triangle to vertex indexing
V0 = mesh$loc[TV[,1],Dset]   # V = vertices for each triangle
V1 = mesh$loc[TV[,2],Dset]
V2 = mesh$loc[TV[,3],Dset]
E0 = V2 - V1                      # E = edge for each triangle
E1 = V0 - V2
E2 = V1 - V0
# Calculate Areas
TmpFn = function(Vec1, Vec2) abs(det( rbind(Vec1, Vec2) ))
Tri_Area = rep(NA, nrow(E0))
for(i in 1:length(Tri_Area)) Tri_Area[i] = TmpFn( E0[i,],E1[i,] )/2   # T = area of each triangle
spde_aniso <- list(
  "n_s"      = spde$n.spde,
  "n_tri"    = nrow(TV),
  "Tri_Area" = Tri_Area,
  "E0"       = E0,
  "E1"       = E1,
  "E2"       = E2,
  "TV"       = TV - 1,
  "G0"       = spde$param.inla$M0,
  "G0_inv"   = as(as(as(diag(1/diag(spde$param.inla$M0)),"dMatrix"), "generalMatrix"), "TsparseMatrix"))


tmb_data<-list()
tmb_data$I<-rep(1,nrow(sim_obs))
tmb_data$IR<-rep(1,nrow(sim_obs))
tmb_data$prop<-rep(1,nrow(sim_obs))
tmb_data$C<-matrix(data=0,nrow=nknots,ncol=NY+1)
tmb_data$area<-as.numeric(stratarea$area)

#Indices
tmb_data$n_i<-as.numeric(length(tmb_data$I))
tmb_data$n_t<-as.numeric(NY)
tmb_data$n_s<-as.numeric(length(unique(sim_obs$knotID)))
tmb_data$n_m<-as.numeric(mesh$n)

#Factors
tmb_data$s_i<-as.integer(sim_obs$knotID-1L)
tmb_data$t_i<-as.integer(sim_obs$year-1)
tmb_data$v_i<-as.integer(mesh$idx$loc-1L)
tmb_data$s_a<-as.integer(rep(0,nknots))

tmb_data$prop[which(tmb_data$t_i==0)]<-NA

#Probability of encounters
tmb_data$n_tows<-rep(50,NY)
tmb_data$pos_tows_I<-rep(1,NY)
tmb_data$pos_tows_IR<-rep(1,NY)
# tmb_data$pos_tows_IR[1:10]<-NA

#Growths
set.seed(904231)
tmb_data$gI<-matrix(rnorm(NY+1,1.3,0.1),nrow=1)
tmb_data$gR<-matrix(rnorm(NY+1,2.7,0.4),nrow=1)

#Components for SPDE approach for isotropic random field
tmb_data$spde<-spde$param.inla[c("M0","M1","M2")]

#Choices on informing catchabilities or not
tmb_data$prior_q<-0
tmb_data$prior_qR<-1
tmb_data$inf_qI<-0.9
tmb_data$inf_qR<-0.5
# tmb_data$inf_qR<-0
tmb_data$sim_C_choice<-0
tmb_data$sim_IR_pres<-NA

#Setting up starting values for parameters
#These parameters were successful at checkConvergence! But reminder
#these models can create some real bullshit sometimes through sheer probability
par<-list()
par$log_sigma_epsilon<--1
par$log_sigma_upsilon<--1
par$log_kappa_B<--3
par$log_tau_B<-3.5
par$log_kappa_R<--4
par$log_tau_R<-4.4

par$log_R0<-4
par$log_B0<-8
par$log_m0<-log(0.3)
par$logit_p_I<-3
par$logit_p_IR<-1.5

#Fixing qI at 1, so must all be ==1 here
par$log_qI<-rep(log(1),tmb_data$n_s)
par$log_qR<-log(0.66)
par$log_sigma_m<--2
par$log_sd_prop<--2
par$beta<--0.3
par$beta_eff<-0.6

#Random effects
par$omega_B<-matrix(rep(0,mesh$n*((tmb_data$n_t+1))),ncol=(tmb_data$n_t+1))
par$omega_R<-matrix(rep(0,mesh$n*(tmb_data$n_t)),ncol=(tmb_data$n_t))
par$log_m<-rep(log(0.6),(NY+1))

random<-c("omega_R","omega_B","log_m","log_qI")
#Maps is where parameters are fixed
#This fixes qI at the values set above (i.e., 1)
maps = list(log_qI=as.factor(rep(NA,tmb_data$n_s)))

# obj_list = MakeADFun(data=tmb_data,
#                      parameters=par,
#                      map=maps,
#                      random=random,
#                      DLL="seam_LF_sims",
#                      silent = F)
# testyboi<-obj_list$simulate(complete=T)
# testyboi$nll_comp
# # hist(testyboi$omega_B)
# plot.ts(testyboi$totB)
# checkyboi<-checkConsistency(obj_list,n=100)

NY<-30
n_obs<-60
set.seed(293)
for (y in 1:NY){
  if (y==1){
    sim_obs<-data.frame(geometry=st_sample(sim_area,n_obs),year=y) %>% st_as_sf()
  } else {
    sim_obs<-rbind(sim_obs,st_as_sf(data.frame(geometry=st_sample(sim_area,n_obs),year=y)))
  }
}

nknots<-15
set.seed(932)
knots<-kmeans(st_coordinates(sim_obs),nknots)
sf.knots<-st_as_sf(as.data.frame(knots$centers),coords=c("X","Y"))
ggplot()+geom_sf(data=sim_area)+geom_sf(data=sf.knots)

sim_obs$knotID<-knots$cluster

big_grid<-st_make_grid(sim_area,cellsize=1) %>% st_as_sf()
big_grid$area<-st_area(big_grid)

big_grid$knotID<-st_nearest_feature(big_grid,sf.knots)
stratarea<-aggregate(area~knotID,data=big_grid,FUN=sum)

mesh<-inla.mesh.2d(knots$centers,
                   max.edge=10,
                   boundary=inla.sp2segment(as_Spatial(sim_area)))
# plot(mesh);points(st_coordinates(sf.knots.check),col="red",pch=15)

spde<-inla.spde2.matern(mesh)
# Triangle info
Dset = 1:2
TV = mesh$graph$tv           # Triangle to vertex indexing
V0 = mesh$loc[TV[,1],Dset]   # V = vertices for each triangle
V1 = mesh$loc[TV[,2],Dset]
V2 = mesh$loc[TV[,3],Dset]
E0 = V2 - V1                      # E = edge for each triangle
E1 = V0 - V2
E2 = V1 - V0
# Calculate Areas
TmpFn = function(Vec1, Vec2) abs(det( rbind(Vec1, Vec2) ))
Tri_Area = rep(NA, nrow(E0))
for(i in 1:length(Tri_Area)) Tri_Area[i] = TmpFn( E0[i,],E1[i,] )/2   # T = area of each triangle
spde_aniso <- list(
  "n_s"      = spde$n.spde,
  "n_tri"    = nrow(TV),
  "Tri_Area" = Tri_Area,
  "E0"       = E0,
  "E1"       = E1,
  "E2"       = E2,
  "TV"       = TV - 1,
  "G0"       = spde$param.inla$M0,
  "G0_inv"   = as(as(as(diag(1/diag(spde$param.inla$M0)),"dMatrix"), "generalMatrix"), "TsparseMatrix"))


tmb_data<-list()
tmb_data$I<-rep(1,nrow(sim_obs))
tmb_data$IR<-rep(1,nrow(sim_obs))
tmb_data$prop<-rep(1,nrow(sim_obs))
tmb_data$C<-matrix(data=0,nrow=nknots,ncol=NY+1)
tmb_data$area<-as.numeric(stratarea$area)

#Indices
tmb_data$n_i<-as.numeric(length(tmb_data$I))
tmb_data$n_t<-as.numeric(NY)
tmb_data$n_s<-as.numeric(length(unique(sim_obs$knotID)))
tmb_data$n_m<-as.numeric(mesh$n)

#Factors
tmb_data$s_i<-as.integer(sim_obs$knotID-1L)
tmb_data$t_i<-as.integer(sim_obs$year-1)
tmb_data$v_i<-as.integer(mesh$idx$loc-1L)
tmb_data$s_a<-as.integer(rep(0,nknots))

tmb_data$prop[which(tmb_data$t_i==0)]<-NA

#Probability of encounters
tmb_data$n_tows<-rep(n_obs,NY)
tmb_data$pos_tows_I<-rep(1,NY)
tmb_data$pos_tows_IR<-rep(1,NY)
# tmb_data$pos_tows_IR[1:10]<-NA

#Growths
set.seed(904231)
tmb_data$gI<-matrix(rnorm(NY+1,1.3,0.1),nrow=1)
tmb_data$gR<-matrix(rnorm(NY+1,2.7,0.4),nrow=1)

#Components for SPDE approach for isotropic random field
tmb_data$spde<-spde$param.inla[c("M0","M1","M2")]

#Choices on informing catchabilities or not
tmb_data$prior_q<-0
tmb_data$prior_qR<-1
tmb_data$inf_qI<-0.9
tmb_data$inf_qR<-0.5
# tmb_data$inf_qR<-0
tmb_data$sim_C_choice<-3
#This sets C to be ~10% at all above average density knots, and ~2% everywhere else
tmb_data$sim_IR_pres<-NA

#Setting up starting values for parameters
#These parameters were successful at checkConvergence! But reminder
#these models can create some real bullshit sometimes through sheer probability
par<-list()
par$log_sigma_epsilon<-0
par$log_sigma_upsilon<-0
par$log_kappa_B<--3
par$log_tau_B<-3
par$log_kappa_R<--4
par$log_tau_R<-4

par$log_R0<-4
par$log_B0<-8
par$log_m0<-log(0.3)
par$logit_p_I<-3
par$logit_p_IR<-1.5

#Fixing qI at 1, so must all be ==1 here
par$log_qI<-rep(log(1),tmb_data$n_s)
par$log_qR<-log(0.66)
par$log_sigma_m<--0.5
par$log_sd_prop<--0.5
par$beta<--0.3
par$beta_eff<-0.6

#Random effects
par$omega_B<-matrix(rep(0,mesh$n*((tmb_data$n_t+1))),ncol=(tmb_data$n_t+1))
par$omega_R<-matrix(rep(0,mesh$n*(tmb_data$n_t)),ncol=(tmb_data$n_t))
par$log_m<-rep(log(0.6),(NY+1))

random<-c("omega_R","omega_B","log_m","log_qI")
#Maps is where parameters are fixed
#This fixes qI at the values set above (i.e., 1)
maps = list(log_qI=as.factor(rep(NA,tmb_data$n_s)))

sim_obj = MakeADFun(data=tmb_data,
                    parameters=par,
                    map=maps,
                    random=random,
                    DLL="seam_LF_sims",
                    silent = F)
# testyboi<-sim_obj$simulate(complete=T)
# plot.ts(testyboi$m)



fit_par<-list()
fit_par$log_sigma_epsilon<-1
fit_par$log_sigma_upsilon<-1
fit_par$log_kappa_B<--2
fit_par$log_tau_B<-2
fit_par$log_kappa_R<--2
fit_par$log_tau_R<-2

fit_par$log_R0<-6
fit_par$log_B0<-9
fit_par$log_m0<-log(0.4)
fit_par$logit_p_I<-0
fit_par$logit_p_IR<-0

#Fixing qI at 1, so must all be ==1 here
fit_par$log_qI<-rep(log(1),tmb_data$n_s)
fit_par$log_qR<-log(0.8)
fit_par$log_sigma_m<-0
fit_par$log_sd_prop<-0
fit_par$beta<-0
fit_par$beta_eff<-0

#Random effects
fit_par$omega_B<-matrix(rep(0,mesh$n*((tmb_data$n_t+1))),ncol=(tmb_data$n_t+1))
fit_par$omega_R<-matrix(rep(0,mesh$n*(tmb_data$n_t)),ncol=(tmb_data$n_t))
fit_par$log_m<-rep(log(0.4),(NY+1))

random<-c("omega_R","omega_B","log_m","log_qI")
maps = list(log_qI=as.factor(rep(NA,tmb_data$n_s)))

#Experiment 1:
#Checking if works with full data
#This is the baseline, so just 1 setting, no need to test many things

sim_dat<-list()
mess_list<-list()
rep_list<-list()
Report_list<-list()
fit_time<-list()

n_sims<-200

set.seed(4718)
for (sim in 1:n_sims){
  sim_dat[[sim]]<-sim_obj$simulate(complete=T)
  
  time1<-Sys.time()
  fit_obj<-MakeADFun(data=sim_dat[[sim]],
                     parameters=fit_par,
                     map=maps,
                     random=random,
                     DLL="seam_LF_sims",
                     silent = F)
  
  fit_Opt<-optimx::optimr(fit_obj$par,
                          fit_obj$fn,
                          fit_obj$gr,
                          method="nlminb")
  
  #Forcing it to keep optimizing, presence of super large amount of catches slows it down drastically
  while (fit_Opt$message=="iteration limit reached without convergence (10)"){
    fit_obj$par<-fit_obj$env$last.par.best[-which(names(fit_obj$env$last.par.best)%in% random)]
    fit_Opt<-optimx::optimr(fit_obj$par,fit_obj$fn,fit_obj$gr,control=list(maxit=100000),method="nlminb")
  }
  mess_list[[sim]]<-fit_Opt$message
  
  #Calculating standard errors for parameters and values of interest, i.e. biomass
  rep_list[[sim]]<-sdreport(fit_obj,bias.correct=T)
  
  #Reporting estimates of specific outcomes
  Report_list[[sim]]<-fit_obj$report()
  time2<-Sys.time()
  fit_time[[sim]]<-time2-time1
}


#Setting 2:
#Checking if capable of hindcasting
#3 settings with different amounts of hindcasting: 5 years, 10 years, 15 years


sim_dat<-list(list(),list(),list())
mess_list<-list(list(),list(),list())
rep_list<-list(list(),list(),list())
Report_list<-list(list(),list(),list())
fit_time<-list(list(),list(),list())

n_sims<-200

settings<-c("5y","10y","15y")

set.seed(5649)
for (set in 1:length(settings)){
  for (sim in 1:n_sims){
    sim_dat[[paste(settings[set])]][[sim]]<-sim_obj$simulate(complete=T)
    
    if (set==1){
      sim_dat[[paste(settings[set])]][[sim]]$IR[which(sim_dat[[paste(settings[set])]][[sim]]$t_i %in% c(0:4))]<-NA
    } else if (set==2){
      sim_dat[[paste(settings[set])]][[sim]]$IR[which(sim_dat[[paste(settings[set])]][[sim]]$t_i %in% c(0:9))]<-NA
    } else if (set==3){
      sim_dat[[paste(settings[set])]][[sim]]$IR[which(sim_dat[[paste(settings[set])]][[sim]]$t_i %in% c(0:14))]<-NA
    }
    
    time1<-Sys.time()
    fit_obj<-MakeADFun(data=sim_dat[[paste(settings[set])]][[sim]],
                       parameters=fit_par,
                       map=maps,
                       random=random,
                       DLL="seam_LF_sims",
                       silent = F)
    
    fit_Opt<-optimx::optimr(fit_obj$par,
                            fit_obj$fn,
                            fit_obj$gr,
                            method="nlminb")
    
    #Forcing it to keep optimizing, presence of super large amount of catches slows it down drastically
    while (fit_Opt$message=="iteration limit reached without convergence (10)"){
      fit_obj$par<-fit_obj$env$last.par.best[-which(names(fit_obj$env$last.par.best)%in% random)]
      fit_Opt<-optimx::optimr(fit_obj$par,fit_obj$fn,fit_obj$gr,control=list(maxit=100000),method="nlminb")
    }
    mess_list[[paste(settings[set])]][[sim]]<-fit_Opt$message
    
    #Calculating standard errors for parameters and values of interest, i.e. biomass
    rep_list[[paste(settings[set])]][[sim]]<-sdreport(fit_obj,bias.correct=T)
    
    #Reporting estimates of specific outcomes
    Report_list[[paste(settings[set])]][[sim]]<-fit_obj$report()
    time2<-Sys.time()
    fit_time[[paste(settings[set])]][[sim]]<-time2-time1
  }
}







