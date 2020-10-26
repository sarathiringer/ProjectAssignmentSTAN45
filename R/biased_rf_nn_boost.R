library(ranger)
library(DALEX)
library(fairmodels)
library(tidymodels)
library(modeldata)
library(tidyverse)
library(DALEXtra)
library(gbm)

rm(list = ls())
set.seed(123)
compas <- fairmodels::compas
compas <- filter(compas, Ethnicity == "African_American" | Ethnicity == "Caucasian")
# compas$Two_yr_Recidivism <- as.factor(ifelse(compas$Two_yr_Recidivism == '1', '0', '1'))

compas$Ethnicity <- droplevels(compas$Ethnicity)

split <- initial_split(compas, prop = 0.8, strata = "Two_yr_Recidivism")
compas_train <- training(split)
compas_test <- testing(split)

y_numeric <- as.numeric(compas_train$Two_yr_Recidivism)-1
y_numeric_test <- as.numeric(compas_test$Two_yr_Recidivism)-1

resamples <- vfold_cv(compas_train, 5)

###########################
#### RANDOM FOREST ########
###########################

rf_mod <-
  rand_forest() %>%
  set_args(mtry = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_rec <- 
  recipe(Two_yr_Recidivism ~., data = compas_train)  %>%
  # step_bagimpute(everything()) %>%
  step_nzv(everything(), -all_outcomes())

rf_wf <- 
  workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(rf_rec)

rf_res <-  
  rf_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = TRUE))

rf_best <-
  rf_res %>% 
  select_best(metric = "accuracy")

# finalize workflow
final_wf <- 
  rf_wf  %>%
  finalize_workflow(rf_best)
 
rf_fit <-
  final_wf %>%
  fit(data = compas_train)


rf_explainer <- explain_tidymodels(rf_fit, data = compas_train[,-1],
                                   y = y_numeric,
                                   label = "RF")
rf_test_exp <- update_data(rf_explainer, data = compas_test[,-1],
                                   y = y_numeric_test, verbose = FALSE)

# rf_compas <- ranger(Two_yr_Recidivism ~., data = compas, probability = TRUE)
# rf_explainer <- DALEX::explain(rf_compas, data = compas[,-1], y = as.numeric(compas$Two_yr_Recidivism)-1, colorize = FALSE)

rm(rf_mod, rf_rec, rf_wf, rf_res, final_wf, rf_best)


###########################
##### NEURAL NET ##########
###########################


nn_mod <- 
  mlp(hidden_units = tune()) %>%
  set_engine("nnet") %>%
  set_mode("classification")

nn_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train) %>%
  # step_bagimpute(everything()) %>%
  step_dummy(Sex, Ethnicity, Age_Above_FourtyFive,
             Age_Below_TwentyFive, Misdemeanor) %>%
  step_nzv(everything(), -all_outcomes()) %>%
  step_normalize(everything(), -all_outcomes())


nn_wf <- 
  workflow() %>%
  add_model(nn_mod) %>%
  add_recipe(nn_rec)

nn_res <- 
  nn_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = TRUE))

nn_best <- 
  nn_res %>% 
  select_best(metric = "accuracy")


final_nn_wf <- 
  nn_wf  %>%
  finalize_workflow(nn_best)


nn_fit <-
  final_nn_wf %>%
  fit(data = compas_train)

nn_explainer <- explain_tidymodels(nn_fit, data = compas_train[,-1],
                                   y = y_numeric,
                                   label = "ANN")
nn_test_exp <- update_data(nn_explainer, data = compas_test[,-1],
                           y = y_numeric_test, verbose = FALSE)

rm(nn_mod, nn_rec, nn_wf, nn_res, final_nn_wf, nn_best)


#######################
#### LOG REG ##########
#######################

## LOGISTIC REGRESSION

lr_mod <- 
  logistic_reg(penalty = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

lr_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train) %>% 
  #step_bagimpute(everything(), all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_nzv(everything(), -all_outcomes()) %>%
  step_normalize(everything(), -all_outcomes())

lr_wf <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(lr_rec)

lr_res <- 
  lr_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = T))

lr_best <-
  lr_res %>%
  select_best(metric = "accuracy")

final_lr_wf <- 
  lr_wf  %>%
  finalize_workflow(lr_best)

lr_fit <- 
  final_lr_wf %>%
  fit(data = compas_train)

# final_wf <- 
#   finalize_workflow(lr_wf, lr_best)


# Create explainer by DALEX
lr_explainer <- explain_tidymodels(lr_fit,
                                   data = compas_train[,-1],
                                   y = y_numeric, 
                                   label = "LR")
lr_test_exp <- update_data(lr_explainer, data = compas_test[,-1],
                           y = y_numeric_test, verbose = FALSE)

rm(lr_mod, lr_rec, lr_wf, lr_res, final_lr_wf, lr_best)

#######################
#### KNN ##############
#######################


# knn_mod <-
#   nearest_neighbor(neighbors = tune()) %>%
#   set_engine('kknn') %>%
#   set_mode("classification")
# 
# knn_rec <-
#   recipe(Two_yr_Recidivism ~ ., data = compas_train) %>%
#   step_bagimpute(everything()) %>%
#   step_scale(all_numeric(), -all_outcomes()) %>%
#   step_dummy(Sex)
# 
# knn_wf <-
#   workflow() %>%
#   add_model(knn_mod) %>%
#   add_recipe(knn_rec)
# 
# knn_res <-
#   knn_wf %>%
#   tune_grid(resamples = resamples,
#             metrics = metric_set(roc_auc, accuracy),
#             control = control_grid(save_pred = TRUE))
# 
# knn_best <-
#   knn_res %>%
#   select_best(metric = "accuracy")
# 
# 
# final_knn_wf <-
#   knn_wf  %>%
#   finalize_workflow(knn_best)
# 
# knn_fit <-
#   final_knn_wf %>%
#   fit(data = compas_train)
# 
# knn_explainer <- explain_tidymodels(knn_fit,
#                                    data = compas_train[,-1],
#                                    y = y_numeric,
#                                    label = "KNN")
# rm(knn_mod, knn_rec, knn_wf, knn_res, final_knn_wf, knn_best)


#######################
#### BOOOSTING ########
#######################

ab_mod <- 
  boost_tree(mtry = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

ab_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train) %>% 
  #step_bagimpute(everything(), all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_nzv(everything(), -all_outcomes()) %>%
  step_normalize(everything(), -all_outcomes())

ab_wf <- 
  workflow() %>% 
  add_model(ab_mod) %>% 
  add_recipe(ab_rec)

ab_res <- 
  ab_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = T))

ab_best <-
  ab_res %>%
  select_best(metric = "accuracy")

final_ab_wf <- 
  ab_wf  %>%
  finalize_workflow(ab_best)

ab_fit <- 
  final_ab_wf %>%
  fit(data = compas_train)

boost_explainer <- DALEX::explain(ab_fit, data = compas_train[,-1],
                                  y = y_numeric,
                                  label = "AdaBoost")
boost_test_exp <- update_data(boost_explainer, data = compas_test[,-1],
                           y = y_numeric_test, verbose = FALSE)
rm(ab_mod, ab_rec, ab_wf, ab_res, final_ab_wf, ab_best)

#########################
####### METRICS #########
#########################




fobject1 <- fairness_check(rf_explainer, nn_explainer, boost_explainer, lr_explainer, #knn_explainer,
                          protected = compas_train$Ethnicity,
                          privileged = 'Caucasian',
                          verbose = FALSE,
                          colorize = FALSE)



# plot(fobject)

# cm <- choose_metric(fobject, "TPR")
# plot(cm)
# 
# 
# sm <- stack_metrics(fobject)
# plot(sm)
# 
# 
# fair_pca <- fairness_pca(fobject)
# print(fair_pca)
# plot(fair_pca)
# 
# fheatmap <- fairness_heatmap(fobject)
# plot(fheatmap, text_size = 3)

fap <- performance_and_fairness(fobject, fairness_metric = "STP")
plot(fap)


#############################################################################
###### MITIGATION ###########################################################
#############################################################################

compas_train_mit <- compas_train %>% 
  mutate(Number_of_Priors = as.numeric(Number_of_Priors)) %>% 
  disparate_impact_remover(protected = compas_train$Ethnicity, 
                           features_to_transform = c("Number_of_Priors"))
  

resamples <- vfold_cv(compas_train_mit, 5)

#######################
#### RANDOM FOREST ####
#######################

rf_mod <-
  rand_forest() %>%
  set_args(mtry = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_rec <- 
  recipe(Two_yr_Recidivism ~., data = compas_train_mit) %>%
  # step_bagimpute(everything()) %>%
  step_nzv(everything(), -all_outcomes())

rf_wf <- 
  workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(rf_rec)

rf_res <-  
  rf_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = TRUE))

rf_best <-
  rf_res %>% 
  select_best(metric = "accuracy")


# finalize workflow
final_wf <- 
  rf_wf  %>%
  finalize_workflow(rf_best)

# fit the final model
# rf_fit <- 
#   final_wf %>%
#   last_fit(split = split)

rf_fit <-
  final_wf %>%
  fit(data = compas_train_mit)

rf_explainer_mit <- explain_tidymodels(rf_fit, data = compas_train[,-1],
                                   y = y_numeric,
                                   label = "RF rem")
rf_test_exp_mit <- update_data(rf_explainer_mit, data = compas_test[,-1],
                           y = y_numeric_test, verbose = FALSE)

# rf_compas <- ranger(Two_yr_Recidivism ~., data = compas, probability = TRUE)
# rf_explainer <- DALEX::explain(rf_compas, data = compas[,-1], y = as.numeric(compas$Two_yr_Recidivism)-1, colorize = FALSE)

rm(rf_mod, rf_rec, rf_wf, rf_res, final_wf, rf_best)


###########################
##### NEURAL NET ##########
###########################


nn_mod <- 
  mlp(hidden_units = tune()) %>%
  set_engine("nnet") %>%
  set_mode("classification")

nn_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train_mit) %>%
  # step_bagimpute(everything()) %>%
  step_dummy(Sex, Ethnicity, Age_Above_FourtyFive,
             Age_Below_TwentyFive, Misdemeanor) %>%
  step_nzv(everything(), -all_outcomes()) %>%
  step_normalize(everything(), -all_outcomes())


nn_wf <- 
  workflow() %>%
  add_model(nn_mod) %>%
  add_recipe(nn_rec)

nn_res <- 
  nn_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = TRUE))

nn_best <- 
  nn_res %>% 
  select_best(metric = "accuracy")


final_nn_wf <- 
  nn_wf  %>%
  finalize_workflow(nn_best)

# fit the final model
# nn_fit <-
#   final_nn_wf %>%
#   last_fit(split = split)

nn_fit <-
  final_nn_wf %>%
  fit(data = compas_train_mit)

nn_explainer_mit <- explain_tidymodels(nn_fit, data = compas_train[,-1],
                                   y = y_numeric,
                                   label = "ANN rem")
nn_test_exp_mit <- update_data(nn_explainer_mit, data = compas_test[,-1],
                               y = y_numeric_test, verbose = FALSE)

rm(nn_mod, nn_rec, nn_roc, nn_wf, nn_res, final_nn_wf, nn_best)

#######################
#### LOG REG ##########
#######################

## LOGISTIC REGRESSION

lr_mod <- 
  logistic_reg(penalty = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

lr_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train_mit) %>% 
  #step_bagimpute(everything(), all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_nzv(everything(), -all_outcomes()) %>%
  step_normalize(everything(), -all_outcomes())

lr_wf <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(lr_rec)

lr_res <- 
  lr_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = T))

lr_best <-
  lr_res %>%
  select_best(metric = "accuracy")

final_lr_wf <- 
  lr_wf  %>%
  finalize_workflow(lr_best)

lr_fit <- 
  final_lr_wf %>%
  fit(data = compas_train_mit)

# Create explainer by DALEX
lr_explainer_mit <- explain_tidymodels(lr_fit,
                                   data = compas_train[,-1],
                                   y = y_numeric, 
                                   label = "LR rem")
lr_test_exp_mit <- update_data(lr_explainer_mit, data = compas_test[,-1],
                               y = y_numeric_test, verbose = FALSE)
rm(lr_mod, lr_rec, lr_wf, lr_res, final_lr_wf, lr_best)

#######################
#### KNN ##############
#######################


# knn_mod <-
#   nearest_neighbor(neighbors = 4) %>%
#   set_engine('kknn') %>%
#   set_mode("classification")
# 
# knn_rec <-
#   recipe(Two_yr_Recidivism ~ ., data = compas_train_mit) %>%
#   step_bagimpute(everything()) %>%
#   step_scale(all_numeric(), -all_outcomes()) %>%
#   step_dummy(Sex)
# 
# knn_wf <-
#   workflow() %>%
#   add_model(knn_mod) %>%
#   add_recipe(knn_rec)
# 
# knn_res <-
#   knn_wf %>%
#   tune_grid(resamples = resamples,
#             metrics = metric_set(roc_auc, accuracy),
#             control = control_grid(save_pred = TRUE))
# 
# knn_best <-
#   knn_res %>%
#   select_best(metric = "accuracy")
# 
# 
# final_knn_wf <-
#   knn_wf  %>%
#   finalize_workflow(knn_best)
# 
# knn_fit <-
#   final_knn_wf %>%
#   fit(data = compas_train_mit)
# 
# knn_explainer_mit <- explain_tidymodels(knn_fit,
#                                     data = compas_train_mit[,-1],
#                                     y = y_numeric,
#                                     label = "KNN mit")
# rm(knn_mod, knn_rec, knn_wf, knn_res, final_knn_wf, knn_best)

#######################
#### BOOOSTING ########
#######################

ab_mod <- 
  boost_tree(mtry = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

ab_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train_mit) %>% 
  #step_bagimpute(everything(), all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_nzv(everything(), -all_outcomes()) %>%
  step_normalize(everything(), -all_outcomes())

ab_wf <- 
  workflow() %>% 
  add_model(ab_mod) %>% 
  add_recipe(ab_rec)

ab_res <- 
  ab_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = T))

ab_best <-
  ab_res %>%
  select_best(metric = "accuracy")

final_ab_wf <- 
  ab_wf  %>%
  finalize_workflow(ab_best)

ab_fit <- 
  final_ab_wf %>%
  fit(data = compas_train_mit)

boost_explainer_mit <- DALEX::explain(ab_fit, data = compas_train[,-1],
                                  y = y_numeric,
                                  label = "AdaBoost rem")
boost_test_exp_mit <- update_data(boost_explainer_mit, data = compas_test[,-1],
                              y = y_numeric_test, verbose = FALSE)
rm(ab_mod, ab_rec, ab_wf, ab_res, final_ab_wf, ab_best)


#########################
####### METRICS #########
#########################




fobject <- fairness_check(fobject1,
                          rf_explainer_mit, nn_explainer_mit, boost_explainer_mit,
                          lr_explainer_mit, #knn_explainer_mit,
                          protected = compas_train$Ethnicity,
                          privileged = 'Caucasian',
                          verbose = FALSE,
                          colorize = FALSE)



# plot(fobject,)
# 
# cm <- choose_metric(fobject, "TPR")
# plot(cm)


# sm <- stack_metrics(fobject)
# plot(sm)
# 
# 
# fair_pca <- fairness_pca(fobject)
# print(fair_pca)
# plot(fair_pca)
# 
# fheatmap <- fairness_heatmap(fobject)
# plot(fheatmap, text_size = 3)

fap <- performance_and_fairness(fobject, fairness_metric = "STP")
plot(fap)


model_performance(rf_explainer)
print(fobject, colorize = FALSE)



################################
###### RESAMPLING ##############
################################

uniform_indexes <- resample(protected = compas_train$Ethnicity,
                            y = y_numeric)


resamples <- vfold_cv(compas_train[uniform_indexes, ], 5)

###############################
##### RANDOM FOREST ###########
###############################

rf_mod <-
  rand_forest() %>%
  set_args(mtry = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_rec <- 
  recipe(Two_yr_Recidivism ~., data = compas_train[uniform_indexes, ]) %>%
  # step_bagimpute(everything()) %>%
  step_nzv(everything(), -all_outcomes())

rf_wf <- 
  workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(rf_rec)

rf_res <-  
  rf_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = TRUE))

rf_best <-
  rf_res %>% 
  select_best(metric = "accuracy")

# Question 12 

# finalize workflow
final_wf <- 
  rf_wf  %>%
  finalize_workflow(rf_best)

# fit the final model
# rf_fit <- 
#   final_wf %>%
#   last_fit(split = split)

rf_fit <-
  final_wf %>%
  fit(data = compas_train[uniform_indexes, ])

rf_explainer_resa_uni <- explain_tidymodels(rf_fit, data = compas_train[,-1],
                                   y = y_numeric,
                                   label = "RF resa unif")
rf_test_exp_resa_uni <- update_data(rf_explainer_resa_uni, data = compas_test[,-1],
                               y = y_numeric_test, verbose = FALSE)

# rf_compas <- ranger(Two_yr_Recidivism ~., data = compas, probability = TRUE)
# rf_explainer <- DALEX::explain(rf_compas, data = compas[,-1], y = as.numeric(compas$Two_yr_Recidivism)-1, colorize = FALSE)

rm(rf_mod, rf_rec, rf_wf, rf_res, final_wf, rf_best)


###########################
##### NEURAL NET ##########
###########################


nn_mod <- 
  mlp(hidden_units = tune()) %>%
  set_engine("nnet") %>%
  set_mode("classification")

nn_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train[uniform_indexes, ]) %>%
  # step_bagimpute(everything()) %>%
  step_dummy(Sex, Ethnicity, Age_Above_FourtyFive,
             Age_Below_TwentyFive, Misdemeanor) %>%
  step_nzv(everything(), -all_outcomes()) %>%
  step_normalize(everything(), -all_outcomes())


nn_wf <- 
  workflow() %>%
  add_model(nn_mod) %>%
  add_recipe(nn_rec)

nn_res <- 
  nn_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = TRUE))

nn_best <- 
  nn_res %>% 
  select_best(metric = "accuracy")

nn_roc <- 
  nn_res %>% 
  collect_predictions(parameters = nn_best) %>% 
  roc_curve(Two_yr_Recidivism, .pred_0) %>% 
  mutate(model = "neuralnet")


final_nn_wf <- 
  nn_wf  %>%
  finalize_workflow(nn_best)

# fit the final model
# nn_fit <-
#   final_nn_wf %>%
#   last_fit(split = split)

nn_fit <-
  final_nn_wf %>%
  fit(data = compas_train[uniform_indexes, ])

nn_explainer_resa_uni <- explain_tidymodels(nn_fit, data = compas_train[,-1],
                                   y = y_numeric,
                                   label = "ANN resa unif")
nn_test_exp_resa_uni <- update_data(nn_explainer_resa_uni, data = compas_test[,-1],
                                    y = y_numeric_test, verbose = FALSE)
rm(nn_mod, nn_rec, nn_roc, nn_wf, nn_res, final_nn_wf, nn_best)

#######################
#### LOG REG ##########
#######################

## LOGISTIC REGRESSION

lr_mod <- 
  logistic_reg(penalty = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

lr_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train[uniform_indexes, ]) %>% 
  #step_bagimpute(everything(), all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_nzv(everything(), -all_outcomes()) %>%
  step_normalize(everything(), -all_outcomes())

lr_wf <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(lr_rec)

lr_res <- 
  lr_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = T))

lr_best <-
  lr_res %>%
  select_best(metric = "accuracy")

final_lr_wf <- 
  lr_wf  %>%
  finalize_workflow(lr_best)

lr_fit <- 
  final_lr_wf %>%
  fit(data = compas_train[uniform_indexes, ])

# Create explainer by DALEX
lr_explainer_resa_uni <- explain_tidymodels(lr_fit,
                                       data = compas_train[,-1],
                                       y = y_numeric, 
                                       label = "LR resa unif")
lr_test_exp_resa_uni <- update_data(lr_explainer_resa_uni, data = compas_test[,-1],
                                    y = y_numeric_test, verbose = FALSE)
rm(lr_mod, lr_rec, lr_wf, lr_res, final_lr_wf, lr_best)

#######################
#### KNN ##############
#######################


# knn_mod <-
#   nearest_neighbor(neighbors = tune()) %>%
#   set_engine('kknn') %>%
#   set_mode("classification")
# 
# knn_rec <-
#   recipe(Two_yr_Recidivism ~ ., data = compas_train[uniform_indexes, ]) %>%
#   step_bagimpute(everything()) %>%
#   step_scale(all_numeric(), -all_outcomes()) %>%
#   step_dummy(Sex)
# 
# knn_wf <-
#   workflow() %>%
#   add_model(knn_mod) %>%
#   add_recipe(knn_rec)
# 
# knn_res <-
#   knn_wf %>%
#   tune_grid(resamples = resamples,
#             metrics = metric_set(roc_auc, accuracy),
#             control = control_grid(save_pred = TRUE))
# 
# knn_best <-
#   knn_res %>%
#   select_best(metric = "accuracy")
# 
# 
# final_knn_wf <-
#   knn_wf  %>%
#   finalize_workflow(knn_best)
# 
# knn_fit <-
#   final_knn_wf %>%
#   fit(data = compas_train[uniform_indexes, ])
# 
# knn_explainer_resa_uni <- explain_tidymodels(knn_fit,
#                                         data = compas_train[,-1],
#                                         y = y_numeric,
#                                         label = "KNN resa uni")
# rm(knn_mod, knn_rec, knn_wf, knn_res, final_knn_wf, knn_best)

#######################
#### BOOOSTING ########
#######################

ab_mod <- 
  boost_tree(mtry = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

ab_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train[uniform_indexes, ]) %>% 
  #step_bagimpute(everything(), all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_nzv(everything(), -all_outcomes()) %>%
  step_normalize(everything(), -all_outcomes())

ab_wf <- 
  workflow() %>% 
  add_model(ab_mod) %>% 
  add_recipe(ab_rec)

ab_res <- 
  ab_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = T))

ab_best <-
  ab_res %>%
  select_best(metric = "accuracy")

final_ab_wf <- 
  ab_wf  %>%
  finalize_workflow(ab_best)

ab_fit <- 
  final_ab_wf %>%
  fit(data = compas_train[uniform_indexes, ])

boost_explainer_resa_uni <- DALEX::explain(ab_fit, data = compas_train[,-1],
                                  y = y_numeric,
                                  label = "AdaBoost resa unif")
boost_test_exp_resa_uni <- update_data(boost_explainer_resa_uni, data = compas_test[,-1],
                              y = y_numeric_test, verbose = FALSE)

rm(ab_mod, ab_rec, ab_wf, ab_res, final_ab_wf, ab_best)

#########################
####### METRICS #########
#########################




fobject <- fairness_check(fobject, rf_explainer_resa_uni, nn_explainer_resa_uni,
                          boost_explainer_resa_uni, lr_explainer_resa_uni, #knn_explainer_resa_uni,
                          verbose = FALSE,
                          colorize = FALSE)

fap <- performance_and_fairness(fobject, fairness_metric = "STP")
plot(fap)

################################
###### RESAMPLING PREF #########
################################


# getting probs for resampling "preferential" but well just do uniform for now
probs <- glm(Two_yr_Recidivism ~., data = compas_train, family = binomial())$fitted.values
pref_ind <- resample(protected = compas_train$Ethnicity,
                     y = y_numeric,
                     type = "preferential",
                     probs = probs)


resamples <- vfold_cv(compas_train[pref_ind, ], 5)

###############################
##### RANDOM FOREST ###########
###############################

rf_mod <-
  rand_forest() %>%
  set_args(mtry = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_rec <- 
  recipe(Two_yr_Recidivism ~., data = compas_train[pref_ind, ]) %>%
  # step_bagimpute(everything()) %>%
  step_nzv(everything(), -all_outcomes())

rf_wf <- 
  workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(rf_rec)

rf_res <-  
  rf_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = TRUE))

rf_best <-
  rf_res %>% 
  select_best(metric = "accuracy")

# Question 12 

# finalize workflow
final_wf <- 
  rf_wf  %>%
  finalize_workflow(rf_best)

# fit the final model
# rf_fit <- 
#   final_wf %>%
#   last_fit(split = split)

rf_fit <-
  final_wf %>%
  fit(data = compas_train[pref_ind, ])

rf_explainer_resa_pref <- explain_tidymodels(rf_fit, data = compas_train[,-1],
                                        y = y_numeric,
                                        label = "RF resa pref")
rf_test_exp_resa_pref <- update_data(rf_explainer_resa_pref, data = compas_test[,-1],
                                    y = y_numeric_test, verbose = FALSE)

# rf_compas <- ranger(Two_yr_Recidivism ~., data = compas, probability = TRUE)
# rf_explainer <- DALEX::explain(rf_compas, data = compas[,-1], y = as.numeric(compas$Two_yr_Recidivism)-1, colorize = FALSE)

rm(rf_mod, rf_rec, rf_wf, rf_res, final_wf, rf_best)


###########################
##### NEURAL NET ##########
###########################


nn_mod <- 
  mlp(hidden_units = tune()) %>%
  set_engine("nnet") %>%
  set_mode("classification")

nn_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train[pref_ind, ]) %>%
  # step_bagimpute(everything()) %>%
  step_dummy(Sex, Ethnicity, Age_Above_FourtyFive,
             Age_Below_TwentyFive, Misdemeanor) %>%
  step_nzv(everything(), -all_outcomes()) %>%
  step_normalize(everything(), -all_outcomes())


nn_wf <- 
  workflow() %>%
  add_model(nn_mod) %>%
  add_recipe(nn_rec)

nn_res <- 
  nn_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = TRUE))

nn_best <- 
  nn_res %>% 
  select_best(metric = "accuracy")

nn_roc <- 
  nn_res %>% 
  collect_predictions(parameters = nn_best) %>% 
  roc_curve(Two_yr_Recidivism, .pred_0) %>% 
  mutate(model = "neuralnet")


final_nn_wf <- 
  nn_wf  %>%
  finalize_workflow(nn_best)

# fit the final model
# nn_fit <-
#   final_nn_wf %>%
#   last_fit(split = split)

nn_fit <-
  final_nn_wf %>%
  fit(data = compas_train[pref_ind, ])

nn_explainer_resa_pref <- explain_tidymodels(nn_fit, data = compas_train[,-1],
                                        y = y_numeric,
                                        label = "ANN resa pref")

nn_test_exp_resa_pref <- update_data(nn_explainer_resa_pref, data = compas_test[,-1],
                                     y = y_numeric_test, verbose = FALSE)

rm(nn_mod, nn_rec, nn_roc, nn_wf, nn_res, final_nn_wf, nn_best)

#######################
#### LOG REG ##########
#######################

## LOGISTIC REGRESSION

lr_mod <- 
  logistic_reg(penalty = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

lr_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train[pref_ind, ]) %>% 
  #step_bagimpute(everything(), all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_nzv(everything(), -all_outcomes()) %>%
  step_normalize(everything(), -all_outcomes())

lr_wf <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(lr_rec)

lr_res <- 
  lr_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = T))

lr_best <-
  lr_res %>%
  select_best(metric = "accuracy")

final_lr_wf <- 
  lr_wf  %>%
  finalize_workflow(lr_best)

lr_fit <- 
  final_lr_wf %>%
  fit(data = compas_train[pref_ind, ])

# Create explainer by DALEX
lr_explainer_resa_pref <- explain_tidymodels(lr_fit,
                                        data = compas_train[,-1],
                                        y = y_numeric, 
                                        label = "LR resa pref")
lr_test_exp_resa_pref <- update_data(lr_explainer_resa_pref, data = compas_test[,-1],
                                     y = y_numeric_test, verbose = FALSE)
rm(lr_mod, lr_rec, lr_wf, lr_res, final_lr_wf, lr_best)

#######################
#### KNN ##############
#######################


# knn_mod <-
#   nearest_neighbor(neighbors = tune()) %>%
#   set_engine('kknn') %>%
#   set_mode("classification")
# 
# knn_rec <-
#   recipe(Two_yr_Recidivism ~ ., data = compas_train[pref_ind, ]) %>%
#   step_bagimpute(everything()) %>%
#   step_scale(all_numeric(), -all_outcomes()) %>%
#   step_dummy(Sex)
# 
# knn_wf <-
#   workflow() %>%
#   add_model(knn_mod) %>%
#   add_recipe(knn_rec)
# 
# knn_res <-
#   knn_wf %>%
#   tune_grid(resamples = resamples,
#             metrics = metric_set(roc_auc, accuracy),
#             control = control_grid(save_pred = TRUE))
# 
# knn_best <-
#   knn_res %>%
#   select_best(metric = "accuracy")
# 
# 
# final_knn_wf <-
#   knn_wf  %>%
#   finalize_workflow(knn_best)
# 
# knn_fit <-
#   final_knn_wf %>%
#   fit(data = compas_train[pref_ind, ])
# 
# knn_explainer_resa_pref <- explain_tidymodels(knn_fit,
#                                          data = compas_train[,-1],
#                                          y = y_numeric,
#                                          label = "KNN resa pref")
# rm(knn_mod, knn_rec, knn_wf, knn_res, final_knn_wf, knn_best)

#######################
#### BOOOSTING ########
#######################

ab_mod <- 
  boost_tree(mtry = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

ab_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train[pref_ind, ]) %>% 
  #step_bagimpute(everything(), all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_nzv(everything(), -all_outcomes()) %>%
  step_normalize(everything(), -all_outcomes())

ab_wf <- 
  workflow() %>% 
  add_model(ab_mod) %>% 
  add_recipe(ab_rec)

ab_res <- 
  ab_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = T))

ab_best <-
  ab_res %>%
  select_best(metric = "accuracy")

final_ab_wf <- 
  ab_wf  %>%
  finalize_workflow(ab_best)

ab_fit <- 
  final_ab_wf %>%
  fit(data = compas_train[pref_ind, ])

boost_explainer_resa_pref <- DALEX::explain(ab_fit, data = compas_train[,-1],
                                            y = y_numeric,
                                            label = "AdaBoost resa pref")
boost_test_exp_resa_pref <-  update_data(boost_explainer_resa_pref, data = compas_test[,-1],
                                           y = y_numeric_test, verbose = FALSE)

rm(ab_mod, ab_rec, ab_wf, ab_res, final_ab_wf, ab_best)

#########################
####### METRICS #########
#########################




fobject_all <- fairness_check(fobject, rf_explainer_resa_pref, nn_explainer_resa_pref,
                          boost_explainer_resa_pref, lr_explainer_resa_pref, #knn_explainer_resa_pref,
                          verbose = FALSE,
                          colorize = FALSE)

fobject_test <- fairness_check(rf_test_exp, nn_test_exp, lr_test_exp, boost_test_exp,
                               rf_test_exp_mit, nn_test_exp_mit, lr_test_exp_mit, boost_test_exp_mit,
                               rf_test_exp_resa_uni, nn_test_exp_resa_uni, lr_test_exp_resa_uni, boost_test_exp_resa_uni,
                               rf_test_exp_resa_pref, nn_test_exp_resa_pref, lr_test_exp_resa_pref, boost_test_exp_resa_pref ,
                               protected = compas_test$Ethnicity,
                               privileged = 'Caucasian',
                               verbose = FALSE,
                               colorize = FALSE)

plot(performance_and_fairness(fobject_test, fairness_metric = "STP"))

save(fobject1, file = "report/models/fobject1.Rdata")
save(fobject_all, file = "report/models/fobject_all.Rdata")
save(fobject_test, file = "report/models/fobject_test.Rdata")



fobject_co <- fairness_check(rf_explainer_resa_uni, nn_explainer_resa_uni,
                               boost_explainer_resa_uni, rf_explainer_resa_pref,
                               protected = compas_train$Ethnicity,
                               privileged = 'Caucasian',
                               verbose = FALSE,
                               colorize = FALSE)
save(fobject_co, file = "report/models/fobject_co.Rdata")

co <- ceteris_paribus_cutoff(fobject_co,
                            subgroup = "African_American",
                            fairness_metrics = c("TPR", "STP"))
plot(ceteris_paribus_cutoff(fobject_co,
                            subgroup = "African_American",
                            fairness_metrics = c("TPR","STP")))



fo_comp <- fairness_check(nn_explainer_resa_uni, rf_explainer_resa_uni, rf_explainer_resa_pref,
                          boost_explainer_resa_uni,
                          protected = compas_train$Ethnicity,
                          privileged = 'Caucasian',
                          verbose = FALSE,
                          colorize = FALSE)
fo_comp <- fairness_check(fo_comp, rf_explainer_resa_uni, 
                          label = "RF resa unif cutoff",
                          cutoff = list("African_American" = co[["min_data"]][["mins"]][1]),
                          verbose = FALSE,
                          colorize = FALSE)
fo_comp <- fairness_check(fo_comp, nn_explainer_resa_uni, 
                          label = "ANN resa unif cutoff",
                          cutoff = list("African_American" = co[["min_data"]][["mins"]][2]),
                          verbose = FALSE,
                          colorize = FALSE)
fo_comp <- fairness_check(fo_comp, boost_explainer_resa_uni, 
                          label = "AdaBoost resa unif cutoff",
                          cutoff = list("African_American" = co[["min_data"]][["mins"]][3]),
                          verbose = FALSE,
                          colorize = FALSE)
fo_comp <- fairness_check(fo_comp, rf_explainer_resa_pref, 
                          label = "RF resa pref cutoff",
                          cutoff = list("African_American" = co[["min_data"]][["mins"]][4]),
                          verbose = FALSE,
                          colorize = FALSE)
plot(performance_and_fairness(fo_comp, fairness_metric = "STP"))

save(fo_comp, file = "report/models/fo_comp.Rdata")


#####################
#### TESTING ########
#####################

?last_fit


