library(ranger)
library(DALEX)
library(fairmodels)
library(tidymodels)
library(modeldata)
library(tidyverse)
library(DALEXtra)
library(gbm)

compas <- fairmodels::compas
compas <- filter(compas, Ethnicity == "African_American" | Ethnicity == "Caucasian")
compas$Two_yr_Recidivism <- as.factor(ifelse(compas$Two_yr_Recidivism == '1', '0', '1'))
compas$Ethnicity <- droplevels(compas$Ethnicity)

split <- initial_split(compas, prop = 0.8, strata = "Two_yr_Recidivism")
compas_train <- training(split)
compas_test <- testing(split)

y_numeric <- as.numeric(compas_train$Two_yr_Recidivism)-1

resamples <- vfold_cv(compas_train, 5)

rf_mod <-
  rand_forest() %>%
  set_args(mtry = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_rec <- 
  recipe(Two_yr_Recidivism ~., data = compas_train) %>%
  step_bagimpute(everything()) %>%
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
  select_best(metric = "roc_auc")

rf_roc <- 
  rf_res %>% 
  collect_predictions(parameters = rf_best) %>% 
  roc_curve(Two_yr_Recidivism, .pred_0) %>% 
  mutate(model = "RandomForest")


bind_rows(rf_roc) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) + 
  geom_path() +
  geom_abline(lty = 3) + 
  coord_equal()


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
  fit(data = compas_train)

rf_explainer <- explain_tidymodels(rf_fit, data = compas_train[,-1],
                                   y = y_numeric,
                                   label = "Random Forest")

# rf_compas <- ranger(Two_yr_Recidivism ~., data = compas, probability = TRUE)
# rf_explainer <- DALEX::explain(rf_compas, data = compas[,-1], y = as.numeric(compas$Two_yr_Recidivism)-1, colorize = FALSE)

rm(rf_mod, rf_rec, rf_roc, rf_wf, rf_res, final_wf, rf_best)


###########################
##### NEURAL NET ##########
###########################


nn_mod <- 
  mlp(hidden_units = tune()) %>%
  set_engine("nnet") %>%
  set_mode("classification")

nn_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train) %>%
  step_bagimpute(everything()) %>%
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
  select_best(metric = "roc_auc")

nn_roc <- 
  nn_res %>% 
  collect_predictions(parameters = nn_best) %>% 
  roc_curve(Two_yr_Recidivism, .pred_0) %>% 
  mutate(model = "neuralnet")


bind_rows(nn_roc, rf_roc) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) + 
  geom_path() +
  geom_abline(lty = 3) + 
  coord_equal()


final_nn_wf <- 
  nn_wf  %>%
  finalize_workflow(nn_best)

# fit the final model
# nn_fit <-
#   final_nn_wf %>%
#   last_fit(split = split)

nn_fit <-
  final_nn_wf %>%
  fit(data = compas_train)

nn_explainer <- explain_tidymodels(nn_fit, data = compas_train[,-1],
                                   y = y_numeric,
                                   label = "ANN")

rm(nn_mod, nn_rec, nn_roc, nn_wf, nn_res, final_nn_wf, nn_best)



#######################
#### BOOOSTING ########
#######################


df <- compas_train
df$Two_yr_Recidivism <- as.numeric(compas_train$Two_yr_Recidivism)-1
boost_fit <- gbm(Two_yr_Recidivism~., data = df) 

boost_explainer <- DALEX::explain(boost_fit, data = compas_train[,-1],
                                  y = y_numeric,
                                  label = "AdaBoost")


#########################
####### METRICS #########
#########################




fobject <- fairness_check(rf_explainer, nn_explainer, boost_explainer,
                          protected = compas_train$Ethnicity,
                          privileged = 'Caucasian',
                          verbose = FALSE,
                          colorize = FALSE)



plot(fobject)

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


model_performance(rf_explainer)
print(fobject, colorize = FALSE)

#################################
###### MITIGATION ###############
#################################

compas_train_mit <- compas_train %>% 
  mutate(Number_of_Priors = as.numeric(Number_of_Priors)) %>% 
  disparate_impact_remover(protected = compas_train$Ethnicity, 
                           features_to_transform = c("Number_of_Priors"))
  

resamples <- vfold_cv(compas_train_mit, 5)

rf_mod <-
  rand_forest() %>%
  set_args(mtry = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_rec <- 
  recipe(Two_yr_Recidivism ~., data = compas_train_mit) %>%
  step_bagimpute(everything()) %>%
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
  select_best(metric = "roc_auc")

rf_roc <- 
  rf_res %>% 
  collect_predictions(parameters = rf_best) %>% 
  roc_curve(Two_yr_Recidivism, .pred_0) %>% 
  mutate(model = "RandomForest")


bind_rows(rf_roc) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) + 
  geom_path() +
  geom_abline(lty = 3) + 
  coord_equal()


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
  fit(data = compas_train_mit)

rf_explainer_mit <- explain_tidymodels(rf_fit, data = compas_train_mit[,-1],
                                   y = y_numeric,
                                   label = "Random Forest Mit")

# rf_compas <- ranger(Two_yr_Recidivism ~., data = compas, probability = TRUE)
# rf_explainer <- DALEX::explain(rf_compas, data = compas[,-1], y = as.numeric(compas$Two_yr_Recidivism)-1, colorize = FALSE)

rm(rf_mod, rf_rec, rf_roc, rf_wf, rf_res, final_wf, rf_best)


###########################
##### NEURAL NET ##########
###########################


nn_mod <- 
  mlp(hidden_units = tune()) %>%
  set_engine("nnet") %>%
  set_mode("classification")

nn_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train_mit) %>%
  step_bagimpute(everything()) %>%
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
  select_best(metric = "roc_auc")

nn_roc <- 
  nn_res %>% 
  collect_predictions(parameters = nn_best) %>% 
  roc_curve(Two_yr_Recidivism, .pred_0) %>% 
  mutate(model = "neuralnet")


bind_rows(nn_roc, rf_roc) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) + 
  geom_path() +
  geom_abline(lty = 3) + 
  coord_equal()


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

nn_explainer_mit <- explain_tidymodels(nn_fit, data = compas_train_mit[,-1],
                                   y = y_numeric,
                                   label = "ANN Mit")

rm(nn_mod, nn_rec, nn_roc, nn_wf, nn_res, final_nn_wf, nn_best)



#######################
#### BOOOSTING ########
#######################


df <- compas_train_mit
df$Two_yr_Recidivism <- as.numeric(compas_train_mit$Two_yr_Recidivism)-1
boost_fit <- gbm(Two_yr_Recidivism~., data = df) 

boost_explainer_mit <- DALEX::explain(boost_fit, data = compas_train_mit[,-1],
                                  y = y_numeric,
                                  label = "AdaBoost mit")



#########################
####### METRICS #########
#########################




fobject <- fairness_check(rf_explainer, nn_explainer, boost_explainer,
                          rf_explainer_mit, nn_explainer_mit, boost_explainer_mit,
                          protected = compas_train$Ethnicity,
                          privileged = 'Caucasian',
                          verbose = FALSE,
                          colorize = FALSE)



plot(fobject,)

cm <- choose_metric(fobject, "TPR")
plot(cm)


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






