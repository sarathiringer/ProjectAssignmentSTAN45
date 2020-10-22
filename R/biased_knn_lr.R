
library(fairmodels)
library(tidymodels)
library(DALEX)
library(DALEXtra)

# Data prep
data("compas")

summary(compas)

# Training and test set
set.seed(42)

compas <- 
  compas %>% 
  filter(Ethnicity == "African_American" | Ethnicity == "Caucasian") %>% 
  mutate(Two_yr_Recidivism = as.factor(ifelse(compas$Two_yr_Recidivism == '1', '0', '1'))) %>% 
  droplevels()

y_numeric <- as.numeric(compas_train$Two_yr_Recidivism)-1

summary(compas)

split <- initial_split(compas, prop = 0.8, strata = "Two_yr_Recidivism")

compas_test <- testing(split)

test_ethnicity <-
  compas_test %>%
  select(Ethnicity)

compas_train <- training(split)

train_ethnicity <-
  compas_train %>%
  select(Ethnicity)

resamples <- vfold_cv(compas_train, 5)

####################
##### KNN ##########
####################

knn_mod <- 
  nearest_neighbor(neighbors = tune()) %>% 
  set_engine('kknn') %>% 
  set_mode("classification")

knn_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train) %>% 
  step_bagimpute(everything()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>% 
  step_dummy(Sex)

knn_wf <- 
  workflow() %>% 
  add_model(knn_mod) %>% 
  add_recipe(knn_rec)

knn_res <- 
  knn_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = TRUE))

knn_best <-
  knn_res %>%
  select_best(metric = "accuracy")

knn_roc <- 
  knn_res %>% 
  collect_predictions(parameters = knn_best) %>% 
  roc_curve(Two_yr_Recidivism, .pred_0) %>% 
  mutate(model = "Ridge LR")

knn_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
  geom_path() +
  geom_abline(lty = 3) + 
  coord_equal() 

knn_acc <- 
  knn_res %>% 
  collect_predictions(parameters = knn_best) %>% 
  accuracy(Two_yr_Recidivism, .pred_class) %>% 
  mutate(model = "Ridge LR")

knn_results <- 
  knn_res %>% 
  collect_predictions(parameters = knn_best)

final_wf <-
  finalize_workflow(knn_wf, knn_best)

# Fit 
knn_fitted <- 
  final_wf %>% 
  fit(data = compas_train)

##### Bias evaluation

# Dataframe prep
pred <- data.frame(knn_results, train_ethnicity)

# Quick vis
ggplot(data = pred, aes(.pred_class, group = Ethnicity)) +
  geom_bar(aes(y=..prop.., fill = factor(..x..)), stat = "count") +
  scale_y_continuous(labels=scales::percent) +
  ylab("Relative frequencies") +
  xlab("Prediction") +
  facet_grid(~Ethnicity) +
  theme(legend.position = "none")

# Create explainer by DALEX
knn_explainer <- explain_tidymodels(knn_fitted, 
                                    data = compas_train[,-1], 
                                    y = y_numeric,
                                    label = "KNN")




######################
##### LOG REG ########
######################

lr_mod <- 
  logistic_reg(penalty = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

lr_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train) %>% 
  step_rm(Ethnicity) %>% 
  step_bagimpute(everything(), -all_outcomes()) %>% 
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

lr_roc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(Two_yr_Recidivism, .pred_0) %>% 
  mutate(model = "Ridge LR")

lr_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
  geom_path() +
  geom_abline(lty = 3) + 
  coord_equal() 

lr_acc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  accuracy(Two_yr_Recidivism, .pred_class) %>% 
  mutate(model = "Ridge LR")

lr_results <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best)

final_wf <-
  finalize_workflow(lr_wf, lr_best)

# Fit 
lr_fitted <- 
  final_wf %>% 
  fit(data = compas_train)



##### Bias evaluation

# Dataframe prep
pred <- data.frame(lr_results, train_ethnicity)

# Quick vis
ggplot(data = pred, aes(.pred_class, group = Ethnicity)) +
  geom_bar(aes(y=..prop.., fill = factor(..x..)), stat = "count") +
  scale_y_continuous(labels=scales::percent) +
  ylab("Relative frequencies") +
  xlab("Prediction") +
  facet_grid(~Ethnicity) +
  theme(legend.position = "none")

# Create explainer by DALEX
lr_explainer <- explain_tidymodels(lr_fitted, 
                                   data = compas_train[,-1], 
                                   y = y_numeric,
                                   label = "Logistic Regression")

#########################
####### METRICS #########
#########################


# Fairnes object
fobject_knn_lr <- fairness_check(knn_explainer, lr_explainer, 
                                 protected = compas_train$Ethnicity,
                                 privileged = 'Caucasian')

# Inspectt fairness
print(fobject_knn_lr)
plot(fobject_knn_lr)

# Metric scores
ms <- metric_scores(fobject_knn_lr)
plot(ms)

# Performance and fairness
paf <- performance_and_fairness(fobject_knn_lr, fairness_metric = "STP")
plot(paf)

# All cut_offs
ac <- all_cutoffs(fobject_knn_lr,
                  fairness_metrics = c("TPR",
                                       "FPR"))
plot(ac)



gm_knn <- group_matrices(protected = as.vector(compas_train$Ethnicity), 
                         probs = "pred_0", 
                         preds = as.vector(pred$.pred_class), 
                         cutoff = 0.5)


group_m <- group_metric(fobject_knn_lr)
print(group_m)
plot(group_m)

model_performance(knn_explainer)
model_performance(lr_explainer)

cpc <- ceteris_paribus_cutoff(fobject_knn_lr, "African_American")
plot(cpc)

#################################
###### MITIGATION ###############
#################################

compas_train_mit <- compas_train %>% 
  mutate(Number_of_Priors = as.numeric(Number_of_Priors)) %>% 
  disparate_impact_remover(protected = compas_train$Ethnicity, 
                           features_to_transform = c("Number_of_Priors"))


resamples <- vfold_cv(compas_train_mit, 5)

## KNN mitigated

knn_mod <- 
  nearest_neighbor(neighbors = tune()) %>% 
  set_engine('kknn') %>% 
  set_mode("classification")

knn_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train_mit) %>% 
  step_bagimpute(everything()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>% 
  step_dummy(Sex)

knn_wf <- 
  workflow() %>% 
  add_model(knn_mod) %>% 
  add_recipe(knn_rec)

knn_res <- 
  knn_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = TRUE))

knn_best <-
  knn_res %>%
  select_best(metric = "accuracy")

knn_roc <- 
  knn_res %>% 
  collect_predictions(parameters = knn_best) %>% 
  roc_curve(Two_yr_Recidivism, .pred_0) %>% 
  mutate(model = "Ridge LR")

knn_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
  geom_path() +
  geom_abline(lty = 3) + 
  coord_equal() 

knn_acc <- 
  knn_res %>% 
  collect_predictions(parameters = knn_best) %>% 
  accuracy(Two_yr_Recidivism, .pred_class) %>% 
  mutate(model = "Ridge LR")

knn_results <- 
  knn_res %>% 
  collect_predictions(parameters = knn_best)

final_wf <-
  finalize_workflow(knn_wf, knn_best)

# Fit 
knn_mit_fitted <- 
  final_wf %>% 
  fit(data = compas_train_mit)

# Explainer
knn_mit_explainer <- explain_tidymodels(knn_mit_fitted, 
                                        data = compas_train_mit[,-1], 
                                        y = as.numeric(compas_train_mit$Two_yr_Recidivism)-1,
                                        label = "KNN Mit")

## LR mitigated

lr_mod <- 
  logistic_reg(penalty = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

lr_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train_mit) %>% 
  step_rm(Ethnicity) %>% 
  step_bagimpute(everything(), -all_outcomes()) %>% 
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

lr_roc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(Two_yr_Recidivism, .pred_0) %>% 
  mutate(model = "Ridge LR")

lr_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
  geom_path() +
  geom_abline(lty = 3) + 
  coord_equal() 

lr_acc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  accuracy(Two_yr_Recidivism, .pred_class) %>% 
  mutate(model = "Ridge LR")

lr_results <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best)

final_wf <-
  finalize_workflow(lr_wf, lr_best)

# Fit 
lr_mit_fitted <- 
  final_wf %>% 
  fit(data = compas_train_mit)


# Explainer
lr_mit_explainer <- explain_tidymodels(lr_mit_fitted, 
                                       data = compas_train_mit[,-1], 
                                       y = as.numeric(compas_train_mit$Two_yr_Recidivism)-1,
                                       label = "Logistic Regression Mit")


## Fairness object
fobject_knn_lr_mit <- fairness_check(knn_explainer, lr_explainer, knn_mit_explainer, lr_mit_explainer,
                                 protected = compas_train_mit$Ethnicity,
                                 privileged = 'Caucasian')

print(fobject_knn_lr_mit)
plot(fobject_knn_lr_mit)




#############################
###### RESAMPLING ###########
#############################

uniform_indexes <- fairmodels::resample(protected = compas_train$Ethnicity,
                            y = y_numeric)

compas_resampled <- compas_train[uniform_indexes, ]

resamples <- vfold_cv(compas_resampled, 5)


## KNN resampled

knn_mod <- 
  nearest_neighbor(neighbors = tune()) %>% 
  set_engine('kknn') %>% 
  set_mode("classification")

knn_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_resampled) %>% 
  step_bagimpute(everything()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>% 
  step_dummy(Sex)

knn_wf <- 
  workflow() %>% 
  add_model(knn_mod) %>% 
  add_recipe(knn_rec)

knn_res <- 
  knn_wf %>% 
  tune_grid(resamples = resamples,
            metrics = metric_set(roc_auc, accuracy),
            control = control_grid(save_pred = TRUE))

knn_best <-
  knn_res %>%
  select_best(metric = "accuracy")

knn_roc <- 
  knn_res %>% 
  collect_predictions(parameters = knn_best) %>% 
  roc_curve(Two_yr_Recidivism, .pred_0) %>% 
  mutate(model = "Ridge LR")

knn_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
  geom_path() +
  geom_abline(lty = 3) + 
  coord_equal() 

knn_acc <- 
  knn_res %>% 
  collect_predictions(parameters = knn_best) %>% 
  accuracy(Two_yr_Recidivism, .pred_class) %>% 
  mutate(model = "Ridge LR")

knn_results <- 
  knn_res %>% 
  collect_predictions(parameters = knn_best)

final_wf <-
  finalize_workflow(knn_wf, knn_best)

# Fit 
knn_mit_resampled <- 
  final_wf %>% 
  fit(data = compas_resampled)

# Explainer
knn_res_explainer <- explain_tidymodels(knn_mit_resampled, 
                                        data = compas_resampled[,-1], 
                                        y = y_numeric,
                                        label = "KNN Resampled")

## LR resampled

lr_mod <- 
  logistic_reg(penalty = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

lr_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_resampled) %>% 
  step_rm(Ethnicity) %>% 
  step_bagimpute(everything(), -all_outcomes()) %>% 
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

lr_roc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(Two_yr_Recidivism, .pred_0) %>% 
  mutate(model = "Ridge LR")

lr_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
  geom_path() +
  geom_abline(lty = 3) + 
  coord_equal() 

lr_acc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  accuracy(Two_yr_Recidivism, .pred_class) %>% 
  mutate(model = "Ridge LR")

lr_results <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best)

final_wf <-
  finalize_workflow(lr_wf, lr_best)

# Fit 
lr_mit_resampled <- 
  final_wf %>% 
  fit(data = compas_resampled)


# Explainer
lr_res_explainer <- explain_tidymodels(lr_mit_resampled, 
                                       data = compas_resampled[,-1], 
                                       y = as.numeric(compas_resampled$Two_yr_Recidivism)-1,
                                       label = "Logistic Regression Resampled")


## Fairness object
fobject_knn_lr_res <- fairness_check(knn_explainer, lr_explainer, knn_res_explainer, lr_res_explainer,
                                     protected = compas_resampled$Ethnicity,
                                     privileged = 'Caucasian')

print(fobject_knn_lr_res)
plot(fobject_knn_lr_res)
