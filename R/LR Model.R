
library(fairmodels)
library(tidymodels)
library(glmnet)
library(DALEX)
library(DALEXtra)

# Data prep
data("compas")

summary(compas)

# Training and test set
set.seed(42)

split <- initial_split(compas, prop = 0.8, strata = "Two_yr_Recidivism")

compas_test <- testing(split)

test_ethnicity <-
  compas_test %>% 
  select(Ethnicity) %>% 
  mutate(as.factor(Ethnicity))

compas_test <- 
  compas_test %>% 
  select(-Ethnicity)

compas_train <- training(split)

train_ethnicity <-
  compas_train %>% 
  select(Ethnicity) %>% 
  mutate(as.factor(Ethnicity))

compas_train <- 
  compas_train %>% 
  select(-Ethnicity)

resamples <- vfold_cv(compas_train, 5)

## LOGISTIC REGRESSION

lr_mod <- 
  logistic_reg(penalty = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

lr_rec <- 
  recipe(Two_yr_Recidivism ~ ., data = compas_train) %>% 
  step_bagimpute(everything(), all_outcomes()) %>% 
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
  roc_curve(Two_yr_Recidivism, .pred_1) %>% 
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

model_fitted <- 
  lr_wf %>%
  fit(data = compas_train)

# final_wf <- 
#   finalize_workflow(lr_wf, lr_best)

##### Bias evaluation

# Dataframe prep
pred <- data.frame(lr_results, ethnicity = train_ethnicity)

# Quick vis
ggplot(data = pred, aes(.pred_class, group = Ethnicity)) +
  geom_bar(aes(y=..prop.., fill = factor(..x..)), stat = "count") +
  scale_y_continuous(labels=scales::percent) +
  ylab("Relative frequencies") +
  xlab("Prediction") +
  facet_grid(~Ethnicity) +
  theme(legend.position = "none")

# Create explainer by DALEX
lr_explainer <- explain_tidymodels(model_fitted, data = compas_train[,-1], y = as.numeric(compas_train$Two_yr_Recidivism))

fobject <- fairness_check(lr_explainer,
                          protected = train_ethnicity,
                          privileged = 'Caucasian')



# Exempel från dokumentationen
### EXAMPLE

summary(compas_train)

compas_train$Sex <- as.factor(ifelse(compas_train$Sex == "Female", 1, 0))

y_numeric <- as.numeric(compas_train$Two_yr_Recidivism) - 1

lr_model <- glm(Two_yr_Recidivism~.,
                data = compas_train,
                family=binomial(link="logit"))

explainer_lr <- DALEX::explain(lr_model, data = compas_train[,-1], y = y_numeric)

fobject_lr <- fairness_check(explainer_lr,
                          protected = train_ethnicity,
                          privileged = "Caucasian")

plot(fobject_lr)


glm_compas <- glm(Two_yr_Recidivism~., data=compas_train, family=binomial(link="logit"))

y_prob <- glm_compas$fitted.values

y_numeric <- as.numeric(compas_train$Two_yr_Recidivism) - 1

gm <- group_matrices(compas_train$Ethnicity,
                     y_prob,
                     y_numeric,
                     cutoff = list(Asian = 0.45,
                                   African_American = 0.5,
                                   Other = 0.5,
                                   Hispanic = 0.5,
                                   Caucasian = 0.4,
                                   Native_American = 0.5))

gm # FP är allvarligast, skulle kunna plottas




