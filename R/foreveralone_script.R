library(tidyverse)
alone <- read.csv("data/foreveralone.csv")

alone %>% select(-employment, -time, -edu_level, -job_title, -improve_yourself_how, -what_help_from_others) %>% 
  mutate_if(is.character, as_factor) -> alone

library(tidymodels)
library(modeldata)

set.seed(923)

split <- initial_split(alone, prop = 0.75, strata = "depressed")
alone_train <- training(split)
alone_test <- testing(split)

resamples <- vfold_cv(alone_train, 5)


# L2-regularized logistic regression
lr_mod <- 
  logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

lr_rec <- 
  recipe(depressed ~ ., data = alone_train) %>%
  step_rm(gender) %>% 
  step_bagimpute(everything()) %>%
  step_dummy(race, sexuallity, bodyweight, virgin, income,
             prostitution_legal, pay_for_sex, social_fear, attempt_suicide) %>%
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
            control = control_grid(save_pred = TRUE))

lr_best <-
  lr_res %>%
  select_best(metric = "roc_auc")

lr_roc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(depressed, .pred_Yes) %>% 
  mutate(model = "Ridge LR")

lr_pred <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best)

lr_pred$gender <- alone_train$gender

accuracy(filter(lr_pred, gender == "Female"), truth = depressed, estimate = .pred_class)
accuracy(lr_pred, truth = depressed, estimate = .pred_class)


bind_rows(lr_roc) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) + 
  geom_path() +
  geom_abline(lty = 3) + 
  coord_equal()

