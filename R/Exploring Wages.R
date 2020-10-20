library(tidyverse)
library(glmnet)
library(tidymodels)


# Wages (ej ISLR)

wage2 <- read.csv("data/wages.csv")

summary(wage2)

# Glassdoor

wage3 <- read.csv("data/Glassdoor Gender Pay Gap.csv")

summary(wage3)



# ISLR Wage
library(ISLR)
data(Wage)

# Data preparation

# Initial
wage1 <- 
  Wage %>% 
  mutate(race_cat = case_when(race == "1. White" ~ "White",
                              race == "2. Black" ~ "Black")) %>% 
  mutate(high_wage = as.factor(ifelse(wage > quantile(wage,  0.75), 1, 0))) %>%
  select(-race) %>% 
  na.omit()

# Check difference
ggplot(data = wage1, mapping = aes(x = high_wage, fill = race_cat)) +
  geom_bar(position = "fill")

# Another way of plotting the differences
ggplot(data = wage1, aes(high_wage, group = race_cat)) +
  geom_bar(aes(y=..prop.., fill = factor(..x..)), stat = "count") +
  scale_y_continuous(labels=scales::percent) +
  ylab("relative frequencies") +
  facet_grid(~race_cat) +
  theme(legend.position = "none")
  

# Pick out race, for evaluation
race_cat <- wage1$race_cat

# Take away what we don't want for training
wage_train <- 
  wage1 %>% 
  select(-c(race_cat, logwage, wage))


# Tuned model

resamples <- vfold_cv(wage_train, 5)

lr_mod <- 
  logistic_reg(penalty = tune()) %>%
  set_engine("glmnet") %>% 
  set_mode("classification")

lr_rec <- 
  recipe(high_wage ~ ., data = wage_train) %>%
  step_bagimpute(everything()) %>%
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
            control = control_grid(save_pred = TRUE))
lr_best <-
  lr_res %>%
  select_best(metric = "accuracy")

lr_roc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(high_wage, .pred_0) %>% 
  mutate(model = "Ridge LR")

lr_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
  geom_path() +
  geom_abline(lty = 3) + 
  coord_equal() 

lr_pred <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best)
  
lr_pred$race = wage1$race_cat

accuracy(filter(lr_pred, race == "White"), truth = high_wage, estimate = .pred_class)

accuracy(filter(lr_pred, race == "Black"), truth = high_wage, estimate = .pred_class)

accuracy(filter(lr_pred, race == "White"), estimate = .pred_1)

# Plot of counts of the different prediction classes
lr_pred %>% 
  ggplot(aes(x = .pred_class, fill = race)) +
  geom_bar()

lr_pred %>% 
  ggplot(aes(x = .pred_class)) +
  geom_bar() +
  facet_wrap(~race)

ggplot(data = lr_pred, aes(.pred_class, group = race)) +
  geom_bar(aes(y=..prop.., fill = factor(..x..)), stat = "count") +
  scale_y_continuous(labels=scales::percent) +
  ylab("relative frequencies, predicted") +
  facet_grid(~race) +
  theme(legend.position = "none")

lr_pred_white <- subset(lr_pred, race = "White")

lr_pred_black <- subset(lr_pred, race = "Black")



# Model without tuning

# Build
lr_mod <- 
  logistic_reg(penalty = 0) %>%
  set_engine("glmnet") %>% 
  set_mode("classification") %>% 
  fit(high_wage ~ ., data = wage_train)

# Predict
lr_pred <- 
  predict(lr_mod, wage_train)
  
# Inspect result
result_df <- data.frame(pred = lr_pred$.pred_class, true = wage1$high_wage, race = wage1$race_cat)
table(result_df)


