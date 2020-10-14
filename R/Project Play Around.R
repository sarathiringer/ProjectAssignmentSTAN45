
# Preparations
library(ISLR)
library(tidymodels)
library(ggplot2)
data(Wage)

# Fixing binary category
wage <- Wage %>% 
  mutate(race_cat = case_when(race == "1. White" ~ "White",
                              race == "2. Black" ~ "Non-white",
                              race == "3. Asian" ~ "Non-white",
                              race == "4. Other" ~ "Non-white")) %>% 
  select(-race)

# Setting aside test data
set.seed(1)
split <- initial_split(wage, prop = 0.85)
wage_train <- training(split)
wage_test <- testing(split)

resamples <- vfold_cv(wage_train, 5)
# Check number of observations
summary(wage_train$race)

# Boxplot for all racial categories
ggplot(data = wage_train, mapping = aes(x = race, y = wage)) +
  geom_boxplot()

# Boxplot binary categories
ggplot(data = wage_train, mapping = aes(x = race_cat, y = wage)) +
  geom_boxplot()

mean <- wage_train %>% 
  group_by(race_cat) %>% 
  summarise(mean_wage = mean(wage))
  
## Process for t test
df <- wage_train %>%
  filter(race_cat == "White" | race_cat == "Non-white") %>%
  select(race_cat, wage)

# Summary whites
summary(df %>% filter(race_cat == "White") %>% .$wage)

# Summary non-whites
summary(df %>% filter(race_cat == "Non-white") %>% .$wage)

# Boxplots again
ggplot(df, aes(race_cat, wage)) +
  geom_boxplot()

# Check distribution
ggplot(df, aes(wage)) +
  geom_density(aes(fill = race_cat), alpha = 0.6)

ggplot(df, aes(wage)) +
  geom_histogram(fill = "white", color = "grey30") +
  facet_wrap(~ race_cat)

# t-test
t.test(wage ~ race_cat, data = df)

t.test(log(wage) ~ race_cat, data = df)

wilcox.test(wage ~ race_cat, data = df)


# Building a multiple regression model

lr_mod <- 
  linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

lr_rec <- 
  recipe(wage ~ ., data = wage_train) %>%
  step_bagimpute(everything()) %>%
  step_dummy(all_nominal()) 

# %>%
#   step_nzv(everything(), -all_outcomes()) %>%
#   step_normalize(everything(), -all_outcomes())

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
  roc_curve(Status, .pred_bad) %>% 
  mutate(model = "regression")

wage$region <- droplevels(wage$region)

summary(fit <- lm(wage ~ maritl + education + jobclass + health + health_ins + age + race_cat, data = wage))

wage$pred <- predict(fit)

wage %>% group_by(race_cat) %>%
  summarise(mean_pred = mean(pred))

