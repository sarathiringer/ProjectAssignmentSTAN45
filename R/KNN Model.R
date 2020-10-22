
install.packages('fairmodels')
library(fairmodels)
library(tidymodels)
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
  select(Ethnicity)

compas_test <- 
  compas_test %>% 
  select(-Ethnicity)

compas_train <- training(split)

train_ethnicity <-
  compas_train %>% 
  select(Ethnicity)

compas_train <- 
  compas_train %>% 
  select(-Ethnicity)

resamples <- vfold_cv(compas_train, 5)

## KNN MODEL

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
  roc_curve(Two_yr_Recidivism, .pred_1) %>% 
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

model_fitted <- 
  knn_wf %>%
  fit(data = compas_train)

##### Bias evaluation

# Dataframe prep
pred <- data.frame(knn_results, ethnicity = train_ethnicity)

# Quick vis
ggplot(data = pred, aes(.pred_class, group = Ethnicity)) +
  geom_bar(aes(y=..prop.., fill = factor(..x..)), stat = "count") +
  scale_y_continuous(labels=scales::percent) +
  ylab("Relative frequencies") +
  xlab("Prediction") +
  facet_grid(~Ethnicity) +
  theme(legend.position = "none")

# Group confusion matrix
group_matrices(protected = pred$Ethnicity, probs = ".pred_1", preds = pred$.pred_1, cutoff = 0.5)









