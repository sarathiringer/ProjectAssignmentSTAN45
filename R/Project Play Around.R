
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
                              race == "4. Other" ~ "Non-white"))

# Setting aside test data
set.seed(1)
split <- initial_split(wage, prop = 0.85)
wage_train <- training(split)
wage_test <- testing(split)

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
