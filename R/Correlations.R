# Correlation checking

library(psych)

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

# Setting aside test data
set.seed(42)
split <- initial_split(wage1, prop = 0.85)
wage_train <- training(split)
race_cat_train <- wage_train$race_cat

# What variables to investigate?
summary(wage_train)

corr_data <- cbind(wage_train, race_cat = as.factor(race_cat_train))

summary(corr_data)

# Race och wage till att börja med
summary(aov(wage ~ race_cat, data = wage1))
kruskal.test(wage ~ race_cat, data = wage1)

# year, continous (?)
summary(aov(year ~ race_cat, data = corr_data))
kruskal.test(year ~ race_cat, data = corr_data)

# age, continous
summary(aov(age ~ race_cat, data = corr_data))
kruskal.test(age ~ race_cat, data = corr_data)

# maritl, 5 categories
install.packages('GoodmanKruskal')
library(GoodmanKruskal)
plot(GKtauDataframe(corr_data[,c(3,10)]))

# education, ordinal, 5 levels
plot(GKtauDataframe(corr_data[,c(4,10)]))

corr_data$education_num <- ifelse(corr_data$education == '1. < HS Grad', 1, 
                                  ifelse(corr_data$education == '2. HS Grad', 2,
                                         ifelse(corr_data$education == '3. Some College', 3, 
                                                ifelse(corr_data$education == '4. College Grad', 4,
                                                       ifelse(corr_data$education == '5. Advanced Degree', 5, NA)))))

ggplot(corr_data, aes(x = race_cat, y = education_num)) +
  geom_boxplot()

corr_data %>% 
  select(race_cat, education_num) %>% 
  group_by(race_cat) %>% 
  summarize(mean(education_num))

t.test(education_num ~ race_cat, data = corr_data)

# region, same for all, should be dropped

# jobclass, 2 levels
phi(table(wage_train$jobclass, race_cat_train), 2)

# health, 2 levels
phi(table(wage_train$health, race_cat_train), 2)

# health_ins, 2 levels
phi(table(wage_train$health_ins, race_cat_train), 2)
