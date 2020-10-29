load("report/models/fobject1.Rdata")

load("report/models/rf_comp.Rdata")

fap <- performance_and_fairness(fobject1, fairness_metric = "STP",
                                performance_metric = "accuracy")
x <- plot(fap)
x + 
  ggtitle("") +
  labs(x = "Accuracy", y = "Inversed parity loss (demographic parity)", color = "Model") +
  geom_point(size = 5)




df <- data.frame(fap$paf_data)

ggplot(df, aes(x = performance_metric, y = fairness_metric, col = labels)) +
  geom_point(size = 4) +
  geom_text(aes(label=labels), hjust=1, vjust=2) +
  labs(x = "Accuracy", y = "Inversed parity loss (demographic parity)", color = "Model")

load("report/models/fobject_all.Rdata")
fap_all <- performance_and_fairness(fobject_all, fairness_metric = "STP",
                                performance_metric = "accuracy")

x <- plot(fap_all)
x + 
  ggtitle("") +
  labs(x = "Accuracy", y = "Inversed parity loss (demographic parity)") +
  geom_point(aes(col = df$group), size = 3) +
  scale_color_manual(name="Model",
                      values=rep(brewer.pal(4, 'Set1'), 4),
                      guide = guide_legend(override.aes=aes(fill=NA)))



df <- data.frame(fap_all$paf_data)
df$group <- rep(c("RF", "ANN", "AdaBoost", "LR"),4)
df$method <- c(rep("Resampling pref", 4), 
               rep("Resmapling uni", 4), 
               rep("Disparate impact remover", 4), 
               rep("None", 4))
df$method_label <- c(rep("resa pref", 4), 
                     rep("resa uni", 4), 
                     rep("rem", 4), 
                     rep("none", 4))

df$jit<-with(df, ifelse(labels == "RF resa unif" | labels == "LR rem", 0.01, 0.02))

ggplot(df, aes(x = performance_metric, y = fairness_metric, col = group)) +
  geom_point(aes(shape = method), size = 4) +
  geom_text(aes(label = method_label),
                fontface = "bold",
                position=position_jitter(width=df$jit,height=df$jit)) +
  labs(x = "Accuracy", 
       y = "Inversed parity loss (demographic parity)", 
       color = "Model",
       shape = "Bias Mitigation Method")

df$labels <- str_wrap(df$labels, 5)

ggplot(df, aes(x = performance_metric, y = fairness_metric, col = group)) +
  geom_point(size = 4) +
  geom_text(aes(label = labels), nudge_y = 0.02, nudge_x = -0.003, lineheight = 0.9) +
  labs(x = "Accuracy", 
       y = "Inversed parity loss (demographic parity)", 
       color = "Model") +
  theme_bw() +
  xlim(c(0.63, 0.69)) +
  ylim(c(0.1, 0.8))


## Disparate impact remover

summary(compas_train)

df_rem <- compas_train[,c("Number_of_Priors", "Ethnicity")]
df_rem$Number_of_Priors_Rem1.0 <- disparate_impact_remover(df_rem, protected = 'Ethnicity', features_to_transform = 'Number_of_Priors', lambda = 1.0)$Number_of_Priors

g1 <- ggplot(df_rem, aes(x = Number_of_Priors, color = Ethnicity, fill = Ethnicity)) +
  geom_density(alpha = 0.4) +
  theme_bw() +
  labs(subtitle = "No removal") +
  xlim(c(0, 20)) +
  theme(legend.position = 'none')

g2 <- ggplot(df_rem, aes(x = Number_of_Priors_Rem1.0, color = Ethnicity, fill = Ethnicity)) +
  geom_density(alpha = 0.4) +
  labs(subtitle = "1.0 removal") +
  theme_bw() +
  xlim(c(0, 20)) +
  theme(legend.position = 'none')

gridExtra::grid.arrange(g1, g2, ncol=2)


## Descriptives

df_des <- compas

ggplot(df_des, aes(x = Ethnicity, fill = Ethnicity)) +
  geom_bar() +
  theme_bw() +
  scale_fill_brewer(palette="Dark2") +
  labs(y = "Number of observations")

ggplot(df_des, aes(x = Sex, fill = Sex)) +
  geom_bar() +
  theme_bw() +
  scale_fill_brewer(palette="Dark2") +
  labs(y = "Number of observations")

ggplot(df_des, aes(x = Two_yr_Recidivism, fill = Two_yr_Recidivism)) +
  geom_bar() +
  theme_bw() +
  scale_fill_brewer(palette="Dark2") +
  labs(y = "Number of observations")

summary(compas)



load("models/fobject_all.Rdata")
fap <- performance_and_fairness(fobject_all, fairness_metric = "STP",
                                performance_metric = "accuracy")
x <- plot(fap)
x + 
  ggtitle("") +
  labs(x = "Accuracy", y = "Inversed parity loss (demographic parity)", color = "Model") +
  theme(legend.position = "",
        axis.title=element_text(size=14))

plot(x)


## 

load("models/fobject1.Rdata")

fobject1.2 <- expand_fairness_object(fobject1)

fobject_2 <- fobject1.2 %>% 
  filter(metric == "TPR" | metric == "STP") %>% 
  mutate(metric = case_when(metric == "TPR" ~ "Equalized odds",
                            metric == "STP"  ~ "Demographic parity"))
  
ggplot(fobject_2, aes(x = model, y = score, fill = model)) +
  geom_col() +
  geom_hline(yintercept = 0.2, linetype = 'dashed') +
  coord_flip() +
  theme_minimal() +
  labs(x = '', y = 'Fairness Metric', fill = 'Model') +
  scale_fill_brewer(palette="Set2") +
  facet_grid(metric ~ .) +
  theme(text = element_text(size = 15))

ggplot(fobject_2, aes(x = metric, y = score, fill = model)) +
  geom_bar(stat="identity", position="dodge") +
  geom_hline(yintercept = 0.2, linetype = 'dashed') +
  coord_flip() +
  theme_minimal() +
  labs(x = '', y = 'Fairness Metric', fill = 'Model') +
  scale_fill_brewer(palette="Set2")
>>>>>>> sara
