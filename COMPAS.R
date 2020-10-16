# Packages


# Load data
compas <- read.csv("data/compas-analysis-master/compas-scores-raw.csv")

summary(compas)

ggplot(compas, aes(x = Ethnic_Code_Text, y = RawScore)) +
  geom_boxplot()
