# Packages


# Load "raw" data
compas <- read.csv("data/compas-analysis-master/compas-scores-raw.csv")

summary(compas)

ggplot(compas, aes(x = Ethnic_Code_Text, y = RawScore)) +
  geom_boxplot()

# Load two years later
compas_2_years <- read.csv("data/compas-analysis-master/compas-scores-two-years.csv")

summary(compas_2_years)
