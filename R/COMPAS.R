# Packages


# Load "raw" data
compas <- read.csv("data/compas-analysis-master/compas-scores-raw.csv")

summary(compas)

ggplot(compas, aes(x = Ethnic_Code_Text, y = RawScore)) +
  geom_boxplot()

# Load two years later
compas_2_years <- read.csv("data/compas-analysis-master/compas-scores-two-years.csv")

summary(compas_2_years)


library(grid)
library(gridExtra)
pblack <- ggplot(data=filter(compas_2_years, race =="African-American"), aes(ordered(decile_score))) + 
  geom_bar() + xlab("Decile Score") +
  ylim(0, 650) + ggtitle("Black Defendant's Decile Scores")
pwhite <- ggplot(data=filter(compas_2_years, race =="Caucasian"), aes(ordered(decile_score))) + 
  geom_bar() + xlab("Decile Score") +
  ylim(0, 650) + ggtitle("White Defendant's Decile Scores")
grid.arrange(pblack, pwhite,  ncol = 2)

pblack_v <- ggplot(data=filter(compas_2_years, race =="African-American"), aes(ordered(v_decile_score))) + 
  geom_bar() + xlab("Violent Decile Score") +
  ylim(0, 700) + ggtitle("Black Defendant's Violent Decile Scores")
pwhite_v <- ggplot(data=filter(compas_2_years, race =="Caucasian"), aes(ordered(v_decile_score))) + 
  geom_bar() + xlab("Violent Decile Score") +
  ylim(0, 700) + ggtitle("White Defendant's Violent Decile Scores")
grid.arrange(pblack_v, pwhite_v,  ncol = 2)

