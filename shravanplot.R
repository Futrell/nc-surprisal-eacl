library(dplyr)
library(tidyr)
library(ggplot2)

# Numbers from Vasishth et al. (2010), generously shared by Shravan
d = data.frame(c(796.5574, 488.9209, 1198.114, 1897.468), c(54.91069, 23.40487, 71.50435, 185.43846))
names(d) = c("m", "se")
d$lang = c("en", "en", "de", "de")
d$gram = c("gram", "ungram", "gram", "ungram")

# To generate the model numbers below, in Python do:
# import experiments as e
# _, result = e.verb_forgetting_conditions(m=.5, r=.5, e=.2, s=.8)
# for English, and s=0 for German
# then divide the results by log(2) to convert to log base 2
d$model = c(1.9762488824551934, 1.0599082761959622, 1.0072683912728893, 1.9574831221960485)

# These numbers are the same setup except e=.1 (lower noise rate)
# d$model = c(1.3804167980470512, 1.2491010858905676, 0.6182342702110893, 2.476816002815201)

ymax = max(d$model) + .05

dgram = filter(d, gram == "gram")
dungram = filter(d, gram == "ungram")

ddiff = data.frame(dungram$m - dgram$m,
                   c(51.09891, 153.8608),
                   dungram$model - dgram$model,
		   c("English", "German"))
names(ddiff) = c("m", "se", "model", "lang")

ddiff = mutate(ddiff, upper=m+1.96*se, lower=m-1.96*se)

ggplot(ddiff, aes(x=lang, y=m, fill=lang, ymin=lower, ymax=upper)) +
  geom_bar(stat='identity') + ylab("(Ungrammatical - Grammatical) RT (ms)") +
  geom_errorbar() +
  guides(fill=F) +
  ylim(-500, 1001) +
  xlab("")

ggsave("../output/rt_bars.pdf", height=3.8, width=2.3)

ggplot(ddiff, aes(x=lang, y=model, fill=lang)) +
  geom_bar(stat='identity') + ylab("(Ungrammatical - Grammatical) surprisal (bits)") +
  guides(fill=F) +
  ylim(ymax*(-500/1001), ymax) + 
  xlab("")

ggsave("../output/model_bars.pdf", height=3.8, width=2.3)
