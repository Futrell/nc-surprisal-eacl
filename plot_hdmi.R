library(ggplot2)
library(dplyr)
library(tidyr)

typology = read.csv("typology.csv")

d = read.csv("hdmi_with_permutation_test.csv") %>%
  select(-X)

## Langs with less than 500 sentences in UD 1.4
bad_langs = c('swl', 'cop', 'sa', 'uk')

d %>%
  filter(variable %in% c("hdmi", "gdmi", "ssmi")) %>%
  gather(lang, value, -variable) %>%
  filter(!(lang %in% bad_langs)) %>%
  inner_join(typology) %>%
  mutate(variable=ifelse(variable == "hdmi", "Head-Dependent",
                  ifelse(variable == "ssmi", "Sister-Sister", "Grandparent-Dependent"))) %>%
  mutate(variable=factor(variable, levels=c("Head-Dependent",
                                             "Sister-Sister",
					     "Grandparent-Dependent"))) %>%
  rename(Relation=variable) %>%					     
  ggplot(aes(x=Relation, y=value, color=Relation, fill=Relation)) +
    geom_bar(stat='identity') +
    facet_wrap(~lang_name, ncol=8) +
    guides(color=F) +
    ylab("Mutual information (bits)") +
    theme_bw() + 
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
	  legend.position=c(.84, .07)
	  )

ggsave("hdmi_topologies.pdf", width=9, height=6)
