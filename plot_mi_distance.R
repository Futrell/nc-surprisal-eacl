library(ggplot2)
library(dplyr)
library(tidyr)
library(latex2exp)

P_CUTOFF = .005

# Less than 500 sentences in UD 1.4
bad_langs = c('swl', 'cop', 'sa', 'uk')

typology = read.csv("typology.csv") %>% select(lang, lang_name)
d = read.csv("hdmi_distance.csv") %>%
  select(-X) %>%
  filter(!(lang %in% bad_langs)) %>%
  inner_join(typology) %>%
  mutate(lang=lang_name) %>%
  select(-lang_name)

d %>%
 select(mi_base, mi_dep, mi_p, k, lang) %>%
 gather(key, value, -k, -lang, -mi_p) %>%
 mutate(mi_p=ifelse(mi_p < P_CUTOFF, "*", "")) %>%
 mutate(key=ifelse(key=="mi_base", "Baseline", "Head-Dependent")) %>%
 mutate(key=factor(key, levels=c("Head-Dependent", "Baseline"))) %>%
 mutate(k=k+1) %>%
 ggplot(aes(x=k, y=value, color=key)) +
  geom_line() +
  facet_wrap(~lang, ncol=8) +
  theme_bw() +
  xlim(1, 5) +
  ylim(0, 1.1) +
  geom_text(aes(x=k, y=1, label=mi_p), color="black", show.legend=F) +
  ylab(TeX("I(w_i; w_{i+k})")) +
  xlab("k (Distance in words)") +
  theme(legend.position=c(.84, .07), legend.title=element_blank())  

ggsave("hdmi_position_mi.pdf", height=6, width=9)
  

d %>%
 select(pmi_base, pmi_dep, pmi_p, k, lang) %>%
 gather(key, value, -k, -lang, -pmi_p) %>%
 mutate(pmi_p=ifelse(pmi_p < P_CUTOFF, "*", "")) %>%
 mutate(key=ifelse(key=="pmi_base", "Baseline", "Head-Dependent")) %>%
 mutate(key=factor(key, levels=c("Head-Dependent", "Baseline"))) %>%
 mutate(k=k+1) %>%
 ggplot(aes(x=k, y=value, color=key)) +
  geom_line() +
  facet_wrap(~lang, ncol=8) +
  theme_bw() +
  xlim(1, 5) +
  ylim(-.1, .8) +
  geom_text(aes(x=k, y=.7, label=pmi_p), color="black", show.legend=F) +
  ylab(TeX("pmi(w_i; w_{i+k})")) +
  xlab("k (Distance in words)") +  
  theme(legend.position=c(.84, .07), legend.title=element_blank())

ggsave("hdmi_position.pdf", height=6, width=9)

