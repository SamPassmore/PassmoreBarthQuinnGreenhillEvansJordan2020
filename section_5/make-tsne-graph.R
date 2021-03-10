library(stringr)
library(RColorBrewer)
library(ggplot2)
library(dplyr)

tsne = read.csv('public/analysis/g0_clustering_uniqueglottocodes_removeids_mcs10ms3.csv', stringsAsFactors = FALSE)
tsne = read.csv('Supplementary_code/section_5/results/g0_embeddings.csv', header = FALSE)
colnames(tsne) = c("x", "y")

hdbscan = read.csv('Supplementary_code/section_5/results/g0.csv')

tsne = cbind(tsne, hdbscan)

tsne$glottocode = str_extract(tsne$Language_ID, "[a-z]{4}[0-9]{4}")

#### Get new labels & Colours ####
new_labels = read.csv('Supplementary_code/section_5/data/cluster_labels.csv')

tsne = left_join(tsne, new_labels)

# organise for plotting
tsne$alpha = scales::rescale(tsne$cluster_prob, c(0.5, 1), c(0, 1))


plot_1 = ggplot(tsne, aes(x=x,y=y,group=cluster_label,col=cluster_label)) + 
  geom_point(cex = 4) + 
  scale_color_manual(values=new_labels$cols) + 
  xlab("") + ylab("") + 
  theme_minimal() + 
  # annotate("text", x = 214, y = 223.65, label = "Cross-parallel & Sudanese", size = 7) + 
  # annotate("text", x = 208, y = 225.75, label = "Eskimo", size = 7) + 
  # annotate("text", x = 210, y = 220.85, label = "Hawaiian", size = 7) +
  guides(col=guide_legend(nrow=3,byrow=FALSE)) + 
  theme(legend.title = element_blank(), axis.text=element_blank(),
        legend.position = c(0.8, 0.25), 
        legend.box.background = element_rect(colour = "black"), 
        text = element_text(size=20)) 
  
plot_1
ggsave('results/plot_by_type.pdf', plot_1)

## link with dplace
dplace = read.csv('~/Projects_Git/dplace-data/datasets/EA/data.csv') %>% 
  filter(var_id == "EA027") %>% 
  select(soc_id, code)
societies = read.csv('~/Projects_Git/dplace-data/datasets/EA/societies.csv') %>% 
  select(id, glottocode)
codes = read.csv('~/Projects_Git/dplace-data/datasets/EA/codes.csv') %>% 
  filter(var_id == "EA027") %>% 
  select(code, name)

## collapse Iroquois, Omaha, & Crow
codes$recode = ifelse(codes$name %in% c("Iroquois", "Omaha", "Crow"), "Cross-Parallel", as.character(codes$name))

# link society ids w/ glottocodes  
cousin_typology = left_join(dplace, societies, by = c("soc_id" = "id")) %>%
  left_join(., codes, "code")
cousin_typology$recode = ifelse(cousin_typology$recode %in% c("Mixed", "Sudanese", "Missing data"), NA, as.character(cousin_typology$recode))


# link tsne data with dplace data
tsne = left_join(tsne, cousin_typology, "glottocode")
tsne$code = as.factor(tsne$code)

# plot coloured points ontop of missing
tsne_missing = tsne[is.na(tsne$recode),]
tsne_dplace = tsne[!is.na(tsne$recode),]

highlight = tsne[tsne$Language_ID %in% c("v_mode1248_281", "p_chep1245_611"),]
highlight$name = c("Chepang", "Modern Greek")

plot_2 = ggplot() + 
  geom_point(data = highlight, aes(x =x, y=y), cex = 7, colour="black") + 
  geom_point(data = tsne_missing, aes(x =x, y=y), col = "grey50", cex = 5, alpha = 0.5) + 
  geom_point(data = tsne_dplace, aes(x =x, y=y, col = recode), cex = 5) +
  geom_text(data = highlight, aes(x =x, y=y, label=name),hjust=1.17, vjust=0) + 
  xlab("") + ylab("") + 
  scale_color_manual(values=c("#1389E6", "#FED976", "#1C8364", "#F46F9C", "red")) + 
  theme_minimal() + 
  theme(legend.title = element_blank(), axis.text=element_blank(),
        legend.position = c(0.8, 0.25), 
        legend.box.background = element_rect(colour = "black"), 
        text = element_text(size=20)) 

plot_2

ggsave("results/plot_by_dplace.pdf", plot_2)
