library(ggplot2)
library(dplyr)
library(tidyr)
library(forcats)
library(ggtext)
library(stringr)

# https://community.rstudio.com/t/font-gets-really-small-when-saving-to-png-using-ggsave-and-showtext/147029/7
# library(showtext)
# showtext_auto(enable=FALSE)
# showtext_opts(dpi=300)
# bug with gggtext https://github.com/yixuan/showtext/issues/67#issue-1913616939

PATH = "segmentation3d"
PATH_INPUT = "data"
PATH_OUTPUT = file.path(PATH, "results")

df_b50_raw = read.csv(file.path(PATH_INPUT, "b50_explanations_metrics_ig_tsv2.csv"), row.names=1)
df_tsv2_raw = read.csv(file.path(PATH_INPUT, "tsv2_explanations_metrics_ig.csv"), row.names=1)

dict_class_group = list( 
    # dict from tsv2 https://github.com/wasserth/TotalSegmentator/blob/master/resources/imgs/overview_classes_2.png
    "aorta"="cardiovascular system",
    "autochthon left"="muscles",
    "autochthon right"="muscles",
    "costal cartilages"="skeleton", 
    "heart"="cardiovascular system",
    "lung lower lobe left"="other organs",
    "lung lower lobe right"="other organs",
    "lung middle lobe right"="other organs",
    "lung upper lobe left"="other organs",
    "lung upper lobe right"="other organs",
    "pulmonary vein"="cardiovascular system",
    "ribs"="skeleton", 
    "sternum"="skeleton",
    "thyroid gland"="other organs",
    "trachea"="other organs",
    "vertebraes"="skeleton",
    "pleural effusion"="pathology",
    "consolidation"="pathology"
)

dict_group_color = list(
    "cardiovascular system"="#DB4437",
    "muscles"="#F4B400",
    "skeleton"="#4285F4",
    "other organs"="gray50",
    "pathology"="black"
)

convert_from_wide_to_long = function(result, aggregate=FALSE) {
    df = data.frame()
    for (column_name in colnames(result)) {
        if (str_detect(column_name, "mass")) {
            to_from = str_split(str_remove(column_name, "_mass"), "_explanation_in_")[[1]]
            if (aggregate) {
                importance = mean(result[,column_name], na.rm=TRUE)
            } else {
                importance = na.omit(result[,column_name])
            }
            df = rbind(df, data.frame(
                from = str_replace_all(to_from[2], "_", " "),
                to = str_replace_all(to_from[1], "_", " "),
                importance = importance
            ))
        }
    }
    df
}

# --------------------------- pathologies in b50
df_b50_pleural_effusion_raw = read.csv("data/b50_explanations_metrics_ig_tsv2_fluid.csv", row.names=1)
df_b50_consolidation_raw = read.csv("data/b50_explanations_metrics_ig_tsv2_ggo_consolidation.csv", row.names=1) %>% 
    select(contains("joined"))
colnames(df_b50_pleural_effusion_raw) = str_replace(colnames(df_b50_pleural_effusion_raw), "fluid", "pleural_effusion")
colnames(df_b50_consolidation_raw) = str_replace(colnames(df_b50_consolidation_raw), "joined_custom_mask", "consolidation")
df_b50_pathologies_long = rbind(
    convert_from_wide_to_long(df_b50_pleural_effusion_raw, aggregate=FALSE),
    convert_from_wide_to_long(df_b50_consolidation_raw, aggregate=FALSE)
)
dim(df_b50_pathologies_long)
df_b50_pathologies_long_aggregated = rbind(
    convert_from_wide_to_long(df_b50_pleural_effusion_raw, aggregate=TRUE),
    convert_from_wide_to_long(df_b50_consolidation_raw, aggregate=TRUE)
)
dim(df_b50_pathologies_long_aggregated)
# ---------------------------


df_b50_long = convert_from_wide_to_long(df_b50_raw, aggregate=FALSE)
dim(df_b50_long)
df_tsv2_long = convert_from_wide_to_long(df_tsv2_raw, aggregate=FALSE)
dim(df_tsv2_long)

df_b50_long_aggregated = convert_from_wide_to_long(df_b50_raw, aggregate=TRUE)
dim(df_b50_long_aggregated)
df_tsv2_long_aggregated = convert_from_wide_to_long(df_tsv2_raw, aggregate=TRUE)
dim(df_tsv2_long_aggregated)


# ---------------------------
df_long = df_b50_long
df_long_aggregated = df_b50_long_aggregated
DATASET_FILE = "b50"
DATASET_NAME = "B50"

df_long = rbind(df_b50_long, df_b50_pathologies_long)
df_long_aggregated = rbind(df_b50_long_aggregated, df_b50_pathologies_long_aggregated)
DATASET_FILE = "b50plus"
DATASET_NAME = "B50"

df_long = df_tsv2_long 
df_long_aggregated = df_tsv2_long_aggregated
DATASET_FILE = "tsv2"
DATASET_NAME = "TSV2"
# ---------------------------


df_long = df_long %>%
    filter(from != "background", to != "background") %>%
    mutate(group=unlist(dict_class_group[as.character(from)]))

df_long_aggregated = df_long_aggregated %>%
    filter(from != "background", to != "background") %>%
    mutate(group=unlist(dict_class_group[as.character(from)]))


theme_custom <- function(color="black") {
    theme_bw(base_line_size = 0) %+replace%
    theme(
        text = element_text(color = "black", size = 12, family = "serif"),
        axis.ticks = element_blank(), 
        legend.background = element_blank(),
        legend.key = element_blank(), 
        panel.background = element_blank(),
        panel.border = element_blank(), 
        strip.background = element_blank(),
        plot.background = element_blank(), 
        complete = TRUE,
        legend.direction = "horizontal", 
        legend.position = "none",
        plot.title = element_markdown(color = color, size = 14, hjust = 0), #, face = "bold"),
        plot.subtitle = element_text(color = color, hjust = 0),
        axis.title = element_text(color = color, size = 13), #, face = "bold"),
        axis.text = element_text(color = color, size = 12),
        strip.text = element_text(color = color, size = 12, hjust = 0),
        panel.grid.major.x = element_line(color = "grey90", linewidth = 0.5, linetype = 1),
        panel.grid.minor.x = element_line(color = "grey90", linewidth = 0.5,  linetype = 1),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.y = element_blank()
    )
}

df0 = df_long %>%
    filter(from %in% c("pulmonary vein", "aorta", "heart"),
        to %in% c("pulmonary vein", "aorta", "heart"))
p0 = ggplot(df0) +
    geom_histogram(aes(x=importance, fill = ifelse(from == to, "1", "0")), color="black", bins=13) +
    scale_fill_manual(values=c("white", "#DB4437")) +
    facet_grid(cols=vars(from), rows=vars(to)) +
    # scale_y_log10() +
    scale_x_continuous(labels = function(x) stringr::str_replace(sprintf("%g", x), "0.0", ".0")) +
    labs(
        title=paste0(DATASET_NAME, ": Global aggregated importance between <span style = 'color: #DB4437'>heart-related</span> classes"),
        y="Count", x="Importance"
    ) + theme_custom()
p0
ggsave(file.path(PATH_OUTPUT, paste0("global_heart_", DATASET_FILE, ".pdf")), width=7, height=5.5)


df1 = df_long %>% 
    filter(to=="lung lower lobe left") %>%
    mutate(from=fct_reorder(from, importance, .na_rm = TRUE))
p1 = ggplot(df1) +
    geom_boxplot(aes(x=importance, y=from, color=group)) +
    scale_color_manual(values=unlist(dict_group_color)) +
    labs(
        title=paste0(DATASET_NAME, ": Explanation of segmenting <span style = 'color: gray50'>lung lower lobe left</span>"),
        y="Feature", x="Importance"
        ) +
    theme_custom() +
    theme(
        axis.text.y = element_text(colour = unlist(dict_group_color[unlist(dict_class_group[levels(df1$from)])])),
        plot.title = element_markdown(hjust = 0)
    )
p1
ggsave(file.path(PATH_OUTPUT, paste0("global_lung_lower_lobe_left_", DATASET_FILE, ".pdf")), width=7, height=5.5)

df2 = df_long_aggregated %>% 
    filter(to=="lung lower lobe left") %>%
    mutate(from=fct_reorder(from, importance, .na_rm = TRUE))
p2 = ggplot(df2) +
    geom_col(aes(x=importance, y=from, fill=group), color="black") +
    scale_fill_manual(values=unlist(dict_group_color)) +
    labs(
        title=paste0(DATASET_NAME, ": Explanation of segmenting <span style = 'color: gray50'>lung lower lobe left</span> for patient ID 1"),
        y="Feature", x="Importance"
        ) +
    theme_custom() +
    theme(
        axis.text.y = element_text(colour = unlist(dict_group_color[unlist(dict_class_group[levels(df2$from)])])),
        plot.title = element_markdown(hjust = 1.1)
    )
p2
ggsave(file.path(PATH_OUTPUT, paste0("local_lung_lower_lobe_left_", DATASET_FILE, ".pdf")), width=7, height=5.5)

# ---------------------------

df3 = df_long %>% 
    filter(to=="ribs") %>%
    mutate(from=fct_reorder(str_replace_all(from, "_", " "), importance, .na_rm = TRUE))  %>%
    mutate(group=unlist(dict_class_group[as.character(from)]))
p3 = ggplot(df3) +
    geom_boxplot(aes(x=importance, y=from, color=group)) + 
    scale_color_manual(values=unlist(dict_group_color)) +
    labs(
        title=paste0(DATASET_NAME, ": Explanation of segmenting <span style = 'color: #4285F4'>ribs</span>"),
        y="Feature", x="Importance"
        ) +
    theme_custom() +
    theme(
        axis.text.y = element_text(colour = unlist(dict_group_color[unlist(dict_class_group[levels(df3$from)])])),
        plot.title = element_markdown(hjust = 0)
    )
p3
ggsave(file.path(PATH_OUTPUT, paste0("global_ribs_", DATASET_FILE, ".pdf")), width=5, height=4)

df4 = df_long %>% 
    filter(to=="pulmonary vein") %>%
    mutate(from=fct_reorder(str_replace_all(from, "_", " "), importance, .na_rm = TRUE)) %>%
    mutate(group=unlist(dict_class_group[as.character(from)]))
p4 = ggplot(df4) +
    geom_boxplot(aes(x=importance, y=from, color=group)) + 
    scale_color_manual(values=unlist(dict_group_color)) +
    labs(
        title=paste0(DATASET_NAME, ": Explanation of segmenting <span style = 'color: #DB4437'>pulmonary vein</span>"),
        y=NULL, x="Importance"
        ) +
    theme_custom() +
    theme(
        axis.text.y = element_text(colour = unlist(dict_group_color[unlist(dict_class_group[levels(df4$from)])])),
        plot.title = element_markdown(hjust = 0)
    )
p4
ggsave(file.path(PATH_OUTPUT, paste0("global_pulmonary_vein_", DATASET_FILE, ".pdf")), width=6, height=4)
# ---------------------------

###-------------------------- graph
library(igraph)

plot_graph = function(graph, layout=layout_nicely) {
    plot(
        graph, 
        # vertex.label.font=2,
        vertex.label.family="Times",
        vertex.label.color="black",
        vertex.label.cex=0.75,
        vertex.label.dist=2,
        edge.color="black",
        edge.lty=1,
        edge.arrow.size=.33,
        edge.curved=.1,
        layout=layout
    )
}
plot_graph(tnet)

plot_graph2 = function(graph, layout=layout_nicely) {
    plot(
        graph, 
        vertex.shape="circle",
        vertex.size=12,
        # vertex.label.family doesnt work
        vertex.label.font=2,
        vertex.label.color="black",
        vertex.label.cex=0.75,
        vertex.label.dist=2,
        edge.color="black",
        edge.lty=1,
        edge.arrow.size=.33,
        edge.curved=.1,
        layout=layout
    )
}
set.seed(1)
plot_graph2(tnet, layout=layout_with_gem)


#--------------------------------
df_graph = df_long_aggregated %>%
    mutate(
        from=str_replace_all(from, "_", " "),
        to=str_replace_all(to, "_", " "),
        weight=importance
    ) %>%
    select(-importance) %>%
    group_by(to) %>%
    arrange(by=weight) %>%
    top_n(5) %>%
    ungroup()

# top35mass
df_graph = df_long_aggregated %>%
    mutate(
        from=str_replace_all(from, "_", " "),
        to=str_replace_all(to, "_", " "),
        weight=importance
    ) %>%
    select(-importance) %>%
    group_by(to) %>%
    arrange(by=desc(weight)) %>%
    mutate(cumweight=cumsum(weight)) %>%
    filter(cumweight <= 0.35) %>% # 0.35 tsv2, 0.37 b50
    ungroup() %>%
    mutate(-cumweight)
#--------------------------------

table(df_graph$from)

net = graph_from_data_frame(d=df_graph, directed=T) 

plot_graph(net)

E(net)$weight
summary(E(net)$weight)

# tnet = delete_edges(net, E(net)[weight<=quantile(weight, probs=0.75)])
tnet = net
V(tnet)$color = unlist(dict_group_color[unlist(dict_class_group[names(V(tnet))])])
hist(E(tnet)$weight)

eps = 0.02
E(tnet)$width = 2*(E(tnet)$weight + eps) / (max(E(tnet)$weight) + eps)
# E(tnet)$width <- log(1+E(tnet)$weight)
# E(tnet)$width <- 3*E(tnet)$weight
# E(tnet)$width <- 2*sqrt(E(tnet)$weight)
# eps = 0.05
# E(tnet)$width =  (E(tnet)$weight + eps) / (max(E(tnet)$weight) + eps)
# eps = 0.1
# eb = edge_betweenness(tnet, directed=TRUE, weights=E(tnet)$weight)
# E(tnet)$width = 2 * (eb + eps) / max(eb)

hist(E(tnet)$width)

#-- nicely, dh, fr, gem

#-- b50
set.seed(81) # b50 top5 nicely
plot_graph(tnet, layout=layout_nicely)
# set.seed(28) # b50 top37mass gem
# plot_graph(tnet, layout=layout_with_gem)


#-- b50plus
set.seed(9) # top5 nicely
plot_graph2(tnet, layout=layout_nicely)
set.seed(41) # top35 gem
plot_graph2(tnet, layout=layout_with_gem)


#-- tsv
set.seed(20) # tsv top5 nicely
plot_graph2(tnet, layout=layout_nicely)
# set.seed(8) # tsv top35mass dh 
# plot_graph(tnet, layout=layout_with_dh)
set.seed(102) # tsv top35mass gem
plot_graph2(tnet, layout=layout_with_gem)


###-------------------------- clustering
library(reshape2)

df_clust = df_long_aggregated %>%
    mutate(
        from=str_replace_all(from, "_", " "),
        to=str_replace_all(to, "_", " "),
        weight=sqrt(importance)
    ) %>%
    select(-importance) %>%
    filter(from != "background", to != "background")

##
df_clust = rbind(df_clust, data.frame(
        from=c("pleural effusion", "consolidation"),
        to=c("pleural effusion", "consolidation"),
        weight=max(df_clust$weight),
        group="misc"
    ))
##

df_dist = acast(
    df_clust, 
    to ~ from, 
    value.var = 'weight', 
    margins = FALSE
)

##
df_dist[is.na(df_dist)] = 0
##

d0 = as.dist(max(df_dist) - df_dist)

## https://stats.stackexchange.com/a/408842
# Hierarchical Agglomerative Clustering
h1=hclust(d0,method='average')
h2=hclust(d0,method='complete')
h3=hclust(d0,method='ward.D')
h4=hclust(d0,method='single')
# Cophenetic Distances, for each linkage
c1=cophenetic(h1)
c2=cophenetic(h2)
c3=cophenetic(h3)
c4=cophenetic(h4)
# Correlations
cor(d0,c1) # 0.83, 0.82
cor(d0,c2) # 0.77, 0.74
cor(d0,c3) # 0.49, 0.49
cor(d0,c4) # 0.69, 0.66
plot(h1)

library(ggdendro)
ggdendrogram(h1, rotate = TRUE) +
    theme(text=element_text(family="Times"))

ggsave(file.path(PATH_OUTPUT, paste0("cluster_", DATASET_FILE, ".pdf")), width=3, height=4.5)
# ---------------------------


# --------------------------- context
DATASET_NAME = "TSV2"
df_context = read.csv(file.path(PATH, "joined_label_mass_in_label_explanations.csv"), row.names=1)
df_context = df_context %>% filter(dataset==DATASET_NAME)
df_context$group = unlist(dict_class_group[as.character(df_context$Class)])
df_context$Class = fct_reorder(df_context$Class, df_context$value)
# df_context$Class = fct_relevel(
#     df_context$Class, 
#     df_context %>% 
#         filter(dataset=="TSV2") %>% 
#         group_by(Class) %>% 
#         summarise(avg=mean(value)) %>%
#         arrange(avg) %>%
#         pull(Class) %>%
#         as.character()
# )
# df_context$dataset = fct_relevel(df_context$dataset, c("TSV2", "B50")) 

p5 = ggplot(df_context) +
    geom_boxplot(aes(x=value, y=Class, color=group)) + 
    scale_color_manual(values=unlist(dict_group_color)) +
    labs(
        title=DATASET_NAME,
        y=NULL, x="Mass of explanation inside the segmented class"
        ) +
    theme_custom() +
    theme(
        axis.text.y = element_text(colour = unlist(dict_group_color[unlist(dict_class_group[levels(df_context$Class)])])),
        plot.title = element_markdown(hjust = 0)
    )
p5
ggsave(file.path(PATH_OUTPUT, paste0("context_", DATASET_NAME, ".pdf")), width=6, height=4)
