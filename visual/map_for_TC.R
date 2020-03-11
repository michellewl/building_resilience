## uncomment to install packages for the first time
#install.packages(c("plyr", "ggplot2", "maps", "ggthemes", "plotly", "RColorBrewer")) 

library(plyr)
library(ggplot2)
library(maps)
library(ggthemes)
library(plotly)
library("RColorBrewer")
myPalette <- colorRampPalette(rev(brewer.pal(11, "Spectral")))


elec_00 =read.csv("/data/electricity_usage_per_country/electricity_data_2000.csv")
elec_14 =read.csv("/data/electricity_usage_per_country/electricity_data_2014.csv")

## Rename first column to match join hereunder
## Remove words after comma, make lowercase
## adjust 
clean_country_names = function(df){
  colnames(df)[1] = "region"
  df$region = gsub("(.*),.*", "\\1", df$region)
  df$region = tolower(df$region)
  df$region[127] = "usa"
  df$region[102] = "russia"
  df$region[126] = "uk"
  return(df)
}

elec_00 = clean_country_names(elec_00)
elec_14 = clean_country_names(elec_14)


country_lat_lon = map_data("world")
country_lat_lon$region = tolower(country_lat_lon$region)

elec_00_lat_lon = join(country_lat_lon, elec_00, by = "region", type = 'left')
elec_all = join(elec_00_lat_lon, elec_14, by='region', type = 'left')

colnames(elec_all)[18] = "elec_14"
colnames(elec_all)[8] = 'elec_00'
elec_all$elec_14 = round(elec_all$elec_14, 2)
elec_all$elec_00 = round(elec_all$elec_00 , 2)
elec_all$percent_change = (elec_all$elec_14 - elec_all$elec_00) /  elec_all$elec_00
elec_all = elec_all[,c(1:8, 18, 27)]

## change elec_00 to elec_14 for 2014 consumption or to percent_change (change limits to c(-3, 3))  in fill. change title accordingly.
map_elec = ggplot(elec_all, aes(long, lat, group=group, fill=percent_change, text = "")) + 
  geom_polygon(show.legend = T) + 
  scale_fill_gradientn(colours = myPalette(100),  limits=c(-3, 3))+
  ggtitle("Electricity percent change from 2000 to 2014") + xlab("") + ylab("") + theme_classic() + theme(legend.title = element_blank()) +   theme(axis.line=element_blank(),
                                                                axis.text.x=element_blank(),
                                                                axis.text.y=element_blank(),
                                                                axis.ticks=element_blank(),
                                                                axis.title.x=element_blank(),
                                                                axis.title.y=element_blank(),
                                                                panel.background=element_blank(),
                                                                panel.border=element_blank(),
                                                                panel.grid.major=element_blank(),
                                                                panel.grid.minor=element_blank(),
                                                                plot.background=element_blank())


ggplotly(map_elec, tooltip = c("group", "fill"))
