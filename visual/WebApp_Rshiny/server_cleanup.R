library(scales)      # pairs nicely with ggplot2 for plot label formatting
library(gridExtra)   # a helper for arranging individual ggplot objects
library(ggthemes)    # has a clean theme for ggplot2
library(viridis)     # best. color. palette. evar.
library(ggridges)
library(gridExtra)
library(ggplot2)



days_abv_decade <- function(input, row, decade) {
  renderValueBox({
    city = input$City
    thres = input$slider_max
    num_days =  cities[[city]]$abv_dec[row, as.character(thres)] / 3650
    valueBox(
      subtitle = paste0("decade ",decade, "s"),
      color = "red",
      print(round(num_days, 2))
    )
  })
}


days_below_thres_decade <-
  function(input, row, decade) {
    renderValueBox({
      city = input$City
      thres = input$slider_min
      num_days =  cities[[city]]$be_dec[row, as.character(thres)] / 3650
      valueBox(
        subtitle = paste0("decade ",decade, "s"),
        color = "light-blue",
        print(round(num_days, 2))
      )
    })
    
  }


heatmap <- function(input) {
  renderPlot ({
    city = input$City
    thres = as.character(input$slider_max)
    gg <-
      ggplot(
        cities[[city]]$abv_yr_mon,
        aes_string(
          x = factor(cities[[city]]$abv_yr_mon$mon_no),
          y = cities[[city]]$abv_yr_mon$year,
          fill = cities[[city]]$abv_yr_mon[, thres]
        )
      )
    gg <- gg + geom_tile(color = "white", size = 0.1)
    gg <- gg + scale_fill_viridis(name = "# Days", label = comma)
    gg <- gg + coord_equal()
    gg <-
      gg + labs(x = NULL, y = NULL, title = "# of days w/ temperature above threshold \n per year & month")
    gg <- gg + theme(axis.ticks = element_blank())
    gg <- gg + theme(axis.text = element_text(size = 12))
    gg <- gg + theme(legend.title = element_text(size = 12))
    gg <- gg + theme(legend.text = element_text(size = 11))
    gg + coord_flip()
  })
}



ridges = function(input) {
  renderPlot({
    city = input$City
    thres_max = input$slider_max
    thres_min = input$slider_min
    partial_df = cities[[city]]$df
    partial_df = partial_df[partial_df$year %in% c(1983, 1987, 1991, 1995, 1999, 2004, 2008, 2012, 2016), ]
    (
      ggplot(partial_df, aes(x = mx2t_cel, y = factor(year)), fill = factor(mon_no)) +
        geom_density_ridges2(alpha = 0.2, scale = 0.8) + theme_ridges() + geom_vline(
          xintercept = thres_max,
          colour = "red",
          linetype = "longdash"
        )
      + geom_vline(
        xintercept = thres_min,
        colour = "blue",
        linetype = "longdash"
      ) + xlab('Temp (C)') + ylab('Year')
    )
  })
}




line_abv_thres = function(input) {
  renderPlot({
    city = input$City
    thres_max = input$slider_max
    thres_min = input$slider_min
    (ggplot(days_above_df_summary, aes(year, days_above_thres)) 
      + geom_line() + geom_line(aes(y = rollmean(days_above_thres, 5, na.pad = TRUE)), colour = 'red') 
      + scale_y_continuous(limits = c(0, 120)) 
      + ylab('# Days') 
      + ggtitle("# Days above 30C in London") 
      + xlab('Year'))
    
  })
}

mappy = function(input) {
  renderPlot({
    city = input$City
    London_map <- qmap(location  = "london", zoom = 12, color = "bw", legend = "topleft", 
                       base_layer = ggplot(aes(x=lon, y=lat, colour = en_consm_curr, size = en_consm_curr, key = address), data = df2)) +
      geom_point(show.legend = FALSE) + geom_jitter(position = position_jitter(width = 0.05, height = 0.04))  +
      scale_color_viridis(option = "D") + theme(legend.position = "none")
    London_map
    ggplotly(London_map, tooltip = c("key") )
  
  })
}