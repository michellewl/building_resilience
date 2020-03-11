#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

files <- list.files("/Users/omer/Documents/studies/Cambridge/cool_heat/", full.names=TRUE)

# Grab only R files
files <- files[ grepl("\\.[rR]$", files) ]
files <- files[!(files %in% "/Users/omer/Documents/studies/Cambridge/cool_heat//app_playground.R")]

for (f in files) {
  source(f)
}

library(png)
library(shinyjs)
library(shinyWidgets)
library(shiny)
library(shinydashboard)
library(shinycssloaders)
library(shinycustomloader)
library(ggplot2)
library(viridis)
library(scales)      # pairs nicely with ggplot2 for plot label formatting
library(ggthemes)    # has a clean theme for ggplot2
library(ggridges)
library(gridExtra)
library(ggmap)
library(plotly)
library(rhandsontable)
library(DT)

load(file = '/Users/omer/Downloads/energy_map/energy_london')
df2 = df[df$en_consm_curr > 500, ]

ui <-
  fluidPage(
    useShinyjs(),
    includeCSS("www/bootsrap.css"),
    inlineCSS(appCSS),
    tags$head(tags$style('h4 {color:navy; font-family: Bentham;}')),

    # Loading message
    div(
      id = "loading-content",
      tags$div(class="landing-wrapper",

          tags$div(class="landing-block background-content",
               
               # top left
               fluidPage(
                 fluidRow(),
                 fluidRow(),
                 fluidRow(
                 column(10),
                 column(width=2, align = "center",
                        withLoader(plotOutput('tb'), type = 'html', loader = 'dnaspin')))
                         
                 ),
                  
          tags$div(class="landing-block foreground-content",
               tags$div(class="foreground-text",
                        tags$h4("Did you know? In 2010 buildings accounted for 32% of total global final energy use")))
                        )
    )
    ),  # The main app code goes here
    hidden(
      div(
        id = "app-content",
        
        
        # Page
        dashboardPage(
          
          dashboardHeader(title = tags$a(href='http://mycompanyishere.com',
                                         tags$img(src='logo.png'))),
          dashboardSidebar(disable = T, 
                           sidebarMenu(
                             menuItem("Dashboard", tabName = "dashboard"),
                             menuItem("Widgets", tabName = "widgets",
                                      badgeLabel = "new", badgeColor = "green")
                           )
          ),
          ## Body
          dashboardBody(
            ### Panels
            tabsetPanel(
              #### Panel 1
              tabPanel("# extreme temparture days", 
                       column(4, p(HTML("<a href='#slider_max'>KPI & FACTORS BY month </a>"))), 
                       column(4, p(HTML("<a href='#obs_vs_model'>Observations vs. model comparison</a>"))),
                       #column(4, imageOutput('image',height = "200px", width = "50%", inline = TRUE)),
                       ##### wellPanel 1 
                       wellPanel(h1("KPI"), 
                                 ###### Row 1
                                 fluidRow(
                                   box(title = "Input", solidHeader = TRUE, width = 4, status = "primary",
                                       ####### Row 1.1
                                       fluidRow(column(12, selectInput(inputId ="City", selected = "London" , label = "City", choices = names(cities)))),
                                       ####### Row 1.2
                                       fluidRow(column(6,
                                       knobInput(
                                         inputId = "slider_max",
                                         immediate = FALSE,
                                         label = "Max threshold",
                                         value = 30,
                                         min = 20,
                                         max = 35,
                                         displayPrevious = FALSE, 
                                         height = 50,
                                         width =50,
                                         lineCap = "round",
                                         fgColor = "#800000",
                                         inputColor = "#800000"
                                       )
                                      ),
                                       #fluidRow(column(12, sliderInput("slider_max", label = "Max threshold", value = 30, min = 20, max = 35))),
                                       ####### Row 1.3
                                       #fluidRow(column(12, sliderInput("slider_min", label = "Min threshold", value = 0, min = -10, max = 10))),
                                       column(6,
                                       knobInput(
                                         inputId = "slider_min",
                                         immediate = FALSE,
                                         label = "Min threshold",
                                         value = 0,
                                         min = -10,
                                         max = 15,
                                         displayPrevious = FALSE, 
                                         height = 50,
                                         width =50,
                                         lineCap = "round",
                                         fgColor = "#428BCA",
                                         inputColor = "#428BCA"
                                       )))),
                                       
                                       #actionButton(inputId ="update_thres",label = "Update")), 
                                   h5("Share days above"),
                                   valueBoxOutput("dat80" ,width = 2), valueBoxOutput("dat90" ,width = 2), valueBoxOutput("dat00" ,width = 2), valueBoxOutput("dat10" ,width = 2),
                                   h5("Share days below"),
                                   valueBoxOutput("dbt80" ,width = 2), valueBoxOutput("dbt90" ,width = 2), valueBoxOutput("dbt00" ,width = 2), valueBoxOutput("dbt10", width = 2)
                                 )
                       ),
                       ##### wellPanel 2 
                       wellPanel(h1("Year & month overview"), 
                                 ##### Row 1 
                                 fluidRow(
                                   box(title = "", width = 12, solidHeader = TRUE, status = "primary",
                                       plotOutput("heatmap")),   
                                   box(title = "", width = 6, solidHeader = TRUE, status = "primary",
                                       plotOutput("ridges")), 
                                   box(title = "", width = 6, solidHeader = TRUE, status = "primary",
                                       plotOutput("line_above_thres"))
                                 )
                       ),
                       
                       ##### wellPanel 3 
                       wellPanel(h1("Energy survey"), 
                                 fluidRow(
                                   box(title = "", width = 12, solidHeader = TRUE, status = "primary",
                                       plotOutput("mapplot"))
                                 )
                       )
              ),
              tabPanel("Consultancy case",
               wellPanel("AA"),        
              ),
              tabPanel("Raw data",fluidPage(
                sidebarLayout(
                  sidebarPanel(
                    selectInput("dataset", "Choose a dataset:", 
                                choices = c(names(cities)), selected = "Beijing"
                  )),
                  mainPanel(
                    dataTableOutput('table')
                  )
                )
              ))
              ### Close tabset
            )
            ## Close body
          )
          # Close page
        )
      )
    )
  )
  

server <- function(input, output, session) {
  output$tb <- renderPlot({
    Sys.sleep(2) # system sleeping for 3 seconds for demo purpose
  })
  
  beijing = cities$Beijing$df[,c(1:15)]
  london = cities$London$df[,c(1:15)]
  nyc = cities$NYC$df[,c(1:15)]
  tokyo = cities$Tokyo$df[,c(1:15)]
  


  Sys.sleep(8)
  # Simulate work being done for 1 second

  
  # Hide the loading message when the rest of the server function has executed
  hide(id = "loading-content", anim = TRUE, animType = "fade")    
  show("app-content")
  
  output$image <- renderImage({
    list(src = "www/max.jpg",
         alt = "This is alternate text"
    )
  }, deleteFile = FALSE)
  
  output$dat80 <- days_abv_decade(input, 1, 80)
  
  output$dat90 <- days_abv_decade(input, 2, 90)
  
  output$dat00 <- days_abv_decade(input, 3, 00)
  
  output$dat10 <- days_abv_decade(input, 4, 10)
  
  ###########
  output$dbt80 <- days_below_thres_decade(input, 1, 80)
  
  output$dbt90 <- days_below_thres_decade(input, 2, 90)
  
  output$dbt00 <- days_below_thres_decade(input, 3, 00)
  
  output$dbt10 <- days_below_thres_decade(input, 4, 10)
  
  ###########
  output$heatmap <- heatmap(input)
  
  #######
  output$ridges <- ridges(input)
  
  ########
  output$line_above_thres <- line_abv_thres(input)
  
  #######
  #output$mapplot <- mappy(input)
  datasetInput <- reactive({
    switch(input$dataset,
      "Beijing" = beijing, "London" = london, "NYC" = nyc, "Tokyo" = tokyo)
  })
  
  
  output$table <- DT::renderDataTable({
    
    DT::datatable(datasetInput(), options = list(scrollX = TRUE), filter = "top")
  })
}

# Run the application 
app = shinyApp(ui = ui, server = server)

runApp(app, host = getOption("shiny.host", "0.0.0.0"), port = 7638)