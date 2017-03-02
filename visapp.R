# This app is available at https://camillegirabawe.shinyapps.io/nycgreentaxi
# Full modeling is also hosted at http://github.com/kthouz/nyc_green_taxi
# This visualization helps to look at the space-time distribution of New York City Green Taxi pickups
# This a demo app that is only showing data trips of 2015-09-02
# The data was obtained from the TLC (http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml)
# and was cleaned in python.
# The cleaning consisted in:
# - removing all data point with 0 latitude and 0 longitude
# - negative fare
# - negative tips
# Note: To run this app, both data source and script should be in the same folder. 
# Consider using setwd() to set the working directory

library(RCurl)
library(shiny)
library(leaflet)
library(shinydashboard)
library(leaflet)

#setwd('/Volumes/EXDRIVE1/nyctaxi')
# import data from TLC
data <- read.csv('tlc_green_15_09_02.csv');

# the entire dataset on one day has 42 trips. Let's sample 10000
#data <- data[sample(1:nrow(data),10000,replace = FALSE),]

# calculate the tip percentage
data$tippercentage <- round(100* data$tipamount/data$totalamount,2);

### Build the UI
# create header
header <- dashboardHeader(
  title = "NYC Green Cab"
)
# create the body
body <- dashboardBody(
  fluidRow(
    column(width = 6,
           box(width = NULL, solidHeader = TRUE,
               leafletOutput("map", height = 500)
           )
    ),
    # create sliders
    column(6,
           fluidRow(
             column(6,h5("Demo visualization of New York Green Cab pickups on September 2, 2015."))
           ),
           fluidRow(
             column(6,sliderInput("range_passenger",label = "Passengers",
                                  value = range(data$passengercount), 
                                  min = min(data$passengercount), 
                                  max = max(data$passengercount), 
                                  step = 1))
           ),
           fluidRow(
             column(6,sliderInput("range_fare",label = "Fare Amount",
                                  value = range(data$totalamount), 
                                  min = min(data$totalamount), 
                                  max = max(data$totalamount), 
                                  pre = '$'))
           ),
           fluidRow(
             column(6,sliderInput("range_distance",label = "Distance",
                                  value = range(data$tripdistance), 
                                  min = min(data$tripdistance), 
                                  max = max(data$tripdistance), 
                                  post = 'miles'))
           ),
           fluidRow(
             column(6,sliderInput("range_hour",label = "Hour past midnight",
                                  value = min(data$hour), 
                                  min = min(data$hour), 
                                  max = max(data$hour),
                                  animate = TRUE,
                                  step = 1))
           ),
           fluidRow(
             column(3,checkboxGroupInput("payment_type",label = "Payment Mode",
                                         choices = list("Credit Card"=1, "Cash"=2),
                                         selected = c(1,2))),
             column(3,checkboxGroupInput("trip_type",label = "Trip Type",
                                         choices = list("Street-Hail"=1, "Dispatch"=2),
                                         selected = c(1,2)))
           )
    )
  )
)

# Put together all UI constructors
ui <- dashboardPage(
  header,
  dashboardSidebar(disable = TRUE),
  body
)

# build the server
server <- function(input, output) {
  # filter data based on sliderInputs values
  filteredData <- reactive({
    subset(data,passengercount>=input$range_passenger[1] & passengercount<=input$range_passenger[2] & 
             totalamount>=input$range_fare[1] & totalamount<=input$range_fare[2] & 
             tripdistance>=input$range_distance[1] & tripdistance<=input$range_distance[2] &
             hour==input$range_hour & paymenttype %in% input$payment_type & 
             triptype %in% input$trip_type)
  })
  
  # create color palette. We will color by tippercentage
  colorpalette <- reactive({
    colorNumeric('Spectral',data$tippercentage)
  })
  
  # generate the map
  output$map <- renderLeaflet({
    leaflet(data) %>% addTiles(options = providerTileOptions(opacity = .5)) %>% 
      #addProviderTiles(providers$CartoDB.Positron) %>%
      fitBounds(~min(longitude), ~min(latitude), ~max(longitude), ~max(latitude))
  })
  
  # generate an observer object that will overlay circles whenever there is an update on the slider
  observe({
    pal <- colorpalette()
    
    leafletProxy("map", data=filteredData()) %>% 
      clearShapes() %>% 
      addCircles(radius = 200, weight = 1, color = "#888888", fillColor = ~pal(tippercentage),
                 fillOpacity = 0.7, popup = ~paste(tipamount)) %>% 
      clearControls() %>%
      addLegend(position = "bottomleft",pal = pal, values = ~tippercentage, opacity = 0.7, title = 'Tip (%)')
  })

}

# run the application
shinyApp(ui,server)