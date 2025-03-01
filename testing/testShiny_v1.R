library(shiny)
library(DiagrammeR)

ui <- fluidPage(
    titlePanel("Simple Mermaid Diagram Example"),
    mainPanel(
        DiagrammeROutput("mermaidDiagram")
    )
)

server <- function(input, output) {
    output$mermaidDiagram <- renderDiagrammeR({
        mermaid("
            graph TD
            A[Start] --> B{Decision}
            B -->|Yes| C[Action 1]
            B -->|No| D[Action 2]
            C --> E[End]
            D --> E
        ")
    })
}

shinyApp(ui = ui, server = server)
