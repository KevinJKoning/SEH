from shiny import App, ui, render

app_ui = ui.page_fluid(
    # Include Mermaid JS in the head
    ui.tags.head(
        ui.tags.script(src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js")
    ),
    ui.h1("Simple Mermaid Diagram Example"),
    # Add ui.output to connect with server
    ui.output_ui("mermaid_diagram"),
    # Initialize Mermaid
    ui.tags.script("""
        mermaid.initialize({ startOnLoad: true });
        // Re-run mermaid rendering after dynamic content loads
        $(document).on('shiny:value', function(event) {
            if(event.name === 'mermaid_diagram') {
                setTimeout(function() {
                    mermaid.init();
                }, 100);
            }
        });
    """)
)

def server(input, output, session):
    @output
    @render.ui
    def mermaid_diagram():
        return ui.div(
            ui.tags.pre(
                ui.tags.code(
                    """
                    graph TD
                        A[Start] --> B{Decision}
                        B -->|Yes| C[Action 1]
                        B -->|No| D[Action 2]
                        C --> E[End]
                        D --> E
                    """,
                    {"class": "mermaid"}
                )
            )
        )

app = App(app_ui, server)