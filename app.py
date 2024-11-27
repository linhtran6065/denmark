from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import graph_helpers

def load_data():
    data_1 = pd.read_excel("data/ai_related_courses.xlsx")
    data_2 = pd.read_excel("data/employment_status_3_month_graduates_2013_2020.xlsx")
    data_3 = pd.read_excel("data/graduation_employment_status_2013_2020.xlsx")
    data_4 = pd.read_excel("data/monthly_earning_by_occupation_2013_2023.xlsx")
    data_5 = pd.read_excel("data/num_students_entrants_complete_by_age_sex_2013_2023.xlsx")
    return data_1, data_2, data_3, data_4, data_5

# Initialize the app
app = Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])

# Load data when the app starts
data_1, data_2, data_3, data_4, data_5 = load_data()
    

app.layout = html.Div([
    # Dashboard Title
    html.H1("AI Education in Denmark", className='title'),
    
    # Section 1: AI-related Courses
    html.Div([
        html.H2("AI-related Courses", className='section-title'),
        
        # Row 1: Donut chart on the left (40%) and Map on the right (60%)
        html.Div([
            html.Div([
                html.Div([  # Container for the donut chart
                    dcc.Graph(id='donut-degree-type', figure=graph_helpers.donut_chart_percentage_degree_type(data_1)),
                ], className='graph-container'),
            ], className='graph-half', style={'width': '35%', 'display': 'inline-block', 'vertical-align': 'top'}),  # Set width to 40%
            
            html.Div([
                html.Div([  # Container for the map
                    dcc.Graph(id='map-ai-courses', figure=graph_helpers.map_ai_related_course(data_1)),
                ], className='graph-container'),
            ], className='graph-half', style={'width': '65%', 'display': 'inline-block', 'vertical-align': 'top'}),  # Set width to 60%
        ], className='row'),

    ], className='section-container'),

    # Section 2: Graduates vs Entrants Data
    html.Div([
        html.H2("Graduates vs Entrants Data", className='section-title'),
        
        # Row 1: Scatter plot (50%) and Bar plot (50%)
        html.Div([
            html.Div([
                html.Div([  # Container for scatter plot
                    dcc.Graph(id='scatter-enrolled-vs-graduated', figure=graph_helpers.scatter_plot_graduates_and_entrants__2013_2023(data_5)),
                ], className='graph-container'),
            ], className='graph-half'),
            
            html.Div([
                html.Div([  # Container for bar plot
                    dcc.Graph(id='bar-graduates-entrants', figure=graph_helpers.bar_plot_graduates_and_entrants__2013_2023(data_5)),
                ], className='graph-container'),
            ], className='graph-half'),
        ], className='row'),
        
        # Row 2: Histogram (Full row) and Contour plot (Full row)
        html.H2("Age and Sex Distribution", className='section-title'),
        html.Div([
            # Slider for year selection
            html.Div([
                dcc.Slider(
                    id='year-slider',
                    min=2013,
                    max=2023,
                    step=1,
                    value=2023,
                    marks={year: str(year) for year in range(2013, 2024)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag',
                ),
            ], className='slider-container'),

            html.Div([
                html.Div([  # Container for histogram
                    dcc.Graph(id='age-gender-distribution-year', figure=graph_helpers.hist_binning_age_and_gender_distribution(data_5, "2023")),
                ], className='graph-container'),
            ], className='graph-full'),

            html.Div([
                html.Div([  # Container for contour plot
                    dcc.Graph(id='contour-age-gender', figure=graph_helpers.contour_plot_age_and_gender_distribution_2013_2023(data_5)),
                ], className='graph-container'),
            ], className='graph-full'),
        ]),
        
    ], className='section-container'),

    # Section 3: Employment Data
    html.Div([
        html.H2("Employment Data", className='section-title'),
        
        # Row 1: Line plot
        html.Div([
            html.Div([
                html.Div([  # Container for line plot
                    dcc.Graph(id='line-earnings-by-occupation', figure=graph_helpers.line_graph_monthly_earning_2013_2023(data_4)),
                ], className='graph-container'),
            ], className='graph-full'),
        ], className='row'),
        
        # Row 2: Error plot (50%) and 3D plot (50%)
        html.Div([
            html.Div([
                html.Div([  # Container for error plot
                    dcc.Graph(id='error-employment-rate', figure=graph_helpers.error_employment_rate_3m_2013_2020(data_2)),
                ], className='graph-container'),
            ], className='graph-half'),
            html.Div([
                html.Div([  # Container for 3D plot
                    dcc.Graph(id='3d-heatmap-employment-status', figure=graph_helpers.three_dim_graduation_employment_status_2013_2020(data_3)),
                ], className='graph-container'),
            ], className='graph-half'),
                    
        ], className='row'),
        
    ], className='section-container'),

    # Interval component to refresh data every 10 seconds (or as needed)
    dcc.Interval(
        id='data-update-interval',
        interval=2000,  # 2 seconds (in milliseconds)
        n_intervals=0
    ),
])


# Callback for updating the data and graphs every 2 seconds
@app.callback(
    [
        Output('map-ai-courses', 'figure'),
        Output('donut-degree-type', 'figure'),
        Output('bar-graduates-entrants', 'figure'),
        Output('scatter-enrolled-vs-graduated', 'figure'),
        Output('contour-age-gender', 'figure'),
        Output('error-employment-rate', 'figure'),
        Output('3d-heatmap-employment-status', 'figure'),
        Output('line-earnings-by-occupation', 'figure')
    ],
    Input('data-update-interval', 'n_intervals')  # Trigger update when interval occurs
)
def update_all_graphs(n_intervals):
    # Reload data every time the interval triggers
    data_1, data_2, data_3, data_4, data_5 = load_data()
    
    # Return updated figures except for the year-based one
    return (
        graph_helpers.map_ai_related_course(data_1),
        graph_helpers.donut_chart_percentage_degree_type(data_1),
        graph_helpers.bar_plot_graduates_and_entrants__2013_2023(data_5),
        graph_helpers.scatter_plot_graduates_and_entrants__2013_2023(data_5),
        graph_helpers.contour_plot_age_and_gender_distribution_2013_2023(data_5),
        graph_helpers.error_employment_rate_3m_2013_2020(data_2),
        graph_helpers.three_dim_graduation_employment_status_2013_2020(data_3),
        graph_helpers.line_graph_monthly_earning_2013_2023(data_4)
    )

# Callback to update the histogram for age and gender distribution based on year slider
@app.callback(
    Output('age-gender-distribution-year', 'figure'),
    Input('year-slider', 'value')
)
def update_histogram(year):
    data_1, data_2, data_3, data_4, data_5 = load_data()
    return graph_helpers.hist_binning_age_and_gender_distribution(data_5, str(year))

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)