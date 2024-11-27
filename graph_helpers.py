import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def three_dim_graduation_employment_status_2013_2020(data):
    # Load your data
    # df = pd.read_excel('data/graduation_employment_status_2013_2020.xlsx', sheet_name="Filtered_OVGARB10")

    # Filter the DataFrame for employed students
    employed_df = data[data['Employment_Status'] == 'Employed']

    # Define the correct order for the periods
    period_order = ['3 month', '9 month', '15 month', '21 month']

    # Convert 'Period' column to a categorical type with the specified order
    employed_df['Period'] = pd.Categorical(employed_df['Period'], categories=period_order, ordered=True)

    # Group by 'Period' and sum the counts for each year
    employed_counts = employed_df.groupby('Period').sum(numeric_only=True).reset_index()

    # Extract unique periods and years
    x = np.array(employed_counts.columns[3:].astype(int))  # Convert years to integer for x-axis
    y = employed_counts['Period'].unique()  # Unique periods: ['3 month', '9 month', '15 month', '21 month']
    z = []

    # Populate the z array for each period
    for period in y:
        row = employed_counts[employed_counts['Period'] == period].iloc[:, 3:].values.flatten()  # Get the year data as a flat array
        z.append(row)

    z = np.array(z)

    # Create the 3D heatmap
    fig = go.Figure(data=[go.Surface(
        z=z,  # Values to plot (number of employed students)
        x=x,  # X-axis values (Years)
        y=y,  # Y-axis values (Periods)
        colorscale='Viridis',  # Color scale for better contrast
        colorbar=dict(
            title='Number of Students',
            title_font=dict(size=11, color='white'),  # White color for colorbar title
            tickfont=dict(color='white'),  # White color for colorbar ticks
        ),  # Colorbar title
        opacity=0.9,  # Add some transparency to the surface
    )])

    # Set camera angle for better view
    fig.update_layout(
        title='3D Heatmap of Employment Status after certain period of time',
        title_font=dict(size=16, color="white"),  # White title
        title_x=0.5,
        scene=dict(
            xaxis_title='Year',
            yaxis_title='Period After Graduation',
            zaxis_title='Number of Employed Students',
            xaxis=dict(
                title_font=dict(size=12, color='white'),  # White axis titles
                tickfont=dict(size=10, color='white'),  # White ticks
                tickangle=45,  # Rotate x-axis tick labels to avoid overlap
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.3)',  # Light gridlines
            ),
            yaxis=dict(
                title_font=dict(size=12, color='white'),  # White y-axis title
                tickfont=dict(size=10, color='white'),  # White ticks
                tickangle=45,  # Rotate y-axis tick labels to avoid overlap
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.3)',  # Light gridlines
            ),
            zaxis=dict(
                title_font=dict(size=12, color='white'),  # White z-axis title
                tickfont=dict(size=10, color='white'),  # White ticks
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.3)',  # Light gridlines
            ),
            camera=dict(
                eye=dict(x=1.2, y=-1.5, z=1.6)  # Adjust the camera angle for a clearer view
            )
        ),
        plot_bgcolor='#22305e',  # Dark background color
        paper_bgcolor='#22305e',  # Ensure the entire paper area has the same background color
        # template='plotly_dark',  # Dark template for consistency
        width=650,
        height=600,
        margin=dict(
            l=50,  # Left margin
            r=50,  # Right margin
            t=70,  # Top margin
            b=50   # Bottom margin
        )  # Adding space around the plot
    )

    return fig


def error_employment_rate_3m_2013_2020(data):
    # data = pd.read_excel('data/employment_status_3_month_graduates_2013_2020.xlsx', sheet_name="OVGARB10")

    # Convert from wide to long format
    df_long = data.melt(id_vars=['Status', 'Course'], var_name='Year', value_name='Students')
    df_long['Year'] = df_long['Year'].astype(int) 

    # Aggregate data: sum employed and unemployed students across all courses by year
    summary = df_long.groupby(['Year', 'Status'])['Students'].sum().unstack()

    # Calculate total students and employment rate for each year
    summary['Total'] = summary['Employed'] + summary['Unemployed']
    summary['Employment_Rate'] = summary['Employed'] / summary['Total']

    # Calculate standard error
    summary['CI_95'] = 1.96 * np.sqrt(summary['Employment_Rate'] * (1 - summary['Employment_Rate']) / summary['Total'])

    # Create the Plotly error bar chart
    fig = go.Figure()

    # Add error bars for the employment rate
    fig.add_trace(go.Scatter(
        x=summary.index,
        y=summary['Employment_Rate'],
        mode='lines+markers',
        name='Employment Rate',
        line=dict(color='#00FFFF', width=2),  # White line for better visibility
        marker=dict(size=8, color='rgba(255, 255, 255, 0.8)', symbol='circle'),  # White markers
        error_y=dict(
            type='data',
            array=summary['CI_95'],
            visible=True,
            color='rgba(255, 0, 0, 0.7)',  # Red error bars
            thickness=2,
            width=4,
        )
    ))

    # Adjust the layout of the plot
    fig.update_layout(
        title="Overall Employment Rates Among AI Graduates with Error Bars (2013-2020)",
        title_font=dict(size=16, color="white"),  # White title
        title_x=0.5,
        plot_bgcolor='#22305e',  # Dark background color
        paper_bgcolor='#22305e',  # Ensure the entire paper area has the same background color
        xaxis=dict(
            title="Year",
            title_font=dict(color='white'),  # White axis titles
            tickfont=dict(color='white'),  # White ticks
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.3)',  # Light gridlines
        ),
        yaxis=dict(
            title="Employment Rate",
            title_font=dict(color='white'),  # White y-axis title
            tickfont=dict(color='white'),  # White ticks
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.3)',  # Light gridlines
        ),
        showlegend=False,
        height=600,
        width=650,
        margin=dict(
            l=50,  # Left margin
            r=50,  # Right margin
            t=70,  # Top margin
            b=50   # Bottom margin
        )  # Adding space around the plot
    )

    return fig
    

def contour_plot_age_and_gender_distribution_2013_2023(data):

    # Read data
    # data = pd.read_excel('data/num_students_entrants_complete_by_age_sex_2013_2023.xlsx', sheet_name="INST20")

    # Group the data by Age and sum the number of students across years
    df_grouped = data.groupby('Age').sum().drop(columns=['Sex', 'Status'])

    # Reorder the Age groups to ensure 'Under 20 years' comes first
    age_order = ['Under 20 years', '20-24 years', '25-29 years', '30-34 years', '35-39 years', 
                '40-44 years', '45-49 years', '50 years and over']
    df_grouped = df_grouped.loc[age_order]

    # Reshape the data: make years as columns, age groups as rows
    df_reshaped = df_grouped.T  # Transpose the DataFrame to have years as columns and age groups as rows

    # Create the Contour Plot using Plotly
    fig = go.Figure(go.Contour(
        z=df_reshaped.values,  # Values are the number of students
        x=df_reshaped.columns,  # Years
        y=df_reshaped.index,  # Age groups
        colorscale='Viridis',  # Use a diverging color scale (good for both high and low values)
        colorbar=dict(
            title="Number of Students",  # Color bar title
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        ),
        contours=dict(
            coloring='heatmap',  # Fill the contour with colors
            showlabels=True,  # Show contour labels
            labelfont=dict(size=10, color='Black', family="Arial"),  # Customize label font
        ),
    ))

    # Update layout for better clarity
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper background to transparent
        title="Contour Plot of Student Numbers by Age Group (2013-2023)",
        title_font=dict(size=16, color='white', family='Arial, sans-serif'),
        title_x=0.5,
        xaxis_title="Year",
        yaxis_title="Age Group",
        template="plotly_white",  
        xaxis=dict(tickmode='linear', tickangle=45, title_font=dict(color='white'), tickfont=dict(color='white')),  # White axis text
        yaxis=dict(showgrid=True, title_font=dict(color='white'), tickfont=dict(color='white')),  # White axis text
        autosize=True,
        height=700,  # Adjust height for better visual proportion
        width=1360,   # Adjust width for better visual proportion
        legend=dict(
            font=dict(color='white')  # Set legend text to white
        ),
    )

    # Show the plot
    # fig.show()
    return fig


    

def hist_binning_age_and_gender_distribution(data, year):
    # data = pd.read_excel('data/num_students_entrants_complete_by_age_sex_2013_2023.xlsx', sheet_name="INST20")

    # Filter data for graduates and entrants separately
    graduates_df = data[data['Status'] == 'Completed']
    entrants_df = data[data['Status'] == 'Entrants']
    
    # Aggregate by Age and Sex for each group
    graduates_by_age_sex = graduates_df.groupby(['Age', 'Sex'])[year].sum().unstack(fill_value=0)
    entrants_by_age_sex = entrants_df.groupby(['Age', 'Sex'])[year].sum().unstack(fill_value=0)
    
    # Define age group order with "Under 20 years" at the top
    age_order = ["Under 20 years", "20-24 years", "25-29 years", "30-34 years", "35-39 years", "40-44 years", "45-49 years", "50 years and over"][::-1]
    graduates_by_age_sex = graduates_by_age_sex.reindex(age_order)
    entrants_by_age_sex = entrants_by_age_sex.reindex(age_order)
    
    # Create a subplot with two columns: one for entrants, one for graduates
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Entrants", "Graduates"), shared_yaxes=True)

    # Colors for men and women across both charts
    men_color = '#17becf'
    women_color = '#ff6347'

    # Entrants bar chart (left subplot)

    fig.add_trace(go.Bar(
        y=entrants_by_age_sex.index,
        x=entrants_by_age_sex['Women'],
        name='Women',
        marker_color=women_color,
        orientation='h',
        customdata=[f"Entrants: Women, Age: {age}" for age in entrants_by_age_sex.index],
        hovertemplate="%{customdata}<br>Number of Students: %{x}<extra></extra>"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=entrants_by_age_sex.index,
        x=entrants_by_age_sex['Men'],
        name='Men',
        marker_color=men_color,
        orientation='h',
        customdata=[f"Entrants: Men, Age: {age}" for age in entrants_by_age_sex.index],
        hovertemplate="%{customdata}<br>Number of Students: %{x}<extra></extra>"
    ), row=1, col=1)

    # Graduates bar chart (right subplot)
    
    fig.add_trace(go.Bar(
        y=graduates_by_age_sex.index,
        x=graduates_by_age_sex['Women'],
        name='Women',
        marker_color=women_color,
        orientation='h',
        showlegend=False,  # Hide duplicate legend entry
        customdata=[f"Graduates: Women, Age: {age}" for age in graduates_by_age_sex.index],
        hovertemplate="%{customdata}<br>Number of Students: %{x}<extra></extra>"
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        y=graduates_by_age_sex.index,
        x=graduates_by_age_sex['Men'],
        name='Men',
        marker_color=men_color,
        orientation='h',
        showlegend=False,  # Hide duplicate legend entry
        customdata=[f"Graduates: Men, Age: {age}" for age in graduates_by_age_sex.index],
        hovertemplate="%{customdata}<br>Number of Students: %{x}<extra></extra>"
    ), row=1, col=2)

    # Set consistent x-axis range
    max_x = max(entrants_by_age_sex.values.max(), graduates_by_age_sex.values.max())
    fig.update_xaxes(range=[0, max_x * 1.01])  # Extend slightly above max value for visual clarity

     # Customize layout
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper background to dark blue
        title=f'Age and Gender Distribution of Entrants and Graduates in {year}',
        title_font=dict(size=16, color='white', family='Arial, sans-serif'),
        title_x=0.5,
        yaxis_title='Age Group',
        xaxis=dict(title='Number of Students'),
        template='plotly_dark',  # Use the dark template for contrast
        legend_title='Gender',
        bargap=0.15,
        margin=dict(t=50, b=50, l=50, r=50)  # Adjust margins for better spacing
    )

    # Update x-axis and y-axis titles for each subplot
    fig.update_xaxes(title_text="Number of Students", row=1, col=1)
    fig.update_xaxes(title_text="Number of Students", row=1, col=2)

    # Ensure y-axis uses the specified age order
    fig.update_yaxes(categoryorder='array', categoryarray=age_order, row=1, col=1)

    # fig.show()
    
    return fig

def scatter_plot_graduates_and_entrants__2013_2023(data):
    # data = pd.read_excel('data/num_students_entrants_complete_by_age_sex_2013_2023.xlsx', sheet_name="INST20")
    # Filter for Completed and Entrants
    completed_data = data[data['Status'] == 'Completed']
    entrants_data = data[data['Status'] == 'Entrants']

    # Summarize total completed students by year
    completed_summary = completed_data[['2013', '2014', '2015', 
                                    '2016', '2017', '2018', '2019', '2020', '2021', 
                                    '2022', '2023']].sum()

    # Summarize total entrants by year
    entrants_summary = entrants_data[['2013', '2014', '2015', 
                                    '2016', '2017', '2018', '2019', '2020', '2021', 
                                    '2022', '2023']].sum()
    # Prepare the data for the scatter plot
    graduation = completed_summary
    enrollment = entrants_summary

    # Create a Plotly scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=enrollment, 
        y=graduation, 
        mode='markers', 
        marker=dict(color='#FF00FF', size=10, opacity=0.7, symbol='circle'),
        name='Enrollment vs. Graduation',
        hovertemplate='<b>Enrollment</b>: %{x}<br><b>Graduation</b>: %{y}<br><extra></extra>'  # Custom hover text
    ))

    # Add trend line using numpy polyfit
    z = np.polyfit(enrollment, graduation, 1)  # Fit a 1st-degree polynomial (linear)
    p = np.poly1d(z)

    # Create a smoother trend line by generating x values
    x_values = np.linspace(min(enrollment), max(enrollment), 100)
    y_values = p(x_values)

    fig.add_trace(go.Scatter(
        x=x_values, 
        y=y_values, 
        mode='lines', 
        line=dict(color='yellow', dash='dash', width=2),
        name='Trend Line',
        hovertemplate='<b>Trend Line</b>: %{x} -> %{y}<extra></extra>'  # Custom hover text for trend line
    ))


    # Update layout for dark background and contrasting elements
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper background to dark blue
        title='Enrolled vs. Graduated Number of Students in AI Programs',
        title_font=dict(size=16, color='white', family='Arial, sans-serif'),
        title_x=0.5,
        xaxis_title='Total Enrollment in AI Programs',
        yaxis_title='Total Graduated in AI Programs',
        template='plotly_dark',  # Use a dark template for better contrast
        showlegend=True,
        hovermode='closest',
        xaxis=dict(showspikes=True, spikethickness=1, showgrid=False, zeroline=False),  # Hide gridlines for a cleaner look
        yaxis=dict(showspikes=True, spikethickness=1, showgrid=False, zeroline=False),  # Hide gridlines for a cleaner look
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.8)',  # Slightly transparent white background
            font_size=12,  # Font size for hover text
            font_color='black'  # Font color for hover text
        ),
        margin=dict(t=50, b=50, l=50, r=50)  # Adjust margins for a cleaner look
    )

    # fig.show()
    return fig

def donut_chart_percentage_degree_type(data):
    # # Load data
    # data = data.read_excel('data/ai_related_courses.xlsx')
    data = data.drop(columns=["course_link", "institution_link"])

    # Extract degree types from the degree_name column
    degree_types = data['degree_name'].str.split(' ', n=1).str[0]  # Get the first part (e.g., 'Master', 'Bachelor', etc.)

    # Count occurrences of each degree type
    degree_counts = degree_types.value_counts()

    # Define custom labels based on degree types
    label_mapping = {
        'Master': 'Master Degree',
        'Summer': 'Summer School Certificate',
        'Bachelor': 'Bachelor Degree',
        'Top-up': 'Top-up Degree',
        'Academy': 'Academy Profession Degree'
    }

    # Create custom labels for the pie chart
    custom_labels = [label_mapping.get(degree, degree) for degree in degree_counts.index]

    # Calculate percentages for the legend
    percentages = [f'{count / degree_counts.sum() * 100:.1f}%' for count in degree_counts]

   # Create the Plotly donut chart with updated colors and styling
    fig = go.Figure(go.Pie(
        labels=custom_labels,
        values=degree_counts,
        hoverinfo='label+percent',  # Show label and percentage on hover
        textinfo='percent',  # Display percentage inside the chart
        hole=0.7,  # Make it a donut chart
        marker=dict(
            colors=px.colors.qualitative.Plotly_r[:len(degree_counts)],  # Change to a sequential color palette for contrast
            line=dict(color='rgba(0, 0, 0, 0.5)', width=2)  # Add outline for contrast
        ),
    ))

    # Update layout to set background color, font, and legend positioning
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper background to dark blue
        title='Percentage of Degree Type in AI-Related Courses',
        title_font=dict(size=16, color='white', family='Arial, sans-serif'),
        title_x=0.5,
        showlegend=True,
        legend=dict(
            orientation="h",  # Horizontal legend
            y=-0.2,           # Position legend below chart
            x=0.5,
            xanchor="center",
            font=dict(size=12, color='white')  # Set legend font size and color
        ),
        width=450,  # Set custom width for a larger chart
        height=600,
        margin=dict(t=50, b=50, l=50, r=50),  # Adjust margins for a cleaner look
        font=dict(color='white')  # Set font color to white for better contrast on dark background
    )
    return fig

def map_ai_related_course(data):
    # # Load your dataset (replace 'ai_related_courses.xlsx' with your actual file path)
    # data = pd.read_excel('data/ai_related_courses.xlsx')

    # Group the data by institution and combine courses offered by each institution
    df_grouped = data.groupby(['institution_name', 'latitude', 'longitude']).apply(
        lambda group: group[['course_name', 'tuition_fee_per_term']].to_dict('records')
    ).reset_index(name='courses')

    # Create a list of latitudes, longitudes, and institution names for the map markers
    latitudes = df_grouped['latitude'].tolist()
    longitudes = df_grouped['longitude'].tolist()
    institutions = df_grouped['institution_name'].tolist()
    courses = df_grouped['courses'].tolist()

    # Create hover text to display only the institution name
    hover_texts = []
    for inst, course_list in zip(institutions, courses):
        course_count = len(course_list)
        hover_text = f"<b>{inst} | {course_count} courses</b><br><br>"
        hover_text += "<b>Course Name</b> | <b>Tuition Fee per Term</b><br>"
        hover_text += "<br>".join([f"{course['course_name']} | {course['tuition_fee_per_term']}" for course in course_list])
        hover_texts.append(hover_text)

    fig = go.Figure(go.Scattermap(
        lat=latitudes,
        lon=longitudes,
        mode='markers',
        marker=dict(
            size=14,  # Adjust marker size
            color='red',  # Set the marker color to blue
            opacity=0.8,  # Set marker opacity (0.0 to 1.0)
            symbol='circle',  # Set the marker shape to a circle (other options include 'square', 'diamond', etc.)
        ),
        text=hover_texts,  # Use the formatted hover text
        hoverinfo='text'  # Only show the text in hover
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
        paper_bgcolor='#22305e',  # Set paper background to dark blue
        title='Institutions provide AI-Related Courses',
        title_font=dict(size=18, color='white', family='Arial, sans-serif'),
        title_x=0.5,
        autosize=True,
        width=850,  # Set custom width for a larger display
        height=600,
        hovermode='closest',
        map=dict(
            center=dict(lat=56.26392, lon=9.501785),  # Center the map around Denmark (latitude, longitude of Denmark)
            zoom=6,  # Adjust zoom level to show Denmark and nearby regions
        ),
        hoverlabel=dict(
            bgcolor='white',  # Set the background color of the hover text to white
            font_size=10,  # Set the font size of the hover text
            font_family="Arial",  # Set the font family
            font_color="black",  # Set the font color to black
        ),
        margin=dict(t=50, b=50, l=50, r=50),  # Adjust margins for a cleaner look
        font=dict(color='white')  # Set font color to white for better contrast on dark background
    )
    # fig.show()
    return fig
    

def line_graph_monthly_earning_2013_2023(data):
    # data = pd.read_excel('data/monthly_earning_by_occupation_2013_2023.xlsx', sheet_name="LONS20")

    # Create the line plot
    fig = go.Figure()

    # Add traces for each occupation
    for _, row in data.iterrows():
        fig.add_trace(go.Scatter(
            x=data.columns[1:],  # Years
            y=row[1:],  # Monthly earnings
            mode='lines+markers',
            name=row['occupation'],
            hovertemplate='<b>%{y}</b>',
        ))

    # Customize layout with spike lines
    fig.update_layout(
        plot_bgcolor='#22305e',  # Dark background color
        paper_bgcolor='#22305e',  # Ensure the entire paper area has the same background color
        title="Monthly Earnings by Occupation (2013-2023)",
        title_font=dict(size=18, color='white', family='Arial, sans-serif'),
        title_x=0.5,
        xaxis=dict(
            title="Year",
            title_font=dict(color='white'),
            showspikes=True,
            spikemode='across',
            spikesnap="cursor",
            spikecolor='white',
            spikethickness=1,
            tickfont=dict(color='white'),
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.3)',  # Light gridlines
        ),
        yaxis=dict(
            title="Monthly Earnings (DKK)",
            title_font=dict(color='white'),
            showspikes=True,
            spikemode='across',
            spikesnap="cursor",
            spikecolor='white',
            spikethickness=1,
            tickfont=dict(color='white'),
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.3)',  # Light gridlines
        ),
        legend_title="Occupation",
        legend=dict(
            title_font=dict(color='white'),
            font=dict(color='white'),
            bordercolor='white',  # Border color for legend
            borderwidth=1,
        ),
        hovermode="x unified",
    )
    # fig.show()

    return fig


def bar_plot_graduates_and_entrants__2013_2023(data):
    # data = pd.read_excel('data/num_students_entrants_complete_by_age_sex_2013_2023.xlsx', sheet_name="INST20")

    # Filter for Completed and Entrants
    completed_data = data[data['Status'] == 'Completed']
    entrants_data = data[data['Status'] == 'Entrants']

    # Summarize total completed students by year
    completed_summary = completed_data[['2013', '2014', '2015', 
                                    '2016', '2017', '2018', '2019', '2020', '2021', 
                                    '2022', '2023']].sum()

    # Summarize total entrants by year
    entrants_summary = entrants_data[['2013', '2014', '2015', 
                                    '2016', '2017', '2018', '2019', '2020', '2021', 
                                    '2022', '2023']].sum()

    # Set up the bar plot
    labels = completed_summary.index
    fig = go.Figure()

    # Create bars for Completed and Entrants
    fig.add_trace(go.Bar(
        x=labels,
        y=completed_summary.values,
        name='Graduates',
        marker=dict(color='#2596be')
    ))

    fig.add_trace(go.Bar(
        x=labels,
        y=entrants_summary.values,
        name='Entrants',
        marker=dict(color='#fb9b51')
    ))

    # Configure the plot
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper background to dark blue
        title='Total Graduates and Entrants from AI-related Courses in Denmark (2013-2023)',
        title_font=dict(size=16, color='white', family='Arial, sans-serif'),
        title_x=0.5,
        xaxis_title='Year',
        yaxis_title='Number of Students',
        xaxis=dict(tickmode='linear', tickangle=45, tickfont=dict(color='white')),
        yaxis=dict(titlefont=dict(color='white'), tickfont=dict(color='white')),
        barmode='group',  # Group bars side by side
        showlegend=True,
        legend=dict(
            font=dict(color='white'),
            orientation='h',
            y=-0.2,
            x=0.5,
            xanchor="center"
        ),
        template='plotly_dark',  # Use the dark template for contrast
        margin=dict(
            t=50,  # Top margin (increase space for the title)
            b=50,  # Bottom margin (increase space for the x-axis labels)
            l=40,  # Left margin (adjust for axis labels)
            r=40   # Right margin (adjust for the legend)
        )
    )

    return fig 

if __name__ == "__main__":
    pass