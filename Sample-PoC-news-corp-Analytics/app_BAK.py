import os
import json
import base64
import io
from flask import Flask, request, jsonify, render_template
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import google.generativeai as genai

matplotlib.use('Agg')
app = Flask(__name__)

try:
    api_key = os.environ["GEMINI_API_KEY"]
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None


knowledge_base_data = [
    # Added 'publish_hour', 'country_code', 'sub_topic'
    {'id': 1, 'publication': 'Wall Street Journal', 'title': 'Global Tech Stocks Rally', 'publish_date': '2025-06-01', 'publish_hour': 9, 'topic': 'Finance', 'sub_topic': 'Stocks', 'engagement_score': 850, 'word_count': 1200, 'shares': 450, 'country_code': 'USA', 'keywords': ['tech', 'stocks', 'rally', 'inflation']},
    {'id': 2, 'publication': 'Financial Times', 'title': 'Central Banks Consider Rate Moves', 'publish_date': '2025-06-03', 'publish_hour': 11, 'topic': 'Finance', 'sub_topic': 'Macroeconomics', 'engagement_score': 920, 'word_count': 1400, 'shares': 510, 'country_code': 'GBR', 'keywords': ['banks', 'rates', 'economy', 'ecb']},
    {'id': 3, 'publication': 'Wall Street Journal', 'title': 'Future of Crypto Regulation', 'publish_date': '2025-06-08', 'publish_hour': 14, 'topic': 'Finance', 'sub_topic': 'Cryptocurrency', 'engagement_score': 980, 'word_count': 1800, 'shares': 720, 'country_code': 'USA', 'keywords': ['crypto', 'regulation', 'sec', 'bitcoin']},
    {'id': 7, 'publication': 'TechCrunch', 'title': 'New AI Model Achieves SOTA Results', 'publish_date': '2025-06-02', 'publish_hour': 15, 'topic': 'Technology', 'sub_topic': 'AI & ML', 'engagement_score': 1500, 'word_count': 1100, 'shares': 1250, 'country_code': 'USA', 'keywords': ['ai', 'llm', 'deep learning', 'sota']},
    {'id': 8, 'publication': 'Wired', 'title': 'The Ethical Dilemmas of Gene Editing', 'publish_date': '2025-06-06', 'publish_hour': 10, 'topic': 'Science', 'sub_topic': 'Biotechnology', 'engagement_score': 1200, 'word_count': 2500, 'shares': 800, 'country_code': 'USA', 'keywords': ['crispr', 'genetics', 'ethics', 'science']},
    {'id': 12, 'publication': 'Associated Press', 'title': 'Tensions Rise in South China Sea', 'publish_date': '2025-06-09', 'publish_hour': 8, 'topic': 'World News', 'sub_topic': 'Geopolitics', 'engagement_score': 810, 'word_count': 1500, 'shares': 400, 'country_code': 'CHN', 'keywords': ['geopolitics', 'military', 'asia']},
    {'id': 15, 'publication': 'Science Daily', 'title': 'Breakthrough in Alzheimer\'s Research', 'publish_date': '2025-06-05', 'publish_hour': 18, 'topic': 'Health', 'sub_topic': 'Neuroscience', 'engagement_score': 1800, 'word_count': 1700, 'shares': 1500, 'country_code': 'DEU', 'keywords': ['health', 'alzheimers', 'research', 'brain']},
    {'id': 16, 'publication': 'Nature', 'title': 'Hubble Captures Images of Distant Galaxy', 'publish_date': '2025-06-07', 'publish_hour': 20, 'topic': 'Science', 'sub_topic': 'Astronomy', 'engagement_score': 1650, 'word_count': 1200, 'shares': 1300, 'country_code': 'USA', 'keywords': ['space', 'hubble', 'galaxy', 'astronomy']},
    {'id': 20, 'publication': 'BBC Sport', 'title': 'Manchester United Wins FA Cup Final', 'publish_date': '2025-05-18', 'publish_hour': 19, 'topic': 'Sports', 'sub_topic': 'Football', 'engagement_score': 1400, 'word_count': 950, 'shares': 2200, 'country_code': 'GBR', 'keywords': ['football', 'soccer', 'manchester', 'fa cup']},
    {'id': 23, 'publication': 'Variety', 'title': 'Summer Blockbuster Smashes Records', 'publish_date': '2025-06-02', 'publish_hour': 22, 'topic': 'Entertainment', 'sub_topic': 'Movies', 'engagement_score': 1600, 'word_count': 900, 'shares': 2500, 'country_code': 'USA', 'keywords': ['movies', 'box office', 'summer', 'blockbuster']},
    {'id': 35, 'publication': 'Nature', 'title': 'Ocean Cleanup Project Reports Milestone', 'publish_date': '2025-06-28', 'publish_hour': 12, 'topic': 'Science', 'sub_topic': 'Environment', 'engagement_score': 1700, 'word_count': 1400, 'shares': 1450, 'country_code': 'NLD', 'keywords': ['ocean', 'plastic', 'environment', 'cleanup']},
    {'id': 41, 'publication': 'Science Daily', 'title': 'Mars Rover Discovers Evidence of Ancient Lake', 'publish_date': '2025-07-04', 'publish_hour': 17, 'topic': 'Science', 'sub_topic': 'Astronomy', 'engagement_score': 1950, 'word_count': 1500, 'shares': 1800, 'country_code': 'USA', 'keywords': ['mars', 'nasa', 'rover', 'space']},
    {'id': 45, 'publication': 'TechCrunch', 'title': 'The Booming Market for Electric Vehicles', 'publish_date': '2025-07-08', 'publish_hour': 11, 'topic': 'Technology', 'sub_topic': 'Automotive', 'engagement_score': 1600, 'word_count': 1300, 'shares': 1400, 'country_code': 'CHN', 'keywords': ['ev', 'cars', 'tesla', 'battery']},
    {'id': 47, 'publication': 'Nature', 'title': 'CRISPR Technology Used to Treat Genetic Disorder', 'publish_date': '2025-07-10', 'publish_hour': 9, 'topic': 'Health', 'sub_topic': 'Biotechnology', 'engagement_score': 2000, 'word_count': 1800, 'shares': 1900, 'country_code': 'USA', 'keywords': ['crispr', 'genetics', 'health', 'biotech']}
]
kb_df = pd.DataFrame(knowledge_base_data)
kb_df['publish_date'] = pd.to_datetime(kb_df['publish_date'])
kb_df['day_of_week'] = kb_df['publish_date'].dt.day_name()

# --- 3. AI-DRIVEN ANALYSIS FUNCTION (Upgraded Prompt) ---
def get_gemini_analysis(question, df_columns):
    if not model: return {'error': 'Gemini model is not configured.'}

    prompt = f"""
    You are an expert data analyst for a media corporation. A user asked: "{question}"
    The dataset has columns: {df_columns}

    Your task is to respond in a single, clean JSON object with no other text.
    The JSON object must have keys:
    1. "text_response": A conversational summary of the findings.
    2. "chart_type": Choose the best chart from this list: ['bar', 'line', 'scatter', 'pie', 'heatmap', 'treemap', 'map', 'wordcloud', 'table'].
       - Choose 'heatmap' for density over two categories (like time).
       - Choose 'treemap' for multi-level hierarchical breakdowns.
       - Choose 'map' for geographic data.
       - Choose 'wordcloud' for keyword frequency.
       - Choose 'table' for detailed lists of raw data.
    3. "chart_params": A JSON object with parameters for the chosen chart.
       - For bar/line/scatter: {{"x_axis": "column_name", "y_axis": "column_name"}}
       - For pie: {{"data_field": "column_name"}}
       - For heatmap: {{"index_col": "column_name", "columns_col": "column_name", "values_col": "column_name"}}
       - For treemap: {{"path_cols": ["col1", "col2"]}}
       - For map: {{"locations_col": "column_name", "color_col": "column_name"}}
       - For wordcloud: {{"text_col": "column_name"}}
       - For table: {{"columns_to_show": ["col1", "col2"]}}
    4. "monetization_opportunity": A brief, actionable suggestion on how this insight could drive revenue.

    Example for a heatmap question "When are readers most active?":
    {{
      "text_response": "The analysis shows peak reader engagement occurs during midday on weekdays, with a significant drop-off during weekends. This suggests our audience is highly engaged during the business day.",
      "chart_type": "heatmap",
      "chart_params": {{
          "index_col": "day_of_week",
          "columns_col": "publish_hour",
          "values_col": "engagement_score"
      }},
      "monetization_opportunity": "Schedule premium content releases and targeted ad campaigns for 11 AM - 2 PM on weekdays to maximize visibility and click-through rates."
    }}
    """
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(json_text)
    except Exception as e:
        print(f"Error communicating with Gemini or parsing JSON: {e}")
        return {'error': f'AI model communication error: {e}'}

# --- 4. ADVANCED CHART GENERATION ---
def generate_visualization(df, analysis):
    chart_type = analysis.get('chart_type')
    params = analysis.get('chart_params', {})

    # Handle Table generation separately as it returns HTML, not an image
    if chart_type == 'table':
        try:
            cols_to_show = params.get('columns_to_show', list(df.columns))
            # Return a styled HTML table
            return df[cols_to_show].to_html(classes='data-table', index=False)
        except Exception as e:
            return f"<p>Error creating table: {e}</p>"

    # Image-based charts
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    fig = None # For plotly charts

    try:
        if chart_type == 'heatmap':
            pivot_data = df.pivot_table(
                index=params['index_col'],
                columns=params['columns_col'],
                values=params['values_col'],
                aggfunc='mean'
            )
            sns.heatmap(pivot_data, cmap="viridis", annot=True, fmt=".0f")
        elif chart_type == 'treemap':
            agg_df = df.groupby(params['path_cols']).size().reset_index(name='count')
            fig = px.treemap(agg_df, path=params['path_cols'], values='count', title='Hierarchical Breakdown')
        elif chart_type == 'map':
            agg_df = df.groupby(params['locations_col'])[params['color_col']].mean().reset_index()
            fig = px.choropleth(
                agg_df,
                locations=params['locations_col'],
                color=params['color_col'],
                hover_name=params['locations_col'],
                color_continuous_scale=px.colors.sequential.Plasma,
                title="Geographic Distribution"
            )
        elif chart_type == 'wordcloud':
            text = ' '.join(word for sublist in df[params['text_col']] for word in sublist)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
        else: # Seaborn charts
            # Fallback to existing seaborn logic for bar, line, pie, scatter
            # This part can be copied from the previous app.py version
            if chart_type == 'bar': sns.barplot(data=df.nlargest(10, params['y_axis']), x=params['x_axis'], y=params['y_axis'])
            elif chart_type == 'line': sns.lineplot(data=df, x=params['x_axis'], y=params['y_axis'], marker='o')
            elif chart_type == 'pie': plt.pie(df[params['data_field']].value_counts(), labels=df[params['data_field']].value_counts().index, autopct='%1.1f%%')
            elif chart_type == 'scatter': sns.scatterplot(data=df, x=params['x_axis'], y=params['y_axis'], hue='topic')
            if chart_type in ['bar', 'line']: plt.xticks(rotation=45, ha='right')

    except Exception as e:
        print(f"Error generating visualization: {e}")
        return None

    # Save to buffer and encode
    buf = io.BytesIO()
    if fig: # If it's a plotly figure
        fig.write_image(buf, format="png")
    else: # If it's a matplotlib/seaborn figure
        plt.tight_layout()
        plt.savefig(buf, format="png")

    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# --- 5. FLASK ROUTES ---
@app.route('/')
def home(): return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.get_json().get('question')
    if not question: return jsonify({'error': 'No question provided.'}), 400

    analysis = get_gemini_analysis(question, list(kb_df.columns))
    if 'error' in analysis: return jsonify(analysis), 500

    visualization_output = generate_visualization(kb_df, analysis)

    response_data = {
        'text_response': analysis.get('text_response'),
        'monetization_opportunity': analysis.get('monetization_opportunity'),
        # Determine if the output is an image or HTML table
        'chart_image': visualization_output if analysis.get('chart_type') != 'table' else None,
        'html_table': visualization_output if analysis.get('chart_type') == 'table' else None
    }
    return jsonify(response_data)


# --- 6. RUN THE APP ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
