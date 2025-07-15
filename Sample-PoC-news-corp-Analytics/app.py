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
from datetime import datetime
import random

matplotlib.use('Agg')
app = Flask(__name__)

# --- 1. CONFIGURE GEMINI API ---
try:
    api_key = os.environ["GEMINI_API_KEY"]
    print(api_key)
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

# --- 2. CURATED & ENRICHED KNOWLEDGE BASE ---
# Engagement Score = (Time on Page in seconds * 0.5) + (Scroll Depth % * 1) + (Comments * 8) + (Shares * 10) + (Likes * 2)
# This static list contains all necessary fields for all query types.
knowledge_base_data = [
    {'id': 1, 'publication': 'Wall Street Journal', 'title': 'Global Tech Stocks Rally', 'publish_date': '2025-06-01', 'publish_hour': 9, 'topic': 'Finance', 'sub_topic': 'Stocks', 'engagement_score': 850, 'word_count': 1200, 'shares': 450, 'country_code': 'USA', 'keywords': ['tech', 'stocks', 'rally', 'inflation'], 'is_licensable': True, 'content_type': 'News', 'data_quality_issue': 'ok'},
    {'id': 2, 'publication': 'Financial Times', 'title': 'Central Banks Consider Rate Moves', 'publish_date': '2025-06-03', 'publish_hour': 11, 'topic': 'Finance', 'sub_topic': 'Macroeconomics', 'engagement_score': 920, 'word_count': 1400, 'shares': 510, 'country_code': 'GBR', 'keywords': ['banks', 'rates', 'economy', 'ecb'], 'is_licensable': True, 'content_type': 'News', 'data_quality_issue': 'ok'},
    {'id': 3, 'publication': 'Wall Street Journal', 'title': 'Future of Crypto Regulation', 'publish_date': '2025-06-08', 'publish_hour': 14, 'topic': 'Finance', 'sub_topic': 'Cryptocurrency', 'engagement_score': 980, 'word_count': 1800, 'shares': 720, 'country_code': 'USA', 'keywords': ['crypto', 'regulation', 'sec', 'bitcoin'], 'is_licensable': False, 'content_type': 'News', 'data_quality_issue': 'ok'},
    {'id': 4, 'publication': 'New York Post', 'title': 'City Politics Heats Up', 'publish_date': '2025-06-11', 'publish_hour': 18, 'topic': 'Politics', 'sub_topic': 'Local News', 'engagement_score': 750, 'word_count': 950, 'shares': 350, 'country_code': 'USA', 'keywords': ['politics', 'city', 'election'], 'is_licensable': True, 'content_type': 'Opinion', 'data_quality_issue': 'ok'},
    {'id': 5, 'publication': 'TechCrunch', 'title': 'New AI Model Achieves SOTA Results', 'publish_date': '2025-06-02', 'publish_hour': 15, 'topic': 'Technology', 'sub_topic': 'AI & ML', 'engagement_score': 1500, 'word_count': 1100, 'shares': 1250, 'country_code': 'USA', 'keywords': ['ai', 'llm', 'deep learning', 'sota'], 'is_licensable': True, 'content_type': 'Feature', 'data_quality_issue': 'ok'},
    {'id': 6, 'publication': 'Wired', 'title': 'The Ethical Dilemmas of Gene Editing', 'publish_date': '2025-06-06', 'publish_hour': 10, 'topic': 'Science', 'sub_topic': 'Biotechnology', 'engagement_score': 1200, 'word_count': 2500, 'shares': 800, 'country_code': 'USA', 'keywords': ['crispr', 'genetics', 'ethics', 'science'], 'is_licensable': False, 'content_type': 'Feature', 'data_quality_issue': 'ok'},
    {'id': 7, 'publication': 'New York Post', 'title': 'Summer Restaurant Guide', 'publish_date': '2025-06-10', 'publish_hour': 13, 'topic': 'Lifestyle', 'sub_topic': 'Food', 'engagement_score': 880, 'word_count': 1000, 'shares': 950, 'country_code': 'USA', 'keywords': ['food', 'restaurant', 'summer'], 'is_licensable': True, 'content_type': 'Feature', 'data_quality_issue': 'ok'},
    {'id': 8, 'publication': 'Associated Press', 'title': 'Tensions Rise in South China Sea', 'publish_date': '2025-06-09', 'publish_hour': 8, 'topic': 'World News', 'sub_topic': 'Geopolitics', 'engagement_score': 810, 'word_count': 1500, 'shares': 400, 'country_code': 'CHN', 'keywords': ['geopolitics', 'military', 'asia'], 'is_licensable': True, 'content_type': 'News', 'data_quality_issue': 'ok'},
    {'id': 9, 'publication': 'Science Daily', 'title': 'Breakthrough in Alzheimer\'s Research', 'publish_date': '2025-06-05', 'publish_hour': 18, 'topic': 'Health', 'sub_topic': 'Neuroscience', 'engagement_score': 1800, 'word_count': 1700, 'shares': 1500, 'country_code': 'DEU', 'keywords': ['health', 'alzheimers', 'research', 'brain'], 'is_licensable': True, 'content_type': 'News', 'data_quality_issue': 'ok'},
    {'id': 10, 'publication': 'Nature', 'title': 'Hubble Captures Images of Distant Galaxy', 'publish_date': '2025-06-07', 'publish_hour': 20, 'topic': 'Science', 'sub_topic': 'Astronomy', 'engagement_score': 1650, 'word_count': 1200, 'shares': 1300, 'country_code': 'USA', 'keywords': ['space', 'hubble', 'galaxy', 'astronomy'], 'is_licensable': True, 'content_type': 'Feature', 'data_quality_issue': 'ok'},
    {'id': 11, 'publication': 'BBC Sport', 'title': 'Manchester United Wins FA Cup Final', 'publish_date': '2025-05-18', 'publish_hour': 19, 'topic': 'Sports', 'sub_topic': 'Football', 'engagement_score': 1400, 'word_count': 950, 'shares': 2200, 'country_code': 'GBR', 'keywords': ['football', 'soccer', 'manchester', 'fa cup'], 'is_licensable': True, 'content_type': 'News', 'data_quality_issue': 'ok'},
    {'id': 12, 'publication': 'Variety', 'title': 'Summer Blockbuster Smashes Records', 'publish_date': '2025-06-02', 'publish_hour': 22, 'topic': 'Entertainment', 'sub_topic': 'Movies', 'engagement_score': 1600, 'word_count': 900, 'shares': 2500, 'country_code': 'USA', 'keywords': ['movies', 'box office', 'summer', 'blockbuster'], 'is_licensable': False, 'content_type': 'Feature', 'data_quality_issue': 'ok'},
    {'id': 13, 'publication': 'Nature', 'title': 'Ocean Cleanup Project Reports Milestone', 'publish_date': '2025-06-28', 'publish_hour': 12, 'topic': 'Science', 'sub_topic': 'Environment', 'engagement_score': 1700, 'word_count': 1400, 'shares': 1450, 'country_code': 'NLD', 'keywords': ['ocean', 'plastic', 'environment', 'cleanup'], 'is_licensable': True, 'content_type': 'News', 'data_quality_issue': 'ok'},
    {'id': 14, 'publication': 'TechCrunch', 'title': 'The Booming Market for Electric Vehicles', 'publish_date': '2025-07-08', 'publish_hour': 11, 'topic': 'Technology', 'sub_topic': 'Automotive', 'engagement_score': 1600, 'word_count': 1300, 'shares': 1400, 'country_code': 'CHN', 'keywords': ['ev', 'cars', 'tesla', 'battery'], 'is_licensable': True, 'content_type': 'News', 'data_quality_issue': 'ok'},
    {'id': 15, 'publication': 'Nature', 'title': 'CRISPR Technology Used to Treat Genetic Disorder', 'publish_date': '2025-07-10', 'publish_hour': 9, 'topic': 'Health', 'sub_topic': 'Biotechnology', 'engagement_score': 2000, 'word_count': 1800, 'shares': 1900, 'country_code': 'USA', 'keywords': ['crispr', 'genetics', 'health', 'biotech'], 'is_licensable': True, 'content_type': 'Feature', 'data_quality_issue': 'ok'},
    {'id': 16, 'publication': 'Wall Street Journal', 'title': 'Opinion: The State of the Economy', 'publish_date': '2025-07-01', 'publish_hour': 7, 'topic': 'Finance', 'sub_topic': 'Macroeconomics', 'engagement_score': 1100, 'word_count': 2000, 'shares': 600, 'country_code': 'USA', 'keywords': ['opinion', 'economy', 'commentary'], 'is_licensable': False, 'content_type': 'Opinion', 'data_quality_issue': 'ok'},
    {'id': 17, 'publication': 'Wall Street Journal', 'title': 'WSJ Audit Entry 1', 'publish_date': '2024-11-15', 'publish_hour': 10, 'topic': 'Finance', 'sub_topic': 'Macroeconomics', 'engagement_score': 150, 'word_count': 1300, 'shares': 50, 'country_code': 'USA', 'keywords': ['inflation', 'market', 'audit'], 'is_licensable': True, 'content_type': 'News', 'data_quality_issue': 'duplicate'},
    {'id': 18, 'publication': 'Wall Street Journal', 'title': 'WSJ Audit Entry 2', 'publish_date': '2024-11-20', 'publish_hour': 14, 'topic': 'Business', 'sub_topic': 'Logistics', 'engagement_score': 250, 'word_count': 1100, 'shares': 70, 'country_code': 'USA', 'keywords': ['supply chain', 'audit'], 'is_licensable': False, 'content_type': 'News', 'data_quality_issue': 'missing_metadata'},
    {'id': 19, 'publication': 'New York Post', 'title': 'Opinion: City Hall Gridlock', 'publish_date': '2025-07-03', 'publish_hour': 12, 'topic': 'Politics', 'sub_topic': 'Local News', 'engagement_score': 800, 'word_count': 1100, 'shares': 400, 'country_code': 'USA', 'keywords': ['opinion', 'politics', 'city'], 'is_licensable': False, 'content_type': 'Opinion', 'data_quality_issue': 'ok'},
]
kb_df = pd.DataFrame(knowledge_base_data)
kb_df['publish_date'] = pd.to_datetime(kb_df['publish_date'])
kb_df['day_of_week'] = kb_df['publish_date'].dt.day_name()
print(f"Loaded an enriched knowledge base with {len(kb_df)} articles.")


# --- 3. UNIFIED AI & VISUALIZATION LOGIC ---

def get_gemini_analysis(question, df_columns):
    """A single, powerful prompt that generates a complete plan for all query types."""
    if not model: return {'error': 'Gemini model is not configured.'}
    # This prompt is now more detailed to guide the AI correctly for all chart types
    prompt = f"""
    You are an expert data analyst for a media corporation. A user asked: "{question}"
    The dataset of articles has columns: {df_columns}.

    Your task is to respond in a single, clean JSON object with a complete analysis plan.
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

    **VERY IMPORTANT CHARTING RULES:**
    - For questions about "average daily publishing rate", use 'bar'.
    - For "licensable content" or "opinion content", use a stacked 'bar' by setting "stacked": true in chart_params.
    - For "data quality audit" or "WSJ November", use 'pie'.
    - For "daily volume trend", use 'line'.
    - For "engagement by day and hour", you MUST use 'heatmap'.
    - For "breakdown of content from publication, to topic", you MUST use 'treemap'.
    - For "coverage most popular" by geography, you MUST use 'map'.
    """
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(json_text)
    except Exception as e:
        return {'error': f'AI model error: {e}'}

def generate_visualization(df, analysis):
    """A single, master visualization function that handles all data processing and charting."""
    chart_type = analysis.get('chart_type')
    params = analysis.get('chart_params', {})
    if not params: return None, "AI did not provide valid chart parameters."

    # This function now handles both the data pre-processing and the charting
    processed_df = df.copy()

    # --- Data Pre-processing for Specific BI Queries ---
    q_lower = analysis.get('user_question', '').lower()
    if 'average daily' in q_lower:
        processed_df['date_only'] = processed_df['publish_date'].dt.date
        daily_counts = processed_df.groupby(['publication', 'date_only']).size().reset_index(name='count')
        processed_df = daily_counts.groupby('publication')['count'].mean().round(2).reset_index(name='daily_publish_rate')
        processed_df['avg_daily_publications'] = processed_df['daily_publish_rate']
    elif 'licensable' in q_lower:
        processed_df = processed_df.groupby(['publication', 'is_licensable']).size().unstack(fill_value=0)
        if True not in processed_df.columns: processed_df[True] = 0
        if False not in processed_df.columns: processed_df[False] = 0
        processed_df = processed_df.rename(columns={True: 'Licensable', False: 'Not Licensable'})
    elif 'opinion content' in q_lower:
        processed_df = processed_df.groupby(['publication', 'content_type']).size().unstack(fill_value=0)
    elif 'cleanup' in q_lower or 'november 2024' in q_lower:
        processed_df = processed_df[(processed_df['publication'] == 'Wall Street Journal') & (processed_df['publish_date'].dt.year == 2024) & (processed_df['publish_date'].dt.month == 11)]
        if processed_df.empty: return None, "No data found for WSJ in November 2024 to audit."
    elif 'compare' in q_lower and 'wsj' in q_lower and 'nypost' in q_lower:
        processed_df = df[df['publication'].isin(['Wall Street Journal', 'New York Post'])]

    # --- Chart Generation ---
    buf = io.BytesIO(); fig = None; plt.figure(figsize=(10, 6)); sns.set_theme(style="whitegrid")
    try:
        x_axis=params.get('x_axis'); y_axis=params.get('y_axis')

        if chart_type == 'table':
            return processed_df[params.get('columns_to_show')].to_html(classes='data-table', index=False), None
        elif chart_type == 'heatmap':
            pivot_data = processed_df.pivot_table(index=params['index_col'], columns=params['columns_col'], values=params['values_col'], aggfunc='mean')
            sns.heatmap(pivot_data, cmap="viridis", annot=True, fmt=".0f")
        elif chart_type == 'treemap':
            fig = px.treemap(processed_df, path=params['path_cols'], values=params.get('values', 'engagement_score'))
        elif chart_type == 'map':
            agg_df = processed_df.groupby(params['locations_col'])[params['color_col']].mean().reset_index()
            fig = px.choropleth(agg_df, locations=params['locations_col'], color=params['color_col'], hover_name=params['locations_col'])
        elif chart_type == 'wordcloud':
            text = ' '.join(word for sublist in processed_df[params['text_col']].dropna() for word in sublist);
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text); plt.imshow(wordcloud, interpolation='bilinear'); plt.axis("off")
        elif chart_type == 'bar':
            if params.get('stacked'):
                processed_df.plot(kind='bar', stacked=True, colormap='viridis'); plt.xticks(rotation=45, ha='right')
            else:
                agg_df = processed_df.groupby(x_axis)[y_axis].mean().nlargest(10).reset_index() if x_axis in processed_df.columns else processed_df
                sns.barplot(data=agg_df, x=x_axis, y=y_axis, palette="viridis"); plt.xticks(rotation=45, ha='right')
        elif chart_type == 'line':
            sns.lineplot(data=processed_df.sort_values(by=x_axis), x=x_axis, y=y_axis, hue=params.get('color'), marker='o'); plt.xticks(rotation=45, ha='right')
        elif chart_type == 'pie':
            processed_df[params['data_field']].value_counts().plot.pie(autopct='%1.1f%%', startangle=90); plt.ylabel('')
        elif chart_type == 'scatter':
            sns.scatterplot(data=processed_df, x=x_axis, y=y_axis, hue=params.get('hue', 'topic'), alpha=0.7)
        else:
            return None, f"Chart type '{chart_type}' is not supported."

        if fig: fig.write_image(buf, format="png")
        else: plt.tight_layout(); plt.savefig(buf, format="png")
        plt.close(); buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8'), None
    except Exception as e:
        return None, f"An error occurred during chart generation: {e}"

# --- FLASK ROUTE ---
@app.route('/')
def home(): return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.get_json().get('question')
    if not question: return jsonify({'error': 'No question provided.'}), 400

    plan = get_gemini_analysis(question, list(kb_df.columns))
    if 'error' in plan: return jsonify(plan), 500

    plan['user_question'] = question # Pass question to viz function for context
    visualization_output, chart_error = generate_visualization(kb_df, plan)

    return jsonify({
        'text_response': plan.get('text_response'),
        'monetization_opportunity': plan.get('monetization_opportunity'),
        'chart_image': visualization_output if plan.get('chart_type') != 'table' else None,
        'html_table': visualization_output if plan.get('chart_type') == 'table' else None,
        'chart_error': chart_error
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
