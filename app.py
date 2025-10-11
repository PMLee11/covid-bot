from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
import os

app = Flask(__name__)

# Load data
df = pd.read_csv('worldometer_data.csv')
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_llm(messages, temperature=0.7):
    """Call Groq API"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def answer_question(question):
    """Main conversational response"""
    
    # Get relevant data summary
    data_context = f"""
COVID-19 Dataset Information:
- Total countries: {len(df)}
- Columns available: {', '.join(df.columns)}
- Top 5 countries by cases: {df.nlargest(5, 'TotalCases')['Country/Region'].tolist()}
- Top 5 by deaths: {df.nlargest(5, 'TotalDeaths')['Country/Region'].tolist()}
- Continents: {df['Continent'].unique().tolist()}

Sample data statistics:
- Total global cases: {df['TotalCases'].sum():,.0f}
- Total global deaths: {df['TotalDeaths'].sum():,.0f}
- Average cases per country: {df['TotalCases'].mean():,.0f}
"""
    
    messages = [
        {
            "role": "system",
            "content": """You are a helpful COVID-19 data analyst assistant. You have access to global COVID-19 statistics.

Your job is to:
1. Answer questions about COVID-19 data naturally and conversationally
2. Provide specific numbers and insights when available
3. Be empathetic when discussing deaths and health impacts
4. Explain trends and patterns in the data
5. If you can't find exact data, say so and offer related information
6. Keep responses concise but informative (2-4 paragraphs)
7. Use a warm, professional tone

Don't write code or show technical details. Just have a natural conversation about the data."""
        },
        {
            "role": "user",
            "content": f"""Based on this COVID-19 dataset:

{data_context}

User question: {question}

Please provide a helpful, conversational answer. Include specific numbers from the data when relevant."""
        }
    ]
    
    # Try to get specific data if needed
    try:
        # Check if question needs specific calculation
        if any(word in question.lower() for word in ['compare', 'difference', 'vs', 'versus']):
            # Extract country names for comparison
            countries_mentioned = [c for c in df['Country/Region'].values if c.lower() in question.lower()]
            if len(countries_mentioned) >= 2:
                comparison_data = df[df['Country/Region'].isin(countries_mentioned)][
                    ['Country/Region', 'TotalCases', 'TotalDeaths', 'TotalRecovered']
                ].to_string(index=False)
                messages[1]['content'] += f"\n\nSpecific comparison data:\n{comparison_data}"
        
        elif 'continent' in question.lower():
            continent_data = df.groupby('Continent')[['TotalCases', 'TotalDeaths']].sum().to_string()
            messages[1]['content'] += f"\n\nContinent summary:\n{continent_data}"
        
        elif any(word in question.lower() for word in ['top', 'most', 'highest', 'worst']):
            top_data = df.nlargest(10, 'TotalCases')[['Country/Region', 'TotalCases', 'TotalDeaths']].to_string(index=False)
            messages[1]['content'] += f"\n\nTop 10 countries data:\n{top_data}"
    except:
        pass
    
    response = call_llm(messages)
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    question = request.json.get('message', '')
    
    if not question:
        return jsonify({'error': 'No question provided'})
    
    try:
        response = answer_question(question)
        return jsonify({
            'success': True,
            'message': response
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f"I'm having trouble answering that. Error: {str(e)}"
        })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

