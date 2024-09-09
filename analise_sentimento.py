import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Carregar o dataset
file_path = 'dataset/tasks.xlsx'  # Substitua pelo caminho do seu arquivo
df = pd.read_excel(file_path)

# Inicializar o analisador VADER
analyzer = SentimentIntensityAnalyzer()

# Função para analisar o sentimento
def analyze_sentiment(text):
    if pd.isna(text):  # Verificar se o texto é NaN
        return None
    return analyzer.polarity_scores(text)['compound']

# Aplicar a análise de sentimento nas colunas
df['sentiment_summary'] = df['fields_summary'].apply(analyze_sentiment)
df['sentiment_description'] = df['fields_description'].apply(analyze_sentiment)

# Exibir os resultados
print(df[['fields_summary', 'sentiment_summary', 'fields_description', 'sentiment_description']].head())

# Salvar os resultados em um novo arquivo Excel
df.to_excel('sentiment_analysis_results.xlsx', index=False)

