from search import SearchAndLoad
import pandas as pd


# Пути к индексам и данным
data_path = 'data/corpus.csv'
tfidf_index_path = 'indexes/tfidf_index.pkl'
bert_index_path = 'indexes/bert_index.pkl'

# Инициализация поисковой системы
search_try = SearchAndLoad(tfidf_path=tfidf_index_path, bert_path=bert_index_path, data_path=data_path)

# Загрузка существующих индексов
search_try.tfidf_indexer.load_index()
search_try.bert_indexer.load_index()
lyrics = pd.read_csv('data/corpus.csv')

# Запрос для поиска
query = input()

# Поиск с использованием TF-IDF индекса
tfidf_results = search_try.search(query=query, index_type='tf-idf', top_n=5)
print(f"Топ-5 результатов по TF-IDF:")
for result in tfidf_results:
    print(lyrics.loc[int(result)])

# Поиск с использованием BERT индекса
bert_results = search_try.search(query=query, index_type='bert', top_n=5)
print(f"Топ-5 результатов по BERT:")
for result in bert_results:
    print(lyrics.loc[int(result)])

