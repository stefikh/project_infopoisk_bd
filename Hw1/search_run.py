from search import SearchAndLoad

# Пути к индексам и данным
data_path = 'data/songs.csv'
tfidf_index_path = 'indexes/tfidf_index.pkl'
bert_index_path = 'indexes/bert_index.pkl'

# Инициализация поисковой системы
search_try = SearchAndLoad(tfidf_path=tfidf_index_path, bert_path=bert_index_path, data_path=data_path)

# Загрузка существующих индексов
search_try.tfidf_indexer.load_index()
search_try.bert_indexer.load_index()

# Запрос для поиска
query = input()

# Поиск с использованием TF-IDF индекса
tfidf_results = search_try.search(query=query, index_type='tfidf', top_n=5)
print(f"Топ-5 результатов по TF-IDF: {tfidf_results}")

# Поиск с использованием BERT индекса
bert_results = search_try.search(query=query, index_type='bert', top_n=5)
print(f"Топ-5 результатов по BERT: {bert_results}")