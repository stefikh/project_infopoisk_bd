from search import SearchAndLoad

# Пути к данным и индексам
data_path = 'data/corpus.csv'
tfidf_index_path = 'indexes/tfidf_index.pkl'
bert_index_path = 'indexes/bert_index.pkl'

# Инициализация поисковой системы
do_index = SearchAndLoad(tfidf_path=tfidf_index_path, bert_path=bert_index_path, data_path=data_path)

# Загрузка и предобработка данных
do_index.load_data()

# Построение индексов
do_index.build_indexes()

print("Индексы успешно построены и сохранены.")