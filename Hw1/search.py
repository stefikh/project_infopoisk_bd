import pandas as pd
from preprocessing import preprocess_for_tfidf, preprocess_for_bert
from tfidf_index import TFIDFIndexer
from bert_index import BERTIndexer
from typing import List


class SearchAndLoad:
    def __init__(self, tfidf_path: str, bert_path: str, data_path: str):
        self.data_path = data_path
        self.tfidf_indexer = TFIDFIndexer(tfidf_path)
        self.bert_indexer = BERTIndexer(bert_path)
        self.documents = []

    def load_data(self):
        """Загружает данные из CSV файла и сохраняет в память."""
        df = pd.read_csv(self.data_path)
        self.documents = df['full_text'].tolist()

    def build_indexes(self):
        """Строит оба индекса и сохраняет их."""
        # Предобработка для TF-IDF
        processed_texts_tfidf = [preprocess_for_tfidf(doc) for doc in self.documents]
        self.tfidf_indexer.build_index(processed_texts_tfidf)

        # Предобработка для BERT
        processed_texts_bert = [preprocess_for_bert(doc) for doc in self.documents]
        self.bert_indexer.build_index(processed_texts_bert)

    def search(self, query: str, index_type: str = 'tf-idf', top_n: int = 5) -> List[str]:
        """
        Выполняет поиск по заданному индексу.

        :param query: Запрос для поиска
        :param index_type: Тип индексации ('tf-idf' или 'bert')
        :param top_n: Количество возвращаемых результатов
        :return: Индексы релевантных документов
        """
        if index_type == 'tf-idf':
            processed_query = preprocess_for_tfidf(query)
            return self.tfidf_indexer.search(processed_query, top_n)
        elif index_type == 'bert':
            processed_query = preprocess_for_bert(query)
            return self.bert_indexer.search(processed_query, top_n)
        else:
            raise ValueError("Неверный тип индекса. Используйте 'tf-idf' или 'bert'.")

