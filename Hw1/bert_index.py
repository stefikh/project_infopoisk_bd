import pickle  # Для сохранения и загрузки индексов
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from typing import List


class BERTIndexer:
    def __init__(self, save_path: str, model_name: str = 'DeepPavlov/rubert-base-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.save_path = save_path
        self.embeddings = None

    def build_index(self, texts: List[str]):
        """
        Строит индекс на основе эмбеддингов BERT и сохраняет его.

        :param texts: Список текстов для индексации
        """
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)  # Максимальная
            # длина последовательности + берём pytorch
            outputs = self.model(**inputs)
            embeddings.append(outputs.pooler_output.detach().numpy())

        with open(self.save_path, 'wb') as f:
            pickle.dump(embeddings, f)

    def load_index(self):
        """Загружает индекс."""
        with open(self.save_path, 'rb') as f:
            self.embeddings = pickle.load(f)

    def search(self, query: str, top_n: int = 5) -> List[int]:
        """
        Выполняет поиск по запросу с использованием эмбеддингов BERT и возвращает индексы наиболее релевантных документов.

        :param query: Текст запроса
        :param top_n: Количество топ результатов
        :return: Список индексов документов
        """
        inputs = self.tokenizer(query, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        query_embedding = outputs.pooler_output.detach().numpy()

        # Рассчет косинусной близости. Берём из pytorch функцию, берём запрос и документы.
        scores = [torch.nn.functional.cosine_similarity(torch.tensor(query_embedding), torch.tensor(doc_embedding),
                                                        dim=1).item()
                  for doc_embedding in self.embeddings]
        scores_array = np.array(scores)
        ranked_indices = scores_array.argsort()[::-1][:top_n]
        return ranked_indices
