# Импортируем библиотеки 
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Создаем токенизатор
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Создаем модель
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Загружаем данные. Отдадим дань уважения классике.
input_text = "Привет, мир!"

# Токенизируем данные
tokenized_input = tokenizer.tokenize(input_text)

# Преобразуем токены в индексы
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_input)

# Создаем маску токенов
segments_ids = [0] * len(tokenized_input)

# Переводим данные в тензоры
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Предсказываем класс
prediction = model(tokens_tensor, segments_tensors)


