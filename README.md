# Обучение модели ML
Необходимые библиотеки.
```python
import pandas as pd  # библеотека для работы с таблицами
from xgboost import XGBClassifier  # ускоренный градиентный спуск
from sklearn.linear_model import LogisticRegression  # Логистическая регрессия
from sklearn.ensemble import RandomForestClassifier  # Случайный лес
from sklearn.model_selection import train_test_split  # разбиение выборки на тестовую и тренировочную
import pickle  # бинаризация объектов
from nltk.corpus import stopwords  # список стопслов (you, are, were, is и т.д.)
import nltk  # для работы с текстом
import gensim.downloader  # установщик предобученных векторов слов
import numpy as np  # для работы с векторами
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score  # метрики 
```
Функции, которые будут использоваться.
```python
def tokenize_text(text):  # создает список из слов, находящихся в тексте, и убирает лишние слова
    tokens = []
    stop_words = set(stopwords.words('english'))
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) >= 2 and word not in stop_words:
                tokens.append(word)
    return tokens

def avg_word(wv, words):  # считает средний вектор для текста
    shape = wv['nice'].shape
    word_avg = []

    for word in words:
        if isinstance(word, np.ndarray):
            word_avg.append(word)
        elif word in wv.vocab:
            word_avg.append(wv.vectors_norm[wv.vocab[word].index])

    if not word_avg:
        print('cannot compute similarity for:', words)
        return np.zeros(shape)

    return np.array(word_avg).mean(axis=0)


def avg_word_list(wv, text_list):  # считает средний вектор для каждого текста из списка
    return np.vstack([avg_word(wv, sent) for sent in text_list])
```
Далее идет код с комментариями.
```python
df = pd.read_csv('data/CSV_files/Clear_data/new_set.csv')  # считываем датасет

X = df.text  # обозначаем тестовые данные
y = df.spam  # обозначаем целевые данные

google_vectors = pickle.load(open('data/google_vectors.pickle', 'rb'))  # скачиваем предобученную модель Word2Vec

tokenized_X = X.apply(lambda x: tokenize_text(x))  # токенизируем текст датасета
X_word_average = avg_word_list(google_vectors, tokenized_X) # векторизируем токенизированные предложения и получаем вектора средних значений для каждого сообщения

X_train, X_test, y_train, y_test = train_test_split(X_word_average, y, test_size=0.15, random_state=222, stratify=y)  # разбиваем выборку на тестовые и тренировочные данные
```
Тестирование моделей.
```python
# обучаем модель логистической регрессии
model = LogisticRegression(max_iter=200, random_state=444)
model.fit(X_train, y_train)
# предсказываем результат
predicted = model.predict(X_test)
# высчитываем точность
accuracy_score(y_test, predicted)
```
90.8 %
```python
# обучаем модель случайного леса
rand_forest = RandomForestClassifier(n_estimators=60, max_depth=7, random_state=222)
rand_forest.fit(X_train, y_train)
# предсказываем результат
predicted = rand_forest.predict(X_test)
# высчитываем точность
accuracy_score(y_test, predicted)
```
91 %
```python
# обучаем модель градиентного бустинга
clf = XGBClassifier(n_estimators=60, max_depth=7, learning_rate=0.05, random_state=222)
clf.fit(X_train, y_train)
# предсказываем результат
predicted = clf.predict(X_test)
# высчитываем точность
accuracy_score(y_test, predicted)
```
94.4 %

XGBoost оказался наилучшей моделью, поэтому теперь обучаем эту модель на всех данных и сохраняем ее.
```python
# обучаем модель градиентного бустинга
clf = XGBClassifier(n_estimators=60, max_depth=7, learning_rate=0.05, random_state=222)
clf.fit(X_word_average, y, eval_metric='logloss')
# бинаризируем нашу модель и сохраняем ее
pickle.dump(clf, open(file.pickle, 'wb'))
```
