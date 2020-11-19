# Источники датасетов
1)https://www.kaggle.com/uciml/sms-spam-collection-dataset \n
2)https://www.kaggle.com/ozlerhakan/spam-or-not-spam-dataset
3)https://www.kaggle.com/karthickveerakumar/spam-filter
4)https://www.kaggle.com/mandygu/lingspam-dataset
5)http://www.aueb.gr/users/ion/data/enron-spam/
6)https://www.kaggle.com/shravan3273/sms-spam?select=spamraw.csv
### Библеотеки
```python
import os #библеотека для использования команд ОС
import tarfile #библеотека для работы с .tar файлами
import pandas as pd #библеотека для работы с датасетами
import numpy as np #библеотека для работы с датасетами(почти все объекты из таблиц pandas являются объектами numpy)
from tqdm import tqdm #библеотека для визуализации итераций
import re #библеотека для регулярных выражений
```
### Начало обработки датасетов
Сначала enron(http://www.aueb.gr/users/ion/data/enron-spam/), надо разархивировать папки и из текстовых файлов, хранящихся в них достать текст и разметить его.
```python
for i in range(6):# разархивируем файлы
    tar = tarfile.open(f'Enron_dataset/enron{i+1}.tar.gz', 'r:gz')
    tar.extractall('Enron_dataset')
    tar.close()
```
```python
df = pd.DataFrame(columns=['text', 'spam'])#создаем таблицу для данных

dirs = ['ham', 'spam'] #добавляем текст из файлов в таблицу
for i in tqdm(range(6)):
    for dr in dirs:
        
        if dr == 'ham':
            spam_ind = 0
        else:
            spam_ind = 1
            
        for ftxt in os.listdir(f'Enron_dataset/enron{i+1}/{dr}'):
            with open(f'Enron_dataset/enron{i+1}/{dr}/{ftxt}', 'rb') as f:
                text = f.read()
                df = df.append({'text':str(text), 'spam':spam_ind}, ignore_index = True)
```
Удаляем лишние знаки, которые появляются после перевода бинаризированного типа в строчный.
```python
chars_to_remove = ['\\r','\\n','-','|','b\'','+','\\','/','\'','b\"','Subject:']

for j in tqdm(range(len(df))):# удаляем лишние знаки
    for i in chars_to_remove:
        df.text.iloc[j] = df.text.iloc[j].replace(i, '')
```
```python
df.to_csv('CSV_data/enron.csv', index=False)# переводим таблицу в csv файл
```
### Следующий этап
Другие датасеты были взяты с kaggle, где они хранились в удобном нам csv формате (с помощью pandas можнно просто считывать такие файлы в таблицы)
```python
df1 = pd.read_csv('CSV_data/emails.csv')
df2 = pd.read_csv('CSV_data/enron.csv')
df3 = pd.read_csv('CSV_data/messages.csv')
df4 = pd.read_csv('CSV_data/spam.csv', encoding='cp1251')
df5 = pd.read_csv('CSV_data/spam_or_not_spam.csv')
df6 = pd.read_csv('CSV_data/spamraw.csv')
```
Дальше простые преобразования: меняем названия столбцов, удалем лишние
```python
df3 = df3.drop(['subject'], axis=1).rename(columns={'message':'text', 'label':'spam'})
df4 = df4.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1).rename(columns={'v1':'spam', 'v2':'text'})
df4.spam = df4.spam.map({'ham':0, 'spam':1})
df5.rename(columns={'email':'text', 'label':'spam'}, inplace=True)
df6.rename(columns={'type':'spam'}, inplace=True)
```
После этого соединяем в одну таблицу
```python
df_new = pd.concat([df1, df2, df3, df4, df5, df6])
df_new.dropna(inplace=True)#удаляем строки с пустыми значениями
df_new.drop_duplicates(inplace=True)#удаляем одинаковые строки
df_new.reset_index(drop=True, inplace=True)# переобозначаем индексы
df_new.to_csv('CSV_data/main_file.csv', index=False)# создаем основной файл из всех наших таблиц
```
### Небольшие преобразования текста
В дальнейшем нужно будет удалять все знаки, поэтому знаковые для нас знаки меняем на слова и цифры тоже, так как значения их не очень важны и их разнообразие будет только мешать, поэтому заменить на слово будет лучше
```python
for i in tqdm(range(len(df_new))):
    df_new.text.iloc[i] = re.sub('$', 'CURRENCY', df_new.text.iloc[i])
    df_new.text.iloc[i] = re.sub('\d+', 'NUMBER', df_new.text.iloc[i])
    df_new.text.iloc[i] = re.sub('%', 'PERCENT', df_new.text.iloc[i])
    df_new.text.iloc[i] = re.sub('£', 'POUND', df_new.text.iloc[i])
    df_new.text.iloc[i] = re.sub('¥', 'YEN', df_new.text.iloc[i])
```
И смотрим размер нанешнего датасета
```python
df_new.shape
```
(52213, 2)
