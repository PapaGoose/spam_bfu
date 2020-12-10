# Добавление новых данных из спамового почтового адреса в основной датасет 

Для добавления данных из писем необходимо спарсить текст сообщений из .eml файлов. Так как некоторые файлы имеют в названии символы кодировки отличной от базовой(ascii), необходимо удалить нечитабельные символы. 
Необходимые библиотеки для этого кода.
```python
from email import policy
from email.parser import BytesParser
import os
import pandas as pd
import numpy as np
```
Ниже приведены функции, которые буду использовать в коде.
```python
def text_extractor(path, file):  # функция парсит текст из .eml файла и удаляет из него лишние символы
    with open(f'{path}/{file}', 'rb') as fhdl:
        msg = BytesParser(policy=policy.default).parse(fhdl)
    try:
        text = msg.get_body(preferencelist='plain').get_content()
        chars_to_remove = ['\n*', '\n', '\t', '\'', '\"']

        for i in chars_to_remove:
            text = text.replace(i, ' ')
    except:
        return

    return str(text)


def delemoji(string):  # функция удаляющая символы нечитабельной кодировки
    return string.encode('ascii', 'ignore').decode('ascii')


def rename_files(dirc):  # функция переименовывает в файлы для избежания ошибок
    lst = os.listdir(dirc)
    for file in lst:
        if file[-4:] == '.eml':
            new_name = file.replace('\'', '`')
            new_name = new_name.replace('\"', '`')

            os.rename(f'{dirc}/{file}', f'{dirc}/{delemoji(new_name)}')
```

Далее небольшой код, который выполняет необходимые нам действия.
```python
texts = []
rename_files(path)  # переименовываем файлы
for file in os.listdir(path):  # достаем текст из файлов
    if file[-4:] == '.eml':
      text = text_extractor(path, file)
      texts.append(text)
data = {'text': [i for i in texts], 'spam': np.ones(len(texts))}  # создние словря для данных
df = pd.DataFrame(data)  # создание таблицы с данными
```
Добавление новых данных в основной датасет.
```python
df_main = df.read_csv(path_to_main_file)  # считывание основного датасет
df_new = pd.concat([df_main, df])  # соединение новыого и основного датасетов
df_new.reset_index(drop=True, inplace=True)  # создание новых индексов
df_new.to_csv(path_for_new_dataset, index=False)  # сохранение
```
