# Создание сортировщика и графического дизайна для него
Необходимые библиотеки.
```python
import pickle  # бинаризация объектов
from tkinter import *  #  для работы с графическим дизайном
from tkinter.ttk import *  #  для работы с графическим дизайном
from threading import *  #  для реализации многопоточности 
import os  # для использования функций командной строки
from datetime import datetime  # для работы со временем
from PIL import ImageTk  # для изображений
import shutil  # для перемещения файлов
import gensim.downloader  # установщик предобученных векторов слов
from email import policy
from email.parser import BytesParser  # парсер текста из писем
from nltk.corpus import stopwords  # список стопслов (you, are, were, is и т.д.)
import numpy as np  # для работы с векторами
import nltk  # для работы с текстом
from xgboost import XGBClassifier  # ускоренный градиентный спуск
```
Функции для сортировщика.
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
            
            
def mail_sorter(model, wv, list_dir, 
                spam_dir, ham_dir, path_to_eml_files):  # предсказывает класс для текста и перемещает файл в соответствующую папку
    for file in list_dir:
        if file[-4:] == '.eml':
            text = text_extractor(path, file)
            try:
                tokenized_text = tokenize_text(text)
            except:
                tokenized_text = tokenize_text(file)
            word_array = avg_word(wv, tokenized_text).reshape((1, 300))
            prediction = model.predict(word_array)
            if prediction == 1:
                shutil.move(f'{path}/{file}', f'{spam_dir}/{file}')
            elif prediction == 0:
                shutil.move(f'{path}/{file}', f'{ham_dir}/{file}')


def eml_check(lst):  # проверяет наличие .eml файлов в директории
    for i in lst:
        if i[-4:] == '.eml':
            return True
        else:
            return False


def sorter(path, model, wv):  # переименовывает файлы в читабельные, проверяет на наличие .eml файлов и сортирует почту
    lst = os.listdir(path)
    if eml_check(lst):
        rename_files(path)
        lst = os.listdir(path)
        mail_sorter(model=model, wv=wv, list_dir=lst)
```
Так как в интерфейсе есть полоска прогресса, то нужно создать два процесса, которые будут идти параллельно, чтобы полоска заполнялось вместе с прогрессом сортировки.

Далее функции, которые необходимы для работы интерфейса.
```python
def progress(path):  # эта функция заполняет полоску прогресса и выводит сообщения в зависимости от результата работы ПО
    spam_vol_start = len(os.listdir(path_to_spam_dir))
    ham_vol_start = len(os.listdir(path_to_ham_dir))
    st_time = datetime.now()  # засекает время начала выполнения кода
    ln = len(os.listdir(path))
    coef = 0.1

    if len(os.listdir(path)) != 0:
        # очищает интерфейс от старых сообщений
        frame2.place_forget()
        label_empty.pack_forget()
        label_spam.pack_forget()
        label_ham.pack_forget()
        label_time.pack_forget()
        flag = False

        prog_bar.place(relx=.5, rely=.5, anchor="c")
        while not flag:
            if len(os.listdir(path)) <= (ln - ln*coef):  # заполняет полоску прогресса
                prog_bar['value'] += 10
                coef += 0.1
            if len(os.listdir(path)) == 0:
                prog_bar['value'] = 100
                spam_vol_end = len(os.listdir(path_to_spam_dir))
                ham_vol_end = len(os.listdir(path_to_ham_dir))
                # вывод результатов работы ПО на экран
                label_spam.configure(text=f'Spam messages = {spam_vol_end - spam_vol_start}')
                label_ham.configure(text=f'Ham messages = {ham_vol_end - ham_vol_start}')
                label_time.configure(text=f'Total time = {(datetime.now() - st_time).total_seconds()}')
                label_spam.pack()
                label_ham.pack()
                label_time.pack()
                frame1.place(relx=.5, rely=.6, anchor="c")
                flag = True
    else:
        # очищает интерфейс от старых сообщений
        prog_bar.place_forget()
        label_spam.pack_forget()
        label_ham.pack_forget()
        label_time.pack_forget()
        label_empty.pack_forget()
        frame1.place_forget()
        # вывод сообщения о том, что папка пуста
        label_empty.pack()
        frame2.place(relx=.5, rely=.6, anchor="c")

def start():  # функция внедренная в кнопку, начинает функции progress и sorter
    but.configure(state=DISABLED)
    # задаются процессы, которые будут выполняться параллельно
    t1 = Thread(target=sorter, args=(path_to_eml_files, model, google_vectors, lbl1['text'], lbl2['text'], lbl3['text']))
    t2 = Thread(target=progress, args=(path_to_eml_files,))
    # запуск процессов
    t1.start()
    t2.start()
    but.configure(state=NORMAL)
    
def browse_button():  # функция выбора директории для писем
    eml_folder_path = filedialog.askdirectory()
    lbl1.configure(text=eml_folder_path)
    lbl1.place(relx=.5, rely=.2, anchor="c")
    but3.place(relx=.5, rely=.3, anchor="c")
    but2.configure(state=DISABLED)


def browse_button_2():  # функция выбора директории для спам писем
    spam_folder_path = filedialog.askdirectory()
    lbl2.configure(text=spam_folder_path)
    lbl2.place(relx=.5, rely=.4, anchor="c")
    but4.place(relx=.5, rely=.5, anchor="c")
    but3.configure(state=DISABLED)


def browse_button_3():  # функция выбора директории для не спам писем
    ham_folder_path = filedialog.askdirectory()
    lbl3.configure(text=ham_folder_path)
    lbl3.place(relx=.5, rely=.6, anchor="c")
    but5.place(relx=.5, rely=.7, anchor="c")
    but4.configure(state=DISABLED)


def clear_window():  # функция слудующего шага
    lbl1.place_forget()
    lbl2.place_forget()
    lbl3.place_forget()
    but2.place_forget()
    but3.place_forget()
    but4.place_forget()
    but5.place_forget()
    but.place(relx=.5, rely=.3, anchor="c", height=100, width=300)
```
Код графического интерфейса.
```python
# создание и настройка основного окна
root = Tk()
root.geometry('1050x800')
root.title('Spam filter')
root.iconbitmap(path_to_icon_file)
root.resizable(0, 0)  # запрет на изменение разрешения окна
# установка фона
file = path_to_background_file
photo = ImageTk.PhotoImage(file=file)
background_label = Label(root, image=photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
# создание стилей для объектов
style = Style()
style.configure('TButton', font=('Courier', 20))
style.configure('L.TLabel', font=('Courier', 14), relief=RAISED, borderwidth=50, width=40, anchor='c')
# создание объектов интерфейса и их настройка
frame1 = Frame()
frame2 = Frame()

prog_bar = Progressbar(root, orient=HORIZONTAL, length=300, mode='determinate')

label_spam = Label(master=frame1, text='', style='L.TLabel')
label_ham = Label(master=frame1, text='', style='L.TLabel')
label_time = Label(master=frame1, text='', style='L.TLabel')
label_empty = Label(master=frame2, text='Message folder is empty', style='L.TLabel')

lbl1 = Label(root, text='', style='L.TLabel', width=70)
lbl2 = Label(root, text='', style='L.TLabel', width=70)
lbl3 = Label(root, text='', style='L.TLabel', width=70)

but2 = Button(text='Browse directory with .eml files', command=browse_button, width=40)
but2.place(relx=.5, rely=.1, anchor="c")

but3 = Button(text='Browse directory for spam mails', command=browse_button_2, width=40)

but4 = Button(text='Browse directory for ham mails', command=browse_button_3, width=40)

but5 = Button(text='Next step', command=clear_window)

but = Button(root, text='START', command=start, style='TButton')
but.place(relx=.5, rely=.3, anchor="c", height=100, width=300)
# запуск бесконечной петли для работы интерфейса
mainloop()
```
[На главную страницу](https://github.com/PapaGoose/spam_bfu/tree/main)
