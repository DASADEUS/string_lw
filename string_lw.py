# -*- coding: utf-8 -*-
"""
#Обработка текста на естественном языке

### Расстояние редактирования

1.1 Загрузите предобработанные описания рецептов из файла `preprocessed_descriptions.csv`. Получите набор уникальных слов `words`, содержащихся в текстах описаний рецептов (воспользуйтесь `word_tokenize` из `nltk`).
"""

import numpy as np
import pandas as pd
from google.colab import files
import nltk
import random
import Levenshtein
from collections import Counter

pip install python-levenshtein

nltk.download('punkt')

uploaded = files.upload()

data=pd.read_csv('preprocessed_descriptions.csv')

data

strdesc=''
for i in data['description']:
  strdesc=strdesc+str(i)
strdesc

wordsall=nltk.word_tokenize(strdesc)
wordsall=list(np.unique(wordsall))
wordsall

"""1.2 Сгенерируйте 5 пар случайно выбранных слов и посчитайте между ними расстояние редактирования."""

for i in range(5):
  a=random.sample(wordsall,2)
  print('Для слов:',a,'расстояние равно:',Levenshtein.distance(a[0],a[1]))

"""1.3 Напишите функцию, которая для заданного слова `word` возвращает `k` ближайших к нему слов из списка `words` (близость слов измеряется с помощью расстояния Левенштейна)"""

def poslev(x):
  minl=100
  minw=x
  for i in wordsall:
    if (Levenshtein.distance(x,i)<minl)and(x!=i):
      minw=i
      minl=Levenshtein.distance(x,i)
  return minw

poslev('susan')

"""### Стемминг, лемматизация

2.1 На основе результатов 1.1 создайте `pd.DataFrame` со столбцами: 
    * word
    * stemmed_word 
    * normalized_word 

Столбец `word` укажите в качестве индекса. 

Для стемминга воспользуйтесь `SnowballStemmer`, для нормализации слов - `WordNetLemmatizer`. Сравните результаты стемминга и лемматизации.
"""

sn=nltk.SnowballStemmer('english')
wn=nltk.WordNetLemmatizer()

nltk.download('wordnet')

listsn=[sn.stem(i) for i in wordsall]
listwn=[wn.lemmatize(i) for i in wordsall]

snwnpd=pd.DataFrame({'word':wordsall,'stemmed_word':listsn, 'normalized_word':listwn})
snwnpd=snwnpd.set_index('word')
snwnpd[:5000]

"""2.2. Удалите стоп-слова из описаний рецептов. Какую долю об общего количества слов составляли стоп-слова? Сравните топ-10 самых часто употребляемых слов до и после удаления стоп-слов."""

nltk.download('stopwords')

stop=nltk.corpus.stopwords.words('english')
stop[:10]

clear=[i for i in wordsall if i not in stop]
print((len(wordsall)-(len(clear)))/len(wordsall))

Counter(wordsall).most_common()[:10]

Counter(clear).most_common()[:10]

"""### Векторное представление текста

3.1 Выберите случайным образом 5 рецептов из набора данных. Представьте описание каждого рецепта в виде числового вектора при помощи `TfidfVectorizer`
"""

from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer)

cv_news = CountVectorizer(tokenizer=wordsall, stop_words=stop)



tv = TfidfVectorizer(stop_words='english')

corpus_tv = tv.fit_transform(clear)

tv.get_feature_names()

corpus_tv.toarray()

for i in range(5):
  a=random.sample(wordsall,2)
  print('Для слов:',a,'расстояние равно:',Levenshtein.distance(a[0],a[1]))