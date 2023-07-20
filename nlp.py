import pandas as pd
import numpy as np

# Burayı ekstra olarak yazdık veriyi düzgün ayıramadığımız için
data_list = []

with open("data/Restaurant_Reviews.csv", "r") as file:
    lines = file.readlines()

for line in lines:
    last_comma_index = line.rfind(",")  # Cümle içindeki son virgülün indeksini bulma
    review = line[:last_comma_index]    # Cümleyi son virgüle göre ayrıştırma
    liked = line[last_comma_index + 1:].strip()  # Son virgülden sonrasını "Liked" değeri olarak alıyoruz
    data_list.append([review, liked])

print(data_list[1])

data = pd.DataFrame(data_list, columns=["Review", "Liked"])
data.drop(0,inplace = True)
data["Liked"] = data["Liked"].str.strip('"')

#PREPROCESSİNG ÖN TEMİZLEME İŞLEMİ AŞAMASI


#Bizim odağımız kelimeler bu yüzden ilk işlemimiz noktalama işaretlerinden cümleleri arındırmak
import re
#Stop Words Clean (it that is gibi anlam ifade etmeyen kelimeleri temizleme)
import nltk
durma = nltk.download("stopwords") #stopwords listesi indirdik bunu biz türkçe kelimeler için de internetten bularak listeyi kullanabiliriz
from nltk.corpus import stopwords
#Porter Stemmer (Kelimeleri Köklerine ayırma misal runnin --> run)
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

derlem = []
for i in range(1,1001):
    yorum = re.sub("[^a-zA-Z]"," ",data["Review"][i])  #burada "[^a-zA-Z]" bu ifade kelime hariç hepsini al boşlukla değiştir demek
    #sıradaki işimiz büyük harflerin hepsini küçüğe çevirme ya da tam tersi
    yorum = yorum.lower()
    yorum = yorum.split()  # listeye çevirdik str idi
    #alttaki kodun anlamı ps.stem(kelime) --> kelimenin kökünü al fakat ondan önce kontrol yapıyoruz for ile yorumun
    #içindeki hepsini gez eğer o kelime stopwords in içinde yoksa onun kökünü al ve en dıştaki parantezler ise liste
    #olarak döndür demek
    #misal loved stopwords mü hayır al fakat eki at love diye kaydet this al stopwords mü evet direkt at
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    #Count Vektorizer kullanıcaz hazır vektör sayacı ve bunu kullanmak için list çevirdiğimiz yapıyı tekrer str çevirmemiz gerek
    yorum = " ".join(yorum)   #join birleştirme aralara boşluk koy ve str çevir
    derlem.append(yorum)

#Count Vektorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1700)
x = cv.fit_transform(derlem).toarray()
y = data.iloc[:,1].values

#Machine Learning
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()
gb.fit(x_train,y_train)
gb_predict = gb.predict(x_test)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(x_train,y_train)
dtc_predict = dtc.predict(x_test)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train, y_train)
rfc_predict = rfc.predict(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski') #n_neighbors = kaç komşu çizileceği metric = ölçüm stratejisi
knn.fit(x_train,y_train)
knn_predict = knn.predict(x_test)


from sklearn.metrics import confusion_matrix
gb_cm = confusion_matrix(y_test, gb_predict)
dtc_cm = confusion_matrix(y_test,dtc_predict)
rfc_cm = confusion_matrix(y_test, rfc_predict)
knn_cm = confusion_matrix(y_test, knn_predict)

print(gb_cm)
print(dtc_cm)
print(rfc_cm)
print(knn_cm)













