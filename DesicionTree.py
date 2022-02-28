import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("veriseti.csv")

x = dataset.iloc[:,[1,2,3,4]].values
etiket = dataset.iloc[:,[5]].values

for e in x:
	if e[0] == "Erkek":
		e[0] = 1
	elif e[0] == "Kadın":
		e[0] = 0

scaler = StandardScaler()
scaler.fit(x)
parametre = scaler.transform(x)

print(parametre)

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
classifier.fit(parametre, etiket)

cinsiyet = input("Hastanın cinsiyetini girin : (kadın/erkek)")
if cinsiyet == "kadın":
    cinsiyet = 0
else:
    cinsiyet = 1

yas = int(input("Hastanın yaşını girin :"))

akyuvarSayisi = int(input("Hastanın Lökosit değerini (Akyuvar sayısını) girin :"))

kronikRahatsizlik = input("Hastanın kronik bir rahatsızlığı var mı ? (evet/hayır)")
if kronikRahatsizlik=="evet":
    kronikRahatsizlik = 1
else:
    kronikRahatsizlik = 0
     
inputData = np.array([cinsiyet,yas, akyuvarSayisi, kronikRahatsizlik]).reshape(1, -1)


testVector = scaler.transform(inputData)
print(testVector)
print("-"*50)
predictionResult = classifier.predict(testVector)

if predictionResult == 1:
	print("Tahmin edilen durum: ağır seyir beklenilir")
if predictionResult == 0:
	print("Tahmin edilen durum: ayakta iyileşme beklenilir")