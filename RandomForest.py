import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor 


dataset = pd.read_csv("rastsal.csv")

parametre =  dataset.iloc[:,1:2].values
hedef = dataset.iloc[:,2].values

kararAgaci = RandomForestRegressor(n_estimators=20, random_state=0)
kararAgaci.fit(parametre, hedef)

deneyim = input("Çalışanın deneyimini yıl cinsinden giriniz:")
tahmin = kararAgaci.predict(np.array([deneyim]).reshape(1, 1))
print(str(deneyim) + " yıl deneyimi olan birisinin tahmini maaşı : " + str(tahmin[0]) +" TL olur")

X = np.arange(min(parametre), max(parametre), 0.01)
X = X.reshape((len(X), 1))
plt.scatter(parametre, hedef, color = 'red')
plt.plot(X, kararAgaci.predict(X), color = 'blue')
plt.title('Rastgele Orman Algoritması, Regresyon Örneği')
plt.xlabel('Deneyim (Yıl)')
plt.ylabel('Maaş')
plt.show()