#!/usr/bin/env python
# coding: utf-8

# # Prediccion del Precio - Datos Gilmar

# #### Importar Librerias y funciones

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.random import seed
from numpy.random import randn
from scipy.stats import t
from scipy.stats import chi2
from scipy.stats import ttest_ind
import scipy.stats as stats
import scipy


# In[2]:


def Func_indicesOutliers(df, var, coef=1.5):
    q1 = df[var].quantile([0.25])[0.25]
    q3 = df[var].quantile([0.75])[0.75]
    irq = q3-q1
    out_low = q1-(coef*irq)
    out_hig = q3+(coef*irq)
    
    return pd.concat([ df[df[var] > out_hig] , df[df[var] < out_low] ] , axis = 0).index


# In[3]:


def Func_Outliers(df: pd.DataFrame, var: str, coef=1.5) -> float:
    
    q1=df[var].quantile([.25])[0.25]
    q3=df[var].quantile([.75])[0.75]
    irq=q3-q1
    out_low=q1-(coef*irq)
    out_hig=q3+(coef*irq)
    
    
    return  irq, out_low, out_hig, round(100 * (pd.concat([df[df[var]<out_low],df[df[var]>out_hig]],axis=0)).shape[0] / df[var].shape[0],2)


# #### Importar datos

# In[5]:


df = pd.read_csv(r'C:\Users\carolina\Desktop\FormDataAnalyst\ExamenFinal\gilmar.csv', encoding = 'ISO-8859-1',  delimiter=';')


# In[6]:


df.head(3)


# In[7]:


df.shape #  registros 1728 / 16 variables


# In[8]:


df.head(5)


# In[9]:


df.sample(n=5)


# In[10]:


df.info()  ### No hay nulos


# In[11]:


df.columns


# In[12]:


df.corr()


# #### Variable objetivo Precio

# In[13]:


df[['precio']].plot.kde()


# In[14]:


df[['precio']].plot.box()


# In[15]:


df[['precio']].hist()


# In[16]:


indice_atipicos_precio=Func_indicesOutliers(df, 'precio')
len(indice_atipicos_precio), len(indice_atipicos_precio)*100/df.shape[0] 


# In[389]:


Func_Outliers(df, 'precio')


# In[17]:


round(Func_Outliers(df, 'precio')[3],2)


# In[391]:


Func_Outliers(df, 'precio',2)


# In[387]:


df.iloc[list(set(df.index.values).difference(set(indice_atipicos_precio)))]['precio'].count() #.plot.box() # 1636


# In[388]:


+1675+53


# ###  Estudio del resto de las varaibles en relacion con el precio

# #### m2Brutos

# In[18]:


df['m2Brutos'].plot.box()#kde()


# In[19]:


df[df['m2Brutos']<30]['m2Brutos'].count() ## hay demasiados casos con un valor de menos de 30 m2brutos


# In[20]:


df[df['m2Util']<30]['m2Util'].count() # sin embargo no hay casos de m2 utiles de menos de 30


# In[21]:


round(df[df['m2Brutos']<30]['m2Brutos'].count()*100/df.shape[0],2) # la mitad de los registros no tienen un valor correcto


# In[22]:


## esta varaible esta fallida


# In[23]:


df[['m2Brutos','m2Util']].describe()


# In[24]:


df[['m2Brutos','m2Util']].corr() # esto no tiene sentido


# In[25]:


df[['m2Brutos','precio']].corr() # ademas no correlaciona con precio


# In[26]:


df['m2Brutos'].describe()


# In[27]:


df[df['m2Brutos']>0][['m2Brutos', 'precio']].groupby(by='m2Brutos', sort=True).count().sort_values(by='m2Brutos', ascending =True)# 


# In[28]:


df[['m2Brutos', 'm2Util', 'precio']].groupby(by=['m2Brutos', 'm2Util']).count()
## No hay relacion entre los m2 Brutos y los Utiles. Esta variable ademas no correlaciona con precio (.11) 
## Queda fuera del estudio


# In[29]:


df[(df['m2Brutos']>30) & (df['m2Brutos']<5000)][['m2Brutos','precio']].count() # estos podrian ser valores rezonables


# In[30]:


df[(df['m2Brutos']<=30) | (df['m2Brutos']>=5000)][['m2Brutos','precio']].count() # la otra mitad no tiene sentido


# #### edad

# In[31]:


df[['edad','precio']].corr() 


# In[32]:


df[['edad']].plot.box()


# In[33]:


df[df['edad']<100][['edad','precio']].corr() 


# In[34]:


# Tambien presenta datos incoherentes y tiene baja correlacion. 
# Antes de descartarla analizo la correlacion en el rango de datos que podria considerarse normal para esta variable.
#df[['edad', 'precio']].groupby(by=['edad']).count()
df[df['edad']<100][['edad','precio']].corr()  #son 1600 reg aprox Sigue dando muy bajo queda fuera 


# In[35]:


df['edad'].plot.box()


# In[36]:


# Analizo la posibilidad de convertirla en categorica


# In[37]:


df[['edad','precio']].groupby(by='edad').count()


# In[ ]:





# In[ ]:





# #### valorTerreno

# In[38]:


df['valorTerreno'].plot.kde() 
#df['precio'].hist()#plot.kde() 


# In[39]:


df[['valorTerreno','precio']].corr()


# In[40]:


## Valor terreno es unna variable importante corr .58 
df[['valorTerreno']].plot.box()


# In[41]:


df['valorTerreno'].describe()


# In[42]:


# Hay muchos outliers. Hay que tratarlos.

#Cuantos hay?
Por_out_ValorT=Func_Outliers(df, 'valorTerreno',2.5)
Por_out_ValorT
# coef 1.5 10% outliers son muchos
# coef 2 6.5% outliers esto esta mejor 
# coef 2.5 5% outliers 


# In[43]:


ind_Atipicos_ValorT=Func_indicesOutliers(df,'valorTerreno',2.5)
len(ind_Atipicos_ValorT), round(len(ind_Atipicos_ValorT)*100/df.shape[0], 2)


# In[44]:


ind_Atipicos_ValorT=Func_indicesOutliers(df,'valorTerreno',1.5)
len(ind_Atipicos_ValorT), round(len(ind_Atipicos_ValorT)*100/df.shape[0], 2)


# In[45]:


df.iloc[list(set(df.index.values).difference(set(ind_Atipicos_ValorT)))]['valorTerreno'].plot.box() # 1636


# In[46]:


df.iloc[list(set(df.index.values).difference(set(ind_Atipicos_ValorT)))][['valorTerreno','precio']].corr()


# In[47]:


df.iloc[ind_Atipicos_ValorT][['valorTerreno','precio']].corr()


# In[48]:


df.iloc[df.index.values][['valorTerreno','precio']].corr()


# #### Copio a otro df para hacer el tratamiento de los outliers

# In[49]:


df_aux=df.copy() #[list(set(df.index.values).difference(set(ind_Atipicos_ValorT)))]


# In[50]:


df_aux.shape


# In[51]:


Func_Outliers(df_aux, 'valorTerreno',2.5)


# In[52]:


Func_Outliers(df_aux, 'valorTerreno')


# In[53]:


Func_Outliers(df_aux, 'valorTerreno',2.5)


# In[54]:


df_aux[df_aux['valorTerreno']>77850.0]['precio'].count() 


# In[60]:


# Como se observa en la grafica no hay valores por debajo del rango minimo
df_aux[df['valorTerreno']<Func_Outliers(df_aux, 'valorTerreno')[1]]['precio'].count()


# In[61]:


df_aux[df_aux['valorTerreno']>Func_Outliers(df_aux, 'valorTerreno')[2]]['precio'].count() # 121


# In[62]:


df_aux[df_aux['valorTerreno']>Func_Outliers(df_aux, 'valorTerreno',2.5)[2]]['precio'].count() 


# In[63]:


Func_Outliers(df_aux, 'valorTerreno')[2]


# In[64]:


Func_Outliers(df_aux,'valorTerreno') 


# In[66]:


df_aux[df_aux['valorTerreno']<77850.0][['precio','valorTerreno']].corr()


# In[67]:


ind_VT=Func_indicesOutliers(df_aux, 'valorTerreno')


# #### Recuperacion de los outliers

# In[68]:


df_aux.iloc[ind_VT,3]=77850 ## Asignacion del valor mas alto del rango a los outliers


# In[69]:


df_aux[df_aux['valorTerreno']==77850.0]


# In[70]:


df_aux[['valorTerreno']].plot.box()


# In[71]:


df_aux[['valorTerreno','precio']].corr()


# ### m2Util

# In[72]:


df_aux[['m2Util','precio']].corr()


# In[73]:


## Variable m2Util esta variable es importante corr .73
## presenta outliers por arriba 
#df[['m2Util']].plot.kde()
df_aux[['m2Util']].plot.box()


# In[74]:


# Hay outliers. Hay que tratarlos.
Por_out_ValorU=Func_Outliers(df_aux, 'm2Util')
Por_out_ValorU


# In[75]:


ind_Atipicos_ValorU=Func_indicesOutliers(df_aux,'m2Util')
len(ind_Atipicos_ValorU), round(len(ind_Atipicos_ValorU)*100/df.shape[0], 2)


# In[76]:


df_aux.iloc[list(set(df_aux.index.values).difference(set(ind_Atipicos_ValorU)))]['m2Util'].plot.box() 


# In[77]:


df_aux.iloc[list(set(df_aux.index.values).difference(set(ind_Atipicos_ValorU)))][['m2Util','precio']].corr()


# In[78]:


df.iloc[ind_Atipicos_ValorU][['m2Util','precio']].corr() ## esta es la correlacion de los atipicos


# #### Eliminacion de los atipicos

# In[79]:


# 1728-17=1711 correcto


# In[80]:


df_aux.iloc[list(set(df_aux.index.values).difference(set(ind_Atipicos_ValorU)))].count()


# In[82]:


df.shape


# In[86]:


df_aux.loc[list(set(df_aux.index.values).difference(set(ind_Atipicos_ValorU)))].shape


# In[87]:


#### Elimino 17 reg outliers m2Util


# In[89]:


df_aux=df_aux.loc[list(set(df_aux.index.values).difference(set(ind_Atipicos_ValorU)))]


# In[90]:


df_aux.shape


# #### perUni

# In[91]:


df_aux[['perUni','precio']].corr() # correlacion baja


# In[92]:


df_aux[['perUni']].plot.box()


# In[93]:


# Hay outliers. Hay que tratarlos.
Por_out_ValorPU=Func_Outliers(df_aux, 'perUni')
Por_out_ValorPU


# In[94]:


ind_Atipicos_ValorPU=Func_indicesOutliers(df_aux,'perUni')
len(ind_Atipicos_ValorPU), round(len(ind_Atipicos_ValorPU)*100/df.shape[0], 2)


# In[95]:


df_aux.loc[list(set(df_aux.index.values).difference(set(ind_Atipicos_ValorPU)))]['perUni'].plot.box() ## iloc da error por fuera de rango


# In[96]:


df_aux.loc[list(set(df_aux.index.values).difference(set(ind_Atipicos_ValorU)))][['perUni','precio']].corr()


# In[97]:


# df[['perUni', 'precio']].groupby(by='perUni', sort='perUni').count()


# #### numDormi

# In[98]:


## cuantitativa discreta. Miro la frecuencia y veo la distribucion.


# In[99]:


df_aux[['numDormi','precio']].corr() 


# In[100]:


df_aux[['numDormi']].plot.kde() 


# In[101]:


df_aux[['numDormi', 'precio']].groupby(by='numDormi', sort='numDormi').count() 


# In[102]:


## 40% de correlacion
# Vamos a considerar los registros con numDormi entre 2 y 5 dormitorios. Consideraremos outliers las de 1,6 y 7
# Son 15 reg representan menos del 1% (de esta manera quitamos ruido/casos extremos que no interesa considerar)


# In[103]:


df_aux[(df_aux['numDormi']>=2) & (df_aux['numDormi']<=5) ][['numDormi', 'precio']].corr()


# In[104]:


df_aux.head(2)


# #### Recategorizo y Convierto a dummy la variable numDormi 

# In[105]:


# creo las columnas
df_aux['DormiHasta2']=0
df_aux['Dormi3']=0
df_aux['DormiDesde4']=0
df_aux[['numDormi','DormiHasta2','Dormi3','DormiDesde4']]


# In[107]:


df_aux.columns


# In[108]:


## Asigno los valores a las columnas creadas


# In[109]:


df_aux.loc[df_aux[(df_aux['numDormi']<=2)].index,'DormiHasta2']=1
df_aux.loc[df_aux[(df_aux['numDormi']==3)].index,'Dormi3']=1
df_aux.loc[df_aux[(df_aux['numDormi']>=4)].index,'DormiDesde4']=1
df_aux[['numDormi','DormiHasta2','Dormi3','DormiDesde4']].sample(10)


# In[110]:


df_aux[['numDormi', 'precio']].groupby(by='numDormi', sort='numDormi').count() 


# In[111]:


df_aux['DormiHasta2'].sum(), df_aux['Dormi3'].sum(), df_aux['DormiDesde4'].sum()


# In[112]:


df_aux[['numDormi','DormiHasta2','Dormi3','DormiDesde4']].sample(5)


# In[ ]:





# #### numChime

# In[113]:


df_aux[['numChime','precio']].corr() # Baja correlacion


# In[114]:


df_aux[['numChime']].plot.kde() 


# In[115]:


df_aux[['numChime', 'precio']].groupby(by='numChime', sort='numChime').count() 


# In[116]:


# Debido a la distribucion de la variable la convertire en Dummy con/sin chimenea (740 sin/990 con aprox)


# #### ASIGNACION para convertir la variable en booleana

# In[118]:


df_aux.loc[list(df_aux[(df_aux['numChime']==2) | (df_aux['numChime']==3)].index), ['numChime']] = 1 # 37 ok # con loc sino no va


# In[119]:


df_aux.head(2)


# In[120]:


df_aux[['numChime', 'precio']].groupby(by='numChime', sort='numChime').count() 


# In[121]:


df_aux[['numChime','precio']].corr()


# In[122]:


# Evaluo correlacion con t-test


# In[123]:


ttest_ind(df_aux[df_aux['numChime']==0]['precio'], df_aux[df_aux['numChime']==1]['precio']) 


# In[125]:


#El p-value es bajo 


# In[126]:


df_aux[df_aux['numChime']==0][['precio']].sort_values(by='precio')


# In[127]:


df_aux[df_aux['numChime']==0][['precio']].plot.kde()


# In[128]:


df_aux[df_aux['numChime']==1][['precio']].plot.kde()


# #### numServi

# In[129]:


df_aux[['numServi','precio']].corr()


# In[130]:


# esta variable no se corresponde con la descripcion. 
# No se de que dato se trata pero correlaciona al .58. No la puedo desestimar.
# De esta quito 0, 40 (4 reg 0.2%) Tratar 30 y 35 tambien como outliers.


# In[131]:


df_aux[['numServi', 'precio']].groupby(by='numServi', sort='numServi').count()


# In[132]:


df_aux.shape


# In[133]:


df_aux.loc[list(df_aux[(df_aux['numServi']==0) | (df_aux['numServi']==40)].index), ['numServi']] # son estos 4


# In[134]:


list(df_aux[(df_aux['numServi']==0) | (df_aux['numServi']==40)].index)


# In[135]:


len(list(df_aux[(df_aux['numServi']!=0) & (df_aux['numServi']!=40)].index)) # estos deben quedar


# #### ELIIMINACION en numServi 4 reg

# In[136]:


df_aux=df_aux.loc[list(df_aux[(df_aux['numServi']!=0) & (df_aux['numServi']!=40)].index)]


# In[137]:


df_aux.shape


# In[138]:


df_aux[['numServi', 'precio']].groupby(by='numServi', sort='numServi').count() # Los dos ultimos tienen pocos reg.


# In[139]:


Func_Outliers(df_aux, 'numServi') # No cuenta con outliers


# In[140]:


Func_Outliers(df_aux, 'numServi',1) # este parace ser el coeficiente que aplica el box&whiskers


# In[141]:


df_aux[['numServi']].plot.box()


# In[142]:


df_aux[['numServi','precio']].corr()


# #### numHabita

# In[143]:


df_aux[['numHabita','precio']].corr()


# In[144]:


df_aux[['numHabita', 'precio']].groupby(by='numHabita', sort='numHabita').count()  


# In[145]:


len(list(df_aux[(df_aux['numHabita']>3) & (df_aux['numHabita']<11)].index)) # 87%


# In[146]:


df_aux.loc[list(df_aux[(df_aux['numHabita']>3) & (df_aux['numHabita']<11)].index)][['numHabita','precio']].corr()
# Si quito los extremos baja la correlacion. Con tal que se queda asi.


# In[147]:


df_aux[['numHabita']].plot.box()


# #### ESTUDIO DE LA CORRELACION DE LA VARIBLE OBJETIVO CON LAS CUALITATIVAS
# 

# #### calefaccion

# In[148]:


df_aux['calefaccion'].value_counts() # Se ve correcta


# In[149]:


# dummifico calefaccion 


# In[150]:


pd.get_dummies(df_aux['calefaccion'])


# In[151]:


df_aux[['calef_elec','calef_aero','calef_suelo']]=pd.get_dummies(df['calefaccion'])


# In[152]:


df_aux.head(3)


# In[153]:


## Hay menos casos en suelo radiante. Quito esta dummy
   
df_aux=df_aux.drop('calef_suelo',axis=1)


# In[154]:


df_aux.columns


# In[155]:


## t-test  El t-test solo admite binomial. En la combinacion a pares los p-valores dan bajos pero no se si esto es concluyente.


# In[156]:


ttest_ind(df_aux[df_aux['calefaccion']=='aerotermia']['precio'], df_aux[df_aux['calefaccion']=='suelo radiante']['precio']) 


# In[157]:


ttest_ind(df_aux[df_aux['calefaccion']=='Electrica']['precio'], df_aux[df_aux['calefaccion']=='suelo radiante']['precio'])


# In[158]:


ttest_ind(df_aux[df_aux['calefaccion']=='aerotermia']['precio'], df_aux[df_aux['calefaccion']=='Electrica']['precio'])


# In[159]:


# correlacion mediante las dummies. La correlacion es baja pero tampoco se si es correcta esta forma de evaluacion.


# In[160]:


df_aux[['calef_elec','calef_aero','precio']].corr()


# In[162]:


# Hasta ahora las variables cuantitativas que entran en el modelado son 'valorTerreno', 'm2Util', 'numServi','numHabita'


# In[163]:





# #### ELIMINACION dummy redundante en numDormi

# In[169]:


df_aux=df_aux.drop('DormiHasta2',axis=1)


# In[ ]:


# Analizare en que medida las categorias de la variable influyen en la correlacion del resto de variables.


# In[178]:


df_aux[['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[173]:


## Correlacion en las distintas categorias de calefaccion 


# In[179]:


df_aux[df_aux['calefaccion']=='aerotermia'][['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[180]:


df_aux[df_aux['calefaccion']=='Electrica'][['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[177]:


df_aux[df_aux['calefaccion']=='suelo radiante'][['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[181]:


# Hay variaciones en algunas correlaciones en las distintas categorias de 'calefaccion'. Pueden hacer algun aporte al modelo. 


# In[ ]:


# Lo mismo para numChime


# In[189]:


df_aux[['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[193]:


df_aux[df_aux['numChime']==1][['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[191]:


df_aux[df_aux['numChime']==0][['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[ ]:


# Tambien se observan cambios en la correlacion de las cuantitativas por lo que entiendo que pueden servir para ajustar el modelo.


# In[ ]:


# Lo mismo con numChime


# In[194]:


df_aux[['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[195]:


df_aux[df_aux['Dormi3']==0][['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[196]:


df_aux[df_aux['Dormi3']==1][['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[197]:


df_aux[df_aux['DormiDesde4']==0][['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[198]:


df_aux[df_aux['DormiDesde4']==1][['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[ ]:


## Tambien hay cambios. Creo que pueden ajustar el modelo.


# In[ ]:





# #### alimentacion

# In[199]:


df_aux['alimentacion'].value_counts() # Se ve correcta


# In[200]:


df_aux[['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[201]:


df_aux[df_aux['alimentacion']=='gas'][['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[202]:


df_aux[df_aux['alimentacion']=='Electrica'][['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[203]:


df_aux[df_aux['alimentacion']=='Gasoil'][['precio','valorTerreno','m2Util','numServi', 'numHabita']].corr()


# In[204]:


## Las distintas categorias de esta variable producen cambios en la correlacion de las cuantitativas. Se dumifica y se incorpora al modelo.


# In[ ]:


### dummificacion 'alimentacion'


# In[205]:


df_aux.loc[:,'alimentacion'].value_counts()


# In[207]:


pd.get_dummies(df_aux['alimentacion']).head()


# In[209]:


df_aux[['alim_elec','alim_gasoil','alim_gas']]=pd.get_dummies(df_aux['alimentacion'])
df_aux.sample(5)


# In[210]:


df_aux.loc[:,'alim_elec'].value_counts()


# In[211]:


df_aux.loc[:,'alim_gasoil'].value_counts()


# In[212]:


df_aux.loc[:,'alim_gas'].value_counts()


# In[208]:


df_aux.head(5)


# In[213]:


### elimino la dummy con menor frecuencia.


# In[214]:


df_aux=df_aux.drop('alim_gasoil',axis=1)


# #### tipoDesague

# In[216]:


df_aux['tipoDesague'].value_counts()  # elimino los nulos que representan un 0.7% del total


# In[ ]:


### correlacion mediante t-test


# In[217]:


ttest_ind(df[df['tipoDesague']=='comunitario']['precio'], df[df['tipoDesague']=='fosa septica']['precio']) 


# In[ ]:


# El p.valor es bajo. Hay dependencia.
# Elimino los nulos en tipoDesague son pocos registros.


# In[220]:


df_aux=df_aux[df_aux['tipoDesague']!='none']


# In[221]:


df_aux.shape


# In[222]:


df_aux.loc[:,'tipoDesague'].value_counts()


# In[223]:


pd.get_dummies(df_aux['tipoDesague']).head()


# In[224]:


df_aux[['des_comuni','des_fosa']]=pd.get_dummies(df_aux['tipoDesague'])
df_aux.sample(5)


# In[225]:


df_aux.loc[:,'des_comuni'].value_counts()


# In[226]:


df_aux.loc[:,'des_fosa'].value_counts()


# In[213]:


### elimino la dummy con menor frecuencia.


# In[227]:


df_aux=df_aux.drop('des_fosa',axis=1)


# In[229]:


df_aux.loc[:,'tipoDesague'].value_counts()


# In[231]:


df_aux.loc[:,'des_comuni'].value_counts()


# #### conVistas

# In[232]:


df_aux['conVistas'].value_counts() ## booleana


# In[234]:


# t-test
ttest_ind(df_aux[df_aux['conVistas']=='No']['precio'], df_aux[df_aux['conVistas']=='Sí']['precio']) 


# In[235]:


##### asignacion para la conversion a booleana


# In[236]:


df_aux.loc[list(df_aux[(df_aux['conVistas']=='No')].index), ['conVistas']]=0


# In[237]:


df_aux.loc[list(df_aux[(df_aux['conVistas']=='Sí')].index), ['conVistas']] = 1 


# In[238]:


df_aux['conVistas'].value_counts() # ok


# #### construccion

# In[240]:


df_aux['construccion'].value_counts() ## booleana


# In[241]:


ttest_ind(df_aux[df_aux['construccion']=='No']['precio'], df_aux[df_aux['construccion']=='Sí']['precio']) 


# In[242]:


##### asignacion para la conversion a booleana


# In[243]:


df_aux.loc[list(df_aux[(df_aux['construccion']=='No')].index), ['construccion']]=0


# In[244]:


df_aux.loc[list(df_aux[(df_aux['construccion']=='Sí')].index), ['construccion']] = 1 


# In[245]:


df_aux['construccion'].value_counts() # ok


# #### aire

# In[246]:


df_aux['aire'].value_counts() ## booleana


# In[247]:


ttest_ind(df_aux[df_aux['aire']=='No']['precio'], df_aux[df_aux['aire']=='Sí']['precio']) 


# In[248]:


##### asignacion para la conversion a booleana


# In[249]:


df_aux.loc[list(df_aux[(df_aux['aire']=='No')].index), ['aire']]=0


# In[250]:


df_aux.loc[list(df_aux[(df_aux['aire']=='Sí')].index), ['aire']] = 1 


# In[251]:


df_aux['aire'].value_counts() # ok


# ## CASO 1

# ### Variables que no entran en consideracion para el modelo de RLM: m2Brutos, edad, perUni.

# In[257]:


df_aux=df_aux.drop(['m2Brutos','edad','perUni'], axis=1)


# ### Variables a incorporar al modelo RLM: valorTerreno, m2Util, numServi, numHabita, (hasta aqui cuantitativas) Dormi3 y DormiDesde4 , numChime , calef_elec , calef_aero, alim_gas, alim_elec, des_comuni, conVistas, construccion, aire (dummies de categoricas)

# In[258]:


df_aux.columns


# In[286]:


df_aux.info()


# ### Tipificación de variables no numericas a int64

# In[289]:


df_aux.head(3)


# In[287]:


# convertir a int64 : conVistas   , construccion    , aire          


# In[302]:


# convertir a int64: calef_elec    ,calef_aero    , alim_elec     , alim_gas      , des_comuni  


# In[316]:


df_aux[['des_comuni']].value_counts()


# In[292]:


df_aux[['conVistas']]=df_aux[['conVistas']].astype('int64')


# In[295]:


df_aux[['construccion']]=df_aux[['construccion']].astype('int64')


# In[298]:


df_aux[['aire']]=df_aux[['aire']].astype('int64')


# In[303]:


df_aux[['calef_elec']]=df_aux[['calef_elec']].astype('int64')


# In[306]:


df_aux[['calef_aero']]=df_aux[['calef_aero']].astype('int64')


# In[309]:


df_aux[['alim_elec']]=df_aux[['alim_elec']].astype('int64')


# In[312]:


df_aux[['alim_gas']]=df_aux[['alim_gas']].astype('int64')


# In[315]:


df_aux[['des_comuni']]=df_aux[['des_comuni']].astype('int64')


# In[317]:


df_aux.info()


# In[ ]:





# ### Separacion de variable objetivo y variables independientes

# In[318]:


df_aux.iloc[:,[1,2,4,5,6,10,11,12,13,14,15,16,17,18,19]].columns


# In[424]:


x=df_aux.iloc[:,[1,2,4,5,6,10,11,12,13,14,15,16,17,18,19]].values


# In[425]:


y=df_aux.iloc[:,0].values


# In[426]:


df_aux.iloc[:,[1,2,4,5,6,10,11,12,13,14,15,16,17,18,19]].head(1)


# In[427]:


x[0] 


# In[428]:


df_aux.iloc[:,0].head(1)


# In[429]:


y[0]


# ### Division train-test

# In[269]:


from sklearn.model_selection import train_test_split


# In[430]:


x_train,x_test, y_train,y_test = train_test_split(x,y, test_size = 0.30 , random_state = 1987) # ver porcentaje


# In[431]:


x_train.shape, y_train.shape, # 5 son las columnas


# In[432]:


x_test.shape, y_test.shape


# ### Modelado RLM 1

# In[274]:


from sklearn.linear_model import LinearRegression


# In[420]:


r = LinearRegression()


# In[433]:


r.fit(x_train, y_train)


# In[434]:


y_pred=r.predict(x_test)


# ### Iteracion del modelo

# In[278]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[435]:


x = np.append(arr=np.ones((df_aux.shape[0],1)).astype(int), values=x, axis=1 )
                  # matriz de filas totales X una columna que se agregan al comienzo de x
x[0]
# ones 'valorTerreno', 'm2Util', 'numChime', 'numServi', 'numHabita','conVistas', 'construccion', 'aire', 'Dormi3', 
#'DormiDesde4',    'calef_elec', 'calef_aero', 'alim_elec', 'alim_gas', 'des_comuni'


# In[332]:


# significacion=0.05 # umbral del p-valor


# In[333]:


x_optimo=x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]   

# En primer lugar ponemos todas las variables. 
# Luego vamos quitando la de mayor p-valor, hasta p-valor <= significacion (0,05)


# In[334]:


regresion_ols=sm.OLS(endog=y, exog=x_optimo).fit()
regresion_ols.summary()   # P es el p-valor


# In[335]:


x_optimo=x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15]]   
 # quite x13 Cuidado aqui son las columnas de x no de x_optimo!! (alim_elec)
x_optimo[0] 
# ones, 'valorTerreno', 'm2Util', 'numChime', 'numServi', 'numHabita','conVistas', 'construccion', 'aire', 'Dormi3', 
#'DormiDesde4', 'calef_elec', 'calef_aero', 'alim_gas', 'des_comuni'


# In[336]:


regresion_ols=sm.OLS(endog=y, exog=x_optimo).fit()
regresion_ols.summary()   # P es el p-valor


# In[337]:


x_optimo=x[:,[0,1,2,4,5,6,7,8,9,10,11,12,14,15]]   
 # quite x3 que es la 1 Cuidado aqui son las columnas de x no de x_optimo!! ('numChime')
x_optimo[0] 
# ones, 'valorTerreno', 'm2Util', 'numServi', 'numHabita','conVistas', 'construccion', 'aire', 'Dormi3', 
#'DormiDesde4', 'calef_elec', 'calef_aero', 'alim_gas', 'des_comuni'


# In[338]:


regresion_ols=sm.OLS(endog=y, exog=x_optimo).fit()
regresion_ols.summary()   # P es el p-valor


# In[339]:


x_optimo=x[:,[0,1,2,4,5,6,7,8,9,10,11,12,15]]   
 # quite x12  Cuidado aqui son las columnas de x no de x_optimo!! ('alim_gas')
x_optimo[0] 
# ones, 'valorTerreno', 'm2Util', 'numServi', 'numHabita','conVistas', 'construccion', 'aire', 'Dormi3', 
#'DormiDesde4', 'calef_elec', 'calef_aero', 'des_comuni'


# In[340]:


regresion_ols=sm.OLS(endog=y, exog=x_optimo).fit()
regresion_ols.summary()   # P es el p-valor


# In[341]:


x_optimo=x[:,[0,1,2,4,5,6,7,8,9,10,12,15]]   
 # quite x11  Cuidado aqui son las columnas de x no de x_optimo!! ('calef_elec')
x_optimo[0] 
# ones, 'valorTerreno', 'm2Util', 'numServi', 'numHabita','conVistas', 'construccion', 'aire', 'Dormi3', 
#'DormiDesde4', 'calef_aero', 'des_comuni'


# In[342]:


regresion_ols=sm.OLS(endog=y, exog=x_optimo).fit()
regresion_ols.summary()   # P es el p-valor


# In[343]:


x_optimo=x[:,[0,1,2,4,5,6,7,8,10,12,15]]   
 # quite x9  Cuidado aqui son las columnas de x no de x_optimo!! ('Dormi3')
x_optimo[0] 
# ones, 'valorTerreno', 'm2Util', 'numServi', 'numHabita','conVistas', 'construccion', 'aire', 
#'DormiDesde4', 'calef_aero', 'des_comuni'


# In[344]:


regresion_ols=sm.OLS(endog=y, exog=x_optimo).fit()
regresion_ols.summary()   # P es el p-valor


# In[ ]:





# ##### Interpretacion del informe OLS Mínimos cuadrados ordinarios
# 
# 
# Quito la de mayor p-valor absoluto (P en el informe):
# 
#     1. (x13 con p-valor=0,964) alim_elec
#     2. (x3 con p-valor=0,669)  numChime
#     3. (x12 con p-valor=0,256) alim_gas
#     4. (x10 con p-valor=0,407) calef_elec
#     5. (x8 con p-valor=0,174)  Dormi3
#     
# * Para un nivel de confianza del 95 por ciento, un valor p (probabilidad) menor que 0,05 indica una heterocedasticidad o no estacionariedad estadísticamente significativa. 
#     
# * R cuadrado: indica que el modelo (sus variables explicativas modeladas con una regresión lineal) explica aproximadamente ese porcentaje a la variable dependiente. 
# 
#   

# In[ ]:





# ##### CALCULO DE LA PREDICCION CON LAS VARIABLES RESULTANTES DE LA SELECCION

# 
# * Evaluamos el modelo con las variables resultantes:
# 
# ones, 'valorTerreno', 'm2Util', 'numServi', 'numHabita','conVistas', 'construccion', 'aire', 'DormiDesde4', 'calef_aero', 'des_comuni'

# In[345]:


x[:,[0,1,2,4,5,6,7,8,10,12,15]][0] # quito la 0 que es la de ones


# In[ ]:


x[:,[0,1,3,4,5,6,7,9,11,14]][0] 
# quito la 0 que es la de ones y resto 1 al resto de indices porque en x_train y x_test no esta la columna de ones.


#'valorTerreno', 'm2Util', 'numServi', 'numHabita','conVistas', 'construccion', 'aire', 'DormiDesde4', 'calef_aero', 'des_comuni'.


# In[352]:


df_aux[(df_aux['valorTerreno']==20300) & (df_aux['m2Util']==2387)][['valorTerreno', 'm2Util', 'numServi', 'numHabita','conVistas', 'construccion', 'aire', 'DormiDesde4', 'calef_aero', 'des_comuni']]


# In[355]:


x_train[0,[0,1,3,4,5,6,7,9,11,14]] # correcto 


# In[356]:


regresion_auto=LinearRegression() 
regresion_auto.fit(x_train[:,[0,1,3,4,5,6,7,9,11,14]],y_train) ## esto siempre calculado con el lote de entrenamiento 


# In[357]:


y_pred_test=regresion_auto.predict(x_test[:,[0,1,3,4,5,6,7,9,11,14]])


# In[877]:


y_pred_test.shape


# * Evaluamos el modelo en x_train, y_train

# In[358]:


r2_score(y_train, regresion_auto.predict(x_train[:,[0,1,3,4,5,6,7,9,11,14]]))


# * Evaluamos el modelo con las variables predictoras en x_test, y_test

# In[359]:


r2_score(y_test, regresion_auto.predict(x_test[:,[0,1,3,4,5,6,7,9,11,14]]))


# In[ ]:


# No es muy satisfactorio el r2


# In[360]:


len(y_test), len(y_pred_test)


# In[878]:


plt.title('Modelo RLM')
plt.plot(y_test, color='green')
plt.plot(y_pred_test, color='blue')
plt.show


# ### Analisis de las diferencias entre y_test / y_test_predict 

# In[884]:


import seaborn as sn


# In[957]:


# KDE Plot with seaborn
plt.title('Distribución real y predicción en Modelo RLM')
res = sn.kdeplot(y_test, color='red', shade='True', legend='Distribución real')
res2 = sn.kdeplot(y_pred_test, color='blue', shade='False', legend='Distribución predicción')
plt.show()


# In[907]:


Diferencias_RLM=np.round(y_test-y_pred_test)


# In[910]:


Diferencias_RLM=Diferencias_RLM.astype(int)


# In[953]:


pd.DataFrame(Diferencias_RLM).plot.kde()


# In[916]:


pd.DataFrame(Diferencias_RLM).plot.box()


# In[ ]:





# In[377]:


#y_test.reshape(509,1)


# In[381]:


df_aux[['precio','valorTerreno','m2Util','numServi', 'numHabita', 'numDormi','numChime']].corr()


# In[382]:


df[['precio','valorTerreno','m2Util','numServi', 'numHabita', 'numDormi','numChime']].corr()


# In[385]:


df[['precio','valorTerreno','m2Util','numServi', 'numHabita', 'numDormi','numChime']].head(3)


# In[383]:


df.shape


# ### CASO 2

# In[392]:


# Vamos con el df original y las variables que tienen correlacion con precio. Sin quitar ni tratar outliers.
# 'valorTerreno', 'm2Util', 'numServi', 'numHabita', 'numDormi','numChime' (las ultimas dos no tienen mucha pero las incluyo igual)
df.info() 


# In[ ]:


# las variables que voy a usar ya son enteras


# ### Separacion de variable objetivo y variables independientes

# In[393]:


df.columns # 3,4,8,9,6,7


# In[394]:


df.iloc[:,[3,4,8,9,6,7]].columns # serán estas


# In[469]:


x2=df.iloc[:,[3,4,8,9,6,7]].values


# In[470]:


y2=df.iloc[:,0].values


# In[471]:


df.iloc[:,[3,4,8,9,6,7]].head(1)


# In[472]:


x2[0] 


# In[473]:


df.iloc[:,0].head(1)


# In[474]:


y2[0]


# ### Division train-test

# In[269]:


from sklearn.model_selection import train_test_split


# In[475]:


x2_train,x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size = 0.30 , random_state = 1987) 


# In[476]:


x2_train.shape, y2_train.shape, # 5 son las columnas


# In[477]:


x2_test.shape, y2_test.shape


# In[478]:


x2[0]


# In[479]:


x2_train[0]


# In[480]:


x2_test[0]


# In[481]:


len(x2), len(x2_test), len(x2_train)


# In[ ]:


# El primer array de x2 no es primer array ni de train, ni de test.


# In[491]:


x2_train[106] # esta en train pero no sale primero


# In[485]:


np.where(x2_train == 50000) ## para buscar el array de x2 (el primero)


# ### Modelado RLM 2

# In[274]:


from sklearn.linear_model import LinearRegression


# In[445]:


regresion2 = LinearRegression()


# In[492]:


regresion2.fit(x2_train, y2_train)


# In[493]:


y2_pred=regresion2.predict(x2_test)


# ### Iteracion del modelo 2

# In[278]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[557]:


x2 = np.append(arr=np.ones((df.shape[0],1)).astype(int), values=x2, axis=1 )
                  # matriz de filas totales X una columna que se agregan al comienzo de x
x2[0]
# ones, 'valorTerreno', 'm2Util', 'numChime', 'numServi', 'numHabita','conVistas', 'construccion', 'aire', 'Dormi3', 
#'DormiDesde4', 'calef_elec', 'calef_aero', 'alim_elec', 'alim_gas', 'des_comuni'


# In[332]:


# significacion=0.05 # umbral del p-valor


# In[558]:


x2_optimo=x2[:,[0,1,2,3,4,5,6]]   

# En primer lugar ponemos todas las variables. 
# Luego vamos quitando la de mayor p-valor, hasta p-valor <= significacion (0,05)


# In[560]:


regresion_ols=sm.OLS(endog=y2, exog=x2_optimo).fit()
regresion_ols.summary()   # P es el p-valor


# ##### CALCULO DE LA PREDICCION CON LAS VARIABLES RESULTANTES DE LA SELECCION

# 
# * Evaluamos el modelo con las variables resultantes:
# 
#      ones 'valorTerreno', 'm2Util', 'numServi', 'numHabita', 'numDormi'

# In[499]:


x2[:,[1,2,3,4,5]][0] # quito la 0 que es la de ones
'valorTerreno', 'm2Util', 'numServi', 'numHabita', 'numDormi'


# In[500]:


df[(df['valorTerreno']==50000) & (df['m2Util']==906)][['valorTerreno', 'm2Util', 'numServi', 'numHabita', 'numDormi']]


# In[513]:


x2_train[0]#[0,1,2,3,4,5]
#'valorTerreno', 'm2Util', 'numServi', 'numHabita', 'numDormi','numChime'
# tengo que quitar la ultima que se excluyo del modelo
x2_train[[0],[0,1,2,3,4]]


# In[514]:


regresion_auto=LinearRegression() 
regresion_auto.fit(x2_train[:,[0,1,2,3,4]],y2_train) ## esto siempre calculado con el lote de entrenamiento 


# In[515]:


y2_pred_test=regresion_auto.predict(x2_test[:,[0,1,2,3,4]])


# * Evaluamos el modelo en x_train, y_train

# In[516]:


r2_score(y2_train, regresion_auto.predict(x2_train[:,[0,1,2,3,4]]))


# * Evaluamos el modelo con las variables predictoras en x_test, y_test

# In[517]:


r2_score(y2_test, regresion_auto.predict(x2_test[:,[0,1,2,3,4]]))


# ### CASO 2.1 Dejo solo m2Util, valorTerreno, numServi

# In[518]:


df.corr()


# ### Separacion de variable objetivo y variables independientes

# In[393]:


df.columns # m2Util, valorTerreno, numServi 
            4, 3, 8


# In[526]:


df.iloc[:,[4, 3, 8]].columns # serán estas 3


# In[562]:


x21=df.iloc[:,[4, 3, 8]].values


# In[563]:


y21=df.iloc[:,0].values


# In[529]:


df.iloc[:,[4, 3, 8]].head(1)


# In[564]:


x21[0] 


# In[565]:


df.iloc[:,0].head(1)


# In[566]:


y21[0]


# ### Division train-test

# In[269]:


from sklearn.model_selection import train_test_split


# In[567]:


x21_train,x21_test, y21_train, y21_test = train_test_split(x21, y21, test_size = 0.30 , random_state = 1987) 


# In[568]:


x21_train.shape, y21_train.shape, # 5 son las columnas


# In[569]:


x21_test.shape, y21_test.shape


# In[570]:


x21[0]


# In[537]:


x21_train[0]


# In[571]:


x21_test[0]


# In[572]:


len(x21), len(x21_test), len(x21_train)


# In[ ]:


# El primer array de x2 no es primer array ni de train, ni de test.


# In[573]:


x21_train[512] # esta en train pero no sale primero


# In[547]:


np.where(x21_train == 50000) ## para buscar el array de x2 (el primero)


# ### Modelado RLM 2

# In[274]:


from sklearn.linear_model import LinearRegression


# In[574]:


regresion21 = LinearRegression()


# In[575]:


regresion21.fit(x21_train, y21_train)


# In[576]:


y21_pred=regresion21.predict(x21_test)


# ### Iteracion del modelo 2.1

# In[278]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[577]:


x21 = np.append(arr=np.ones((df.shape[0],1)).astype(int), values=x21, axis=1 )
                  # matriz de filas totales X una columna que se agregan al comienzo de x
x21[0]
#


# In[332]:


# significacion=0.05 # umbral del p-valor


# In[578]:


x21_optimo=x21[:,[0,1,2,3]]   

# En primer lugar ponemos todas las variables. 
# Luego vamos quitando la de mayor p-valor, hasta p-valor <= significacion (0,05)


# In[579]:


regresion_ols=sm.OLS(endog=y21, exog=x21_optimo).fit()
regresion_ols.summary()   # P es el p-valor


# ##### CALCULO DE LA PREDICCION CON LAS VARIABLES RESULTANTES DE LA SELECCION

# 
# * Evaluamos el modelo con las variables resultantes:
# 
#      ones m2Util, valorTerreno, numServi 

# In[588]:


x21[:,[1,2,3]] # quito la 0 que es ones 


# In[590]:


df[(df['valorTerreno']==50000) & (df['m2Util']==906)][['m2Util', 'valorTerreno', 'numServi']]


# In[593]:


x21_train[0]#[0,1,2,3,4,5]
#'m2Util', valorTerreno', 'numServi'
# no quito ninguna 


# In[594]:


regresion_auto=LinearRegression()  # todas las columnas porque no quite ninguna en este modelado
regresion_auto.fit(x21_train[:,[0,1,2]],y21_train) ## esto siempre calculado con el lote de entrenamiento 


# In[595]:


y21_pred_test=regresion_auto.predict(x21_test[:,[0,1,2]])


# * Evaluamos el modelo en x_train, y_train

# In[596]:


r2_score(y21_train, regresion_auto.predict(x21_train[:,[0,1,2]]))


# * Evaluamos el modelo con las variables predictoras en x_test, y_test

# In[597]:


r2_score(y21_test, regresion_auto.predict(x21_test[:,[0,1,2]]))


# In[ ]:


### DA LO MISMO QUE EL MODELO 2.


# ### CASO 3 RLS con la variable m2Util (corr .71) y los datos originales

# In[598]:


df.head(2)


# In[600]:


x3=df.iloc[:,4].values
y3=df.iloc[:,0].values


# In[601]:


x3_train, x3_test, y3_train, y3_test = train_test_split(x3,y3,test_size=0.3, random_state=1987)


# In[607]:


len(x3_train)


# In[608]:


len(x3_test)


# In[609]:


len(y3_train)


# In[610]:


len(y3_test)


# #### Creacion del modelo

# In[612]:


regresion=LinearRegression()
regresion.fit(x3_train.reshape(-1,1),y3_train.reshape(-1,1))


# Prediccion

# In[613]:


y3_pred=regresion.predict(x3_test.reshape(-1,1))


# In[617]:


y3_pred.size


# In[616]:


y3_test.size


# #### Evaluación del modelo

# ##### R cuadrado o coheficiente de determinacion
#  
# Refleja la bondad del ajuste de un modelo a la variable que pretender explicar.
# 
# Sera un valor entre -1 (en relaciones inversamente proporcionales) y 1 (en relaciones directamente proporcionales)
# 
# Cuando mas aproxime R^2 a 1 o -1 mejor será el modelo

# In[35]:


from sklearn.metrics import r2_score


# Primero lo vemos sobre los datos de entrenamiento

# In[618]:


# coef R2
r2_score(y3_train.reshape(-1,1), regresion.predict(x3_train.reshape(-1,1)))


# In[619]:


r2_score(y3_test.reshape(-1,1), regresion.predict(x3_test.reshape(-1,1)))


# In[40]:


## no deberia dar mejor en test que en training
# Baja bastante. Aunque la correlacion con la variable es de 0.71, no predice a 0.71, sino a 0.50/0.52 


# In[39]:





# In[621]:


plt.title('Precio del inmueble Vs m2Utiles')
plt.xlabel('m2 Utiles')
plt.ylabel('Precio del inmueble')
plt.scatter(x3_test, y3_test, color='red')
plt.plot(x3_test, regresion.predict(x3_test.reshape(-1,1)), color='blue')
plt.show


# In[ ]:





# ### CASO 3.1 RLS con la variable m2Util (corr .71) y los datos tratados outliers, etc.

# In[622]:


df_aux.head(2)


# In[635]:


x31=df_aux.iloc[:,2].values
y31=df_aux.iloc[:,0].values


# In[636]:


x31_train, x31_test, y31_train, y31_test = train_test_split(x31,y31,test_size=0.3, random_state=1987)


# In[637]:


len(x31_train)


# In[638]:


len(x31_test)


# In[639]:


len(y31_train)


# In[640]:


len(y31_test)


# #### Creacion del modelo

# In[641]:


regresion=LinearRegression()
regresion.fit(x31_train.reshape(-1,1),y31_train.reshape(-1,1))


# Prediccion

# In[642]:


y31_pred=regresion.predict(x31_test.reshape(-1,1))


# In[643]:


y31_pred.size


# In[644]:


y31_test.size


# #### Evaluación del modelo

# ##### R cuadrado o coheficiente de determinacion
#  
# Refleja la bondad del ajuste de un modelo a la variable que pretender explicar.
# 
# Sera un valor entre -1 (en relaciones inversamente proporcionales) y 1 (en relaciones directamente proporcionales)
# 
# Cuando mas aproxime R^2 a 1 o -1 mejor será el modelo

# In[645]:


from sklearn.metrics import r2_score


# Primero lo vemos sobre los datos de entrenamiento

# In[646]:


# coef R2
r2_score(y31_train.reshape(-1,1), regresion.predict(x31_train.reshape(-1,1)))


# In[647]:


r2_score(y31_test.reshape(-1,1), regresion.predict(x31_test.reshape(-1,1)))


# In[40]:


## Aun peor.


# In[648]:


plt.title('Precio del inmueble Vs m2Utiles')
plt.xlabel('m2 Utiles')
plt.ylabel('Precio del inmueble')
plt.scatter(x31_test, y31_test, color='red')
plt.plot(x31_test, regresion.predict(x31_test.reshape(-1,1)), color='blue')
plt.show


# In[ ]:





# ## PRUEBAS CON EL RESTO DE MODELOS, CON LOS DATOS DE DF_AUX Y CON LAS VARIABLES 'm2Util','valorTerreno','numServi','numHabita'

# In[ ]:





# ### Prueba de distintos SVR 

# In[655]:


df_aux[['m2Util','valorTerreno','numServi','numHabita']].head()


# In[659]:


df_aux.head(1)


# ### Separar predictoras y objetivo

# In[673]:


x4=df_aux.iloc[:,[1,2,5,6]].values # [[]] para que sea un array de arrays o sea matriz y luego no tenga que hacer el reshape en los modelos
y4=df_aux.iloc[:,[0]].values


# ### Division train-test

# In[269]:


from sklearn.model_selection import train_test_split


# In[674]:


x4_train,x4_test, y4_train, y4_test = train_test_split(x4, y4, test_size = 0.30 , random_state = 1987) 


# In[675]:


len(x4_train),len(x4_test), len(y4_train), len(y4_test)


# #### Normalizacion / estandarizacion

# * En este tipo de modelo si que es necesario normalizar. Se hace despues de separar training y test.

# In[661]:


from sklearn.preprocessing import StandardScaler


# * Una transformacion para cada variable (en este caso x,y)

# In[676]:


sc_x=StandardScaler()
x4_train=sc_x.fit_transform(x4_train)
x4_train


# In[677]:


sc_y=StandardScaler()
y4_train=sc_y.fit_transform(y4_train)
y4_train


# #### Creacion del modelo
# 
# Pruebo con los distintos kernels

# In[664]:


from sklearn.svm import SVR # de Super vector machine importamos Super vertor regression


# Opciones de kernel de SVR 
# 
# - linear (para modelos lineales)
# - poly (para modelos polinomicos)
# - rbf (radial basic function o gaussiano para modelos radiales)
# - sigmoid (sigmoidal para modelos sigmoides)
# - precomputed (no es usado para ML convencional)
# 
# Otros parametros
# 
# - degree :  para indicar el grado del polinomio en kernel poly (por defecto 3)
# - gamma: para darle escalado al sigmoide, rbf o poly (por defecto auto escalado)

# In[678]:


reg_svr_klineal=SVR(kernel='linear')
reg_svr_klineal.fit(x4_train,y4_train.ravel())  # la advertencia es para el modelo lineal quiere un array 1D. Le agregue ravel para 1D


# In[679]:


reg_svr_kpoly=SVR(kernel='poly')
reg_svr_kpoly.fit(x4_train,y4_train.ravel())  


# In[680]:


reg_svr_kgauss=SVR(kernel='rbf')  # 
reg_svr_kgauss.fit(x4_train,y4_train.ravel())  


# In[681]:


reg_svr_ksigmoide=SVR(kernel='sigmoid')
reg_svr_ksigmoide.fit(x4_train,y4_train.ravel())  


# #### Prediccion

# RECORDAR QUE APLICAMOS LA ESTANDARIZACION!!! 
# 
# * Para transformar fit.transform 
# * Para volver fit.inverse
# 
# 

# In[686]:


y_svr_klineal=reg_svr_klineal.predict(sc_x.fit_transform( x4_test ) )  # Transformar antes de pasar el valor y debe ser array de array
sc_y.inverse_transform(  [y_svr_klineal]  )                            A# hay que hacer la inversa de la transformacion en 'y' para volver al rango de valores original


# In[688]:


y_svr_kpoly=reg_svr_kpoly.predict(sc_x.fit_transform(x4_test))
sc_y.inverse_transform( [y_svr_kpoly] )


# In[690]:


y_svr_kgauss=reg_svr_kgauss.predict(sc_x.fit_transform(x4_test))
sc_y.inverse_transform( [y_svr_kgauss] )  


# In[691]:


y_svr_ksigmoide=reg_svr_ksigmoide.predict(sc_x.fit_transform(x4_test))
sc_y.inverse_transform( [y_svr_ksigmoide] ) 


# #### Evaluacion de los modelos
# 
# 

# In[692]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error 


# En el calculo de los estadisticos no hace falta transformar. Hacemos la prediccion directamente sobre x
# 
# NOTA: En estos ejemplos no estan corregidos los mae y mse. Habria que transformar las variables.
# 
# El R2 da bien aunque no se transforme ni se haga la inversa de la transformacion.

# * Kernel lineal 

# In[ ]:


## tuve que hacer un reshape para que quedara como el y4_test


# In[705]:


r2_klineal=r2_score(y4_test, sc_y.inverse_transform(  [y_svr_klineal]  ).reshape(509,1) ) 
mae_klineal=mean_absolute_error(y4_test, sc_y.inverse_transform(  [y_svr_klineal]  ).reshape(509,1))
mse_klineal=mean_squared_error(y4_test, sc_y.inverse_transform(  [y_svr_klineal]  ).reshape(509,1))
print(f'Estadisticos de la SVR con kernel lineal\n r2: {r2_klineal:.2f} \n mae: {mae_klineal:.2f} \n mse: {mse_klineal:.2f}')


# * Kernel polinomico

# In[707]:


r2_kpoly=r2_score(y4_test, sc_y.inverse_transform(  [y_svr_kpoly]  ).reshape(509,1) ) 
mae_kpoly=mean_absolute_error(y4_test, sc_y.inverse_transform(  [y_svr_kpoly]  ).reshape(509,1)  )
mse_kpoly=mean_squared_error(y4_test, sc_y.inverse_transform(  [y_svr_kpoly]  ).reshape(509,1)   )
print(f'Estadisticos de la SVR con kernel polinomico\n r2: {r2_kpoly:.2f} \n mae: {mae_kpoly:.2f} \n mse: {mse_kpoly:.2f}')


# * Kernel Gaussiano o radial (rbf)

# In[708]:


r2_kgauss=r2_score(y4_test, sc_y.inverse_transform(  [y_svr_kgauss]  ).reshape(509,1)  ) 
mae_kgauss=mean_absolute_error(y4_test, sc_y.inverse_transform(  [y_svr_kgauss]  ).reshape(509,1))
mse_kgauss=mean_squared_error(y4_test, sc_y.inverse_transform(  [y_svr_kgauss]  ).reshape(509,1))
print(f'Estadisticos de la SVR con kernel radial\n r2: {r2_kgauss:.2f} \n mae: {mae_kgauss:.2f} \n mse: {mse_kgauss:.2f}')


# * Kernel Sigmoide

# In[709]:


r2_ksigm=r2_score(y4_test, sc_y.inverse_transform(  [y_svr_ksigmoide]  ).reshape(509,1) ) 
mae_ksigm=mean_absolute_error(y4_test, sc_y.inverse_transform(  [y_svr_ksigmoide]  ).reshape(509,1))
mse_ksigm=mean_squared_error(y4_test, sc_y.inverse_transform(  [y_svr_ksigmoide]  ).reshape(509,1))
print(f'Estadisticos de la SVR con kernel Sigmoide\n r2: {r2_ksigm:.2f} \n mae: {mae_ksigm:.2f} \n mse: {mse_ksigm:.2f}')
# Esto da muy mal


# * Este ultimo da muy mala aproximacion 

# In[726]:


plt.title('linear')
plt.plot(y4_test, color='green')
plt.plot(sc_y.inverse_transform(  [y_svr_klineal]  ).reshape(509,1), color='blue')
plt.show


# In[730]:


plt.title('radial')
plt.plot(y4_test, color='green')
plt.plot(sc_y.inverse_transform(  [y_svr_kgauss]  ).reshape(509,1), color='blue')
plt.show


# ###  ARBOL DE DECISION

# In[277]:


# from sklearn.metrics import r2_score


# In[ ]:


# Sobre las mismas cuatro varibles que mejor correlacionan con precio para no sobrecargar el modelo.


# In[736]:


df_aux[['m2Util','valorTerreno','numServi','numHabita']].head(1)


# ### Separar predictoras y objetivo

# In[731]:


x5=df_aux.iloc[:,[1,2,5,6]].values # [[]] para que sea un array de arrays o sea matriz y luego no tenga que hacer el reshape en los modelos
y5=df_aux.iloc[:,[0]].values


# ### Division train-test

# In[269]:


from sklearn.model_selection import train_test_split


# In[732]:


x5_train,x5_test, y5_train, y5_test = train_test_split(x5, y5, test_size = 0.30 , random_state = 1987) 


# In[733]:


len(x5_train),len(x5_test), len(y5_train), len(y5_test)


# #### Normalizacion / estandarizacion

# * En este tipo de modelo si que es necesario normalizar. 
# * Se hace despues de separar training y test para que los datos de training no interfieran en los de test.

# In[661]:


from sklearn.preprocessing import StandardScaler


# * Una transformacion para cada variable (en este caso x,y)

# In[734]:


sc_x=StandardScaler()
x5_train=sc_x.fit_transform(x5_train)
x5_train


# In[735]:


sc_y=StandardScaler()
y5_train=sc_y.fit_transform(y5_train)
y5_train


# In[ ]:





# #### Crear modelo

# In[737]:


from sklearn.tree import DecisionTreeRegressor


# R2 para distintos criterion
# 
#                 'squared_error' .24
#                 'friedman_mse' .25, con min_samples_split=5 .31, min_samples_split=10 .40, max_leaf_nodes=100
#                 'absolute_error' .15

# In[772]:


reg_arbol=DecisionTreeRegressor(criterion='squared_error', min_samples_split=15, max_leaf_nodes=200, random_state=1987)
reg_arbol.fit(x5_train,y5_train)


# #### Prediccion para test

# In[773]:


y_arbol=reg_arbol.predict(sc_x.fit_transform(x5_test))
sc_y.inverse_transform([y_arbol])


# #### Evaluacion del modelo (r2, mae,mse)

# In[37]:


# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[774]:


r2_arbol=r2_score(y5_test, sc_y.inverse_transform([y_arbol]).reshape(509,1) ) 
mae_arbol=mean_absolute_error(y5_test, sc_y.inverse_transform([y_arbol]).reshape(509,1))
mse_arbol=mean_squared_error(y5_test, sc_y.inverse_transform([y_arbol]).reshape(509,1))
print(f'Estadisticos de la regresion por Arbol de Desicion\n r2: {r2_arbol:.2f} \n mae: {mae_arbol:.2f} \n mse: {mse_arbol:.2f}')


# In[816]:


plt.title('Arbol de Decisión')
plt.plot(y5_test, color='green')
plt.plot(sc_y.inverse_transform(  [y_arbol]  ).reshape(509,1), color='blue')
plt.show


# In[ ]:





# ## Bosques Aleatorios

# In[775]:


# Utilizo mismos datos
df_aux[['m2Util','valorTerreno','numServi','numHabita']].head(1)


# ### Separar predictoras y objetivo

# In[776]:


x6=df_aux.iloc[:,[1,2,5,6]].values # [[]] para que sea un array de arrays o sea matriz y luego no tenga que hacer el reshape en los modelos
y6=df_aux.iloc[:,[0]].values


# ### Division train-test

# In[269]:


from sklearn.model_selection import train_test_split


# In[777]:


x6_train,x6_test, y6_train, y6_test = train_test_split(x6, y6, test_size = 0.30 , random_state = 1987) 


# In[778]:


len(x6_train),len(x6_test), len(y6_train), len(y6_test)


# #### Normalizacion / estandarizacion

# * En este tipo de modelo si que es necesario normalizar. 
# * Se hace despues de separar training y test para que los datos de training no interfieran en los de test.

# In[661]:


from sklearn.preprocessing import StandardScaler


# * Una transformacion para cada variable (en este caso x,y)

# In[779]:


sc_x=StandardScaler()
x6_train=sc_x.fit_transform(x6_train)
x6_train


# In[780]:


sc_y=StandardScaler()
y6_train=sc_y.fit_transform(y6_train)
y6_train


# #### Crear el modelo Random Forest
# 
# 
# * Parametros:
#     
#         - n_estimators: numero de arboles, por defecto 100. Habitualmente entre 300 y 1000.
#         - criterion: squared_error, absolute_error, friedman_mse, poisson. Por defecto squared_error
#         - max_depth: profundidad
#         - min_samples_split: numero de muestras minimo
#         - min_samples_leaf: numero de muestras maximo
#         - max_features: maximo numero de caracteristicas. Por defecto 'auto'. El modelo probará con una, con dos, etc.

# In[781]:


from sklearn.ensemble import RandomForestRegressor


# * criterion=squared_error
#                         n_estimators=100, r2=0.53 (con mas estimadores no cambia)
# * criterion=absolute_error
#                         n_estimators=100, r2=0.54 (con mas estimadores no cambia)
#                         n_estimators=500, r2= 0.55
# * criterion=friedman_mse
#                         n_estimators=500, r2= 0.53 (con mas estimadores no cambia)

# In[813]:


reg_randomForest=RandomForestRegressor(n_estimators=500, criterion='absolute_error', max_features='auto', random_state=1987)
reg_randomForest.fit(x6_train,y6_train) # ya estan normalizadas


# #### Prediccion para test

# In[814]:


y_randomForest=reg_randomForest.predict(sc_x.fit_transform(x6_test))
sc_y.inverse_transform([y_randomForest])


# #### Evaluacion del modelo (r2, mae,mse)

# In[37]:


# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[815]:


r2_arbol=r2_score(y6_test, sc_y.inverse_transform([y_randomForest]).reshape(509,1) ) 
mae_arbol=mean_absolute_error(y5_test, sc_y.inverse_transform([y_randomForest]).reshape(509,1))
mse_arbol=mean_squared_error(y5_test, sc_y.inverse_transform([y_randomForest]).reshape(509,1))
print(f'Estadisticos de la regresion por Bosques Aleatorios\n r2: {r2_arbol:.2f} \n mae: {mae_arbol:.2f} \n mse: {mse_arbol:.2f}')


# In[817]:


plt.title('Bosques Aleatorios')
plt.plot(y4_test, color='green')
plt.plot(sc_y.inverse_transform(  [y_randomForest]  ).reshape(509,1), color='blue')
plt.show


# In[ ]:





# ## PRUEBAS CON EL RESTO DE MODELOS, CON LOS DATOS DE DF_AUX Y CON LAS 10 VARIABLES 

# ## 10 variables mejores segun iteracion ODL

# In[818]:


# 'valorTerreno', 'm2Util', 'numServi', 'numHabita','conVistas', 'construccion', 'aire', 'DormiDesde4', 'calef_aero', 'des_comuni'.


# ### Prueba de distintos SVR 

# In[823]:


df_aux[['valorTerreno', 'm2Util', 'numServi', 'numHabita','conVistas', 'construccion', 'aire', 'DormiDesde4', 'calef_aero', 'des_comuni']].head(3)


# In[822]:


df_aux.iloc[:,[1,2,5,6,10,11,12,14,16,19]].head(3) # son estas las que 10 variables 


# In[659]:


df_aux.head(1)


# ### Separar predictoras y objetivo

# In[824]:


x4=df_aux.iloc[:,[1,2,5,6,10,11,12,14,16,19]].values # [[]] para que sea un array de arrays o sea matriz y luego no tenga que hacer el reshape en los modelos
y4=df_aux.iloc[:,[0]].values


# ### Division train-test

# In[825]:


from sklearn.model_selection import train_test_split


# In[826]:


x4_train,x4_test, y4_train, y4_test = train_test_split(x4, y4, test_size = 0.30 , random_state = 1987) 


# In[827]:


len(x4_train),len(x4_test), len(y4_train), len(y4_test)


# #### Normalizacion / estandarizacion

# * En este tipo de modelo si que es necesario normalizar. Se hace despues de separar training y test.

# In[828]:


from sklearn.preprocessing import StandardScaler


# * Una transformacion para cada variable (en este caso x,y)

# In[829]:


sc_x=StandardScaler()
x4_train=sc_x.fit_transform(x4_train)
x4_train


# In[830]:


sc_y=StandardScaler()
y4_train=sc_y.fit_transform(y4_train)
y4_train


# #### Creacion del modelo
# 
# Pruebo con los distintos kernels

# In[664]:


from sklearn.svm import SVR # de Super vector machine importamos Super vertor regression


# Opciones de kernel de SVR 
# 
# - linear (para modelos lineales)
# - poly (para modelos polinomicos)
# - rbf (radial basic function o gaussiano para modelos radiales)
# - sigmoid (sigmoidal para modelos sigmoides)
# - precomputed (no es usado para ML convencional)
# 
# Otros parametros
# 
# - degree :  para indicar el grado del polinomio en kernel poly (por defecto 3)
# - gamma: para darle escalado al sigmoide, rbf o poly (por defecto auto escalado)

# In[831]:


reg_svr_klineal=SVR(kernel='linear')
reg_svr_klineal.fit(x4_train,y4_train.ravel())  # la advertencia es para el modelo lineal quiere un array 1D. Le agregue ravel para 1D


# In[832]:


reg_svr_kpoly=SVR(kernel='poly')
reg_svr_kpoly.fit(x4_train,y4_train.ravel())  


# In[833]:


reg_svr_kgauss=SVR(kernel='rbf')  # 
reg_svr_kgauss.fit(x4_train,y4_train.ravel())  


# In[834]:


reg_svr_ksigmoide=SVR(kernel='sigmoid')
reg_svr_ksigmoide.fit(x4_train,y4_train.ravel())  


# #### Prediccion

# RECORDAR QUE APLICAMOS LA ESTANDARIZACION!!! 
# 
# * Para transformar fit.transform 
# * Para volver fit.inverse
# 
# 

# In[836]:


y_svr_klineal=reg_svr_klineal.predict(sc_x.fit_transform( x4_test ) )  # Transformar antes de pasar el valor y debe ser array de array
sc_y.inverse_transform(  [y_svr_klineal]  )                            # hay que hacer la inversa de la transformacion en 'y' para volver al rango de valores original


# In[837]:


y_svr_kpoly=reg_svr_kpoly.predict(sc_x.fit_transform(x4_test))
sc_y.inverse_transform( [y_svr_kpoly] )


# In[838]:


y_svr_kgauss=reg_svr_kgauss.predict(sc_x.fit_transform(x4_test))
sc_y.inverse_transform( [y_svr_kgauss] )  


# In[839]:


y_svr_ksigmoide=reg_svr_ksigmoide.predict(sc_x.fit_transform(x4_test))
sc_y.inverse_transform( [y_svr_ksigmoide] ) 


# #### Evaluacion de los modelos
# 
# 

# In[692]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error 


# En el calculo de los estadisticos no hace falta transformar. Hacemos la prediccion directamente sobre x
# 
# NOTA: En estos ejemplos no estan corregidos los mae y mse. Habria que transformar las variables.
# 
# El R2 da bien aunque no se transforme ni se haga la inversa de la transformacion.

# * Kernel lineal 

# In[ ]:


## tuve que hacer un reshape para que quedara como el y4_test


# In[840]:


r2_klineal=r2_score(y4_test, sc_y.inverse_transform(  [y_svr_klineal]  ).reshape(509,1) ) 
mae_klineal=mean_absolute_error(y4_test, sc_y.inverse_transform(  [y_svr_klineal]  ).reshape(509,1))
mse_klineal=mean_squared_error(y4_test, sc_y.inverse_transform(  [y_svr_klineal]  ).reshape(509,1))
print(f'Estadisticos de la SVR con kernel lineal\n r2: {r2_klineal:.2f} \n mae: {mae_klineal:.2f} \n mse: {mse_klineal:.2f}')


# * Kernel polinomico

# In[841]:


r2_kpoly=r2_score(y4_test, sc_y.inverse_transform(  [y_svr_kpoly]  ).reshape(509,1) ) 
mae_kpoly=mean_absolute_error(y4_test, sc_y.inverse_transform(  [y_svr_kpoly]  ).reshape(509,1)  )
mse_kpoly=mean_squared_error(y4_test, sc_y.inverse_transform(  [y_svr_kpoly]  ).reshape(509,1)   )
print(f'Estadisticos de la SVR con kernel polinomico\n r2: {r2_kpoly:.2f} \n mae: {mae_kpoly:.2f} \n mse: {mse_kpoly:.2f}')


# In[ ]:





# * Kernel Gaussiano o radial (rbf)

# In[842]:


r2_kgauss=r2_score(y4_test, sc_y.inverse_transform(  [y_svr_kgauss]  ).reshape(509,1)  ) 
mae_kgauss=mean_absolute_error(y4_test, sc_y.inverse_transform(  [y_svr_kgauss]  ).reshape(509,1))
mse_kgauss=mean_squared_error(y4_test, sc_y.inverse_transform(  [y_svr_kgauss]  ).reshape(509,1))
print(f'Estadisticos de la SVR con kernel radial\n r2: {r2_kgauss:.2f} \n mae: {mae_kgauss:.2f} \n mse: {mse_kgauss:.2f}')


# ### Analisis de las diferencias entre y_test / y_test_predict 

# In[884]:


import seaborn as sn


# In[935]:


# KDE Plot with seaborn
plt.title('Modelo SVR Radial')
res = sn.kdeplot(y4_test.ravel(), color='red', shade='True')
res2 = sn.kdeplot(sc_y.inverse_transform([y_svr_kgauss]).ravel(), color='blue', shade='False')
plt.show()


# In[936]:


Diferencias_SVR_Radial=np.round(y4_test.ravel()-sc_y.inverse_transform([y_svr_kgauss]).ravel())


# In[938]:


pd.DataFrame(Diferencias_SVR_Radial).plot.kde()


# In[939]:


pd.DataFrame(Diferencias_SVR_Radial).plot.box()


# In[ ]:





# In[ ]:





# * Kernel Sigmoide

# In[843]:


r2_ksigm=r2_score(y4_test, sc_y.inverse_transform(  [y_svr_ksigmoide]  ).reshape(509,1) ) 
mae_ksigm=mean_absolute_error(y4_test, sc_y.inverse_transform(  [y_svr_ksigmoide]  ).reshape(509,1))
mse_ksigm=mean_squared_error(y4_test, sc_y.inverse_transform(  [y_svr_ksigmoide]  ).reshape(509,1))
print(f'Estadisticos de la SVR con kernel Sigmoide\n r2: {r2_ksigm:.2f} \n mae: {mae_ksigm:.2f} \n mse: {mse_ksigm:.2f}')
# Esto da muy mal


# In[844]:


plt.title('linear')
plt.plot(y4_test, color='green')
plt.plot(sc_y.inverse_transform(  [y_svr_klineal]  ).reshape(509,1), color='blue')
plt.show


# In[845]:


plt.title('radial')
plt.plot(y4_test, color='green')
plt.plot(sc_y.inverse_transform(  [y_svr_kgauss]  ).reshape(509,1), color='blue')
plt.show


# ###  ARBOL DE DECISION

# In[277]:


# from sklearn.metrics import r2_score


# In[ ]:


# Sobre las mismas cuatro varibles que mejor correlacionan con precio para no sobrecargar el modelo.


# ### Separar predictoras y objetivo

# In[846]:


x5=df_aux.iloc[:,[1,2,5,6,10,11,12,14,16,19]].values # [[]] para que sea un array de arrays o sea matriz y luego no tenga que hacer el reshape en los modelos
y5=df_aux.iloc[:,[0]].values


# ### Division train-test

# In[269]:


from sklearn.model_selection import train_test_split


# In[847]:


x5_train,x5_test, y5_train, y5_test = train_test_split(x5, y5, test_size = 0.30 , random_state = 1987) 


# In[848]:


len(x5_train),len(x5_test), len(y5_train), len(y5_test)


# #### Normalizacion / estandarizacion

# * En este tipo de modelo si que es necesario normalizar. 
# * Se hace despues de separar training y test para que los datos de training no interfieran en los de test.

# In[661]:


from sklearn.preprocessing import StandardScaler


# * Una transformacion para cada variable (en este caso x,y)

# In[849]:


sc_x=StandardScaler()
x5_train=sc_x.fit_transform(x5_train)
x5_train


# In[850]:


sc_y=StandardScaler()
y5_train=sc_y.fit_transform(y5_train)
y5_train


# In[ ]:





# #### Crear modelo

# In[737]:


from sklearn.tree import DecisionTreeRegressor


# R2 para distintos criterion
# 
#                 'squared_error' .24
#                 'friedman_mse' .25, con min_samples_split=5 .31, min_samples_split=10 .40, max_leaf_nodes=100
#                 'absolute_error' .15

# In[851]:


reg_arbol=DecisionTreeRegressor(criterion='squared_error', min_samples_split=15, max_leaf_nodes=200, random_state=1987)
reg_arbol.fit(x5_train,y5_train)


# #### Prediccion para test

# In[852]:


y_arbol=reg_arbol.predict(sc_x.fit_transform(x5_test))
sc_y.inverse_transform([y_arbol])


# #### Evaluacion del modelo (r2, mae,mse)

# In[37]:


# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[853]:


r2_arbol=r2_score(y5_test, sc_y.inverse_transform([y_arbol]).reshape(509,1) ) 
mae_arbol=mean_absolute_error(y5_test, sc_y.inverse_transform([y_arbol]).reshape(509,1))
mse_arbol=mean_squared_error(y5_test, sc_y.inverse_transform([y_arbol]).reshape(509,1))
print(f'Estadisticos de la regresion por Arbol de Desicion\n r2: {r2_arbol:.2f} \n mae: {mae_arbol:.2f} \n mse: {mse_arbol:.2f}')


# In[854]:


plt.title('Arbol de Decisión')
plt.plot(y5_test, color='green')
plt.plot(sc_y.inverse_transform(  [y_arbol]  ).reshape(509,1), color='blue')
plt.show


# In[ ]:





# ## Bosques Aleatorios

# ### Separar predictoras y objetivo

# In[855]:


x6=df_aux.iloc[:,[1,2,5,6,10,11,12,14,16,19]].values # [[]] para que sea un array de arrays o sea matriz y luego no tenga que hacer el reshape en los modelos
y6=df_aux.iloc[:,[0]].values


# ### Division train-test

# In[269]:


from sklearn.model_selection import train_test_split


# In[856]:


x6_train,x6_test, y6_train, y6_test = train_test_split(x6, y6, test_size = 0.30 , random_state = 1987) 


# In[857]:


len(x6_train),len(x6_test), len(y6_train), len(y6_test)


# #### Normalizacion / estandarizacion

# * En este tipo de modelo si que es necesario normalizar. 
# * Se hace despues de separar training y test para que los datos de training no interfieran en los de test.

# In[661]:


from sklearn.preprocessing import StandardScaler


# * Una transformacion para cada variable (en este caso x,y)

# In[858]:


sc_x=StandardScaler()
x6_train=sc_x.fit_transform(x6_train)
x6_train


# In[859]:


sc_y=StandardScaler()
y6_train=sc_y.fit_transform(y6_train)
y6_train


# #### Crear el modelo Random Forest
# 
# 
# * Parametros:
#     
#         - n_estimators: numero de arboles, por defecto 100. Habitualmente entre 300 y 1000.
#         - criterion: squared_error, absolute_error, friedman_mse, poisson. Por defecto squared_error
#         - max_depth: profundidad
#         - min_samples_split: numero de muestras minimo
#         - min_samples_leaf: numero de muestras maximo
#         - max_features: maximo numero de caracteristicas. Por defecto 'auto'. El modelo probará con una, con dos, etc.

# In[781]:


from sklearn.ensemble import RandomForestRegressor


# * criterion=squared_error
#                         n_estimators=100, r2=0.53 (con mas estimadores no cambia)
# * criterion=absolute_error
#                         n_estimators=100, r2=0.54 (con mas estimadores no cambia)
#                         n_estimators=500, r2= 0.55
# * criterion=friedman_mse
#                         n_estimators=500, r2= 0.53 (con mas estimadores no cambia)

# In[874]:


reg_randomForest=RandomForestRegressor(n_estimators=500, criterion='absolute_error', max_features='auto', random_state=1987)
reg_randomForest.fit(x6_train,y6_train) # ya estan normalizadas


# #### Prediccion para test

# In[875]:


y_randomForest=reg_randomForest.predict(sc_x.fit_transform(x6_test))
sc_y.inverse_transform([y_randomForest])


# #### Evaluacion del modelo (r2, mae,mse)

# In[37]:


# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[876]:


r2_arbol=r2_score(y6_test, sc_y.inverse_transform([y_randomForest]).reshape(509,1) ) 
mae_arbol=mean_absolute_error(y5_test, sc_y.inverse_transform([y_randomForest]).reshape(509,1))
mse_arbol=mean_squared_error(y5_test, sc_y.inverse_transform([y_randomForest]).reshape(509,1))
print(f'Estadisticos de la regresion por Bosques Aleatorios\n r2: {r2_arbol:.2f} \n mae: {mae_arbol:.2f} \n mse: {mse_arbol:.2f}')


# ### Analisis de las diferencias entre y_test / y_test_predict 

# In[884]:


import seaborn as sn


# In[949]:


# KDE Plot with seaborn
plt.title('Modelo Random Forest')d
res = sn.kdeplot(y6_test.ravel(), color='red', shade='True')
res2 = sn.kdeplot(sc_y.inverse_transform([y_randomForest]).reshape(509,1).ravel(), color='blue', shade='False')
plt.show()


# In[950]:


Diferencias_RandomForest=np.round(y6_test.ravel()-sc_y.inverse_transform([y_randomForest]).reshape(509,1).ravel())


# In[951]:


pd.DataFrame(Diferencias_RandomForest).plot.kde()


# In[952]:


pd.DataFrame(Diferencias_RandomForest).plot.box()


# In[ ]:





# In[863]:


plt.title('Bosques Aleatorios')
plt.plot(y4_test, color='green')
plt.plot(sc_y.inverse_transform(  [y_randomForest]  ).reshape(509,1), color='blue')
plt.show


# In[ ]:





# In[ ]:





# In[ ]:




