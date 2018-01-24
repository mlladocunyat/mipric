
# coding: utf-8

# #Intentamos analizar de forma previa los datos

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk 
from  sklearn.feature_extraction import  DictVectorizer
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import csr_matrix, hstack,vstack
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import NMF
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


# In[2]:


#Global variables
NUM_TRAIN=.5
MAX_COMPONENTS=2
NUM_FRASES=500
train=pd.DataFrame()
test=pd.DataFrame()
SPARSE=True
NJOBS=2
LB_name= LabelBinarizer( sparse_output=SPARSE)
LB_cat1= LabelBinarizer( sparse_output=SPARSE)
LB_cat2= LabelBinarizer( sparse_output=SPARSE)
LB_cat3= LabelBinarizer( sparse_output=SPARSE)
LB_shipping= LabelBinarizer( sparse_output=SPARSE)
LB_item_condition_id= LabelBinarizer( sparse_output=SPARSE)
T_MODEL='MLPR'
if T_MODEL=='LinearReg':
    MODEL = LinearRegression(fit_intercept=True, normalize=False, copy_X=False, n_jobs=NJOBS)
if T_MODEL=='RBF':    
    MODEL = SVR(kernel='rbf', C=1e3, gamma=0.1)
if T_MODEL=='Ridge':   
    MODEL = Ridge(alpha=100)
if T_MODEL=='MLPR':
    MODEL = MLPRegressor(verbose=True,hidden_layer_sizes=(50,10,5 ),early_stopping=True)
SKALER = MaxAbsScaler()
#REDUCT  =  TruncatedSVD(n_components=MAX_COMPONENTS,algorithm='randomized', n_iter=5, random_state=0, tol=0.0)
REDUCT = NMF(n_components=MAX_COMPONENTS, init='random', random_state=0)
if SPARSE:
    X_train = csr_matrix((3, 4))
Y_train = np.asarray((5))
ID_CAMPO='train_id'
frases=pd.DataFrame()


# In[3]:


#train read
def read_files(icase):
    global train,test
    if icase == 'train':
        train=pd.read_table('../input/train.tsv')
        print("se lee "+'../input/train.tsv'+' filas:',len(train))
    else:
        train=pd.read_table('../input/test.tsv')
        print("se lee "+'../input/test.tsv')
    return train


# In[4]:


def carga_datos(icase):
    global train,NUM_TRAIN,ID_CAMPO
    if icase=='train':
        if NUM_TRAIN != 0:
            train=read_files('train')
            print("LEN train",len(train))
            if NUM_TRAIN > 1:
                    train=train.sample(n=NUM_TRAIN) 
            else:
                np.random.seed(0)
                msk = np.random.rand(len(train)) < NUM_TRAIN
                if icase == 'train':
                    train = train[msk]
                else:
                    train = train[~msk]
            ID_CAMPO="train_id"
            train.reset_index(drop=True,inplace=True)
        else:
            train=read_files(icase)
            ID_CAMPO="test_id"
        train.reindex()    
    else:
            train=read_files(icase)
            ID_CAMPO="test_id"      
    print("LEN train",len(train))
    return


# In[5]:


def fill_missing_data(data):
    data.category_name.fillna(value = "Other/Other/Other", inplace = True)
    data.brand_name.fillna(value = "Unknown", inplace = True)
    data.item_description.fillna(value = "No description yet", inplace = True)
    return data


# In[6]:


def vectorize_sentence_fit():
    global frases,train,test,NUM_TRAIN,NUM_FRASES
    print("vectorize_sentence_fit:Start vectorize_sentence_fit")
    print("vectorize_sentence_fit:LEN train",len(train))
    test=pd.read_table('../input/test.tsv')
    if NUM_TRAIN != 0:
        if NUM_TRAIN > 1:
            test=test.head(NUM_TRAIN)
        else:
            test=test.sample(frac=NUM_TRAIN)
    print("vectorize_sentence_fit:Read:test",len(test))
    test['item_description_upper']= test['item_description'].str.upper()
    txt=train['item_description_upper'].str.cat(sep='.')+test['item_description_upper'].str.cat(sep='.') 
    del test
    print("vectorize_sentence_fit:Star sent_tokenize")
    sent=nltk.sent_tokenize(txt)
    print("vectorize_sentence_fit:Star FreqDist")
    fdist_sent = nltk.FreqDist(sent)
    frases=pd.DataFrame(fdist_sent.most_common(len(fdist_sent)),columns=['Word', 'Frequency'])
    frases['len']=frases['Word'].str.len()
    frases=frases[frases['len']>2] 
    frases=frases[frases['Frequency']>5]
    frases=frases.head(NUM_FRASES)
    frases.reset_index(drop=True,inplace=True)
    print("End vectorize_sentence_fit",len(frases))
    return


# In[7]:


def preproceso():
    global train
    global Y_train,ID_CAMPO
#split category
    train=fill_missing_data(train)
    train[["cat_level_1","cat_level_2","cat_level_3"]]=train["category_name"].str.split('/', expand=True,n=2)
    del train["category_name"]
#Calcule Log price    
    if ID_CAMPO=="train_id":
        train["target"] = np.log1p(train.price)
        Y_train = np.asarray(train['target'],dtype=np.float)  
    else:
        Y_train = np.zeros((train.shape[0],), dtype=float)
    train['item_description_upper']= train['item_description'].str.upper()    
    return


# In[8]:


def vectorizacion_fit():
    global LV_name,LB_cat1,LB_cat2,LB_cat3,LB_shipping,LB_item_condition_id
    global train
    global X_train,NUM_FRASES
    print("vectorizacion_fit:Name")
    LB_name.fit(train['name'])
    print("vectorizacion_fit:cat_level_1")
    LB_cat1.fit(train['cat_level_1'].astype(str))
    print("vectorizacion_fit:cat_level_2")
    LB_cat2.fit(train['cat_level_2'].astype(str))
    print("vectorizacion_fit:cat_level_3")
    LB_cat3.fit(train['cat_level_3'].astype(str))
    print("vectorizacion_fit:shipping")
    LB_shipping.fit(train['shipping'])  
    print("vectorizacion_fit:item_condition_id")
    LB_item_condition_id.fit(train['item_condition_id'])
    if NUM_FRASES!=0:
        print("vectorizacion_fit:vectorize_sentence_fit")
        vectorize_sentence_fit()
    return 


# In[9]:


def normalization_fit():
    global SKALER
    SKALER.fit(X_train)


# In[10]:


def normalization_transform():
    global SKALER
    SKALER.transform(X_train)


# In[11]:


def reduction_fit():
    global REDUCT
    global X_train
    X=REDUCT.fit_transform(X_train)
    X_train=X


# In[12]:


def reduction_transform():
    global REDUCT
    global X_train
    X_train=REDUCT.transform(X_train)


# In[13]:


def add_feature(X,Y):
    global SPARSE
    print("Antes de add",X.shape,Y.shape)
    try:
        if SPARSE:
            X1 =  hstack([X,Y])
        else:
            X1 =  np.hstack((X,X_train))
    except:
        print("Error en add_feature")
        return X
    print("Despues de add",X.shape,Y.shape,X1.shape)    
    return X1    


# In[14]:


def vectorize_sentence_transform():
    global train,frases
    longi=len(train)
    X_train_sent=csr_matrix((longi,0), dtype=int)
    print("vectorize_sentence_transform:Star map con frases=",len(frases),len(train))
    train["tokenized_sents"] = train["item_description_upper"].fillna("").map(nltk.sent_tokenize)
    def index_va(sentence):
        try:
            return train[train['tokenized_sents'].apply(lambda x: sentence in x)].index.values.tolist()
        except:
            []
        return
    print("vectorize_sentence_transform:Star index_va")
    frases['index_va']=frases["Word"].apply(index_va)     
    for index, row in frases.iterrows():
        try:
            if len(row['index_va'])!=0:
                unos=[1 for i  in range(len(row['index_va']))]
                filas=row['index_va']
                columnas=[0 for i  in range(len(row['index_va']))]
                Y=csr_matrix((unos,(filas,columnas)),shape=(longi,1))
            else:
                Y=csr_matrix(([0],([1],[1])),shape=(longi,1))            
            X_train_sent=hstack((X_train_sent,Y))
        except:
            print("Error en vectorize_sentence_transform",X_train_sent.shape,Y.shape) 
            raise 
    return X_train_sent


# In[15]:


def vectorizacion_transform():
    global LV_name,LB_cat1,LB_cat2,LB_cat3,LB_shipping,LB_item_condition_id
    global train
    global X_train,Y_train
    global SPARSE
    print("vectorizacion_transform:Name")
    X =  LB_name.transform(train['name'])
    X_train= X
    print("vectorizacion_transform:cat_level_1")
    X =  LB_cat1.transform(train['cat_level_1'].astype(str))
    X_train=add_feature(X,X_train)
    print("vectorizacion_transform:cat_level_2")
    X =  LB_cat2.transform(train['cat_level_2'].astype(str))
    X_train=add_feature(X,X_train)
    print("vectorizacion_transform:cat_level_3")
    X =  LB_cat3.transform(train['cat_level_3'].astype(str))
    X_train=add_feature(X,X_train)
    print("vectorizacion_transform:shipping")
    X =  LB_shipping.transform(train['shipping'])
    X_train=add_feature(X,X_train) 
    print("vectorizacion_transform:item_condition_id")
    X =  LB_item_condition_id.transform(train['item_condition_id'])
    X_train=add_feature(X,X_train)
    if NUM_FRASES!=0:
        print("vectorizacion_transform:vectorize_sentence_transform")
        X= vectorize_sentence_transform()
        X_train=add_feature(X_train,X)
    del train
    return 


# In[16]:


def vectorizacion():
    vectorizacion_fit()
    vectorizacion_transform()
    normalization_fit()
    normalization_transform()
    return 


# In[17]:


def vectorizacion_transform_norm():
    vectorizacion_transform()
    normalization_transform()
    return 


# In[18]:


def ajuste():
    global MODEL
    MODEL.fit(X_train,Y_train)
    return


# In[19]:


def mpredict(X_train):
    global MODEL
    return MODEL.predict(X_train).clip(min=0)


# In[20]:


def save_result(preds):
    global train,ID_CAMPO
    carga_datos('test')
    train["price"] = np.expm1(preds)
    train[[ID_CAMPO, "price"]].to_csv("submission_ridge.csv", index = False)
    return


# In[21]:


print("Cargardo Datos")
carga_datos('train')
print("Preproseco")
preproceso()
print("Vectorización")
vectorizacion()
print("Fin Vectorización")


# In[ ]:


print("X_train.shape,Y_train.shape",X_train.shape,Y_train.shape)


# In[ ]:


print("Ajuste")
ajuste()


# In[ ]:


try:
    print("Error Fit:",mean_squared_log_error(Y_train,mpredict(X_train)))
except:
    print("NO LOG")


# In[ ]:


print("Carga datos test")
carga_datos('test')


# In[ ]:


print("Preproceso test")
preproceso()


# In[ ]:


print("Vectorización test")
vectorizacion_transform_norm()


# In[ ]:


print("X_train.shape,Y_train.shape",X_train.shape,Y_train.shape)


# In[ ]:


print("Prediccion test")
mpredict(X_train).shape


# In[ ]:


try:
    print("Error zeros:",mean_squared_log_error(Y_train,mpredict(X_train)))
except:
    print("NO LOG TEST")    


# In[ ]:


print("Save test")
save_result(mpredict(X_train))

