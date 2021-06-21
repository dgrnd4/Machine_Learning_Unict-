from data_from_xml import get_data_from_xml,selection_sort
from data_from_db import get_from_db
import numpy as np
from matplotlib import pyplot as plt
import datetime as dt
from datetime import datetime
import math
from matplotlib import rcParams
from pandas import read_csv
from matplotlib import pyplot
from workalendar.europe import Italy
import torch
from torch import nn
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from torch.autograd import Variable
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.ar_model import AutoReg
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

def create_date_featured(x):
    s=0
    a=x[s][0:4]
    multi=0
    uno_da_aggiungere=0

    #####
    cal = Italy()

    while s<len(x):
        anno=[x[s][0:4]]
        if str(a)!=str(anno[0]):
            #print("a:",a , "-anno:", anno[0]) #Cambio dell'anno 
            if multi==0:
                uno_da_aggiungere=uno_da_aggiungere+1
            multi=multi+1
            
        mese=[x[s][5:7]]
        giorno=[x[s][8:]]
        
        ####
        whole_data = datetime(int(anno[0]), int(mese[0]), int(giorno[0]))
        is_working_day=cal.is_working_day(whole_data)
        is_working_day = int(is_working_day)
        

        day_of_the_week = datetime(int(anno[0]), int(mese[0]),int(giorno[0]))
        n_th_day = (dt.date(int(anno[0]), int(mese[0]),int(giorno[0]) ) - dt.date(int(anno[0]),1,1)).days + 1
        a=x[s][0:4]
        x[s]=[int(giorno[0]), int(mese[0]), int(anno[0]), 
            int(day_of_the_week.strftime("%w")), int(n_th_day), #int(n_th_day)+(365+uno_da_aggiungere)*multi ,
            is_working_day ]
        s=s+1
        # array = [0:giorno, 1:mese, 2:anno, 
        #          3:0-6 day,4:n_esimo_anno, 5:n_esimo_multi
        #          6:is_workin_day ]

def visualize_single_array(arr1):
    
    plt.figure(figsize=(12,8))
    plt.plot(arr1)

    scatter=False
    if scatter:
        x=[]
        for i in range(len(arr1)):
            x.append(i)
        plt.scatter(x,arr1) 

    plt.grid()
    plt.legend(['test_value'])
    plt.show()

def visualize_graph(reg,arr1,arr2,iter):
    plt.figure(figsize=(12,8))
    plt.plot(arr1)
    plt.plot(arr2)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['real_value','predicted_value'])
    plt.savefig(reg+str(iter)+".png")
    #plt.show()
    plt.close()

def visualize_graph(reg,arr1,arr2):
    plt.figure(figsize=(12,8))
    plt.plot(arr1)
    plt.plot(arr2)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['real_value','predicted_value'])
    plt.savefig(reg+".png")
    #plt.show()
    plt.close()



class LinearRegression(nn.Module):
    def __init__(self,in_size, out_size): #input size, output size
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        result = self.linear(x)
        return result

class Model(nn.Module):
    def __init__(self,in_size, out_size):
        super(Model, self).__init__()
        hidden_layer=20
        self.layer1 = torch.nn.Linear(in_size, hidden_layer)
        self.activation = nn.LeakyReLU()

        self.layer2 = torch.nn.Linear(hidden_layer, out_size)
        

    def forward(self, X):
        
        X=self.layer1(X)
        X=self.activation(X)
        X=self.layer2(X)
        return X
        ##return self.layer2(nn.LeakyReLU()(self.layer1(x)))
    
class MLP(nn.Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(n_inputs, 10)
        nn.init.xavier_uniform_(self.hidden1.weight)
        self.act1=nn.LeakyReLU()
        
        self.hidden2 = nn.Linear(10, 8)
        nn.init.xavier_uniform_(self.hidden2.weight)
        self.act2=nn.LeakyReLU()

        self.hidden3 = nn.Linear(8, 1)
        nn.init.xavier_uniform_(self.hidden3.weight)
        self.dropout = nn.Dropout(p=0.5,inplace=True)
 
    def forward(self, X):
        
        X = self.hidden1(X)
        X = self.act1(X)
        ##X = self.dropout(X) 
        X = self.hidden2(X)
        X = self.act2(X)
        ##X = self.dropout(X)
        X = self.hidden3(X)
              
        return X

def RMSE(predictions, gt):
    assert predictions.shape == gt.shape
    return ((predictions-gt)**2).mean()**(1/2)




def train_and_creation_model(iteration_number=1,reg="mlp", save=True ):
    reg1=reg
    #x_orig,y_orig, = get_data_from_xml() # len=337
    x_orig,y_orig=get_from_db() # len = 2613
    new_score=-10
    reg1=reg
    create_date_featured(x_orig) #[1, 1, 2018, 1, 1, 0]
    #print(x_orig[0])

    #15% del totale del dataset
    division=int((15/100) *len(x_orig)) +1
    division=len(x_orig)-division
    x_tr = torch.Tensor(x_orig[:division]) 
    y_tr = torch.Tensor(y_orig[:division])
    x_te = torch.Tensor(x_orig[division:])
    y_te = torch.Tensor(y_orig[division:])
    x_tot = torch.Tensor(x_orig)
    y_tot = torch.Tensor(y_orig)

    #print("x_orig:" +  str(len(x_orig)) + " y_orig:" +str(len(y_orig)) + "x_tr:" +  str(len(x_tr)) + " y_tr:" +str(len(y_tr)) + " x_te:" +str(len(x_te)) + " y_te:" +str(len(y_te)) + " " )

    i=0
    while i<iteration_number:
        lr = 0.01
        epochs = 10000
        means = x_tr.mean(0)
        stds = x_tr.std(0)
        X_training_norm = (x_tr-means)/stds
        X_testing_norm = (x_te-means)/stds
        x_tot_norm = (x_tot-means)/stds

        if reg=="linearRegression1":
            reg = LinearRegression(6,1)
            print("linear model")

        if reg=="model":
            reg = Model(6,1)
            print("model")

        if reg=="mlp":
            reg = MLP(6)
            print("mlp")


        optimizer = torch.optim.SGD(reg.parameters(),lr=lr,weight_decay = 0.0001, momentum=0.95) 
        
        losses_train = []
        losses_test = []
        output_te = []
        output_tr = []

        for e in range(epochs):
            reg.train()
            
            output_tr = reg(X_training_norm)

            
            l= RMSE(output_tr.view(-1), y_tr) 
            losses_train.append(l.item())

            l.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            reg.eval() 

            with torch.set_grad_enabled(False):
                output_te = reg(X_testing_norm)
                l= RMSE(output_te.view(-1), y_te)
                losses_test.append(l.item())


        ris_tr = reg(X_training_norm)
        ris_te = reg(X_testing_norm)
        ris_tot = reg(x_tot_norm)
        z_te = RMSE(ris_te.view(-1),y_te)
        z_tot = RMSE(ris_tot.view(-1),y_tot)
        y_te_predicted=[]
        y_tot_predicted=[]

        s=0
        while s<len(ris_te):
            y_te_predicted.append(int(ris_te[s][0].item()))
            #print( str(int(y_te[s].item())) +" - "+ str(int(ris_te[s][0].item()) )) #real value - predicted value
            s=s+1

        s=0
        while s<len(ris_tot):
            y_tot_predicted.append(int(ris_tot[s][0].item()))
            #print( str(int(y_tot[s].item())) +" - "+ str(int(ris_tot[s][0].item()) )) #real value - predicted value
            s=s+1

        y_te = y_te.type(torch.IntTensor)

        score=r2_score(y_te,y_te_predicted)

        if new_score<score:

            new_score=score
            mae=mean_absolute_error(y_te,y_te_predicted)
            rmse = mean_squared_error(y_te_predicted, y_te)
            mse = mean_squared_error(y_te,y_te_predicted, squared=False)

            PATH = "/Users/diegorando/diego/uni/ML/laboratorio/"+reg1+"_"+str(i)+".pth"
            torch.save(reg.state_dict(), PATH)
            print("esecution numb "+ str(i) + ": r2_score: " + str(round(score, 2))+ " - mae: " + str(round(mae,2)) + " - rmse: "+ str(round(rmse,2)) + " - mse: "+ str(round(mse,2)) )

            if save:
                visualize_graph(reg1,y_te,y_te_predicted,iter=i)
        
        i=i+1

def train_net_reglin(iteration_number=1, save=True ):
    x_orig,y_orig=get_from_db() # len = 2613
    new_score=-10

    create_date_featured(x_orig) #[1, 1, 2018, 1, 1, 0]

    division=int((15/100) *len(x_orig)) +1
    division=len(x_orig)-division

    x_tr = torch.Tensor(x_orig[:division]) 
    y_tr = torch.Tensor(y_orig[:division])
    x_te = torch.Tensor(x_orig[division:])
    y_te = torch.Tensor(y_orig[division:])

    i=0
    
    while i<iteration_number:
        lr = 0.01
        epochs = 2000
        means = x_tr.mean(0)
        stds = x_tr.std(0)
        X_training_norm = (x_tr-means)/stds
        X_testing_norm = (x_te-means)/stds

        reg = torch.nn.Sequential(
            torch.nn.Linear(6, 12),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(12, 12),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(12, 1),
        )
        
        optimizer = torch.optim.Adam(reg.parameters(),lr=lr)##,weight_decay = 0.0001, momentum=0.95) #Stocastic Gradient descent
        losses_train = []
        losses_test = []
        output_te = []
        output_tr = []

        loss_func = torch.nn.MSELoss()

        for t in range(epochs):
        
            output_tr = reg(X_training_norm)

            loss = loss_func(output_tr.view(-1), y_tr)  

            optimizer.zero_grad()   
            loss.backward()         
            optimizer.step()

            with torch.set_grad_enabled(False):
                output_te = reg(X_testing_norm)
                l= RMSE(output_te.view(-1), y_te) 
                losses_test.append(l.item())
                #y_t = y_t.type(torch.IntTensor) #per fare il cast a int


        
        ris_te = reg(X_testing_norm)

        y_te_predicted=[]

        s=0
        while s<len(ris_te):
            y_te_predicted.append(int(ris_te[s][0].item()))
            #print( str(int(y_te[s].item())) +" - "+ str(int(ris_te[s][0].item()) )) #real value - predicted value
            s=s+1

        y_te_predicted=np.abs(y_te_predicted)

        y_te = y_te.type(torch.IntTensor)

        score=r2_score(y_te,y_te_predicted)

        if new_score<score:

            new_score=score
            mae=mean_absolute_error(y_te,y_te_predicted)
            rmse = mean_squared_error(y_te_predicted, y_te)
            mse = mean_squared_error(y_te,y_te_predicted, squared=False)

            PATH = "/Users/diegorando/diego/uni/ML/laboratorio/linearRegression2_"+str(i)+".pth"
            torch.save(reg.state_dict(), PATH)
            print("esecution numb "+ str(i) + ": r2_score: " + str(round(score, 2))+ " - mae: " + str(round(mae,2)) + " - rmse: "+ str(round(rmse,2)) + " - mse: "+ str(round(mse,2)) )

            if save:
                visualize_graph(y_te,y_te_predicted,iter=i)
        
        i=i+1

def training_sklearn(model, show=False, show_numbers=False,save=False):
    model1=model
    x_orig,y_orig=get_from_db()
    create_date_featured(x_orig) 

    division=int((15/100) *len(x_orig)) +1
    division=len(x_orig)-division

    x_tr = torch.Tensor(x_orig[:division]) 
    x_tr = np.array(x_tr)
    y_tr = torch.Tensor(y_orig[:division])
    y_tr = np.array(y_tr)
    x_te = torch.Tensor(x_orig[division:])
    x_te = np.array(x_te)
    y_te = torch.Tensor(y_orig[division:])
    y_te = np.array(y_te)


    means = x_tr.mean(0)
    stds = x_tr.std(0)
    X_training_norm = (x_tr-means)/stds
    X_testing_norm = (x_te-means)/stds

    if model1=="rbf":
        model = SVR(kernel='rbf', C=20, epsilon=0.0001,gamma='auto')

    if model1=="poly":
        model = SVR(kernel='poly', C=20, gamma='auto', degree=3, epsilon=0.01, coef0=1) 

    if model1=="decisionTree":
        model = DecisionTreeRegressor()  

    if model1=="kNeighbors":
        model = KNeighborsRegressor() 

    if model1=="ridge":
        model = Ridge()


    param_grid = [
        {'C': [1,5,10,15,20,25,30,35,40,45,50,55,60], 'epsilon': [0.1,0.001,0.0001], 'gamma': [0.01, 0.05, 0.001,'scale','auto'], 'kernel': ['rbf']},
        {'C': [1,5,10,15,20,25,30,35,40,45,50,55,60], 'epsilon': [0.1,0.001,0.0001], 'gamma': [0.01, 0.05, 0.001,'scale','auto'], 'kernel': ['poly']} 
    ]

    #gs = GridSearchCV(SVR(), param_grid ,verbose=1, n_jobs=2)
    #gs.fit(X_training_norm,y_tr)
    #res=gs.best_estimator_
    #print(res)

    model.fit(x_tr, y_tr.ravel())
    ypred = model.predict(x_te) 
    
    #ypred = sc_y.inverse_transform(ypred)
    #y_te = sc_y.inverse_transform(y_te)

    score=r2_score(y_te,ypred)
    new_score=score
    mae=mean_absolute_error(y_te,ypred)
    rmse = mean_squared_error(ypred, y_te)
    mse = mean_squared_error(y_te,ypred, squared=False)
    
    print(": r2_score: " + str(round(score, 2))+ " - mae: " + str(round(mae,2)) + " - rmse: "+ str(round(rmse,2)) + " - mse: "+ str(round(mse,2)) )

    if save:
        PATH = "/Users/diegorando/diego/uni/ML/laboratorio/"+model1+".pth"
        pickle.dump(model, open(PATH, 'wb'))

    if show_numbers==True:
        s=0
        while s<len(ypred):
            #print("real value: ", y_te[s], " - pred value: ", math.ceil(ypred[s]) )
            print("real value: ", y_te[s], " - pred value: ", round(ypred[s]))
            s=s+1

    if show:
        visualize_graph(model1,y_te,ypred)
   
def load_model(reg='mlp',show_numbers=False):
    reg1=reg
    #x_orig,y_orig, = get_data_from_xml() #x = date, y = valori, len=337
    x_orig,y_orig=get_from_db() # len = 2613

    create_date_featured(x_orig) #[1, 1, 2018, 0, 1, 0]
    #print(x_orig[:2])



    #15% del totale del dataset
    division=int((15/100) *len(x_orig)) +1
    division=len(x_orig)-division


    x_tr = torch.Tensor(x_orig[:division]) 
    y_tr = torch.Tensor(y_orig[:division])
    x_te = torch.Tensor(x_orig[division:])
    y_te = torch.Tensor(y_orig[division:])
    x_tot = torch.Tensor(x_orig)
    y_tot = torch.Tensor(y_orig)


    #print("x_orig:" +  str(len(x_orig)) + " y_orig:" +str(len(y_orig)) + "x_tr:" +  str(len(x_tr)) + " y_tr:" +str(len(y_tr)) + " x_te:" +str(len(x_te)) + " y_te:" +str(len(y_te)) + " " )


    lr = 0.01
    epochs = 10000
    means = x_tr.mean(0)
    stds = x_tr.std(0)+0.00001
    X_training_norm = (x_tr-means)/stds
    X_testing_norm = (x_te-means)/stds
    x_tot_norm = (x_tot-means)/stds


    if reg=="linearRegression1":
        reg = LinearRegression(6,1)
    if reg=="model":
        reg = Model(6,1)
    if reg=="mlp":
        reg = MLP(6)
    if reg=="linearRegression2":
        reg = torch.nn.Sequential(
                    torch.nn.Linear(6, 12),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(12, 12),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(12, 1),
                )

    MSE = nn.MSELoss()
    optimizer = torch.optim.SGD(reg.parameters(),lr=lr,weight_decay = 0.0001, momentum=0.90) #Stocastic Gradient descent


            
    PATH = "/Users/diegorando/diego/uni/ML/laboratorio/" + reg1 +".pth"

    reg.load_state_dict(torch.load(PATH))

    ris_tr = reg(X_training_norm)
    ris_te = reg(X_testing_norm)
    ris_tot = reg(x_tot_norm)
    z_te = RMSE(ris_te.view(-1),y_te)
    z_tot = RMSE(ris_tot.view(-1),y_tot)

    y_te_predicted=[]

    s=0
    while s<len(ris_te):
        y_te_predicted.append(int(ris_te[s][0].item()))
        #print( str(int(y_te[s].item())) +" - "+ str(int(ris_te[s][0].item()) )) #real value - predicted value
        s=s+1

    s=0
    if show_numbers:
        while s<len(y_te):
            print( str(int(y_te[s].item())) +" - "+ str(int(y_te_predicted[s])) ) #real value - predicted value
            s=s+1

    y_te = y_te.type(torch.IntTensor)

    score=r2_score(y_te,y_te_predicted)
    mae=mean_absolute_error(y_te,y_te_predicted)
    rmse = mean_squared_error(y_te_predicted, y_te)
    mse = mean_squared_error(y_te,y_te_predicted, squared=False)

    print("r2_score: " + str(round(score, 2))+ " - mae: " + str(round(mae,2)) + " - rmse: "+ str(round(rmse,2)) + " - mse: "+ str(round(mse,2)) )

    visualize_graph(reg1,y_te,y_te_predicted)

def load_model_sklearn(reg="ridge"):

    reg1=reg
    x_orig,y_orig=get_from_db()
    create_date_featured(x_orig)

    division=int((15/100) *len(x_orig)) +1
    division=len(x_orig)-division
    x_tr = torch.Tensor(x_orig[:division]) 
    x_tr = np.array(x_tr)
    y_tr = torch.Tensor(y_orig[:division])
    y_tr = np.array(y_tr)
    x_te = torch.Tensor(x_orig[division:])
    x_te = np.array(x_te)
    y_te = torch.Tensor(y_orig[division:])
    y_te = np.array(y_te)

    means = x_tr.mean(0)
    stds = x_tr.std(0)
    X_training_norm = (x_tr-means)/stds
    X_testing_norm = (x_te-means)/stds

    PATH = "/Users/diegorando/diego/uni/ML/laboratorio/"+reg1+".pth"

    
    model = pickle.load(open(PATH, 'rb'))
    result = model.score(X_testing_norm, y_te)

    y_te_predicted = model.predict(X_testing_norm)
    #print(y_te_predicted)
    
 
    visualize_graph(reg1,y_te,y_te_predicted)
    #print("R2_Score"+ str(result))



def main():

    # TRAINING model

    # la variabile reg puo prendere i seguenti valori = linearRegression1, mlp
    #train_and_creation_model(iteration_number=1,reg="mlp",save=True) 
    
    # tramite il seguente modello otteniamo il risultato relativo al linearRegression2
    #train_net_reglin(iteration_number=100,save=True) 

    # la variabile model puo prendere i seguenti valori = rbf, poly, decisionTree, kNeighbors, ridge
    #training_sklearn(model="poly",show=True,show_numbers=False,save=True) 


    # LOADING model

    # la variabile reg puo prendere i seguenti valori = linearRegression1, linearRegression2, mlp, poly, rbf
    #load_model(reg= "mlp", show_numbers=False)

    # la variabile reg puo prendere i seguenti valori = decisionTree, KNeighbors, Ridge, poly, rbf
    load_model_sklearn(reg="Ridge")

    

if __name__ == "__main__":
    main()




