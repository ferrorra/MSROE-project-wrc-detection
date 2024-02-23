#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load

# ploting option
plt.rcParams['figure.figsize'] = [12.94, 8]
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
#plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 10

SAVE_FIGS = True


dataset = pd.read_csv('data_clean.csv')

def model_theta_MLP(dataset, hidden, print_coefs = True, max_iter= 1000000):
    # à completer (penser à enlever 'code' des données de départ
    X_train, X_test, Y_train, Y_test = train_test_split(

    # à compléter (commencer par les options par défaut)
    model = MLPRegressor(
        
    coefs = model.coefs_
    Yhat = model.predict(X_train)
    Yhat_test = model.predict(X_test)
    loss = np.square(Y_test - Yhat_test).mean()
    hiddens = coefs[0].T
    final_mlp = coefs[1].flatten()
    
    coefs = list(zip([dict(zip(X_train.columns, h)) for h in hiddens],
                     [['output mult:', m] for m in  final_mlp.flatten()], 
                     [['intercept:', i] for i in  model.intercepts_[0]]))
    print('loss:', loss)
    if print_coefs:
        for idx, c in enumerate(coefs):
            f1, o, i = c
            print('feature', idx, '=',
                      f1['P2'].round(2), '* P2 +', 
                      f1['P50'].round(2), '* P50 +',
                      f1['P100'].round(2), '* P100 +',
                      f1['P250'].round(2), '* P250 +',
                      f1['P500'].round(2), '* P500 +',
                      f1['P1000'].round(2), '* P1000 +',
                      f1['P2000'].round(2), '* P2000 +',
                      f1['rho'].round(2), '* rho +',
                      f1['suction'].round(2), '* suction +',
                      i[1].round(2))
        output = 'Yhat = '
        for fidx, v in enumerate(final_mlp):
            output = output + str(v.round(2)) + ' * feat ' + str(fidx) + ' + '
        output = output + str(model.intercepts_[1][0].round(2))
        print(output)
    return model, X_train, X_test, Y_train, Y_test, Yhat, Yhat_test, coefs, loss
# je conseille d'ajouter des sorties pour tracer les graphes demandés    

# je vous laisse le code pour la figure de comparaison (cadeau)
def graph_real_and_predicted(X_train, X_test, Y_train, Y_test, Yhat, Yhat_test, fname=None):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(Y_train, Yhat,'r.')
    ax.plot(Y_train, Y_train,'k-')
    ax.set_xlabel('Training values')
    ax.set_ylabel('Predicted values')
    ax = fig.add_subplot(122)
    ax.plot(Y_test, Yhat_test,'b.')
    ax.plot(Y_test, Y_test,'k-')
    ax.set_xlabel('Testing values')
    ax.set_ylabel('Predicted values')
    if fname is not None:
        plt.savefig('images/' + fname + '.pdf')
    plt.close()


#exemple d'un appel, avec une couche cachée à 10 neurones
model, X_train, X_test, Y_train, Y_test, Yhat, Yhat_test, coefs, loss = model_theta_MLP(dataset, [10])
#score du modèle
print(model.score(X_test,Y_test))
#figures
graph_real_and_predicted(X_train, X_test, Y_train, Y_test, Yhat, Yhat_test, fname='result_1')

#prédiction à partir du modèle sachant:
print('testing the model:')
print(model.predict([[0,0.1,0.2,0.3,0.5,0.6,1,1.7,0.1]]))

#storing the model on disk
dump(model, 'model_2.joblib') 

#test (reload model + prediction):
model = load('model_2.joblib') 
print('testing the model reloaded:')
print(model.predict([[0,0.1,0.2,0.3,0.5,0.6,1,1.7,100]]))

#ajouter:
#- gridsearch + cross validation (cf. fonction GridSearchCV)
# - penser à réentrainer le modèle après GridSearchCV et l'utiliser
