import random
import numpy as np
import time


def sigmoid(x):
    return 1/(1 + np.exp(-x))

class dato:
    def __init__(self, ent1, ent2, salida):
        self.ent1 = ent1
        self.ent2 = ent2
        self.salida = salida  
    
class Capa:
    def __init__(self, cantNeuronas, cantPesosPorNeurona):
        self.neuronas = []
        self.bias = []
        for i in range (cantNeuronas):
            p = []
            p.clear()
            for j in range (cantPesosPorNeurona):
                p.append(random.uniform(-1, 1))
            self.bias.append(random.uniform(-1, 1))
            self.neuronas.append(p.copy())
                

class Neurona:
    def __init__(self, cantPesos):
        self.bias = random.random()
        self.pesos = []
        for i in range(cantPesos):
            self.pesos.append(random.uniform(-1, 1))
        
class Red:
    def __init__(self, neuronasPorCapa, cantEntradas):
        self.capas = []
        for i in range(len(neuronasPorCapa)):
            if i == 0:
                self.capas.append(Capa(neuronasPorCapa[i], cantEntradas))
            else:
                self.capas.append(Capa(neuronasPorCapa[i], len(self.capas[i-1].neuronas)))
    
    
    def evaluar(self, entradas):
        red = []
        salidas = np.array(entradas)
        for capa in self.capas:
            v = np.dot(capa.neuronas, salidas) + capa.bias
            salidas = sigmoid(v) #1 / (1 + np.exp(-v))
            red.append(salidas.copy())
        return red
    
    
    
    def obtenerDeltas(self, y_s, error, salidas):
        deltas = [0]*len(self.capas)
        for i, capa in reversed(list(enumerate(self.capas))):
            deltasPorCapa = []
            if i == len(self.capas)-1:
                for k in  range(len(capa.neuronas)):
                    deltasPorCapa.append(y_s * error)
                deltas[i] = deltasPorCapa.copy()     
            else:
                deltasPorCapa.clear()
                for k in range (len(capa.neuronas)):
                    sumatoria = 0
                    for j in range(len(self.capas[i+1].neuronas)):
                        sumatoria += self.capas[i+1].neuronas[j][k] * deltas[i+1][j]
                    delta = salidas[i][k] * sumatoria #* delta_s
                    deltasPorCapa.append(delta)
                deltas[i] = deltasPorCapa.copy()
        return deltas
    
    
    def actualizarPesos(self, eta, deltas, data, salidas):
        for i in range(len(self.capas)):
            if i == 0:
                for j in range(len(self.capas[i].neuronas)):
                    self.capas[i].bias[j] = self.capas[i].bias[j] + eta * deltas[i][j]
                    self.capas[i].neuronas[j][0] = self.capas[i].neuronas[j][0] + eta * deltas[i][j] * data.ent1
                    self.capas[i].neuronas[j][1] = self.capas[i].neuronas[j][1] + eta * deltas[i][j] * data.ent2     
            else:
                for j in range(len(self.capas[i].neuronas)):
                    self.capas[i].bias[j] = self.capas[i].bias[j] + eta * deltas[i][j]
                    for k in range(len(self.capas[i].neuronas[j])):
                        self.capas[i].neuronas[j][k] = self.capas[i].neuronas[j][k] + eta * deltas[i][j] * salidas[i-1][k]
    
    def entrenar(self, datas, eta, pintarContour, fPorGeneraciones):
        entrenado = False
        cont = 0

        while not entrenado:
            entrenado = True
            print("--------------------------")
            
            print(cont)
            cont+=1
            for data in datas:
                salidas = self.evaluar([data.ent1, data.ent2])
                y_s = salidas[len(self.capas)-1][0]
                error = data.salida - y_s
                     
                print(error)
                
                if abs(error) > 0.1:
                    entrenado = False
                    deltas = self.obtenerDeltas(y_s, error, salidas)
                    
                    self.actualizarPesos(eta, deltas, data, salidas)
            if cont % fPorGeneraciones == 0:
                pintarContour()
            
        pintarContour() 
               






