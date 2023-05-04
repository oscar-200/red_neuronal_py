from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import tkinter as Tk
from tkinter import Tk, Frame,Button,Label, Entry, ttk, StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from red import Red
from numba import njit,vectorize

red = 0

class dato:
    def __init__(self, x1, x2, valor):
        self.ent1 = x1
        self.ent2 = x2
        self.salida = valor

class DataSet:
    def __init__(self, line):

        self.puntos = []

        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
    
    def limpiarPuntos(self):
        self.puntos.clear()

    def __call__(self, event):


        self.puntos.append(dato(event.xdata, event.ydata, 1 if event.button is MouseButton.LEFT else 0))
        if event.button is not MouseButton.LEFT:
            plt.scatter(event.xdata, event.ydata, 
                picker=True, pickradius=1, color="green")
        else:
            plt.scatter(event.xdata, event.ydata, 
                picker=True, pickradius=1, color="blue")
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.figure.canvas.draw()

fig, ax = plt.subplots()

plt.xlim(-3, 3)
plt.ylim(-3, 3)

ax.axhline(linewidth=2, color='r')
ax.axvline(linewidth=2, color='r')

ax.set_title('Red Neuronal')
line, = ax.plot([0], [0])  
dataSet =  DataSet(line)

ventana = Tk()
ventana.geometry('700x550')
ventana.wm_title('Practica 5')
ventana.minsize(width=700,height=646)

frame = Frame(ventana, bg='#3A1111',bd=15)
frame.grid(column=0,row=0)

canvas = FigureCanvasTkAgg(fig, master = frame)  # Crea el area de dibujo en Tkinter
canvas.get_tk_widget().grid(column=0, row=0, columnspan=3, padx=5, pady =5)

Label(frame, width = 15, text="Neuronas por capa", bg="#3A1111", fg="white").grid(column=0, row=1, pady =5)

neuronasPorCapa = Entry(frame, width=15)
neuronasPorCapa.grid(column=1, row=1, pady=5)

Label(frame, width = 15, bg="#3A1111", fg="white").grid(column=2, row=1, pady =5)

Label(frame, width = 15, text="Eta", bg="#3A1111", fg="white").grid(column=0, row=2, pady =5)

Eta = Entry(frame, width=15)
Eta.grid(column=1, row=2, pady=0)

Label(frame, width = 15, bg="#3A1111", fg="white").grid(column=2, row=1, pady =5)

Label(frame, width = 15, text="Generaciones por F", bg="#3A1111", fg="white").grid(column=0, row=3, pady =5)
fPorGeneraciones = Entry(frame, width=15)
fPorGeneraciones.grid(column=1, row=3, pady=0)

def pintarPuntos():
    for i in dataSet.puntos:
        if not i.salida:
                plt.scatter(i.ent1, i.ent2, 
                    picker=True, pickradius=1, color="green")
        else:
            plt.scatter(i.ent1, i.ent2, 
                picker=True, pickradius=1, color="blue")

def limpiar():
    dataSet.limpiarPuntos()
    limpiarPantalla()
    canvas.draw()
    ventana.update()

def limpiarPantalla(): 
    plt.clf()
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axhline(linewidth=2, color='r')
    plt.axvline(linewidth=2, color='r')

    ax.set_title('Red Neuronal')
    plt.plot([0], [0])

@njit
def calculoPorCapa(pesos, bias, salidas):
    v = np.transpose(np.dot(pesos, salidas)) + bias
    return 1/(1 + np.exp(-np.transpose(v)))


def pintarContour():
    
    x = np.arange(-3, 3, .1)
    y = np.arange(-3, 3, .1)

    z = []
    X, Y = np.meshgrid(x,y)
    X = X.flatten()
    Y = Y.flatten()
    
    salidas = np.array([X, Y])
    for j in range (len(red.capas)):
        np.array(salidas, dtype=np.float64)
        salidas = calculoPorCapa(np.array(red.capas[j].neuronas, dtype=np.float64),
                                np.array(red.capas[j].bias, dtype=np.float64), 
                                np.array(salidas, dtype=np.float64))
    z.append(np.round(np.ravel(salidas.copy())))
    
    z = np.array(z)
    z = z.reshape((60, 60))
    X = X.reshape((60, 60))
    Y = Y.reshape((60, 60))
    
    plt.ion()
    plt.clf()
    plt.contourf(X,Y,z, 10, cmap='rainbow')
    plt.colorbar()
    pintarPuntos()
    ventana.update()

def entrenar():
    vec = str(neuronasPorCapa.get()).split(",")
    vec.append("1")
    arreglo_enteros = list(map(int, vec))
    
    global red
    red = Red(arreglo_enteros, 2)
    red.entrenar(dataSet.puntos, float(Eta.get()), pintarContour, int(fPorGeneraciones.get()))
    ventana.update()

    
Button(frame, text='Entrenar', width = 15, bg='black',fg='white', command= entrenar).grid(column=0, row=4, pady =5)
Button(frame, text='Limpiar', width = 15, bg='black',fg='white', command= limpiar).grid(column=1, row=4, pady =5)

style = ttk.Style()
style.configure("Horizontal.TScale", background= '#3A1111')  

ventana.mainloop()
