# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:49:44 2019

@author: varnerj
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial.hermite import *
import scipy as sp
import math
from matplotlib import animation
from numpy.random import choice

m1 = 1.5
m2 = 2
k = 1 ###spring constant
h_bar = 1


n = 0
mu = (m1*m2)/(m1+m2)
nu = 0
####lam
###k =  2*np.pi/lam
alpha = np.sqrt((k*mu)/h_bar**2)
omega = 2*np.pi*nu
x = np.linspace(-5,5,200)



pi = np.pi
def HO_Func(state, xgrid, k, mu):
    
  w = np.sqrt(k/mu)
  psi = []
  herm_coeff = []

  for i in range(state):
      herm_coeff.append(0)
  herm_coeff.append(1)

  for x in xgrid:
    psi.append(math.exp(-mu*w*x**2/(2*h_bar)) * hermval((mu*w/h_bar)**0.5 * x, herm_coeff))
  # normalization factor for the wavefunction:
  psi = np.multiply(psi, 1 / (math.pow(2, state) * math.factorial(state))**0.5 * (mu*w/(pi*h_bar))**0.25)

      
  return psi


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-5, 5), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially to generate the animation


#### give a list of x-values between 0 and L, the length L, and the quantum numbner n
#### return a list of psi_n values between 0 and L
def hom_func(x, n, alpha):
    a = 1/np.sqrt(2**n*np.math.factorial(n))
    b = (alpha/pi)**(1/4)
    An = a*b
    psi_val = An*np.exp((-1* alpha*x**2)/2)
    
    return psi_val

def hom_en(n):
  
    nu = (1/2*np.pi)*np.sqrt(k/mu)
    en = h_bar*nu*(n+(1/2))
    
    return en

def hom_time(n,t):
    E = hom_en(n)
    ci = 0+1j
    phi = np.exp(-1*ci*E*t)
    return phi
# call the animator.  blit=True means only re-draw the parts that have changed.


#anim.save('PIB_EE3.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
      
###plt.show()

#####################################################################


def tri_wave(xarray):
    tw = np.zeros(len(xarray))
    for i in range (0,len(xarray)):
        xval = xarray[i]
        if xval <= 2:
            tw[i] = 0
        elif xval < 3:
            tw[i] = xval-2
        elif xval < 4:
            tw[i] = -xval+ 4
        else:
            tw[i] = 0
    return tw

tri = tri_wave(x)


def Fourier_Analysis(func, xarray, n):
    psi_n = HO_Func(n, xarray, k , mu)
    psi_n_star = np.conj(psi_n)
    dx = np.abs(xarray[0]-xarray[1])
    integrand = psi_n_star * func
    som = 0
    for i in range(0, len(xarray)):
        A =  integrand[i]*dx
        som = som + A
    return som


n_array = np.linspace(0,100, 101)
c_array = np.zeros(len(n_array), dtype = complex)

psi_exp = np.zeros(len(tri), dtype = complex)



for j in range(0,len(n_array)):
    c_array[j] = Fourier_Analysis(tri, x, int(n_array[j]))
    psi_exp = psi_exp + c_array[j] * HO_Func(int(n_array[j]), x, k, mu)

p_array = np.real(np.conj(c_array)*c_array)
N = np.sum(p_array)
p_array = p_array/N
draw = choice(n_array, 1, p=p_array)
print(draw)

def animate(i):
    psi_t = np.zeros(len(tri), dtype = complex)
    for j in range(0,len(n_array)):
        c_array[j] = Fourier_Analysis(tri, x, int(n_array[j]))
        psi_t = psi_t + c_array[j] * HO_Func(int(n_array[j]), x, k, mu)*hom_time(n_array[j], i/10)

    ci = 0.+1j
    psi_1 = np.sqrt(1/2)*HO_Func(6, x, k, mu)

    E1  = hom_en(6)
    ft1  = hom_time(6, i)
    #psi_t = np.zeros(1000,dtype=complex)
    psi_t_star = np.conj(psi_t)

 
    y = np.real(psi_t)
    z = np.imag(psi_t)
    p = np.real(psi_t_star * psi_t)
    
    line.set_data(x, p)
    return line,

plt.plot(x,tri)
#plt.plot(x,psi_exp )



anim = animation.FuncAnimation(fig, animate, init_func=init,
	                               frames=10000, interval=20, blit=True)



plt.show()







