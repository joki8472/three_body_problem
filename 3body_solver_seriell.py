#!/usr/bin/python
"""
Program for finding the stationary states of linear harmonic oscillator.
We assumer here \hbar = m = \omega = 1.
Expected energy eigenvalues are therefore n + 1/2 where n = 0, 1, ...
"""

import sys                              # for emergency exit
import numpy as np                      # for NumPy arrays
from scipy.optimize import brentq       # for root finding
import matplotlib.pyplot as plt         # for plotting
import time

n         = 2000                             # how many points we use for integration
xm        = 10                               # at which value of x we stop
dx        = float(xm) / n                    # step over the x-axis
dx2       = dx * dx                          # dx squared
tolerance = 1e-8                             # how precise we want to satisfy the boundary conditions
emax      = 1500                              # probe for E_b < emax
#
redm      = 1.0/2.0*(938.3+939.6)/2.0        # mass parameter of the radial equation
MeVfm     = 197.3161329                      # c=1 => MeVfm=hbar^2
b3        = 0.2
#
lam       = 4.0
D_1       = 5.5032
R0        = 1.0/lam                          # range of the 3-body interaction [fm]
# for lim x-> infty the solution to the free Schroedinger equation sets the boundary condition
# i.e., an exponential decay
def asymptotic_boundary(E_val): return np.exp(-np.sqrt(2*redm*E_val/MeVfm**2)*xm)

# the (effective) interaction with [V_0] = MeV
# for performance reasons the parameterization of the potential
# is hard-wired
def V(x,lam,co): return co*np.exp(-lam*x**2)
def V_eff(x,ro,d1):
    if x<ro:
        return MeVfm**2/(2.0*redm)*d1
    else:
        return -MeVfm**2/(2.0*redm)*(15.0/4.0 - 1.01251)/x**2

# we solve (d^2/dx^2 + f(x)) y(x) = 0
def f(n,E,lec,lam): return (2*redm/(MeVfm**2)) * (E - V_eff(n * dx,0.1*lam,lec))

# Numerov interation up to xm
# This function is minimized to retrieve binding energies
def psi_xm(E_val,lec,lam):
    """Returns the value of \Psi(x_m)."""
    x = np.linspace(0, xm, n+1)         # grid in the x-direction
    y = np.zeros(n+1)                   # wave-function in individual points
    # initial conditions
    y[0] = 0
    y[1] = 1.0
    #
    for i in range(1,n):
        y[i + 1] = (2 - 5 * dx2 * f(i, E_val,lec,lam) / 6) * y[i] - (1 + dx2 * f(i-1, E_val,lec,lam) / 12) * y[i - 1]
        y[i + 1] /= (1 + dx2 * f(i+1, E_val,lec,lam) / 12)
    return y[n]-asymptotic_boundary(-E_val)


def plot_range_of_energies(Emin, Emax):
    Evals, yfeven, yfodd = [], [], []   # arrays to store the obtained values
    for E in np.linspace(Emin, Emax, 101):
        Evals.append(E)
        yf = psi_xm(E)          # find \Psi(x_m) for an even wave function with energy E
        yfeven.append(yf)
    plt.plot(Evals, yfeven, 'r', linewidth = 2, label = r'$\Psi$')
    plt.plot(Evals, np.zeros(len(Evals)), 'g:', linewidth = 2)
    plt.ylim([-40,40])
    plt.xlabel(r'$E$', fontsize = 20)
    plt.xticks(range(5), fontsize = 16)
    plt.yticks(np.linspace(-40,40,5), fontsize = 16)
    plt.ylabel(r'$\Psi({})$'.format(xm), fontsize = 20)
    plt.legend(loc = 'upper right', prop = {'size': 20})
    plt.savefig('psi_xm.pdf')
    plt.show()


def main():
    plot_range_of_energies(0, 4)        # draw \Psi(x_m) for a wide range of energies
    E_0 = brentq(psi_xm, 0.2, 0.8, args=('even'))
    E_1 = brentq(psi_xm, 1.2, 1.8, args=('odd'))
    print 'found energy of the ground state: {}'.format(E_0)
    print 'found energy of the first excited state: {}'.format(E_1)


def find_shallow(de1,lambd):
    xend   = 0.0
    inc    = 1.0
    const = True
    x0    = 0.0
    signum = np.sign(psi_xm(x0,de1,lambd))
    while (const):
        x0 = x0 - inc
        signum_1 = np.sign(psi_xm(x0,de1,lambd))
        if (signum_1<>signum)|(abs(x0)>emax):
            const = False
            xend = x0
    if abs(xend)<emax:
        E_1 = brentq(psi_xm, x0+inc, xend,args=(de1,lambd))
        #print 'E = %4.4f MeV' %E_1
        return E_1+b3


def produce_limit_cycle(L0,dL,nL):
    H_L = []
    b3 = 1.0
    lamb_start = L0
    lamb_inc   = dL
    for ll in range(0,nL):
        lamb  = float(lamb_start + ll*lamb_inc)
        print 'lambda = %2.4f' %lamb
        const = True
        lec_start = -100.0
        lec_max   = 5000
        lec_inc   = +100.0
        lec       = lec_start - lec_inc
        go        = False
        # increase D1 until 0 < B(3) < B(triton)
        cntr = 0
        while go==False:
            cntr = cntr+1
            lec = lec + lec_inc
            emb_p = find_shallow(lec,lamb)
            print 'E_0(%4.2f) = %4.2f MeV' %(lec,emb_p)
            if emb_p>0:
                go = True
                break
            elif (emb_p=='none')|(emb_p==0.0):
                lec = lec_start - lec_inc
                lec_inc = lec_inc*0.5
            if (lec>lec_max):
                lec_inc = lec_inc*0.5
                lec_max = lec_max/10
                lec     = -100.0
            if lec_max==1:
                break    

        # (i)  increase D1 until B(3)>B(triton)
        # (ii) fit D1 to B(t)
        if go:
            D_1 = brentq(find_shallow, lec-2.0*lec_inc, lec, args=(lamb))
            print '%4.4f  , %12.4f' %(lamb,D_1)
            H_L.append([lamb,D_1])
    return H_L

startL  = 4.5
deL     = 0.07
anzL    = 10
#
res = []
res = produce_limit_cycle(startL,deL,anzL)
print res
#exit()
#
xx=[]
yy=[]

emax = 1500
for mm in range(0,anzL):
    start = startL + deL*mm
    res1  = find_shallow(-100.0,start)
    print res1
    if (type(res1)==float):
        xx.append(start)
        yy.append(res1)

fig = plt.figure()  
ax1 = fig.add_subplot(121)
ax1.set_xlabel(r'$\Lambda$ [fm$^{-1}$]')
ax1.set_ylabel(r'$H(\Lambda)$ [MeV]')
ax1.plot(xx,yy,'ro',label=r'$B_{\min}(3) = $%2.2f MeV' %b3)
ax1 = fig.add_subplot(122)
ax1.plot([LL[0] for LL in res],[HH[1] for HH in res],'ro',label=r'$B_{\min}(3) = 8.482$ MeV')
legend = ax1.legend(loc='lower right', fontsize=12)
#plt.ylim(-80,0)
#
plt.show()

exit()

def calc_spec(emax,de1,lamb):
    spect  = []
    xstart = 0.0
    xend   = 0.0
    inc    = 1.0
    while abs(xstart)<emax:
        const = True
        x0    = xstart
        signum = np.sign(psi_xm(x0,de1,lamb))
        while (const):
            x0 = x0 - inc
            signum_1 = np.sign(psi_xm(x0,de1,lamb))
            if (signum_1<>signum)|(abs(x0)>emax):
                const = False
                xend = x0
        if abs(xend)<emax:
            E_1 = brentq(psi_xm, xstart, xend,args=(de1,lamb))
            print 'E = %4.4f MeV' %E_1
            spect.append(E_1)
            xstart = xend
        else:
            break
    try:
        print 'N = %d bound state(s) found. E_max/E_max-1 = %4.4f' %(len(spect),spect[0]/spect[1])
    except:
        print 'N = %d bound state(s) found.' %(len(spect))
    
    fig = plt.figure()  
    ax1 = fig.add_subplot(111)
    #ax1.set_xlabel(r'$\rho/R_0$')
    ax1.set_ylabel(r'$Bindingenergy [MeV]$')
    for level in spect:
        ax1.plot([1,10],[level,level],'k-',lw=2)
    #plt.show()
calc_spec(100,0.5,4.0)
print find_shallow(0.5,4.0)-b3
#plot_range_of_energies(0,-4)
exit()
#main()
