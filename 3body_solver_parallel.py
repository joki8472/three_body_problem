#!/usr/bin/python
"""
algorithm to solve the three-body problem
- coordinate space
- Feddeev wave-function ansatz
- two identical fermions interacting with a core (neutron-neutron-proton)
- two- and three-body interactions regulated via theta functions ("square wells")
- identical regulator parameters for 2- and 3-body vertices, i.e. one Lambda (!)
- mass normalization equals the mass of the particles which in turn have identical masses:
  m_norm = m_proton = m_neutron
- procedure:
  (i)    obtain angular eigenvalue from transcendental equation det(D)=0 (numerically) => lam = f(a_nn,a_np(s=0),a_np(s=1),Lambda,rho))
  (ii)   solve hyperradial equation via Numerov for the lowest eigenvalue
  (iii)  adjust the hyperradial regulator depth (H(Lambda)) to yield an energy close to zeros
  (iv)   goto (i) and set Lambda = Lambda + delta (eventually, we are interested in H(Lambda) )
"""

import sys                              # for emergency exit
import numpy as np                      # for NumPy arrays
from scipy import optimize
from scipy.optimize import brentq       # for root finding
import matplotlib.pyplot as plt         # for plotting
import time
import multiprocessing as mp

n         = 2000                             # how many points we use for integration
xm        = 10                               # at which value of x we stop
dx        = float(xm) / n                    # step over the x-axis
dx2       = dx * dx                          # dx squared
tolerance = 1e-8                             # how precise we want to satisfy the boundary conditions
emax      = 1500                              # probe for E_b < emax
#
mn        = (938.3+939.6)/2.0
redm      = 1.0/2.0*(938.3+939.6)/2.0        # mass parameter of the radial equation
MeVfm     = 197.3161329                      # c=1 => MeVfm=hbar^2
b3        = 0.2
#
lam       = 4.0
D_1       = 5.5032
R0        = 1.0/lam                          # range of the 3-body interaction [fm]
#
# relate scattering lengths to the 2-body-interaction strengths, V_ff,V_fc+,V_fc-
def A_of_V(vv,lam,afit):
    return (np.tan(np.sqrt(vv)/(lam*np.sqrt(2.)))/np.sqrt(vv)*(lam*np.sqrt(2.))-1.)/lam - afit

fig = plt.figure()
ax1 = fig.add_subplot(111)
xx = np.arange(77,81,0.1)
yy = [ A_of_V( cc,4.0,0.0) for cc in xx]
plt.plot(xx,yy)
#plt.show()

print A_of_V(79.64,4.0,-22.7)   
#exit()

def V_of_A():
    lam   = 4.0
    afit  = -22.7

    res = optimize.root(A_of_V, 9.64, args=(lam,afit) )
    return res

print V_of_A()
exit()

# check parity (A.5)
phi      = np.arctan(np.sqrt(3))
phi_schl = np.arctan(np.sqrt(3))
def klein_f(lambd): return np.sin(np.sqrt(lambd)*(phi-np.pi/2.0))/np.sin(2*phi)
def klein_f_schl(lambd): return np.sin(np.sqrt(lambd)*(phi_schl-np.pi/2.0))/np.sin(2*phi_schl)
#
def alpha0(x):
    return np.arcsin(1/(x*lam*np.sqrt(2)))
def kappa_0(rho,lambd,V_ff):
    return np.sqrt(2*mn/MeVfm**2*V_ff*rho**2+lambd)
def kappa_scp(rho,lambd,V_fc):
    return np.sqrt(2*mn/MeVfm**2*(1+gamma_s/4.0)*V_fc*rho**2+lambd)
def kappa_scm(rho,lambd,V_fc):
    return np.sqrt(2*mn/MeVfm**2*(1-3.0*gamma_s/4.0)*V_fc*rho**2+lambd)
#                                               nn   sc-   sc+
# spin overlap functions C_ij_sisj , (i,s) = (1,0),(2,0),(3,1)
# the overlap is symmetric: C_ij_sisj=C_ji_sjsi
C_12_00 = -1./2.
C_21_00 = C_12_00
C_13_00 = -1./2.
C_31_00 = C_13_00
C_12_01 = np.sqrt(3./4.)
C_21_10 = C_12_01
C_13_01 = -np.sqrt(3./4.)
C_31_10 = C_13_01
C_23_00 = -1./2.
C_32_00 = C_23_00
C_23_11 = -1./2.
C_32_11 = C_23_11
C_23_01 = np.sqrt(3.)/2.
C_32_10 = C_23_01
C_23_10 = -np.sqrt(3.)/2.
C_32_01 = C_23_10
#
def D1(rho,lambd,V_ff):
    return kappa_0(rho,lambd,V_ff) * np.sin((alpha0(rho)-np.pi/2.)*np.sqrt(lambd)) * np.cos(alpha0(rho)*kappa_0(rho,lambd,V_ff)) - np.sqrt(lambd) * np.cos((alpha0(rho)-np.pi/2.)*np.sqrt(lambd)) * np.sin(alpha0(rho)*kappa_0(rho,lambd,V_ff))
def D2(rho,lambd,V_fc):
    return kappa_scm(rho,lambd,V_fc) * np.sin((alpha0(rho)-np.pi/2.)*np.sqrt(lambd)) * np.cos(alpha0(rho)*kappa_scm(rho,lambd,V_fc)) - np.sqrt(lambd) * np.cos((alpha0(rho)-np.pi/2.)*np.sqrt(lambd)) * np.sin(alpha0(rho)*kappa_scm(rho,lambd,V_fc))
def D3(rho,lambd,V_fc):
    return kappa_scp(rho,lambd,V_fc) * np.sin((alpha0(rho)-np.pi/2.)*np.sqrt(lambd)) * np.cos(alpha0(rho)*kappa_scp(rho,lambd,V_fc)) - np.sqrt(lambd) * np.cos((alpha0(rho)-np.pi/2.)*np.sqrt(lambd)) * np.sin(alpha0(rho)*kappa_scp(rho,lambd,V_fc))
#
def F_12(rho,lambd,V_ff):
    return (kappa_0(rho,lambd,V_ff) * np.sin(alpha0(rho)*np.sqrt(lambd)) * np.cos(alpha0(rho)*kappa_0(rho,lambd,V_ff)) - np.sqrt(lambd) * np.cos(alpha0(rho)*np.sqrt(lambd)) * np.sin(alpha0(rho)*kappa_0(rho,lambd,V_ff)))*4./np.sqrt(lambd)*klein_f(lambd)
def F_21(rho,lambd,V_fc):
    return (kappa_scm(rho,lambd,V_fc) * np.sin(alpha0(rho)*np.sqrt(lambd)) * np.cos(alpha0(rho)*kappa_scm(rho,lambd,V_fc)) - np.sqrt(lambd) * np.cos(alpha0(rho)*np.sqrt(lambd)) * np.sin(alpha0(rho)*kappa_scm(rho,lambd,V_fc)))*2./np.sqrt(lambd)*klein_f(lambd)
def F_13(rho,lambd,V_ff):
    return (kappa_0(rho,lambd,V_ff) * np.sin(alpha0(rho)*np.sqrt(lambd)) * np.cos(alpha0(rho)*kappa_0(rho,lambd,V_ff)) - np.sqrt(lambd) * np.cos(alpha0(rho)*np.sqrt(lambd)) * np.sin(alpha0(rho)*kappa_0(rho,lambd,V_ff)))*4./np.sqrt(lambd)*klein_f(lambd)
def F_31(rho,lambd,V_fc):
    return (kappa_scp(rho,lambd,V_fc) * np.sin(alpha0(rho)*np.sqrt(lambd)) * np.cos(alpha0(rho)*kappa_scp(rho,lambd,V_fc)) - np.sqrt(lambd) * np.cos(alpha0(rho)*np.sqrt(lambd)) * np.sin(alpha0(rho)*kappa_scp(rho,lambd,V_fc)))*2./np.sqrt(lambd)*klein_f(lambd)
def F_23(rho,lambd,V_fc):
    return (kappa_scm(rho,lambd,V_fc) * np.sin(alpha0(rho)*np.sqrt(lambd)) * np.cos(alpha0(rho)*kappa_scm(rho,lambd,V_fc)) - np.sqrt(lambd) * np.cos(alpha0(rho)*np.sqrt(lambd)) * np.sin(alpha0(rho)*kappa_scm(rho,lambd,V_fc)))*2./np.sqrt(lambd)*klein_f(lambd)
def F_32(rho,lambd,V_fc):
    return (kappa_scp(rho,lambd,V_fc) * np.sin(alpha0(rho)*np.sqrt(lambd)) * np.cos(alpha0(rho)*kappa_scp(rho,lambd,V_fc)) - np.sqrt(lambd) * np.cos(alpha0(rho)*np.sqrt(lambd)) * np.sin(alpha0(rho)*kappa_scp(rho,lambd,V_fc)))*2./np.sqrt(lambd)*klein_f(lambd)
#
def d11(rho,lambd,V_ff):
    return D1(rho,lambd,V_ff)
def d22(rho,lambd,V_fc):
    return D2(rho,lambd,V_fc)+F_23(rho,lambd,V_fc)*C_23_00
def d33(rho,lambd,V_fc):
    return D3(rho,lambd,V_fc)-F_32(rho,lambd,V_fc)*C_23_11
def d12(rho,lambd,V_fc):
    return F_12(rho,lambd,V_fc)*C_12_00
def d21(rho,lambd,V_fc):
    return F_21(rho,lambd,V_fc)*C_21_00
def d23(rho,lambd,V_fc):
    return F_23(rho,lambd,V_fc)*C_23_01
def d32(rho,lambd,V_fc):
    return F_32(rho,lambd,V_fc)*C_32_10
def d13(rho,lambd,V_fc):
    return F_13(rho,lambd,V_fc)*C_13_01
def d31(rho,lambd,V_fc):
    return F_31(rho,lambd,V_fc)*C_31_10
#
def det_D(rho,lambd,V_fc,V_ff):
    return d11(rho,lambd,V_ff)*d22(rho,lambd,V_fc)*d33(rho,lambd,V_fc)+d12(rho,lambd,V_fc)*d23(rho,lambd,V_fc)*d31(rho,lambd,V_fc)+d13(rho,lambd,V_fc)*d21(rho,lambd,V_fc)*d32(rho,lambd,V_fc)-d12(rho,lambd,V_fc)*d21(rho,lambd,V_fc)*d33(rho,lambd,V_fc)-d11(rho,lambd,V_fc)*d23(rho,lambd,V_fc)*d32(rho,lambd,V_fc)-d13(rho,lambd,V_fc)*d22(rho,lambd,V_fc)*d31(rho,lambd,V_fc)
#
gamma_s = 1.1
print det_D(1,1,1.1,1.3)
exit()
# -------------------------------------------------------------------------------------------------
# angular-eigenvalue equation: two identical fermions (neutron,neutron) interacting with
# a core (proton)
# 


# -------------------------------------------------------------------------------------------------
# for lim x-> infty the solution to the free Schroedinger equation sets the boundary condition
# i.e., an exponential decay
def asymptotic_boundary(E_val): return np.exp(-np.sqrt(2*redm*E_val/MeVfm**2)*xm)

# the (effective) interaction with [V_0] = MeV
# for performance reasons the parameterization of the potential
# is hard-wired
def V(x,lam,co): return co*np.exp(-lam*x**2)

def V_eff_tni(x,ro,d1):
    if x<ro:
        return MeVfm**2/(2.0*redm)*d1
    else:
        return -MeVfm**2/(2.0*redm)*(15.0/4.0 - 1.01251)/x**2

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
            print xend
        else:
            print xstart,xend
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
    plt.show()
    exit()

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

def calc_shallow(de1,lambd,output):
    xend   = 0.0
    inc    = 1.5
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
        output.put([lambd,E_1])

def fit_shallowest(lamb,D10,output):
    const = True
    lec_start = D10
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
        #print 'E_0(%4.2f) = %4.2f MeV' %(lec,emb_p)
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
        output.put([lamb,D_1])

if __name__ == '__main__':
    
    #calc_spec(10000,-200.0,4.0)

    output = mp.Queue()

    L0  = 3.0
    LE  = 4.0
    nL  = 60
    dL  = (LE-L0)/nL
    lambdas = [ float(L0 + ll*dL) for ll in range(0,nL)]
    print lambdas
    b3   = 1.5
    emax = 1200
    D10  = -1020.0
    #exit()
    #processes = [mp.Process(target=fit_shallowest, args=(lambd, D10, output)) for lambd in lambdas]
    processes = [mp.Process(target=calc_shallow, args=(D10,lambd, output)) for lambd in lambdas]
    for p in processes:
        p.start()

    for p in processes:
        p.join()
   
    res = [output.get() for p in processes]

    print(res)

#
    fig = plt.figure()  
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'$\Lambda$ [fm$^{-1}$]')
    ax1.set_ylabel(r'$H(\Lambda)$ [MeV]')
    ax1.set_ylabel(r'$B(3,\Lambda)$ [MeV]')
    #ax1 = fig.add_subplot(122)
    ax1.plot([LL[0] for LL in res],[HH[1] for HH in res],'ro',label=r'$B_{\min}(3,D_1=$%2.2f MeV)' %D10)
    legend = ax1.legend(loc='lower right', fontsize=12)
    #plt.ylim(-80,0)
    #
    plt.show()
    
    exit()
    
