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
# -------------------------------------------------------------------------------------------------
import os
import sys                              # for emergency exit
import numpy as np                      # for NumPy arrays
from scipy import optimize
from scipy.optimize import brentq       # for root finding
import matplotlib.pyplot as plt         # for plotting
import cmath as cm
import time
import multiprocessing as mp

cc_piless_list = {'1.50': [-86.4497,  -58.891727, 21.4797, 21.4797 ],
                         '2.00': [-142.3643, -106.279322, 68.4883, 68.4883 ],
                         '3.00': [-295.9365, -242.792464, 267.5588, 267.5588 ],
                         '4.00': [ -505.1643, -434.958473, 677.7989, 677.7989  ],
                         '6.00': [-1090.584 , -986.251897, 2652.651, 2652.651  ],
                         '8.00': [-1898.622 ,-1760.161732, 7816.228, -2539.22994 ],
                         '10.00':[-2929.277 ,-2756.688416, 20483.217, -3329.278037 ],
                         '14.00':[-6451.9818 , -5419.48037  , 125857.785, 0.0]}

# for debugging the 'paralized' code
def info(title):
    print title
    #print 'module name:', __name__
    #if hasattr(os, 'getppid'):  # only available on Unix
    #    print 'parent process:', os.getppid()
    print 'process id:', os.getpid()
# -------------------------------------------------------------------------------------------------
tolerance    = 1e-8                             # how precise we want to satisfy the boundary conditions
#
mn        = (938.3+939.6)/2.0
redm      = 1.0/2.0*(938.3+939.6)/2.0        # mass parameter of the radial equation
MeVfm     = 197.3161329                      # c=1 => MeVfm=hbar^2
#
a_ff      = -18.0                            # [fm], neutron-neutron scattering length, 'experimental' value 
a_fcm     = -23.7                            # [fm], singlet S=0 neutron-proton scattering length
a_fcp     = +5.42                            # [fm], triplet S=1 neutron-proton scattering length
#
v_0_ff    = float('Inf')
v_0_fcp   = float('Inf')
v_0_fcm   = float('Inf')
gamma_s   = float('Inf')
v_ff      = float('Inf')
v_fc      = float('Inf')
# -------------------------------------------------------------------------------------------------
# relate scattering lengths to the 2-body-interaction strengths, V_ff,V_fc+,V_fc-
# use equ.(96) on p.214
def A_of_V(vv,lam,afit):
    if abs(afit)>10**(-5):
        return abs((np.tan(np.sqrt(vv)/(lam*np.sqrt(2.)))/np.sqrt(vv)*(lam*np.sqrt(2.))-1.)/lam - afit)
    else:
        return (np.tan(np.sqrt(vv)/(lam*np.sqrt(2.)))/np.sqrt(vv)*(lam*np.sqrt(2.))-1.)/lam

def plot_a_of_v(lma):
    xx = np.arange(0.1,30)
    yy = [ A_of_V(vb,lma,0.0) for vb in xx]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xx,yy)
    plt.show()
    exit()

def V_of_A(lam,afit,vo):
    try:
        res = optimize.fmin(A_of_V, vo, args=(lam,afit), disp=0 )
        return res
    except:
        print 'no solution for V_0(a=%2.2f,v0=%4.2f) found.\n Try a different vo' %(afit,vo)
        exit()
# -------------------------------------------------------------------------------------------------
# check parity (A.5)
phi      = np.arctan(np.sqrt(3))
phi_schl = np.arctan(np.sqrt(3))
def klein_f(lambd): return np.sin(np.sqrt(lambd)*(phi-np.pi/2.0))/np.sin(2*phi)
def klein_f_schl(lambd): return np.sin(np.sqrt(lambd)*(phi_schl-np.pi/2.0))/np.sin(2*phi_schl)
#
def alpha0(x,lamm):
    return np.arcsin(1/(x*lamm*np.sqrt(2)))
def kappa_0(rho,lambd,V_ff):
    return np.sqrt(2*mn/MeVfm**2*V_ff*rho**2+lambd)
def kappa_scp(rho,lambd,V_fc,gamm):
    return np.sqrt(2*mn/MeVfm**2*(1+gamm/4.0)*V_fc*rho**2+lambd)
def kappa_scm(rho,lambd,V_fc,gamm):
    return np.sqrt(2*mn/MeVfm**2*(1-3.0*gamm/4.0)*V_fc*rho**2+lambd)
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
def D1(rho,lambd,V_ff,lamm):
    return kappa_0(rho,lambd,V_ff) * np.sin((alpha0(rho,lamm)-np.pi/2.)*np.sqrt(lambd)) * np.cos(alpha0(rho,lamm)*kappa_0(rho,lambd,V_ff)) - np.sqrt(lambd) * np.cos((alpha0(rho,lamm)-np.pi/2.)*np.sqrt(lambd)) * np.sin(alpha0(rho,lamm)*kappa_0(rho,lambd,V_ff))
def D2(rho,lambd,V_fc,gamm,lamm):
    return kappa_scm(rho,lambd,V_fc,gamm) * np.sin((alpha0(rho,lamm)-np.pi/2.)*np.sqrt(lambd)) * np.cos(alpha0(rho,lamm)*kappa_scm(rho,lambd,V_fc,gamm)) - np.sqrt(lambd) * np.cos((alpha0(rho,lamm)-np.pi/2.)*np.sqrt(lambd)) * np.sin(alpha0(rho,lamm)*kappa_scm(rho,lambd,V_fc,gamm))
def D3(rho,lambd,V_fc,gamm,lamm):
    return kappa_scp(rho,lambd,V_fc,gamm) * np.sin((alpha0(rho,lamm)-np.pi/2.)*np.sqrt(lambd)) * np.cos(alpha0(rho,lamm)*kappa_scp(rho,lambd,V_fc,gamm)) - np.sqrt(lambd) * np.cos((alpha0(rho,lamm)-np.pi/2.)*np.sqrt(lambd)) * np.sin(alpha0(rho,lamm)*kappa_scp(rho,lambd,V_fc,gamm))
#
def F_12(rho,lambd,V_ff,lamm):
    return (kappa_0(rho,lambd,V_ff) * np.sin(alpha0(rho,lamm)*np.sqrt(lambd)) * np.cos(alpha0(rho,lamm)*kappa_0(rho,lambd,V_ff)) - np.sqrt(lambd) * np.cos(alpha0(rho,lamm)*np.sqrt(lambd)) * np.sin(alpha0(rho,lamm)*kappa_0(rho,lambd,V_ff)))*4./np.sqrt(lambd)*klein_f(lambd)
def F_21(rho,lambd,V_fc,gamm,lamm):
    return (kappa_scm(rho,lambd,V_fc,gamm) * np.sin(alpha0(rho,lamm)*np.sqrt(lambd)) * np.cos(alpha0(rho,lamm)*kappa_scm(rho,lambd,V_fc,gamm)) - np.sqrt(lambd) * np.cos(alpha0(rho,lamm)*np.sqrt(lambd)) * np.sin(alpha0(rho,lamm)*kappa_scm(rho,lambd,V_fc,gamm)))*2./np.sqrt(lambd)*klein_f(lambd)
def F_13(rho,lambd,V_ff,lamm):
    return (kappa_0(rho,lambd,V_ff) * np.sin(alpha0(rho,lamm)*np.sqrt(lambd)) * np.cos(alpha0(rho,lamm)*kappa_0(rho,lambd,V_ff)) - np.sqrt(lambd) * np.cos(alpha0(rho,lamm)*np.sqrt(lambd)) * np.sin(alpha0(rho,lamm)*kappa_0(rho,lambd,V_ff)))*4./np.sqrt(lambd)*klein_f(lambd)
def F_31(rho,lambd,V_fc,gamm,lamm):
    return (kappa_scp(rho,lambd,V_fc,gamm) * np.sin(alpha0(rho,lamm)*np.sqrt(lambd)) * np.cos(alpha0(rho,lamm)*kappa_scp(rho,lambd,V_fc,gamm)) - np.sqrt(lambd) * np.cos(alpha0(rho,lamm)*np.sqrt(lambd)) * np.sin(alpha0(rho,lamm)*kappa_scp(rho,lambd,V_fc,gamm)))*2./np.sqrt(lambd)*klein_f(lambd)
def F_23(rho,lambd,V_fc,gamm,lamm):
    return (kappa_scm(rho,lambd,V_fc,gamm) * np.sin(alpha0(rho,lamm)*np.sqrt(lambd)) * np.cos(alpha0(rho,lamm)*kappa_scm(rho,lambd,V_fc,gamm)) - np.sqrt(lambd) * np.cos(alpha0(rho,lamm)*np.sqrt(lambd)) * np.sin(alpha0(rho,lamm)*kappa_scm(rho,lambd,V_fc,gamm)))*2./np.sqrt(lambd)*klein_f(lambd)
def F_32(rho,lambd,V_fc,gamm,lamm):
    return (kappa_scp(rho,lambd,V_fc,gamm) * np.sin(alpha0(rho,lamm)*np.sqrt(lambd)) * np.cos(alpha0(rho,lamm)*kappa_scp(rho,lambd,V_fc,gamm)) - np.sqrt(lambd) * np.cos(alpha0(rho,lamm)*np.sqrt(lambd)) * np.sin(alpha0(rho,lamm)*kappa_scp(rho,lambd,V_fc,gamm)))*2./np.sqrt(lambd)*klein_f(lambd)
#
def d11(rho,lambd,V_ff,lamm):
    return D1(rho,lambd,V_ff,lamm)
def d22(rho,lambd,V_fc,gamm,lamm):
    return D2(rho,lambd,V_fc,gamm,lamm)+F_23(rho,lambd,V_fc,gamm,lamm)*C_23_00
def d33(rho,lambd,V_fc,gamm,lamm):
    return D3(rho,lambd,V_fc,gamm,lamm)-F_32(rho,lambd,V_fc,gamm,lamm)*C_23_11
def d12(rho,lambd,V_fc,gamm,lamm):
    return F_12(rho,lambd,V_fc,lamm)*C_12_00
def d21(rho,lambd,V_fc,gamm,lamm):
    return F_21(rho,lambd,V_fc,gamm,lamm)*C_21_00
def d23(rho,lambd,V_fc,gamm,lamm):
    return F_23(rho,lambd,V_fc,gamm,lamm)*C_23_01
def d32(rho,lambd,V_fc,gamm,lamm):
    return F_32(rho,lambd,V_fc,gamm,lamm)*C_32_10
def d13(rho,lambd,V_fc,gamm,lamm):
    return F_13(rho,lambd,V_fc,lamm)*C_13_01
def d31(rho,lambd,V_fc,gamm,lamm):
    return F_31(rho,lambd,V_fc,gamm,lamm)*C_31_10
#
def det_D(lambd,rho,V_ff,V_fc,gamm,lamm):
    return d11(rho,lambd,V_ff,lamm)*d22(rho,lambd,V_fc,gamm,lamm)*d33(rho,lambd,V_fc,gamm,lamm)+d12(rho,lambd,V_fc,gamm,lamm)*d23(rho,lambd,V_fc,gamm,lamm)*d31(rho,lambd,V_fc,gamm,lamm)+d13(rho,lambd,V_fc,gamm,lamm)*d21(rho,lambd,V_fc,gamm,lamm)*d32(rho,lambd,V_fc,gamm,lamm)-d12(rho,lambd,V_fc,gamm,lamm)*d21(rho,lambd,V_fc,gamm,lamm)*d33(rho,lambd,V_fc,gamm,lamm)-d11(rho,lambd,V_fc,lamm)*d23(rho,lambd,V_fc,gamm,lamm)*d32(rho,lambd,V_fc,gamm,lamm)-d13(rho,lambd,V_fc,gamm,lamm)*d22(rho,lambd,V_fc,gamm,lamm)*d31(rho,lambd,V_fc,gamm,lamm)
# -------------------------------------------------------------------------------------------------
def det_D_approx(lam,rhoo,am,ap,sc,ac):
    # dependencies: a+,a-,rho,sc,phi
    phi = np.arctan(np.sqrt(ac*(ac+2)))
    determinant_line1 = lam*cm.cos(np.pi/2.*cm.sqrt(lam))**2+rhoo**2/(ap*am)*cm.sin(cm.sqrt(lam)*np.pi/2.0)**2+1./2.*(rhoo/am+rhoo/ap)*cm.sqrt(lam)*cm.sin(np.pi*cm.sqrt(lam))
    determinant_line2 = 1./(2.*sc+1)*(rhoo/am-rhoo/ap)*2./cm.sin(2*phi)*cm.sin(np.pi/2.*cm.sqrt(lam))*cm.sin((phi-np.pi/2.)*cm.sqrt(lam))-4./cm.sin(2*phi)**2*cm.sin((phi-np.pi/2.)*cm.sqrt(lam))**2
    return (determinant_line1+determinant_line2).real
#
def set_eff_pot_coeff(v00,lamm):
    try:
        v0 = v00;
        v_0_ff = V_of_A(lamm,a_ff,v0)
        print 'a_ff (Lambda=%1.1f fm^-1,v=%2.2f fm^-2) = %2.2f fm' %(lamm,v_0_ff,A_of_V(v_0_ff,lamm,0.0)[0])
        v_0_fcp = V_of_A(lamm,a_fcp,v0-10)
        print 'a_fcp (Lambda=%1.1f fm^-1,v=%2.2f fm^-2) = %2.2f fm' %(lamm,v_0_fcp,A_of_V(v_0_fcp,lamm,0.0)[0])
        v_0_fcm = V_of_A(lamm,a_fcm,v_0_ff)
        print 'a_fcm (Lambda=%1.1f fm^-1,v=%2.2f fm^-2) = %2.2f fm\n--' %(lamm,v_0_fcm,A_of_V(v_0_fcm,lamm,0.0)[0])
    except:
        print 'I failed to determine well depths from the scattering lengths with the specified initial guess.'
        exit()
    # V_original from the effective, scaled values, equ.(103,104,and 74f)
    ff = MeVfm**2/(2.*mn)*v_0_ff
    #
    g  = 4.*(v_0_fcp-v_0_fcm)/(3.*v_0_fcp+v_0_fcm)
    fc = 0.25*MeVfm**2/(2.*mn)*(3*v_0_fcp+v_0_fcm)
    return ff,fc,g
#
def calc_angular_ev(lamm,nn,xmm,output):
    numerov_grid = np.linspace(0, xmm, nn+1)
    dx           = float(xmm)/nn            # step over the x-axis
    dx2          = dx * dx                          # dx squared
    lecs = set_eff_pot_coeff(15.1,lamm)
    lambda_n = []
    lambd0 = 0.0
    outstr = 'ang_ev_'+str(nn)+'_'+str(xmm)+'_%2.2f'%float(lamm)+'.dat'
    #
    if os.path.isfile(outstr):
        lambda_n = [ float(aa) for aa in open(outstr)]
        print 'angular EV from file. Lambda = %2.2f fm^-1' %float(lamm)
        output.put(lambda_n)
        return
    else:
        for gp in numerov_grid[::-1]:
            if (gp<(1./lamm)):
                lambda_n.append(lambd0)
            else:
                lambda_n.append(optimize.fmin(det_D, lambd0, args=(gp,lecs[0][0],lecs[1][0],lecs[2][0],lamm), disp=0 )[0])
                lambd0 = lambda_n[-1]
        lambda_n = lambda_n[::-1]
        with open(outstr,'w') as outfile:
            for llll in lambda_n:
                outfile.write(str(llll)+'\n')
        print 'angular EV calculated and written to file. Lambda = %s fm^-1' %lamm
    output.put(lambda_n)
    return      
#
def calc_angular_ev_approx(lamm,nn,xmm,aminus,aplus,corespin,corenuclearnumber,output):
    numerov_grid = np.linspace(0, xmm, nn+1)
    dx           = float(xmm)/nn            # step over the x-axis
    dx2          = dx * dx                          # dx squared
    lambda_n = []
    lambd0 = 0.000001
    lambd1 = lambd0*1.01
    #
    for gp in numerov_grid[::-1]:
        sign  = np.sign(det_D_approx(lambd0 ,gp ,aminus ,aplus ,corespin ,corenuclearnumber))
        sign2 = np.sign(det_D_approx(lambd1,gp ,aminus ,aplus ,corespin ,corenuclearnumber))
        while sign==sign2:
            lambd1 = lambd1 + 0.002
            sign2 = np.sign(det_D_approx(lambd1,gp ,aminus ,aplus ,corespin ,corenuclearnumber))
            if lambd1>10:
                print 'no solution!'
                continue
        #res = optimize.brentq(det_D_approx, lambd0, lambd1 ,args=(gp,aminus,aplus,corespin,corenuclearnumber), disp=0 )
        #res = optimize.newton(det_D_approx, lambd0, args=(gp,aminus,aplus,corespin,corenuclearnumber) )
        res = optimize.fmin(det_D_approx, np.mean([lambd0,lambd1]), args=(gp,aminus,aplus,corespin,corenuclearnumber), disp=0 )[0]
        # lambda = lambda_tilde - 4 => (lambda+15/4)=(lambda-1/4) is passed to the potential function
        lambda_n.append(res-0.25)
        lambd0 = res*0.92

    output.put(lambda_n[::-1])

#
# -------------------------------------------------------------------------------------------------
# angular-eigenvalue equation: two identical fermions (neutron,neutron) interacting with
# a core (proton)
# 


# -------------------------------------------------------------------------------------------------
# for lim x-> infty the solution to the free Schroedinger equation sets the boundary condition
# i.e., an exponential decay @ not in use @
def asymptotic_boundary(E_val,rho_max): return np.exp(0*np.sqrt(2*mn/MeVfm**2)*rho_max)*np.exp(-np.sqrt(2*mn*E_val/MeVfm**2)*rho_max)
#
def calc_spec(emax,de1,lamb,lambdan,grid):
    spect  = []
    xstart = 0.0
    xend   = 0.0
    inc    = 1.0
    while abs(xstart)<emax:
        const = True
        x0    = xstart
        signum = np.sign(psi_xm(x0,de1,lamb,lambdan,grid))
        while (const):
            x0 = x0 - inc
            signum_1 = np.sign(psi_xm(x0,de1,lamb,lambdan,grid))
            if (signum_1<>signum)|(abs(x0)>emax):
                const = False
                xend = x0
        if abs(xend)<emax:
            E_1 = brentq(psi_xm, xstart, xend,args=(de1,lamb))
            #E_1 = optimize.bisect(psi_xm, xstart, xend,args=(de1,lamb,lambdan,grid))
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
#
# th(r_1-r_2)
def theta(rhoo_1,rhoo_2):
    if rhoo_1>rhoo_2:
        return 1
    else:
        return 0

# we solve (d^2/drho^2 + f(rho)) y(rho) = 0, see, e.g. Hjorth-Jensen notes ch.9 equ.9.11 ff
# ---------
# f(x) = 2*redm/hbc^2 * (v(rho) -B -hbc^2/(2*redm)*(ang_ev/rho^2))
# ---------
# [lamm]  = fm^-2
# [E_val] = MeV > 0 means bound
# [lecc]  = MeV > 0 means attractive
# [rhoo]  = fm
def f_theta(E,rhoo,lecc,lamm,ang_ev):
    # test case: for ang_ev = 0, lamm=4.0, and lecc=505.1643, an appropriate grid should yield B=2.22(4)
    #return 2*redm/(MeVfm**2)*(lecc*np.exp(-lamm**2/4.0*rhoo**2) -E)
    #
    if float(rhoo)<(10**(-9)):
        return 2*redm/(MeVfm**2)*(lecc*theta(1./lamm,rhoo) -E)
    else:
        return 2*redm/(MeVfm**2)*(lecc*theta(1./lamm,rhoo) -E) - theta(rhoo,1.0/lamm)*ang_ev/rhoo**2
def f_gauss(E,rhoo,lecc,lamm,ang_ev):
    # test case: for ang_ev = 0, lamm=4.0, and lecc=505.1643, an appropriate grid should yield B=2.22(4)
    return 2*redm/(MeVfm**2)*(lecc*np.exp(-lamm**2/4.0*rhoo**2) -E)


def plot_potentials(lamth,lth,lamg,lg,xm):
    xx  = np.linspace(0,xm,10000)
    yth = [ f_theta(0.0,rr,lth,lamth,0.0) for rr in xx ]
    yg  = [ f_gauss(0.0,rr,lg,lamg,0.0) for rr in xx ]
    fig = plt.figure()  
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'$\rho$ [fm]')
    ax1.set_ylabel(r'$\frac{2\mu}{\hbar^2}V(\rho)$')
    ax1.plot(xx,yth,label=r'$V(\rho) = $ %4.1f $C_0 e^{-\Lambda^2/4\rho^2}$' %lth)
    ax1.plot(xx,yg,label=r'$V(\rho) = $ %4.1f $C_0 \theta(\Lambda^{-1}-\rho)$' %lg)
    legend = ax1.legend(loc='upper right', fontsize=12)
    plt.show()
    exit()

#plot_potentials(2,200,2,200,12)
# Numerov interation up to xm
# This function is minimized to retrieve binding energies

# numerical outward integration on a fixed grid for a defined interaction and energy
# application, e.g., find ground state via min(psi_joki) and use this function to
# visualize the result
def wfkt(E_val,lec,lammm,ang_ev,grid):
    """Returns the value of \Psi(x_m)."""
    x   = grid                    # grid in the x-direction
    dex = float(grid[-1]-grid[-2])
    dex2= dex*dex
    n   = len(grid)-1
    y   = np.zeros(n+1)                   # wave-function in individual points
    yy   = np.zeros(n+1)                   # wave-function in individual points
    # fin turning point
    y[-1] = 0.0
    y[-2] = 0.5*dex2
    rho_match = 0
    for i in range(1,n)[::-1]:        
        y[i - 1] = (2.0 - 5.0 * dex2 * f_theta(E_val,i*dex,lec,lammm,ang_ev) / 6.0) * y[i] - (1.0 + dex2 * f_theta(E_val,(i+1)*dex,lec,lammm,ang_ev) / 12.0) * y[i + 1]
        y[i - 1] /= (1.0 + dex2 * f_theta(E_val,(i-1)*dex,lec,lammm,ang_ev) / 12.0)

    return y

def psi_joki(E_val,lec,lammm,ang_ev,grid):
    dex  = float(grid[-1]-grid[-2])
    dex2 = dex*dex
    n    = len(grid)-1
    y    = np.zeros(n+1)                   # wave-function for the inward integration
    yy   = np.zeros(n+1)                   # wave-function for the outward integration
    # boundary conditions at infinity
    y[-1] = 0.0
    y[-2] = 0.5*dex2
    rho_match = 0
    # integrate towards zero until y+ < y-, i.e., a turning point is reached
    for i in range(1,n)[::-1]:        
        y[i - 1]  = (2.0 - 5.0 * dex2 * f_theta(E_val,i*dex,lec,lammm,ang_ev[i]) / 6.0) * y[i] - (1.0 + dex2 * f_theta(E_val,(i+1)*dex,lec,lammm,ang_ev[i+1]) / 12.0) * y[i + 1]
        y[i - 1] /= (1.0 + dex2 * f_theta(E_val,(i-1)*dex,lec,lammm,ang_ev[i-1]) / 12.0)
        if (y[i-1]<y[i]):
            rho_match = i+1
            break
    if rho_match==0:
        #print 'no turning point!'
        return 10^8
    # logarithmic derivative at the matching point (crudest approximation)
    log_diff_inward = (y[rho_match]-y[rho_match+1])/(dex*y[rho_match])
    # initial conditions at zero
    yy[0] = 0.0
    yy[1] = 0.5*dex2
    # iterate with Numerov accuracy up to the turning point
    for i in range(1,rho_match+1):
        yy[i + 1] = (2. - 5. * dex2 * f_theta(E_val,i*dex,lec,lammm,ang_ev[i]) / 6.) * yy[i] - (1. + dex2 * f_theta(E_val,(i-1)*dex,lec,lammm,ang_ev[i-1]) / 12.) * yy[i - 1]
        yy[i + 1] /= (1. + dex2 * f_theta(E_val,(i+1)*dex,lec,lammm,ang_ev[i+1]) / 12.)
    # again, logarithmic derivative at the matching point, now from the outward integration
    log_diff_outward = (yy[rho_match]-yy[rho_match-1])/(dex*yy[rho_match])
    # 
    return abs(log_diff_outward-log_diff_inward)

def find_shallow_par(de1,lambdaa,ang_ev,enint,grid,output):
    #info('find_shallow_par')
    x0     = enint[0]
    xend   = enint[1]
    E_1 = optimize.fmin(psi_joki, np.mean([x0,xend]),args=(de1,lambdaa,ang_ev,grid))[0]
    output.put([lambdaa,E_1])
    return
    inc    = 0.1
    const  = True
    signum   = np.sign(psi_joki(x0,de1,lambdaa,ang_ev,grid))
    signum_1 = signum
    while (const):
        x0 = x0 + inc
        signum =signum_1
        signum_1 = np.sign(psi_joki(x0,de1,lambdaa,ang_ev,grid))
        if (signum_1<>signum)&(x0<xend):
            const = False
            xend = x0
            E_1 = brentq(psi_joki, x0-inc, x0,args=(de1,lambdaa,ang_ev,grid))
            
            output.put([lambdaa,E_1])
        elif (x0>xend):
            const = False
            print 'no bound state found for (Lambd, D_1) = (%2.2f,%2.2f) below E_max = %4.2f' %(lambdaa, de1, xend)
            output.put([lambdaa,0.0])

def find_shallow_ser(de1,lambdaa,ang_ev,enint,grid,bfit=0.0):
    x0     = enint[0]
    xend   = enint[1]

    inc    = 10.0
    const  = True
    signum = np.sign(psi_joki(x0,de1,lambdaa,ang_ev,grid))
    while (const):
        x0 = x0 + inc
        signum_1 = np.sign(psi_joki(x0,de1,lambdaa,ang_ev,grid))
        if (signum_1<>signum)&(x0<xend):
            #print [enint[0],x0]
            const = False
            E_1 = brentq(psi_joki, enint[0], x0,args=(de1,lambdaa,ang_ev,grid))
            #E_1 = optimize.bisect(psi_joki, x0, xend,args=(de1,lambd,ang_ev,grid))
            return E_1-bfit
        elif (x0>xend):
            const = False
            print 'no bound state found for (Lambd, D_1) = (%2.2f,%2.2f) below E_max = %4.2f' %(lambdaa, de1, xend)
            #output.put([lambd,0.0])    
            return 'none'

def fit_shallowest(lamb,D10,ang_ev,interv,grid,bfit,output):
    const = True
    lec   = D10
    ueber = False
    unter = False
    # increase D1 until 0 < B(3) < B(triton)
    cntr = 0
    fac  = 4.05
    while (cntr<50)&(lec<10**4):
        cntr = cntr + 1
        emb_p = find_shallow_ser(lec,lamb,ang_ev,interv,grid,bfit)
        # no bs yet => increase D1
        if (emb_p=='none')|(emb_p==0.0):
            lec = lec*fac
        elif (emb_p>0):
            #print 'E_0(%4.2f) = %4.2f MeV' %(lec,emb_p)
            cntr = 0
            ueber = lec
            lec = lec*0.9
            if unter<>False:
                break
        elif (emb_p<0):
            #print 'E_0(%4.2f) = %4.2f MeV' %(lec,emb_p)
            cntr = 0
            unter = lec
            lec = lec*1.1
            if ueber<>False:
                break
    if (ueber==False)|(unter==False):
        output.put([lamb,0.0])
    else:
        D_1 = brentq(find_shallow_ser, unter, ueber, args=(lamb,ang_ev,interv,grid,bfit))
        print '%4.4f  , %12.4f' %(lamb,D_1)
        output.put([lamb,D_1])

def numerov_test():
    # use commented Gaussian version of f_theta(rho) above
    stue         = 4000     # how many points we use for integration
    xmax         = 12        #[ 10*nn for nn in range(1,2) ]   # at which value of x we stop
    dx           = xmax/stue #[ float(xmaxx)/stue for xmaxx in xmax ]                   # step over the x-axis
    dx2          = dx*dx     #[ dxx * dxx for dxx in dx ]                          # dx squared
    numerov_grid = np.linspace(0, xmax, stue+1) #[np.linspace(0, xmaxx, stue+1) for xmaxx in xmax]
    #
    lam = '2.00'
    LEc = abs(cc_piless_list[str(lam)][0])
    inter = [0.001,100]
    eb = find_shallow_ser(LEc,float(lam),0.0,inter,numerov_grid)
    print 'B = %2.2f MeV' %eb
    print 'A Gaussian should yield B=B(d) \n -----------------------------'
    #
    vv =  [ f_theta(0.0,xxx,LEc,float(lam),0.0) for xxx in numerov_grid ]
    yy = wfkt(eb,LEc,float(lam),0.0,numerov_grid)
    #
    fig = plt.figure()  
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel(r'$\rho$ [fm]')
    ax1.set_ylabel(r'$V_{eff}(\rho)$')
    ax1.plot(numerov_grid,vv)
    ax1 = fig.add_subplot(122)
    ax1.set_ylabel(r'$\psi(\rho)$')
    ax1.plot(numerov_grid,yy,label=r'B = %4.4f'%eb)
    legend = ax1.legend(loc='upper right', fontsize=12)
    plt.show()
    exit()

if __name__ == '__main__':
    # --------------------------------------------------------------------------------------------------------
    #numerov_test()


    inter = [0.01,20]
    D10   = 40.0
    bf    = 10.0
    stue  = 5000
    xmax  = 20
    a_m   = -23.7
    a_p   = 5.42
    s_c   = 0.5
    a_c   = 1
    numerov_grid = np.linspace(0, xmax, stue+1)
    L0  = 1.1
    LE  = 2.5
    nL  = 2
    dL  = (LE-L0)/nL
    lambdas = [ float(L0 + ll*dL) for ll in range(0,nL)]
    print lambdas

    output1 = mp.Queue()

    processes1 = [mp.Process(target=calc_angular_ev_approx, args=(lmbd, stue, xmax, a_m, a_p, s_c, a_c, output1)) for lmbd in lambdas]
    for p in processes1:
        p.start()
       
    res1 = [output1.get() for p in processes1]
    # joining dead locks the queue if the returned data exceeds a limit, see https://docs.python.org/2/library/multiprocessing.html#multiprocessing-programming
    for p in processes1:
        p.join()

    pl = False
    if pl:
        fig = plt.figure()  
        ax1 = fig.add_subplot(111)
        for hh in range(0,len(res1)):
            #xx = [ np.log10(fm) for fm in numerov_grid if fm>0]
            #ax1.plot(xx,res1[hh][1:])
            ax1.plot(numerov_grid,res1[hh])
        plt.show()
        exit()
    for worker in processes1:
        assert not worker.is_alive()
    #
    vv = [ [f_theta(0.0,i*float(xmax)/float(stue),D10,lambdas[nn],res1[nn][i]) for i in range(0,stue+1) ] for nn in range(0,len(lambdas)) ]
    #
    fitt = False
    #
    if fitt:
        output2 = mp.Queue()
        processes2 = [mp.Process(target=fit_shallowest, args=(lambdas[ng],D10,res1[ng],inter,numerov_grid, bf ,output2)) for ng in range(0,len(lambdas)) ]
        #
        for rop in processes2:
            rop.start()
        #   
        res2 = [output2.get() for p in processes2]
        #
        for p in processes2:
            p.join(timeout=1)
        #
        print(res2)
    
    output3 = mp.Queue()

    processes3 = [mp.Process(target=find_shallow_par, args=(D10,lambdas[ng],res1[ng],inter,numerov_grid, output3)) for ng in range(0,len(lambdas)) ]
    #
    for rop in processes3:
        rop.start()
    #   
    res3 = [output3.get() for p in processes3]
    #
    for p in processes3:
        p.join(timeout=1)
    #
    print(res3)

    #
    fig = plt.figure()  
    ax1 = fig.add_subplot(133)
    ax1.set_xlabel(r'$\rho$ [fm]')
    ax1.set_ylabel(r'$V_{eff}(\rho)$')
    [ ax1.plot(numerov_grid,ve) for ve in vv ]
    #
    ax1 = fig.add_subplot(131)
    ax1.set_xlabel(r'$\Lambda$ [fm^-1]')
    ax1.set_ylabel(r'$B(3,\Lambda)$ [MeV]')
    ax1.plot([HH[0] for HH in res3],[HH[1] for HH in res3],'o',label=r'$B_{\min}(3,D_1=$%2.2f MeV)' %D10)
    ax1 = fig.add_subplot(132)
    ax1.set_ylabel(r'$H(\Lambda)$ [MeV]')
    if fitt:
        ax1.plot([HH[0] for HH in res2],[HH[1] for HH in res2],'o',label=r'$B_{\min}(3,D_1=$%2.2f MeV)' %D10)

    #legend = ax1.legend(loc='lower right', fontsize=12)
    #plt.ylim(-80,0)
    #
    plt.show()
    
    exit()
    
