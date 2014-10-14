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

import time
import multiprocessing as mp

def info(title):
    print title
    #print 'module name:', __name__
    #if hasattr(os, 'getppid'):  # only available on Unix
    #    print 'parent process:', os.getppid()
    print 'process id:', os.getpid()
# -------------------------------------------------------------------------------------------------
def grid_gen(anz_st,max_rho):
    return np.linspace(0, max_rho, anz_st+1)          # grid in the x-direction

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

def set_eff_pot_coeff(v00,lamm):
    try:
        v0 = v00;
        v_0_ff = V_of_A(lamm,a_ff,v0)
        print 'a_ff (Lambda=%1.1f fm^-1,v=%2.2f fm^-2) = %2.2f fm' %(lamm,v_0_ff,A_of_V(v_0_ff,lamm,0.0)[0])
        v_0_fcp = V_of_A(lamm,a_fcp,v0)
        print 'a_fcp (Lambda=%1.1f fm^-1,v=%2.2f fm^-2) = %2.2f fm' %(lamm,v_0_fcp,A_of_V(v_0_fcp,lamm,0.0)[0])
        v_0_fcm = V_of_A(lamm,a_fcm,v0)
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
    numerov_grid = grid_gen(nn,xmm)
    dx           = float(xmm)/nn            # step over the x-axis
    dx2          = dx * dx                          # dx squared
    lecs = set_eff_pot_coeff(1000.1,lamm)
    lambda_n = []
    lambd0 = 4.0
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
shwo_lam = 0
if shwo_lam:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for n in [5,6]:
        lammm = 4.5
        lambd0  = (4*n-1)**2
        rho_max = 40/int(lam)
        xx = np.arange(0.1,rho_max,float(rho_max)/50.0)
        yy = []
        for rho in xx[::-1]:
            yy.append(optimize.fmin(det_D, lambd0, args=(rho,v_ff,v_fc,lammm), disp=0 ))
            lambd0 = yy[-1]
        yy = yy[::-1]
        xx = [ c*lam for c in xx]
        rho_inf = 20000
        lambd0 = (4*n-1)**2
        labe = r'$\lim_{\rho\to\infty}\lambda(\rho) = $ %4.2f, n=%d' %(float(optimize.fmin(det_D, lambd0, args=(rho_inf,v_ff,v_fc,lammm), disp=0 )),n)
        ax.plot(xx,yy,label=labe)
    
    ax.legend(loc='upper right')
    ax.set_xlabel(r'$\rho\cdot\Lambda$')
    ax.set_ylabel(r'$\lambda(\rho)$')
    plt.show()
    exit()

# -------------------------------------------------------------------------------------------------
# angular-eigenvalue equation: two identical fermions (neutron,neutron) interacting with
# a core (proton)
# 


# -------------------------------------------------------------------------------------------------
# for lim x-> infty the solution to the free Schroedinger equation sets the boundary condition
# i.e., an exponential decay
def asymptotic_boundary(E_val,rho_max): return np.exp(-np.sqrt(2*mn*E_val/MeVfm**2)*rho_max)

# the (effective) interaction with [V_0] = MeV
# for performance reasons the parameterization of the potential
# is hard-wired
def V(x,lam,co): return co*np.exp(-lam*x**2)

def V_eff(x,ro,d1,lll):
    if x<ro:
        return -d1
    else:
        return -(15.0/4.0 + lll)/x**2

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

# we solve (d^2/dx^2 + f(x)) y(x) = 0                           rho    cut off D_1
def f(E,ii,lec_tni,lamm,aev): return 2*mn*E/(MeVfm**2) - V_eff(ii*dx,1.0/lamm,lec_tni,aev)

# Numerov interation up to xm
# This function is minimized to retrieve binding energies
def psi_xm(E_val,lec,lammm,lambdan,grid):
    """Returns the value of \Psi(x_m)."""
    x = grid                    # grid in the x-direction
    n = len(grid)-1
    y = np.zeros(n+1)                   # wave-function in individual points
    # initial conditions
    y[0] = 0
    y[1] = 1.0
    #
    for i in range(1,n):
        y[i + 1] = (2 - 5 * dx2 * f(E_val,i,lec,lammm,lambdan[i]) / 6) * y[i] - (1 + dx2 * f(E_val,i-1,lec,lammm,lambdan[i-1]) / 12) * y[i - 1]
        y[i + 1] /= (1 + dx2 * f(E_val,i+1,lec,lammm,lambdan[i+1]) / 12)
    return y[n]-asymptotic_boundary(-E_val,x[-1])

def find_shallow_ser(de1,lambd,emaxi,ang_ev,grid):
    xend   = 0.0
    inc    = 1.0
    const = True
    x0    = 0.0
    signum = np.sign(psi_xm(x0,de1,lambd,ang_ev,grid))
    while (const):
        x0 = x0 - inc
        signum_1 = np.sign(psi_xm(x0,de1,lambd,ang_ev,grid))
        print psi_xm(x0,de1,lambd,ang_ev,grid),x0,de1,lambd,ang_ev[0],asymptotic_boundary(-x0,grid[-1])
        if (signum_1<>signum)|(abs(x0)>emaxi):
            const = False
            xend = x0
    if abs(xend)<emaxi:
        E_1 = brentq(psi_xm, x0+inc, xend,args=(de1,lambd,ang_ev,grid))
        print 'E = %4.4f MeV' %E_1
        return E_1+b3

def find_shallow_par(de1,lambd,ang_ev,xende,grid,output):
    #info('find_shallow_par')
    xend   = 0.0
    inc    = 1.0
    const = True
    x0    = 0.0
    signum = np.sign(psi_xm(x0,de1,lambd,ang_ev,grid))
    while (const):
        x0 = x0 - inc
        signum_1 = np.sign(psi_xm(x0,de1,lambd,ang_ev,grid))
        if (signum_1<>signum)|(abs(x0)>xende):
            const = False
            xend = x0
    if abs(xend)<xende:
        E_1 = brentq(psi_xm, x0+inc, xend,args=(de1,lambd,ang_ev,grid))
        output.put([lambd,E_1])
    else:
        print 'no bound state found for (Lambd, D_1) = (%2.2f,%2.2f) below E_max = %4.2f' %(lambd, de1, xende)
        output.put([lambd,0.0])

def fit_shallowest(lamb,D10,ang_ev,output):
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
        emb_p = find_shallow_ser(lec,lamb,ang_ev)
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
        D_1 = brentq(find_shallow_ser, lec-2.0*lec_inc, lec, args=(lamb,ang_ev))
        print '%4.4f  , %12.4f' %(lamb,D_1)
        output.put([lamb,D_1])

if __name__ == '__main__':
    b3 = 0.0
    stue         = 5000                             # how many points we use for integration
    xmax         = 90                               # at which value of x we stop
    dx           = float(xmax)/stue                    # step over the x-axis
    dx2          = dx * dx                          # dx squared
    numerov_grid = np.linspace(0, xmax, stue+1)    

    output1 = mp.Queue()

    L0  = 4.0
    LE  = 4.5
    nL  = 2
    dL  = (LE-L0)/nL
    lambdas = [ float(L0 + ll*dL) for ll in range(0,nL)]
    print lambdas
    processes1 = [mp.Process(target=calc_angular_ev, args=(lmbd, stue, xmax, output1)) for lmbd in lambdas]
    for p in processes1:
        p.start()
       
    res1 = [output1.get() for p in processes1]

    # joining dead locks the queue if the returned data exceeds a limit, see https://docs.python.org/2/library/multiprocessing.html#multiprocessing-programming
    for p in processes1:
        p.join()

    fig = plt.figure()  
    ax1 = fig.add_subplot(111)
    for hh in res1:
        ax1.plot(numerov_grid,hh)        
    plt.show()
    for worker in processes1:
        assert not worker.is_alive()

    b3   = 10.0
    emax = 1200
    D10  = 110.0
    #                      de1,lambd,emaxi,ang_ev,grid
    tmp = find_shallow_ser(D10,lambdas[0],120,res1[0],numerov_grid)
    exit()


    output2 = mp.Queue()
    vv =  [ V_eff(xxx,1./lambdas[0],D10,res1[0][int(xxx/dx)]) for xxx in numerov_grid ]
    #print vv
    #processes = [mp.Process(target=fit_shallowest, args=(lambd, D10, output)) for lambd in lambdas]
#    for lmbdnr in range(0,len(lambdas)):
#        find_shallow_ser(D10,lambdas[lmbdnr], res1[lmbdnr])

    processes2 = [mp.Process(target=find_shallow_par, args=(D10,lambdas[lmbdnr], res1[lmbdnr], emax, output2)) for lmbdnr in range(0,len(lambdas))]

    for rop in processes2:
        rop.start()
        
    res2 = [output2.get() for p in processes2]

    for p in processes2:
        p.join(timeout=1)

    print(res2)
#
    fig = plt.figure()  
    ax1 = fig.add_subplot(122)
    ax1.set_xlabel(r'$\rho$ [fm]')
    ax1.set_ylabel(r'$V_{eff}(\rho)$')
    #ax1 = fig.add_subplot(122)
    ax1.plot(numerov_grid,vv)
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel(r'$\Lambda$ [fm$^{-1}$]')
    ax1.set_ylabel(r'$H(\Lambda)$ [MeV]')
    ax1.set_ylabel(r'$B(3,\Lambda)$ [MeV]')
    #ax1 = fig.add_subplot(122)
    ax1.plot([LL[0] for LL in res2],[HH[1] for HH in res2],'ro',label=r'$B_{\min}(3,D_1=$%2.2f MeV)' %D10)
    legend = ax1.legend(loc='lower right', fontsize=12)
    #plt.ylim(-80,0)
    #
    plt.show()
    
    exit()
    
