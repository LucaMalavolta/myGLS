import numpy as np 
import argparse
from astroML.time_series import lomb_scargle
import matplotlib as mpl                                                                                                                                                                                                                                                                         
#mpl.use('Agg') 
import matplotlib.pyplot as plt 

def SineFit(V,Z,W,YY):
    #     fit sine wave y=A*cosx+B*sinx+off
    #     A,B,off - fit parameter
    #      DOUBLE PRECISION v(10000),z(10000),ww(10000),cosx,sinx
    #      COMMON /data/ v,z,ww,YY,N  ! v = phase, z = data, ww = weights
    #common data:      V,Z,W,Y,N
    cosV = np.cos(V,dtype=np.double)
    sinV = np.sin(V,dtype=np.double)
    CC = np.sum(W*cosV**2,dtype=np.double)
    CS = np.sum(W*cosV*sinV,dtype=np.double)
    C  = np.sum(W*cosV,dtype=np.double)
    S  = np.sum(W*sinV,dtype=np.double)
    YC = np.sum(Z*cosV,dtype=np.double)
    YS = np.sum(Z*sinV,dtype=np.double)
    
    SS = 1. - CC
    D  = CC*SS - CS*CS
    powLS = (SS*YC**2/D+CC*YS**2/D-2*CS*YC*YS/D) / YY  # Lomb-Scargle power
    CC = CC - C*C
    SS = SS - S*S
    CS = CS - C*S
    D  = CC*SS - CS*CS
    A = (YC*SS-YS*CS) / D
    B = (YS*CC-YC*CS) / D
    off = -A*C-B*S
    powGLS = (SS*YC**2/D+CC*YS**2/D-2*CS*YC*YS/D) / YY   # GLS power
    return A,B,off,powLS, powGLS

def Spectral_Window(V):
    WC = np.sum(np.cos(V),dtype=np.double)
    WS = np.sum(np.sin(V),dtype=np.double)
    WO = (WC**2+WS**2) / (np.size(V))**2
    return WO

def GLS(X,Y,E,wexp,Freq):
    N = np.size(Freq)
    W = (1./E)**wexp 
    
    WW = W/np.sum(W)
    Ym = np.sum(Y*WW)
    WY = Y-Ym              # centering RVs
    YY = np.sum(WW*WY**2)   # sum for chi2 above the mean
    WY = WY*WW              # attach weights to centered data (saves many multiplications)
    
    # v -> frequency
    # N
    # common data v,wy,ww,YY,N
    twopi = 2. * np.pi
    powSin = 0.
    powGLS = np.zeros(np.size(Freq),dtype=np.double)
    powLS  = np.zeros(np.size(Freq),dtype=np.double)
    SW     = np.zeros(np.size(Freq),dtype=np.double)
    for ii in xrange(0,N):
        V = X * twopi * Freq[ii]
        A, B, C, powLS[ii], powGLS[ii] = SineFit(V,WY,WW,YY)
        SW[ii] = Spectral_Window(V)

        if (powGLS[ii] > powSin):
            powSin = powGLS[ii]
            ph = np.mod(np.arctan2(A,B,dtype=np.double)+twopi, twopi)
            Amp = np.sqrt(A**2+B**2) # = A/sin(ph)
            CBest = Ym+C
            PSin = 1./Freq[ii]
    return powGLS, SW, PSin, ph, Amp, CBest

def GLS_res(X,Y,E,wexp,PSin,ph,Amp,CBest):
    twopi = 2. * np.pi
    N = np.size(X)
    V = X * twopi /PSin
    W = (1./E)**wexp 
    WW = W/np.sum(W)
     
    RVmod = (Amp*np.sin(V+ph)+CBest)
    dRVsin = Y-RVmod
    PhaseSin= (X / PSin) % 1 

    rmssin = np.sum(dRVsin**2)
    chisin = np.sum((dRVsin/E)**2)

    rmssin  = np.sqrt(np.sum(dRVsin**2)/(N-4.)) #! unbiased rms
    wrmssin = np.sqrt(chisin/np.sum(W)*N/(N-4))

    z = np.linspace(0,twopi,1000)
    Xplt =  z/twopi
    Yplt =  Amp*np.sin(z+ph)+CBest
    return PhaseSin, dRVsin, RVmod, rmssin, wrmssin, Xplt, Yplt

parser = argparse.ArgumentParser(prog='PyGLS.py', description='Get GLS periodogram.')
parser.add_argument('input_file', type=str, nargs=1, help='input file')
parser.add_argument('-sk'       , type=np.int, nargs='?', required=False, default=0, help='skip first N rows')
parser.add_argument('-trades'   , type=np.int, nargs='?', required=False, default=False, help='input from TRADES ')
parser.add_argument('-period'     , type=np.double, nargs=2, required=False, default=[0.7, 1000], help='range of periods (in log10)  ')
parser.add_argument('-p'        , type=np.int, nargs='?', required=False, default=False, help='Do plots')
parser.add_argument('-f'        , type=np.int, nargs='?', required=False, default=False, help='Save to file')
parser.add_argument('-i'        , type=np.int, nargs=1, required=False, default=[0], help='Iterate on residuals')
parser.add_argument('-nfreq'    , type=np.int, nargs=1, required=False, default=[0], help='Number of samplings in frequency')
parser.add_argument('-ldiff'    , type=np.double, nargs=2, required=False, default=[0, 0], help='Use the difference of two columns for the RVs (Python notation)')
parser.add_argument('-lin'      , type=np.double, nargs=2, required=False, default=[1, 2], help='Use this columns for the RVs and errors (Python notation)')

args = parser.parse_args()

iterate = args.i[0] + 1

input_data = np.genfromtxt(args.input_file[0],dtype=np.double,skip_header=args.sk)

BJD = input_data[:,0]
RV  = input_data[:,args.lin[0]]
RVe = input_data[:,args.lin[1]]

if np.abs(args.ldiff[0] - args.ldiff[1])>0:
    RV = input_data[:,args.ldiff[0]]- input_data[:,args.ldiff[1]]

if  args.trades != False:
    RV  = input_data[:,1]- input_data[:,4]

if  args.nfreq[0]<1:
  Nfreq = np.asarray((np.amax(BJD)-np.amin(BJD))*40,dtype=np.int64)
else:
  Nfreq=args.nfreq[0]

Period = np.exp(np.linspace(np.log(args.period[0],dtype=np.double), np.log(args.period[1],dtype=np.double), Nfreq))

#Freq = np.linspace(Fbeg,Fend,Fsize)
Freq = 1. / Period
wexp = 2.


print np.sum(RV**2/RVe**2)/(np.size(RV)-1)
print 'STD input RVs ', np.std(RV)


RVin = RV

for ii in xrange(0,iterate):

    powGLS, SW, PSin, ph, Amp, CBest = GLS(BJD[:],RVin[:],RVe[:],wexp,Freq)
    PhaseSin, dRVsin, RVmod, rmssin, wrmssin, Xplt, Yplt = GLS_res(BJD,RVin,RVe,wexp,PSin,ph,Amp,CBest)

    print 'PSin: ',PSin,'   Amp: ', Amp,'   Ph: ',ph , '   RMSsin: ', rmssin	


    if  args.f != False:
	
	file_rad = args.input_file[0][:-4]
	if iterate > 1 and ii > 0:
	    file_rad = file_rad + '_it' + repr(ii)
	
 	fileout = open(file_rad + '_RVfit.dat','w')
	fileout.write('descriptor bjd rv_res,+- rv_in,+- ph_sin rv_mod \n')
	for ii in xrange(0,np.size(BJD)):
	  fileout.write('{0:14f} {1:14f} {2:14f} {3:14f} {4:14f} {5:14f} {6:14f}  \n'
	    .format(BJD[ii],dRVsin[ii],RVe[ii],RVin[ii],RVe[ii],PhaseSin[ii],RVmod[ii]))
	fileout.close()
 
  	fileout = open(file_rad + '_RVmod.dat','w')
	fileout.write('descriptor phase model \n')
	for ii in xrange(0,np.size(Xplt)):
	  fileout.write('{0:14f} {1:14f} \n'.format(Xplt[ii], Yplt[ii]))
	fileout.close()
 
  	fileout = open(file_rad + '_GLS.dat','w')
	fileout.write('descriptor freq period powGLS SW \n')
	for ii in xrange(0,Nfreq):
	  fileout.write('{0:14f} {1:14f} {2:14f} {3:14f} \n'.format(Freq[ii], Period[ii], powGLS[ii], SW[ii]))
	fileout.close()	
 
    if  args.p != False:
	plt.figure(1)
	plt.subplot(411)
	plt.errorbar(PhaseSin, RVin, RVe, fmt='o')
	plt.plot(Xplt, Yplt)
	plt.subplot(412)
	plt.errorbar(PhaseSin, dRVsin, RVe, fmt='o')
	plt.subplot(413)
	plt.plot(Freq,powGLS)
	plt.axvline(1./PSin,c='r')
	plt.subplot(414)
	plt.plot(Freq,SW)
	plt.axvline(1./PSin,c='r')
	plt.show()


    RVin = dRVsin




