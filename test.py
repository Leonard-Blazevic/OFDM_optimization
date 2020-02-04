import numpy as np
import matlab.engine 

# Start matlab engine
eng = matlab.engine.start_matlab()

# Define basic constants for testing acf and zzb
c0                  = 3e8
fc1                 = 1.5e9
fsc                 = 100e3
Tperiod             = 1/fsc
snr_ratio           = 0.01
Tobs                = Tperiod/2

#---------------------Define the test signal------------------#
deltaT_vector       = np.array(np.arange(-Tperiod, Tperiod + Tperiod/10, Tperiod/10))  # za razliku od matlaba, zadnji element nije Tperiod nego onaj prije njega
powerSpectrum       = np.array([1,1,1,1,1,1,1,1,1,1])
freqBins            = np.array(np.arange(0,10,1)*fsc + fc1)
#-------------------------------------------------------------#

#--------------Convert from numpy to matlab type--------------#
matlab_deltaT_vector = matlab.double(deltaT_vector.tolist())
matlab_freqBins = matlab.double(freqBins.tolist())
matlab_powerSpectrum = matlab.double(powerSpectrum.tolist())
#-------------------------------------------------------------#

#------------------------Test ACF and ZZB---------------------#
acf_vec_output = eng.acf(matlab_deltaT_vector, matlab_freqBins, matlab_powerSpectrum, nargout=1)
print(acf_vec_output)

zzb_vec_output = eng.zzb(float(snr_ratio), matlab_freqBins, matlab_powerSpectrum, float(Tobs), nargout=1)
print(zzb_vec_output)
#-------------------------------------------------------------#

# Quit matlab engine
eng.quit()

# To validate, output should be as follows:
# ----------------------------------------------------------------------------------------------------
# [[(1-6.24484455439e-12j),(-6.19349016517e-13+4.01095823221e-12j),
# (-1.83792980835e-12+9.91828841279e-13j),(-1.41979761281e-12-1.29974919716e-12j),
# (8.67084182232e-15+1.15060183603e-12j),3.47038894079e-13j,(2.79343215226e-13+1.3209433547e-13j),
# (2.85832468805e-13-5.62772051182e-14j),(2.45647946429e-13+1.78468351208e-13j),
# (3.5571545709e-14+4.18265422297e-13j),(1+4.79129542954e-11j),(-5.26245713672e-15-3.62065932791e-13j),
# (4.59482452086e-13-2.4794610809e-13j),(7.20035142621e-14+3.50597328946e-13j),
# (-8.40294500648e-13-4.9592552287e-13j),-3.47038894079e-13j,(8.68194405257e-15-4.23006074612e-13j),
# (8.19533330088e-13-3.27204929818e-13j),(-1.83795756392e-12+4.63351579327e-13j),
# (-1.47467593692e-12-2.83369994136e-12j),(1+1.00832293539e-10j)]]

# Warning: Reached the limit on the maximum number of intervals in use. Approximate bound on error is   
# 3.5e-21. The integral may not exist, or it may be difficult to approximate numerically to the requested accuracy.
# > In integralCalc/iterateArrayValued (line 282)
#   In integralCalc/vadapt (line 130)
#   In integralCalc (line 75)
#   In integral (line 88)
#   In zzb (line 9)
# 1.91785204313e-12
# ----------------------------------------------------------------------------------------------------
