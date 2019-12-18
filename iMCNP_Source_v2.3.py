'''
########################################################################################################
# Copyright 2019 F4E | European Joint Undertaking for ITER and the Development                         #
# of Fusion Energy (‘Fusion for Energy’). Licensed under the EUPL, Version 1.2                         #
# or - as soon they will be approved by the European Commission - subsequent versions                  #
# of the EUPL (the “Licence”). You may not use this work except in compliance                          #
# with the Licence. You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl.html       #
# Unless required by applicable law or agreed to in writing, software distributed                      #
# under the Licence is distributed on an “AS IS” basis, WITHOUT WARRANTIES                             #
# OR CONDITIONS OF ANY KIND, either express or implied. See the Licence permissions                    #
# and limitations under the Licence.                                                                   #
########################################################################################################
'''

##----------------------------------------------------------------------------##
##                         ********    **   ********                          ##
##                         ********   ***   ********                          ##
##                         **        ** *   **                                ##
##                         ******   ******  ******                            ##
##                         ******   ******  ******                            ##
##                         **          **   ***                               ##
##                         **          **   ********                          ##
##                         **          **   ********                          ##
##                                                                            ##
##----------------------------------------------------------------------------##
##                       TSS / A&C / Nuclear Section                          ##
##----------------------------------------------------------------------------##
##                                                                            ##
##                         Fusion for Energy                                  ##
##                         c/ Josep Pla, n2                                   ##
##                      Torres Diagonal Litoral B3                            ##
##                         Barcelona (Spain)                                  ##
##                          +34 93 320 1800                                   ##
##                    http://fusionforenergy.europa.eu/                       ##
##                                                                            ##
##----------------------------------------------------------------------------##
 
    # CODE: iMCNP_Source
 
    # LANGUAGE: PYTHON 2.7
	
    # AUTHOR/S: Francisco Ogando, Marco Fabbri
 
    # e-MAIL/S: fogando@ind.uned.es, marco.fabbri@f4e.europa.eu
 
    # DATE: 17/12/2019

    # Copyright F4E 2019
 
    # IDM: F4E_D_2F96Z4  
 
    # DESCRIPTION: This script creates a complete MCNP sdef card for a set of pipes containing activated water.
	#              The activation levels may be set homogeneously or according to a set of cell tag labels.
	#              The SDEF can be generated as a set of cylinders or lines.
	#              The SDEF distribution has been optimized.
	#              Please check the USER PARAMETERS section for the detailed source configuration
	
	# USAGE:      python iMCNP_Source.py IN_FILE
	
	# OUTPUT:     IN_FILE_[SDEF+TYPE] --> text file containing SDEF card to be copy pasted in the MCNP file.

	# VERSIONS: 
	#            1.0 [2017-01-19]  ---> Developed by Francisco Ogando (UNED) under the F4E EXP-238,https://idm.f4e.europa.eu/?uid=27S8RF
	#                                   Starting version.
	#            2.0 [2019-06-18]  ---> Improved by Marco Fabbri (F4E) for further usage. Changes implemented:
	#                                   1) SD distribution optimized. No repetitions are present. EXT and RAD values are unique.
	#   							    2) User has now the possibility to create SDEF as CYLINDERS, SURFACE SOURCE or LINES.
    #                                      Surface source might be usefull for ACP.
    #            2.1 [2019-07-19]  ---> Debugged by Alvaro Cubi. Changes implemented: 
    #                                   1) Variable name properly referenced in line 226. 
    #                                   2) Added CX, CY, CZ in the recognized surface definitions.
    #                                   3) Added the capacity to work with hollow pipes.    
	#            2.2 [2019-11-18]  ---> 1) EUPL License statement added in the memo
    #                                   2) EUPL License statement added in the script   
    #            2.3 [2019-12-17]  ---> 1) EUPL License version updated from v.1.1 to 1.2 both in the memo and in the script
    #                                   2) Order of authors corrected as Francisco Ogando has developped most of the code
	
    # IMPROVEMENTS:   
	#               --> 
	#				--> 
   
# ===========================     FUNCTIONS        =============================
# 
import math, sys
import numpy as np
from geoMod import GeoClass

# Escribe una etiqueta y despues un vector de datos
# con el formato elegido
def writeMcnpArray(lab,fmt,arr):
  maxLen=80
  locStr = '{0:5s}'.format(lab)
  slen = len(locStr)
  if isinstance(arr[0],tuple):
    lf = lambda x:fmt.format(*x)
  else:
    lf = lambda x:fmt.format(x)
  for val in map(lf,arr):
    sval = len(val)
    if slen+sval > maxLen:
      locStr += '\n' + ' '*5
      slen = 5
    locStr += val 
    slen += sval
  return locStr+'\n'
  
# Shows error and 
def salir(r):
  raise SystemExit("ERROR:\n"+r)

#  Finds the lam value of the intersection with plane,
#  in the line parametric equation
# x = x0 + lam*v
# a*x - d = 0
def cutLinePlane(cy,pl):
  x0 = np.array(cy.datos[0:3])
  v  = np.array(cy.datos[3:6])
  a  = np.array(pl.datos[0:3])
  return (pl.datos[3]-np.dot(a,x0))/np.dot(a,v)

# Reads file lines until blank
def skipToBlank(g):
  s = ''
  while s!='\n': s = g.readline()
  return

# Skip file lines
def skipLines(f,n=1):
  while n>0:
    f.readline()
    n -= 1
  return

# Round-off to 2 decimals
def prec2(x):
  r = 100.
  return int(r*x)/r
# ====================   END OF FUNCTIONS ======================================

# ==================     USER PARAMETERS =======================================
# 
defaultActivity  = 8.30e9  # Bq/cm3
gammaPerDis = 0.749578
isTagged = True
actDict  = { 'y_pos_water_top':7.5679e9, 'y_pos_water_bottom':0., \
             'y_neg_water_top':7.5679e9, 'y_neg_water_bottom':0., \
             'c_pos_water_top':7.5679e9, 'c_pos_water_bottom':0., \
             'c_neg_water_top':7.5679e9, 'c_neg_water_bottom':0. }
# Energy spectrum of emitted radiation (MCNP format for SI6 ending in EOL)
ergSpectrum='''C . . Spectrum for N-16
si4 L 0.9865 1.755 1.9548 2.7415 2.8225 6.0482 6.1292 6.9155 7.1151 8.8692
sp4   0.0035 0.14 0.04 0.84 0.0013 0.013 68.8 0.04 5.0 0.08
'''
# Definition of the type of source:
# cyl  --> cylinders
# line --> line
# acp  --> distribution on the surface to simulate activate corrosion product deposits.
source = 'cyl'
# 
# ======================  END OF USER PARAMETERS ===============================


# ====================    PROGRAM PARAMETERS   =================================
# Program parameters
# MCNP comment marks
commStr = 'Cc'
# Number of parameters in surface definition
nparDict={'P':4,'GQ':10,'PX':1,'PY':1,'PZ':1,'C/X':3,'C/Y':3,'C/Z':3,'CX':1,'CY':1,'CZ':1}
# Initial distribution number
ndis0   = 10
# =====================  END OF PROGRAM PARAMETERS =============================


# =========================    ROUTINE      ====================================
try:
  f = open(sys.argv[1],'rt')
except IndexError:
  salir("Missing argument: input file")
except IOError:
  salir("Nonexisting input file")

surfDB = dict()
# First read surface definitions
skipToBlank(f)

while True:
  line = f.readline()
  if line=='' or line=='\n': break
  if line[0] in commStr: continue
  if line[0]==' ':
    print 'Unexpected blank'
    break
  vals = line.split()
  ns = int(vals[0])   # surface number
  tp = vals[1]        # surface type
  prs= list(map(float,vals[2:]))
  # In case of missing parameters, map(float) below will raise exception
  while nparDict[tp]>len(prs):
    line = f.readline()
    if line[:5]!=' '*5: salir('Error in surface definition')
    prs.extend(map(float,line.split()))
  # Records the surface
  surfDB[ns] = GeoClass(tp,prs)

# Cell processing after file rewind
# cell format in following format
# 7     0   ( 1827 -619 -1543)     
#       IMP:N=1.0  IMP:P=1.0  IMP:E=0.0 TMP=2.53005e-008   
#       $c_neg_water_bottom
f.seek(0)

cels = list()
poss = list()
axss = list()
exts = list()
rads = list()
wgts = list()

skipLines(f) # skip title
while True:
  line = f.readline()
  if line=='\n': break
  if line=='': salir('Unexpected file end')
  if line[0] in commStr: continue
  # Process cell parameters
  vals = line.split()
  try:
    ncell = int(vals.pop(0))
  except ValueError:
    salir('Wrong cell number:'+'\n'+line)
  if vals.pop(0)!='0': vals.pop(0)
  # Surface definitions may appear between brakets
  if vals[0]=='(': vals.pop(0)
  if vals[0][0]=='(': vals[0] = vals[0][1:]
  if vals[-1][-1]==')': vals[-1]= vals[-1][:-1]
  try:
    if len(vals)==3:
        vals = (int(vals[0]),int(vals[1]),int(vals[2]))
    elif len(vals)==4: # In case of hollow pipe
        vals = (int(vals[0]),int(vals[1]),int(vals[2]),int(vals[3]))
  except ValueError:
    salir('Wrong character in cell definition'+'\n'+line)
  # skip definition of importances
  skipLines(f)
  if isTagged:
    line = f.readline()  # tag
    tag  = line[line.index('$')+1:-1]
    act  = actDict[tag]
  else:
    act = defaultActivity
  # Inactive pipe does not contribute to source
  if act==0.: continue
  # Records one cylinder and two planes
  # unrecorded surface raises exception
  prs = list()
  cyl = None
  try:
    for x in vals:
      x = abs(x)
      if surfDB[x].tipo=='C':
        if not cyl:
            cyl = surfDB[x]
        if cyl and surfDB[x].datos[6]<cyl.datos[6]: # If hollow pipe, get the cyl with smaller radius
            cyl = surfDB[x]
      else:
        prs.append(surfDB[x])
  except KeyError:
    salir("Unrecorded surface in cell definition: {0:d}".format(ncell))
  # Find intersection between cylinder axis and planes
  L0 = cutLinePlane(cyl,prs[0])
  L1 = cutLinePlane(cyl,prs[1])
  Lmin = min(L0,L1)
  x0 = np.array(cyl.datos[0:3])
  v  = np.array(cyl.datos[3:6])
  R  = cyl.datos[6]
  H  = abs(L0-L1)
  vol = math.pi*R**2*H
  cels.append(ncell)
  poss.append(tuple(x0+Lmin*v))
  axss.append(tuple(v))
  exts.append(prec2(H))  # Limit precision to enable reusing of distributions
  rads.append(prec2(R))
  wgts.append(vol*act)

# done reading mcnp input
f.close()
# Radial values without repetitions
radSI = list(set(rads))

# if len(exts)>len(list(set(exts))): print "EXT could be further simplified!!!"

# Ext values without repetitions
extSI = list(sorted(set(exts)))

if source == 'line': # Source as lines (Degenerate cylinders)
	# Write output
	outfile = sys.argv[1] + '_[SDEF-LINE]'
	f = open(outfile,'wt')
	f.write('C . . Total gamma source: {0:12.5e} g/s\n'.format(sum(wgts)*gammaPerDis))
	f.write('sdef par=p pos=d1 axs=fpos=d2 ext=fpos=d3 erg=d4\n')
	f.write(writeMcnpArray('si1 L ','{0:7.1f} {1:7.1f} {2:7.1f}  ',poss))
	f.write(writeMcnpArray('sp1','{0:12.5e} ',wgts))
	f.write(writeMcnpArray('ds2 L ','{0:7.4f} {1:7.4f} {2:7.4f}  ',axss))
		
	# SI EXT
	ndis  = ndis0
	
	extDS = []
	for item in exts:
		extDS.append(ndis+extSI.index(item))
	
	f.write(writeMcnpArray('ds3 s ','{0:d} ',extDS))
	
	# SI ERG: N-16
	f.write(ergSpectrum)
	# Distribuciones EXT
	ndis = ndis0
	f.write('C . . Distributions EXT\n')
	for x in extSI:
	  f.write('si{0:d} h 0 {1:.2f}\n'.format(ndis,x))
	  ndis += 1
elif source == 'cyl': # Source as Cylinders
	# Write output
	outfile = sys.argv[1] + '_[SDEF-CYL]'
	f = open(outfile,'wt')
	f.write('C . . Total gamma source: {0:12.5e} g/s\n'.format(sum(wgts)*gammaPerDis))
	f.write('sdef par=p pos=d1 axs=fpos=d2 ext=fpos=d3 erg=d4 rad=fpos=d5\n')
	f.write(writeMcnpArray('si1 L ','{0:7.1f} {1:7.1f} {2:7.1f}  ',poss))
	f.write(writeMcnpArray('sp1','{0:12.5e} ',wgts))
	f.write(writeMcnpArray('ds2 L ','{0:7.4f} {1:7.4f} {2:7.4f}  ',axss))
	# SI EXT
	ndis  = ndis0
	
	extDS = []
	for item in exts:
		extDS.append(ndis+extSI.index(item))
	
	f.write(writeMcnpArray('ds3 s ','{0:d} ',extDS))
	# SI RAD
	ndis = max(extDS)+1
		
	radDS = [ radSI.index(x)+ndis for x in rads ]
	f.write(writeMcnpArray('ds5 s ','{0:d} ',radDS))
	# SI ERG: N-16
	f.write(ergSpectrum)
	# Distribuciones EXT
	ndis = ndis0
	f.write('C . . Distributions EXT\n')
	for x in extSI:
	  f.write('si{0:d} h 0 {1:.2f}\n'.format(ndis,x))
	  ndis += 1
	# Distribuciones RAD (w/o repetitions)
	f.write('C . . Distribuciones RAD\n')

	for x in radSI:
	  f.write('si{0:d} h 0 {1:.2f}\nsp{0:d}   -21 1\n'.format(ndis,x))
	  ndis += 1

elif source == 'acp': # Source as surface source
	# Write output
	outfile = sys.argv[1] + '_[SDEF-ACP]'
	f = open(outfile,'wt')
	f.write('C . . Total gamma source: {0:12.5e} g/s\n'.format(sum(wgts)*gammaPerDis))
	f.write('sdef par=p pos=d1 axs=fpos=d2 ext=fpos=d3 erg=d4 rad=fpos=d5\n')
	f.write(writeMcnpArray('si1 L ','{0:7.1f} {1:7.1f} {2:7.1f}  ',poss))
	f.write(writeMcnpArray('sp1','{0:12.5e} ',wgts))
	f.write(writeMcnpArray('ds2 L ','{0:7.4f} {1:7.4f} {2:7.4f}  ',axss))
	# SI EXT
	ndis  = ndis0
	
	extDS = []
	for item in exts:
		extDS.append(ndis+extSI.index(item))
	
	f.write(writeMcnpArray('ds3 s ','{0:d} ',extDS))
	# SI RAD
	ndis = max(extDS)+1
		
	radDS = [ radSI.index(x)+ndis for x in rads ]
	f.write(writeMcnpArray('ds5 s ','{0:d} ',radDS))
	# SI ERG: N-16
	f.write(ergSpectrum)
	# Distribuciones EXT
	ndis = ndis0
	f.write('C . . Distributions EXT\n')
	for x in extSI:
	  f.write('si{0:d} h 0 {1:.2f}\n'.format(ndis,x))
	  ndis += 1
	# Distribuciones RAD (w/o repetitions)
	f.write('C . . Distribuciones RAD\n')
	
	depACP = 1e-6
	layACP = 0.1
	
	for x in radSI:
	  f.write('si{0:d} h 0 {1:.2f}  {2:.2f}\n'.format(ndis,x-layACP,x))
	  f.write('sp{0:d}   0 {1:.1e}  1\n'.format(ndis,depACP))
	  ndis += 1
# Cleanup
f.close()
# =====================  END OF ROUTINE ========================================