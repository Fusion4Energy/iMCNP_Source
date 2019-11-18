'''
########################################################################################################
# Copyright 2019 F4E | European Joint Undertaking for ITER and the Development                         #
# of Fusion Energy (‘Fusion for Energy’). Licensed under the EUPL, Version 1.1                         #
# or - as soon they will be approved by the European Commission - subsequent versions                  #
# of the EUPL (the “Licence”). You may not use this work except in compliance                          #
# with the Licence. You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl.html       #
# Unless required by applicable law or agreed to in writing, software distributed                      #
# under the Licence is distributed on an “AS IS” basis, WITHOUT WARRANTIES                             #
# OR CONDITIONS OF ANY KIND, either express or implied. See the Licence permissions                    #
# and limitations under the Licence.                                                                   #
########################################################################################################
'''

# Module for handling MCNP surfaces (far from complete)
# ==============================================================
# UNED-TECF3IR
# Code author : Francisco Ogando (fogando@ind.uned.es)
# Version date: January 19th, 2017
# ==============================================================
import math
import numpy as np
from numpy import linalg as LA

# =============================
# Norma de un vector 3D
def vnorm(x):
  return math.sqrt(vdot(x,x))

# Producto escalar
def vdot(x,y):
  return x[0]*y[0]+x[1]*y[1]+x[2]*y[2]

# Producto con escalar
def vsprod(s,x):
  return [s*x[0],s*x[1],s*x[2]]

# Producto vectorial
def vcross(x,y):
  return [ x[1]*y[2]-x[2]*y[1], -x[0]*y[2]+x[2]*y[0], x[0]*y[1]-x[1]*y[0] ]

# Definiciones
#   cilindro (infinito): p0(3),v(3),r
#   plano: a,b,c,-d
class GeoClass:
  cart='XYZ'
  # Tolerancia para igualdad de clases
  def __init__(self,t,x):
    self.orgData = x[:]
    self.orgTipo= t.upper()
    lx = list(map(float,x))
    lt = t.upper()
    if lt=='P':
      self.tipo='P'
      self.datos=lx[:]
    elif lt[0]=='P': # PX, PY, PZ
      self.tipo='P'
      self.datos=[0.]*4
      icoor=self.cart.index(lt[1])
      self.datos[icoor] = 1.
      self.datos[3] = lx[0]
    elif lt[0]=='C':
      self.tipo='C'
      self.datos=[0.]*7
      # CX, CY, CZ
      if lt[1] in self.cart:
        icoor = self.cart.index(lt[1])
        self.datos[6] = lx[0]
      elif lt[1]=='/':
        icoor = self.cart.index(lt[2])
        # Coordenada de un punto del eje
        j = 0
        for i in range(3):
          if i==icoor: continue
          self.datos[i]=lx[j]
          j += 1
        # Radio
        self.datos[6] = lx[2]
      # Vector director
      self.datos[icoor+3] = 1.
    elif lt=='GQ':
      self.tipo='C'
      self.datos = gq2cyl(lx)
    else:
      self.tipo=lt
      self.datos=lx[:]
      return
    self.cleanSmall()
    return

  # Quita valores de coordenadas muy pequenos
  def cleanSmall(self):
    minTol=1.e-4
    for j in (0,3):
      s = sum(map(abs,self.datos[j:j+3]))
      for i in range(3):
        if abs(self.datos[i+j])<s*minTol: self.datos[i+j]=0
      if self.tipo=='P': return  # solo hace uno
    return

  # Igualdad entre formas geometricas
  def __eq__(self,othe):
    eqTol=1.e-3
    if self.tipo != othe.tipo: return False
    rv = False
    if self.tipo=='P':
      v1 = self.datos[:3]
      n1 = vnorm(v1)
      d1 = self.datos[3]
      v2 = othe.datos[:3]
      n2 = vnorm(v2)
      d2 = othe.datos[3]
      # Planos paralelos
      if ( 1.-abs(vdot(v1,v2))/n1/n2 < eqTol*eqTol ):
        # Puntos donde corta una perpendicular desde el origen
        vd = [ d1*x/(n1*n1) - d2*y/(n2*n2) for x,y in zip(v1,v2) ]
        rv = ( vnorm(vd) < eqTol )
    elif self.tipo=='C':
      x01 = self.datos[:3]
      v1  = self.datos[3:6]
      n1  = vnorm(v1)
      R1  = self.datos[6]
      x02 = othe.datos[:3]
      v2  = othe.datos[3:6]
      n2  = vnorm(v2)
      R2  = othe.datos[6]
      # Ejes paralelos
      if ( 1.-abs(vdot(v1,v2)/n1/n2) < eqTol*eqTol ):
        # Distancia entre ejes
        rv = ( vnorm(vcross([ x-y for x,y in zip(x01,x02) ],v2))/n2 < eqTol )
        # Radio
        rv &= ( abs(R1-R2) < eqTol )
      # else false 
    return rv

  # Igualdad total en la definicion
  def totalEq(self,othe,fac):
    return (self.tipo==othe.tipo) & \
            all( (float(x)==fac*float(y) for x,y in zip(self.orgData,othe.orgData) ) )


# Conversion de GQ a Cyl
# Ax2+By2+Cz2+Dxy+Eyz+Fxz+Gx+Hy+Jz+K=0
def gq2cyl(x):
  minWTol=5.e-2
  minRTol=1.e-3
  m = np.array( [[x[0],x[3]/2,x[5]/2], \
                 [x[3]/2,x[1],x[4]/2], \
                 [x[5]/2,x[4]/2,x[2]]] )
  w,P = LA.eigh(m)
  lw = list(w)
  sw = sorted(lw)
  if abs(sw[0])>minWTol*sum(sw):
    salir('Esto no es un cilindro: '+str(x)+'\n'+str(w))
  if abs(sw[2]-sw[1])>minRTol*sum(sw):
    salir('Este cilindro no es redondo: '+str(x))
  iaxis=lw.index(sw[0])
  rv = [0.]*7
  # Vector de desplazamiento
  # x0 = -0.5*Pt*D-1*P*b pero ojo que un lambda es cero
  # P es la matriz de valores propios
  b = np.array(x[6:9])
  Pb= np.dot(P,b)
  for i in range(3):
    if i!=iaxis: Pb[i] /= w[i]
  x0 = -0.5*np.dot(P.T,Pb)
  k2 = -0.5*np.dot(x0,b)-x[9]
  # Resultados finales
  rv[0:3] = x0                     # Punto del eje
  rv[3:6] = P[:,iaxis]             # Vector director
  rv[6]   = math.sqrt(k2/sw[1])    # Radio
  return rv
