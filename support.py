import numpy as np 
import matplotlib.pyplot as plt 

"""
THIS IS THE SUPPORT FILE FOR THE PYTHON PROJECT MADE BY BENAISSA LOUAI ABDALLAH 
IT SHOULD CONTAIN ALL THE FUNCTIONS AND CLASSES USED IN THE PROJECT.
SORBONNE UNIVERSITE IS FUN !!!
"""
G = 6.67 * 10**(-11)
M_T = 5.97 * 10**24
M_L = 7.34 * 10**22
R_L = 384400 * 10**3
V_L = 1022



class Planet : 
  def __init__(self,m,r0,v0):
    """
    @ m : Mass of the planet in kg.
    @ r0 : Initial position of the planet, a list with shape (1,2), contains the initial position in x and y (in meters).
    @ v0 : Initial velocity of the planet, a list with shape (1,2), contains the initial velocity in x and y (in meters per second).
    """
    self.m = m
    self.r0 = r0
    self.v0 = v0



# Acceleration simple.
a = lambda r,p2 : -G*p2.m/np.linalg.norm(r)**3 * r 


def a2(r1, r2, other_planet):
    """
    Calculates the gravitational acceleration on a body.
    @ r1 : Position of the target body
    @ r2 : Position of the other body
    @ other_planet : The second planet.
    """
    r = r2 - r1
    distance = np.linalg.norm(r)
    if distance == 0:
        return np.array([0, 0])  # Évite la division par zéro
    return G * other_planet.m / distance**3 * r



def RK2(p1,p2,tf,h) :
  """
  This function uses the Runge-Kutta method of order 2 to find the position of a body orbiting another (Earth-Moon or Planet-Sun).
  @ p1 : first planet (moving).
  @ p2 : second planet (fixed).
  @ tf : simulation time.
  @ h : sampling step size.
  """

  n = int(tf/h)
  r = np.zeros((n,2))
  v = np.zeros((n,2))

  r[0,:] = p1.r0 
  v[0,:] = p1.v0

  a = lambda r : -G*p2.m/np.linalg.norm(r)**3 * r 
  for i in range(1,n) : 
    k1_r = v[i-1]
    k1_v = a(r[i-1])

    k2_r = (v[i-1]+h*k1_v)
    k2_v = a(r[i-1]+h*k1_r)

    r[i] = r[i-1] + 0.5*h*(k1_r+k2_r)
    v[i] = v[i-1] + 0.5*h*(k1_v+k2_v)

  return r 

def Euler(p1,p2,tf,h) : 
  """
  This function uses the Euler method to find the position of a body orbiting another (Earth-Moon or Planet-Sun).

  @ p1 : first planet (moving).
  @ p2 : second planet (fixed).
  @ tf : simulation time.
  @ h : sampling step size.
  """
  n = int(tf/h)
  r = np.zeros((n,2))
  v = np.zeros((n,2))
  r[0,:] = p1.r0 
  v[0,:] = p1.v0

  a = lambda r : -G*p2.m/np.linalg.norm(r)**3 * r 

  for i in range(1,n) : 

    v[i] = v[i-1] + h*a(r[i-1])
    r[i] = r[i-1] + h*v[i-1]

  return r  


def RK4(p1,p2,tf,h):
  """
  This function uses the Runge-Kutta method of order 4 to find the position of a body orbiting another (Earth-Moon or Planet-Sun).

  @ p1 : first planet (moving).
  @ p2 : second planet (fixed).
  @ tf : simulation time.
  @ h : sampling step size.
  """

  n = int(tf/h)
  r= np.zeros((n,2))
  v = np.zeros((n,2))
  r[0,:] = p1.r0 
  v[0,:] = p1.v0
  a = lambda r : -G*p2.m/np.linalg.norm(r)**3 * r 
  for i in range(1,n) : 
    k1_r = h* v[i-1]
    k1_v = h* a(r[i-1])

    k2_r = h*(v[i-1]+0.5*k1_v)
    k2_v = h*a(r[i-1]+0.5*k1_r)

    k3_r = h*(v[i-1]+0.5*k2_v)
    k3_v = h*a(r[i-1]+0.5*k2_r)

    k4_r = h*(v[i-1]+k3_v)
    k4_v = h*a(r[i-1]+k3_r)

    r[i] = r[i-1] + 1/6*(k1_r+2*k2_r+2*k3_r+k4_r)
    v[i] = v[i-1] + 1/6*(k1_v+2*k2_v+2*k3_v+k4_v)

  return r   


def DiffSimple(p1,p2,tf,h) : 
  """
  This function uses the finite difference method to find the position of a body orbiting another (Earth-Moon or Planet-Sun).

  @ p1 : first planet (moving).
  @ p2 : second planet (fixed).
  @ tf : simulation time.
  @ h : sampling step size.
  """

  n = int(tf/h)
  r = np.zeros((n,2))
  v = np.zeros((n,2))
  r[0,:] = p1.r0 
  v[0,:] = p1.v0
  a = lambda r : -G*p2.m/np.linalg.norm(r)**3 * r   
  r[1] = r[0] + h*v[0]

  for i in range(2,n) : 
    r[i] = 2*r[i-1]-r[i-2]+h**2*a(r[i-1])
  return r

def ShowMethods(rL_Euler,rL_RK2,rL_RK4,rL_Diff):
  """
  Function to display the results of the 4 methods (Euler, RK2, RK4, and Finite Differences).
  @ rL_Euler : position vector from Euler method.
  @ rL_RK2 : position vector from RK2 method.
  @ rL_RK4 : position vector from RK4 method.
  @ rL_Diff : position vector from finite differences method.
  """

  fig, axs = plt.subplots(2, 2, figsize=(10, 10))


  axs[0, 0].scatter(0, 0, s=50,color='blue')
  axs[0, 0].plot(rL_Euler[:,0], rL_Euler[:,1], label='Euler',color='gray')
  axs[0, 0].set_title("Euler's method")
  axs[0, 0].set_aspect('equal', adjustable='box')

# RK2
  axs[0, 1].scatter(0, 0, s=50,color='blue')
  axs[0, 1].plot(rL_RK2[:,0], rL_RK2[:,1], label='RK2',color='gray')
  axs[0, 1].set_title('Runge-Kotta 2')
  axs[0, 1].set_aspect('equal', adjustable='box')

# RK4
  axs[1, 0].scatter(0, 0, s=50,color='blue')
  axs[1, 0].plot(rL_RK4[:,0], rL_RK4[:,1], label='RK4',color='gray')
  axs[1, 0].set_title('Runge-Kotta 4')
  axs[1, 0].set_aspect('equal', adjustable='box')

# Diff fin
  axs[1, 1].scatter(0, 0, s=50,color='blue')
  axs[1, 1].plot(rL_Diff[:,0], rL_Diff[:,1], label='DiffSimple',color='gray')
  axs[1, 1].set_title('Finite Differences')
  axs[1, 1].set_aspect('equal', adjustable='box')

  for ax in axs.flat:
      ax.set(xlabel='X (meters)', ylabel='Y (meters)')

  plt.tight_layout()
  plt.show()


def DiffReal(p1,p2,tf,h) : 
  """
  This function uses the finite differences method to calculate the positions of two bodies interacting with each other.
  @ p1 : first planet.
  @ p2 : second planet.
  @ tf : simulation time.
  @ h : sampling step.
  """

  n = int(tf/h)

  rL = np.zeros((n,2))
  rT = np.zeros((n,2))
  vL = np.zeros((n,2))
  vT = np.zeros((n,2))

  rL[0,:] = p1.r0
  rT[0,:] = p2.r0

  vL[0,:] = p1.v0
  vT[0,:] = p2.v0


  rL[1] = rL[0] + h*vL[0]
  rT[1] = rT[0] + h*vT[0]

  for i in range(2,n) : 

    rL[i] = 2*rL[i-1]-rL[i-2]+h**2*(a2(rL[i-1],rT[i-1],p2))
    rT[i] = 2*rT[i-1]-rT[i-2]+h**2*(a2(rT[i-1],rL[i-1],p1))

  return rL,rT

def ShowSimulation2(rL_instable,rT_instable,rL_stable,rT_stable):

  """
  This function is used to display the results of the simulation in step 3.
  @ rL_instable : position of the moon in the unstable system.
  @ rT_instable : position of the Earth in the unstable system.

  @ rL_stable : position of the moon in the stable system.
  @ rT_stable : position of the Earth in the stable system.
  """

  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# Les trajectoires divergentes
  axs[0].plot(rL_instable[:,0], rL_instable[:,1], label='Lune',color='gray')
  axs[0].plot(rT_instable[:,0], rT_instable[:,1], label='Terre',color='blue')

  axs[0].set_title("Instable Moon-Earth system")
  axs[0].set_aspect('equal')

# Les trajectoires stables

  axs[1].plot(rL_stable[:,0], rL_stable[:,1], label='Lune',color='gray')
  axs[1].plot(rT_stable[:,0], rT_stable[:,1], label='Terre',color='blue')

  axs[1].set_title("Stable Moon-Earth system")
  axs[1].set_aspect('equal')


# Setting labels for x and y axis
  for ax in axs.flat:
      ax.set(xlabel='X (meters)', ylabel='Y (meters)')

# Adjust layout to prevent overlap
  plt.tight_layout()
  plt.show()

def DiffSimpleAll(planets,tf,h,Sun): 
  """
  This function uses the finite difference method to calculate the positions of planets in a system around a fixed body.
  @ p1 : first planet.
  @ p2 : second planet.
  @ tf : simulation time.
  @ h : sampling step.
  """

  n= int(tf/h)
  positions = np.zeros((len(planets),n,2))

  for i in range(len(planets)) : 
    positions[i] = DiffSimple(planets[i],Sun,tf,h)
  return positions






def ShowSimulation3(Internal,External,rint,rext) : 
  """
  This function is used to display the results of step 3 simulation.
  @ Internal : inner solar system.
  @ External : outer solar system.
  @ rint : positions of the inner system.
  @ rext : positions of the outer system.
  """

  colors1 = [ '#B0B0B0', '#2A52BE', '#D35400', '#EEDFCC'] 
  labels1 = [ "Mercury", "Earth", "Mars", "Venus"]  

  colors2 = ['#F4A460', '#FAD6A5', '#AFDBF5', '#2C75FF']  
  labels2 = [ "Jupiter", "Saturn", "Uranus", "Neptune"]  

  (a, fig) = plt.subplots(1, 2, figsize=(10, 5))

  for ax in fig:
      ax.set_aspect('equal')

  a.supylabel('y (m)')
  a.supxlabel('x (m)')
  a.suptitle('Solar System')

  fig[0].set_title('Internal Solar System')
  fig[0].scatter(0,0,label="Soleil",color='#FFFF00',s=50)
  for i in range(len(Internal)) : 
      fig[0].plot(rint[i,:,0],rint[i,:,1],label=labels1[i],color=colors1[i])

  fig[0].legend(loc='lower right')



  fig[1].set_title('External Solar System')
  fig[1].scatter(0,0,label="Soleil",color='#FFFF00',s=50)
  for i in range(len(External)) : 
      fig[1].plot(rext[i,:,0],rext[i,:,1],label=labels2[i],color=colors2[i])

  fig[1].legend(loc='lower right')

  plt.show()
 



def DiffRealAll(planets,tf,h) :
  """
  This function uses the finite difference method to find the positions of all celestial bodies in our solar system.

  @ planets : list of planets to simulate (always starting with the Sun).
  @ tf : simulation time.
  @ h : sampling step.
  """


  n = int(tf/h)
  r = np.zeros((len(planets),n,2))
  v = np.zeros((len(planets),n,2))
  for i in range(len(planets)):
    r[i,0,:] = planets[i].r0
    v[i,0,:] = planets[i].v0

    r[i,1,:] = r[i,0,:] + h*v[i,0,:]

  for i in range(2,n) : 
    for j in range(len(planets)) : 
    
      atot = 0

      for k in range(len(planets)) :
        if k!=j : 
          atot += a2(r[j,i-1,:],r[k,i-1,:],planets[k])
    
      r[j,i,:] = 2*r[j,i-1,:] - r[j,i-2,:] + h**2 * atot

  return r


def ShowSimulation4(Internal,External,rint,rext) : 
  """
  This function is used to display the results of step 4 simulation.
  @ Internal : inner solar system.
  @ External : outer solar system.
  @ rint : positions of the inner system.
  @ rext : positions of the outer system.
  """

  colors1 = ['#FFFF00', '#B0B0B0', '#2A52BE', '#D35400', '#EEDFCC'] 
  labels1 = ["Sun", "Mercury", "Earth", "Mars", "Venus"]  

  colors2 = ['#FFFF00', '#F4A460', '#FAD6A5', '#AFDBF5', '#2C75FF']  
  labels2 = ["Sun", "Jupiter", "Saturn", "Uranus", "Neptune"]  

# Create subplots
  (a, fig) = plt.subplots(1, 2, figsize=(10, 5))

# Make both subplots square
  for ax in fig:
      ax.set_aspect('equal')

  # Plot data on the first figure with lab[els
  a.supylabel('y (m)')
  a.supxlabel('x (m)')
  a.suptitle('Solar System')

  fig[0].set_title('Internal solar system')
  fig[0].scatter(rint[0,:,0],rint[0,:,1],label=labels1[0],color=colors1[0],s=50)
  for i in range(1,len(Internal)) : 
      fig[0].plot(rint[i,:,0],rint[i,:,1],label=labels1[i],color=colors1[i])

  fig[0].legend(loc='lower right')



  fig[1].set_title('External solar system')
  fig[1].scatter(rext[0,:,0],rext[0,:,1],label=labels2[0],color=colors2[0],s=50)
  for i in range(1,len(External)) : 
      fig[1].plot(rext[i,:,0],rext[i,:,1],label=labels2[i],color=colors2[i])

  fig[1].legend(loc='lower right')

  plt.show()