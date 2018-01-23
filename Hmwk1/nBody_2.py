# Kathryn Neguent
# Homework 1

import numpy as np
import matplotlib.pyplot as plt

# initialize paramters
# number of particles
N = 100
# box size (m)
L = 1000
# steps to iterate over
steps = 100
# time (s)
t = 5
# mass (kg)
m1 = 1
m2 = 10
# radius (m)
r = .02*L
# Kinetic Energy (J)
KEstart = 10

# Initialize particle properties within grid
x = np.zeros(N)
y = np.zeros(N)
m = np.zeros(N)
v = np.zeros(N)
vx = np.zeros(N)
vy = np.zeros(N)
KE = np.ones(N)*KEstart

# function definitions
# move particles
def move(x,y,vx,vy):
    new_x = x + vx
    new_y = y + vy
    return new_x,new_y

# get distance between two points
def distance(x1,x2,y1,y2):
    return np.sqrt(((x1-x2)**2) + ((y1-y2)**2))

# plot scatter plot of positions
def plot_position(x,y,title,xlab,ylab):
    plt.scatter(x,y)
    plt.title("Position Plot {}".format(title))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig("position_{}.eps".format(title))
    plt.close()

# plot histogram of velocities
def hist_velocity(v,title,xlab,ylab,bin):
    v[np.isnan(v)] = 0
    plt.hist(v,bins=bin)
    plt.title("{} Histogram".format(title))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig("{}_histogram.eps".format(title))
    plt.close()

# randomly assigns unique positions and velocities grid in (x,y) coord system
for i in range(N):
    # floats between 0 and box size (L)
    x[i] = float(np.random.randint(L))
    y[i] = float(np.random.randint(L))
    # allows for a range of masses
#    m[i] = np.random.choice([m1,m2])
    m[i] = 1
    # determine velocities (x and y can be positive or negative)
    v[i] = np.sqrt((2*KE[i]/m[i]))
    vx[i] = np.random.choice(int(v[i]))*np.random.choice([-1.0,1.0])
    vy[i] = np.sqrt(v[i]**2-vx[i]**2)

# plotting initial state
plot_position(x,y,"Before - 2mass","x (m)","y (m)")
hist_velocity(vx,"X Velocity Before - 2mass","velocity (m/s)","N",2)
hist_velocity(vy,"Y Velocity Before - 2mass","velocity (m/s)","N",2)
hist_velocity(v,"Velocity Before - 2mass","velocity (m/s)","N",2)
hist_velocity(KE,"KE Before - 2mass","energy (J)","N",2)

# iterate through time steps
for t in range(0,steps*t):
    # iterate through particles
    for i in range(N):
        # first check and see if particle is near a wall 
        moved=False
        # now that they are within the box, check for collisions
        for j in range(N):
            if j != i:
                # if two particles are within the collision cross-section, they need to bounce
                if distance(x[i],x[j],y[i],y[j]) <= (2.0*r):
                    moved=True
                    # update velocities for particles
                    
                    # calculating impulse
                    J_numerator = (2.0*m[i]*m[j]*((vx[j]-vx[i])*(x[j]-x[i]) + (vy[j]-vy[i])*(y[j]-y[i]))) 
                    J_denominator = (distance(x[i],x[j],y[i],y[j])*(m[i]+m[j]))

                    J = J_numerator / J_denominator
                    Jx = J*(x[j]-x[i]) / distance(x[i],x[j],y[i],y[j])
                    Jy = J*(y[j]-y[i]) / distance(x[i],x[j],y[i],y[j])

                    # calculating the velocity components for both particles
                    vx[i] = vx[i] + Jx/m[i] 
                    vy[i] = vy[i] + Jy/m[i] 

                    vx[j] = vx[j] - Jx/m[j]
                    vy[j] = vy[j] - Jy/m[j]

                    # move to new x and y positions
                    x[i],y[i] =  move(x[i],y[i],vx[i],vy[i])
                    x[j],y[j] =  move(x[j],y[j],vx[j],vy[j])

                    # calculating final velocity
                    v[i] = np.sqrt(vx[i]**2 + vy[i]**2 )
                    KE[i] = 0.5*m[i]*v[i]**2
            if not moved:
                # if they didn't collide, then just move particle 1
                x[i],y[i] =  move(x[i],y[i],vx[i],vy[i])

        # making sure they are inside the enclosure 
        if x[i] > L:
            # past the right wall
            x[i]= 0.0
        elif x[i] < 0.0:
            # past the left wall
            x[i]= float(L)

        if y[i] > L:
            # past the top wall
            y[i]= 0.0
        elif y[i] < 0.0:
            # past the bottom wall
            y[i]= float(L) 
        
    # relaxation
    sortV = np.sort(v)
    max = np.max(v)
    min = np.min(i for i in v if i > 0)
    step = int((max-min)/10)
    for 
    for s in range(0,step*2):
        up = np.sum(med*s,med*(s+1)
    
    print "timestep", t

# plot final results          
plot_position(x,y,"After - 2mass","x (m)","y (m)")
hist_velocity(vx,"X Velocity After - 2mass","velocity (m/s)","N",10)
hist_velocity(vy,"Y Velocity After - 2mass","velocity (m/s)","N",10)
hist_velocity(v,"Velocity After - 2mass","velocity (m/s)","N",10)
hist_velocity(KE,"KE After - 2mass","energy (J)","N",10)
