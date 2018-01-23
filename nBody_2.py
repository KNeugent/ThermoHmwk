# Kathryn Neguent
# Homework 1

import numpy as np
import matplotlib.pyplot as plt

# initialize paramters
# number of particles
N = 100
# box side length (m)
L = 1000
# steps to iterate over
steps = 100
# time (s)
time = 5
# mass (kg)
m1 = 1
m2 = 1
# radius (m)
r = .02*L
# Kinetic Energy (J)
KEstart = 10

# for relaxation time
vSum = np.zeros(steps*time)

# Initialize particle properties within grid
x = np.zeros(N)
y = np.zeros(N)
m = np.zeros(N)
v = np.zeros(N)
vx = np.zeros(N)
vy = np.zeros(N)
KE = np.ones(N)*KEstart

# Initialize pressures
rightM = np.zeros(N)
rightV = np.zeros(N)
rightN = 0
leftM = np.zeros(N)
leftV = np.zeros(N)
leftN = 0
topM = np.zeros(N)
topV = np.zeros(N)
topN = 0
bottomM = np.zeros(N)
bottomV = np.zeros(N)
bottomN = 0

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
    m[i] = np.random.choice([m1,m2])
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
for t in range(0,steps*time):
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
            # save for use of pressure determination
            rightM = m[i]
            rightV = v[i]
            rightN = rightN +1
        elif x[i] < 0.0:
            # past the left wall
            x[i]= float(L)
            leftM = m[i]
            leftV = v[i]
            leftN = leftN +1
        if y[i] > L:
            # past the top wall
            y[i]= 0.0
            topM = m[i]
            topV = v[i]
            topN = topN + 1
        elif y[i] < 0.0:
            # past the bottom wall
            y[i]= float(L) 
            bottomM = m[i]
            bottomV = v[i]        
            bottomN = bottomN + 1
    # relaxation
# what didn't work
#    vSplit = np.split(v,50)
#    v1bin = (np.sum(vSplit[0]) - np.sum(vSplit[9]))uu
#    v2bin = (np.sum(vSplit[1]) - np.sum(vSplit[8]))
#    v3bin = (np.sum(vSplit[2]) - np.sum(vSplit[7]))
#    v4bin = (np.sum(vSplit[3]) - np.sum(vSplit[6]))
#    v5bin = (np.sum(vSplit[4]) - np.sum(vSplit[5]))
#    vSum[t] = v1bin+v2bin+v3bin+v4bin+v5bin
#    print t
#    print vSum[t]

# what did work
    v[np.isnan(v)] = 0
    mean = np.mean(v)
    median = np.median(v)
    vSum[t] = abs(mean - median)

# pressure calculation
# P = N*(2/3)((average mass)*(average velocity)**2)/2
rightP = rightN * (2./3.) * ((np.sum(rightM)/rightN)+(np.sum(rightV)/rightN)**2)/2.
leftP = leftN * (2./3.) * ((np.sum(leftM)/leftN)+(np.sum(leftV)/leftN)**2)/2.
topP = topN * (2./3.) * ((np.sum(topM)/topN)+(np.sum(topV)/topN)**2)/2.
bottomP = bottomN * (2./3.) * ((np.sum(bottomM)/bottomN)+(np.sum(bottomV)/bottomN)**2)/2.
print "right N = ", rightN
print "right KE = ", ((np.sum(rightM)/rightN)+(np.sum(rightV)/rightN)**2)/2.
print "right P = ", rightP

print "left N = ", leftN
print "left KE = ", ((np.sum(leftM)/leftN)+(np.sum(leftV)/leftN)**2)/2.
print "left P = ", leftP

print "top N = ", topN
print "top KE = ", ((np.sum(topM)/topN)+(np.sum(topV)/topN)**2)/2.
print "top P = ", topP

print "bottom N = ", bottomN
print "bottom KE = ", ((np.sum(bottomM)/bottomN)+(np.sum(bottomV)/bottomN)**2)/2.
print "bottom P = ", bottomP
    
# plot final results          
plot_position(x,y,"After - 1g and 10g","x (m)","y (m)")
hist_velocity(vx,"X Velocity After - 1g and 10g","velocity (m/s)","N",10)
hist_velocity(vy,"Y Velocity After - 1g and 10g","velocity (m/s)","N",10)
hist_velocity(v,"Velocity After - 1g and 10g","velocity (m/s)","N",10)
hist_velocity(KE,"KE After - 1g and 10g","energy (J)","N",10)
