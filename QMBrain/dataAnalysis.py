import scipy.io
import scipy.signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

filename = input('Input the EEG filename: ')
filetype = input('Input the EEG filetype: ')

if filetype == 'mat':
    dataSet = scipy.io.loadmat(filename)

elif filetype == 'csv':
    dataSet = pd.read_csv(filename)

data = dataSet['data']
times = dataSet['times']
chanLocs = dataSet['chanlocs']

x = chanLocs['X']
y = chanLocs['Y']

hilbertTransData = scipy.signal.hilbert(data)

amplitude = np.abs(hilbertTransData)

phase = np.unwrap(np.angle(hilbertTransData))

ampMag = np.linalg.norm(amplitude.T)

normAmp = (np.asarray(amplitude.T)/np.asarray(ampMag)).T

probability = normAmp*normAmp

xAvg = probability@x
yAvg = probability@y

xSqrAvg = probability@(x*x)
ySqrAvg = probability@(y*y)

dx = np.sqrt(xSqrAvg-(xAvg*xAvg))
dy = np.sqrt(ySqrAvg-(yAvg*yAvg))

# Create an animation of the average position on the scalp

#for i in range(300):
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)




# First set up the figure, the axis, and the plot element we want to animate

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line


# animation function.  This is called sequentially
def animate(x,y):
    fig.errorbar(xAvg, yAvg, dy, dy, dx, dx, marker='.')
    line.set_data(x, y)
    return line


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)


#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])yAvg[i],dx[i],dy[i])