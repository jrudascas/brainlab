import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def load_matrix(filepath):
    extension = filepath.split('.')[1]
    if str(extension) == 'csv':
        return np.genfromtxt(filepath,delimiter = ',')
    elif str(extension) == 'npy':
        return np.load(filepath)
    elif str(extension) == 'mat':
        return scipy.io.loadmat(filepath)
    elif str(extension) == 'npz':
        return np.load(filepath)


def animation_station(xAvg,yAvg,xInit,yInit):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(xlim=(-10, 10), ylim=(-8, 8))
    line, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line.set_data(xInit, yInit)
        return line,

    # animation function.  This is called sequentially
    def animate(i, xAvg, yAvg):
        # fig.errorbar(xAvg[i], yAvg[i], dy, dy, dx, dx, marker='.')
        x1 = xAvg  # [i]
        y1 = yAvg  # [i]
        line.set_data(x1, y1)
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=300, interval=20)  # , blit=True)

    anim.save('animationTest.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

