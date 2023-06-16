#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf


WINDOW = 200 #ms
DEVICE = 0
CHUNK = 1024
RATE = 44100
DOWNSAMPLE = 10
INTERVAL = 30 #ms
FORMAT = "PCM_24"
CHANNELS = [1]
DURATIOn = 5 #in seconds
FRAMES = []
FILE_NAME = "test.wav"

q = queue.Queue()


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata.copy())


def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines


try:
    if RATE is None:
        device_info = sd.query_devices(DEVICE, 'input')
        RATE = device_info['default_samplerate']

    length = int(WINDOW * RATE / (1000 * DOWNSAMPLE))
    plotdata = np.zeros((length, len(CHANNELS)))

    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)
    if len(CHANNELS) > 1:
        ax.legend(['channel {}'.format(c) for c in CHANNELS],
                  loc='lower left', ncol=len(CHANNELS))
    ax.axis((0, len(plotdata), -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom='off', top='off', labelbottom='off',
                   right='off', left='off', labelleft='off')
    fig.tight_layout(pad=0)

    stream = sd.InputStream(
        device=DEVICE, channels=max(CHANNELS),
        samplerate=RATE, callback=audio_callback)
    ani = FuncAnimation(fig, update_plot, interval=INTERVAL, blit=True)
    with stream:
        with sf.SoundFile(FILE_NAME, mode='x', samplerate=RATE,
                      channels=CHANNELS[0], subtype=FORMAT) as file:
            file.write(q.get())
            plt.show()
        
        
except KeyboardInterrupt:
    print('\nRecording finished.')
    exit()