import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import signal

rcParams['font.family'] = 'Arial'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

def plot_EMG(data, names, fs, plt_start_time, plt_end_time):
    """
    data: the signals you want to plot, a T*n numpy array, T is the number of
          samples, n is the number of channels
    names: a list for the names of EMG channels or labels for forces
    fs: the sampling frequency
    plt_start_time: the start time for plotting, a float number
    plt_end_time: the end time for plotting, a float number
    """
    N = data.shape[1]
    grid = plt.GridSpec(N, 1, wspace=0.5,hspace=0.2)
    for i in range(N):
        ax = plt.subplot(grid[i,0])
        ax.tick_params(axis=u'both', which=u'both',length=4)
        plt.tick_params(labelsize=18)
        sns.despine()
        p = data[int(plt_start_time*fs):int(plt_end_time*fs), :] 
        t = np.arange(np.floor(plt_end_time*fs) - np.floor(plt_start_time*fs))/fs
        plt.text(0, np.max(p[:, i]),'%s' %(names[i]),fontsize = 18,
                 verticalalignment="top",horizontalalignment="left")
        if i<N-1:
            plt.plot(t, p[:, i], 'k')
            plt.setp(ax.get_xticklabels(),visible=False)    
        if i == N-1:
            plt.plot(t, p[:, i], 'k')
            plt.setp(ax.get_xticklabels(),visible=True)
            ax.set_xlabel('Time (s)', fontsize = 18)
            
def plot_EMG_spectrogram(data, EMG_names, fs, plt_start_time, plt_end_time, f_range = [0, 400]):
    """
    This function is used to calculate and plot the spectrogram of multi-ch EMG signals.
    It calls spectrogram in scipy.signal to do the computation
    
    data: the EMG signals you want to analyze, a T*n numpy array, T is the number of
          samples, n is the number of channels
    EMG_names: a list for the names of EMG channels or labels for forces
    fs: the sampling frequency
    plt_start_time: the start time for plotting, a float number
    plt_end_time: the end time for plotting, a float number
    f_range: a two-element list specifying the start and the end frequency you want to plot,
            default is from 0 Hz to 400 Hz
    """
    N = data.shape[1]
    grid = plt.GridSpec(N, 1, wspace=0.5,hspace=0.2)
    cmap = plt.cm.jet
    for i in range(N):
        ax = plt.subplot(grid[i,0])
        ax.set_ylabel('f (Hz)', fontsize = 18)
        ax.tick_params(axis=u'both', which=u'both',length=4)
        plt.tick_params(labelsize=18)
        sns.despine()
        f, t, Sxx = signal.spectrogram(data[int(plt_start_time*fs):int(plt_end_time*fs), i], fs, 
                               scaling = 'density', nperseg = 256, noverlap = 64, nfft = 256)
        f_idx = np.where((f>f_range[0])&(f<f_range[1]))[0]
        plt.text(1, f[f_idx[-1]] ,'%s' %(EMG_names[i]),fontsize = 18,
                 verticalalignment="top",horizontalalignment="left")
        if i<N-1:
            im = ax.pcolormesh(t, f[f_idx], 10*np.log10(Sxx[f_idx, :]), cmap = cmap)
            plt.colorbar(im, ax = ax)
            plt.setp(ax.get_xticklabels(),visible=False)    
        if i == N-1:
            im = ax.pcolormesh(t, f[f_idx], 10*np.log10(Sxx[f_idx, :]), cmap = cmap)
            plt.colorbar(im, ax = ax)
            plt.setp(ax.get_xticklabels(),visible=True)
            ax.set_xlabel('Time (s)', fontsize = 18)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize = 14)
        cbar.set_label('dB', fontsize = 14)

def plot_EMG_spectral(data, EMG_names, fs, plt_start_time, plt_end_time):
    """
    This function is used to calculate and plot the power spectral of multi-ch EMG signals.
    It calls welch in scipy.signal to do the computation
    
    data: the EMG signals you want to analyze, a T*n numpy array, T is the number of
          samples, n is the number of channels
    EMG_names: a list for the names of EMG channels or labels for forces
    fs: the sampling frequency
    plt_start_time: the start time for plotting, a float number
    plt_end_time: the end time for plotting, a float number
    """
    N = data.shape[1]
    grid = plt.GridSpec(N, 1, wspace=0.5,hspace=0.2)
    for i in range(N):
        ax = plt.subplot(grid[i,0])
        plt.tick_params(labelsize = 14)
        plt.ylabel('PSD '+r'$(V^2/Hz)$', fontsize = 18)
        plt.ylim([0.5e-3, 1])
        sns.despine()
        f, Pxx_den = signal.welch(data[:, i], fs, nperseg=65536)
        #plt.grid(which='both', axis='both')
        if i<N-1:
            plt.semilogy(f, Pxx_den)
            plt.setp(ax.get_xticklabels(),visible=False)
        elif i == N-1:
            plt.semilogy(f, Pxx_den)
            plt.setp(ax.get_xticklabels(),visible=True)
            plt.xlabel('Frequency (Hz)', fontsize = 18)

def plot_filter_amp_response(b, a, fs, fc):
    """
    This function is used to plot the frequency response of a digital filter specified by 
    the filter coefficients b and a. To calculate such response, freqz in scipy is called
    b and a: the filter coefficients
    fs: sampling frequency
    fc: the corner frequency
    """
    w, h = signal.freqz(b, a)
    plt.subplot(121)
    plt.semilogx(w/np.pi*fs/2, 20 * np.log10(abs(h)))  
    plt.title('Filter frequency response', fontsize = 18)
    plt.xlabel('Frequency (Hz)', fontsize = 18)
    plt.ylabel('Amplitude (dB)', fontsize = 18)
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(fc, color='green') # cutoff frequency
    plt.tick_params(labelsize = 16)
    plt.ylim([-70, 10])
    
    plt.subplot(122)
    angles = np.unwrap(np.angle(h))
    plt.semilogx(w/np.pi*fs/2, angles)  
    plt.title('Filter frequency response', fontsize = 18)
    plt.xlabel('Frequency (Hz)', fontsize = 18)
    plt.ylabel('Angle (radians)', fontsize = 18)
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(fc, color='green') # cutoff frequency
    plt.tick_params(labelsize = 16)

def plot_pca_var_ratio(my_pca):
    """
    This function is used to plot the explained variance ratio of a PCA transform specified by 'my_pca'
    """
    data = my_pca.explained_variance_ratio_
    sns.barplot(np.arange(len(data))+1, data, facecolor = 'dimgray')
    plt.ylabel('Expl. var. ratio', fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.xlabel('%d components: %.2f' % (len(data), sum(data)), fontsize = 20)
    plt.ylim([0, 1])
    sns.despine()
    plt.tight_layout()

def plot_scatters(trials, target_dir, target_list, color_list):
    """
    This function is used to visulize dimensionality reduced data in 'trials'
    trials: an N*k numpy array. N is the number of trials, k is the number of variables,
            only the first two dimensions are used because this is a 2-D visualization
    target_dir: target directions for each trial
    target_list: a list listing all possible target directions
    color_list: the colors corresponding to each target in target_list
    """
    ax = plt.subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    plt.tick_params(labelsize = 12)
    for i in range(trials.shape[0]):
        if len(color_list) > 0:
            c = np.where(target_list == target_dir[i][0])[0][0]
            plt.scatter(trials[i, 0], trials[i, 1], s = 10, color = color_list[c])
        else:
            plt.scatter(trials[i, 0], trials[i, 1], s = 10, color = 'gray')

def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap=plt.cm.binary):
    """
    cm: the confusion matrix you want to plot. Generally they can be generated by sklearn.metrics.confusion_matrix
    labels: the class labels
    title: the title of your figure
    cmap: the color map you want to use
    """
    tick_marks = np.array(range(len(labels))) + 0.5
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=18, va='center', ha='center')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize = 18)
    #plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation = 45)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label', fontsize = 18)
    plt.xlabel('Predicted label', fontsize = 18)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.tick_params(labelsize=18)
    plt.gcf().subplots_adjust(bottom=0.15)

def plot_predictions(data1, data2, names, fs, plt_start_time, plt_end_time):
    """
    data1: the signals you want to plot, a T*n numpy array, T is the number of
          samples, n is the number of channels
    data2: another groups of signals you want to plot, could be the predicted signals,
           a T*n numpy array, T is the number of samples, n is the number of channels
    names: a list for the names of EMG channels or the labels for forces
    fs: the sampling frequency
    plt_start_time: the start time for plotting, a float number
    plt_end_time: the end time for plotting, a float number
    """
    N = data1.shape[1]
    grid = plt.GridSpec(N, 1, wspace=0.5,hspace=0.2)
    for i in range(N):
        ax = plt.subplot(grid[i,0])
        ax.tick_params(axis=u'both', which=u'both',length=4)
        plt.tick_params(labelsize=18)
        sns.despine()
        p1 = data1[int(plt_start_time*fs):int(plt_end_time*fs), :] 
        p2 = data2[int(plt_start_time*fs):int(plt_end_time*fs), :]
        t = np.arange(np.floor(plt_end_time*fs) - np.floor(plt_start_time*fs))/fs
        plt.text(0, np.max(p1[:, i]),'%s' %(names[i]),fontsize = 18,
                 verticalalignment="top",horizontalalignment="left")
        if i<N-1:
            plt.plot(t, p1[:, i], 'k')
            plt.plot(t, p2[:, i], 'r')
            plt.setp(ax.get_xticklabels(),visible=False)    
        if i == N-1:
            plt.plot(t, p1[:, i], 'k')
            plt.plot(t, p2[:, i], 'r')
            plt.setp(ax.get_xticklabels(),visible=True)
            ax.set_xlabel('Time (s)', fontsize = 18)



            
            
            