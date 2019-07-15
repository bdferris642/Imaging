import matplotlib.pyplot as plt
import numpy as np
import copy
import numpy.fft as fft
import scipy.signal as signal
from scipy import stats
import matplotlib.gridspec as gridspec
import plotting_help as ph
from scipy.interpolate import interp1d
import functions as fc
from matplotlib.ticker import ScalarFormatter
from matplotlib import rcParams

# Vector manipulation
def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta
            
def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def perpendicular(a) :   #not used
    b = np.empty_like(a)
    b[:, 0] = -a[:, 1]
    b[:, 1] = a[:, 0]
    return b

def get_vector(slope, origin, length=20, orthogonal=True):
    # Generate 2D vector for each slope
    v = np.array([np.ones(len(slope)), slope]).T
    v[v.T[1] == np.inf] = np.array([0, 1])
    v[v.T[1] == -np.inf] = np.array([0, -1])
    
    # Set all vectors to the same length
    v = v/np.linalg.norm(v, axis=1).reshape(len(v), 1)
    v *= length/2.
    
    if orthogonal:
        v = perpendicular(v)
    
    # Define a line extending in both directions from the origin
    vf = np.empty([len(v), 2, 2])
    vf[:, 0] = origin - v
    vf[:, 1] = origin + v
    
    return vf

def get_slope(points, edge_order=2):
    x, y = points.T
    dx = np.gradient(x)
    slope = np.gradient(y, dx, edge_order=edge_order)
    return slope

def get_strings(mask):  
    """
    Extracts string of neighbouring masked values - used for finding skel in bridge
    """
    def get_sudoku(p):
            x0 = p[0]
            y0 = p[1]
            sudoku = [(x0-1, y0-1), (x0, y0-1), (x0+1, y0-1), 
                      (x0-1, y0),               (x0+1, y0), 
                      (x0-1, y0+1), (x0, y0+1), (x0+1, y0+1)]
            return sudoku
        
    def get_next_point(points_sorted, points):
        p0 = points_sorted[-1]
        for s in get_sudoku(p0):
            if s in points and s not in points_sorted:
                points_sorted.append(s)
                get_next_point(points_sorted, points)
                break
        return points_sorted
    
    def get_string(start, points):
        string1 = get_next_point([start], points)
        string = get_next_point([string1[-1]], points)
        start2 = min([string[0], string[-1]])
        string = get_next_point([start2], points)
        
        return string
    
    mask = copy.copy(mask)
    x, y = np.where(mask.T)
    points = zip(x, y)
    start = points[0]
    strings = []
    while 1:
        string = get_string(start, points)
        strings.append(string)
        sx, sy = zip(*string)
        mask[sy, sx] = 0
        x, y = np.where(mask.T)
        points = zip(x, y)
        if len(points):
            start = points[0]
        else:
            break
    
    # Order strings
    starts = [min([string[0], string[-1]]) for string in strings]
    starts, strings = zip(*sorted(zip(starts, strings)))  # Sort strings based on starts
    strings = map(np.array, strings)
    
    return strings


# FFT  
def powerspec(data, axis=-1, n=100, show=False, axs=None, fig_filename='', vmin=None,
              vmax=None, gate=[6, 11], cmap=plt.cm.jet, logscale=True, t=None,
              mask=False, norm=True):
    # take your data
    # subtract the mean
    # take the fourier transform along an axis
    # the power = the fourier transform squared
    # if normalized, the power is normalized by the maximum
    # phase = np.angle of the fourier transform
    
    data = copy.copy(data)
    data -= data.mean()
    if mask:
        data[:, 80:101] = 0
    axlen = data.shape[axis]*n
    fft1 = fft.fft(data, axlen, axis)
    power = np.abs(fft1)**2
    if norm:
        power = power / power.max()
    freq = fft.fftfreq(axlen, 1./n)/n
    phase = np.angle(fft1)
    
    midpoint = freq.size/2
    freq = freq[1:midpoint]
    period = (1./freq)
    power = power[:, 1:midpoint]
    phase = phase[:, 1:midpoint]
    
    if show or axs:
        if axs is None:
            plt.figure(1, (3, 7))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6])
            gs.update(hspace=0.2)
            ax1 = plt.subplot(gs[1, 0])
            ax2 = plt.subplot(gs[0, 0])#, sharex=ax1)
        else:
            ax2, ax1 = axs

        # Plot power spectrum over time
        ph.adjust_spines(ax1, ['left', 'top'])
        unit = 'frame #' if (t is None) else 's'
        ax1.set_ylabel('Time (%s)' %unit, rotation='horizontal', ha='right', labelpad=20)
        yedges = np.arange(power.shape[0]) if (t is None) else t
        xedges = period
        img = ax1.pcolormesh(xedges, yedges, power, cmap=cmap)
        if vmax: img.set_clim(vmax=vmax)
        if vmin: img.set_clim(vmin=vmin)
        ax1.set_xlabel('Period (glomeruli)', labelpad=20)
        # ax1.scatter(self.tif.t, periods, color='black', alpha=.8, lw=0, s=5, clip_on=True)
        ax1.set_xlim(2, 32)

        ax1.invert_yaxis()


        # Plot average power spectrum

        ax2.set_xticks([])
        h = power[:, period<=32].mean(axis=0)
        ymax = fc.round(h.max()*10, 1, 'up')/10.
        ax2.set_ylim(0, ymax)
        ax2.set_yticks([0, ymax])
#         ax2.fill_between(period[period<=pmax], h, facecolor=blue, lw=0)
        ax2.plot(period[period<=32], h, c='black', lw=1)
        ax2.set_xlim(2, 30)
        
        if logscale:
            ax1.set_xscale('log', basex=2)
            ax1.xaxis.set_major_formatter(ScalarFormatter())
            ax2.set_xscale('log', basex=2)
            ph.adjust_spines(ax2, ['left'])
        
        if fig_filename:
            plt.savefig(fig_filename, bbox_inches='tight')

        return img
    else:    
        return power, period, phase

def powerspec_peak(data, mode='constant',
                   gate=[6, 11], axis=-1,
                   t=None, cval=None,
                   n=100, show=False,
                   fig_filename=''): #mode used to be 'peak'
    """
    'peak' mode: find largest peak within gate. If no peak is detected, return nan.
    'max' mode: find largest value within gate. Same as peak, except returns largest value if no peak is detected.
    'constant' mode: use same value for each time frame.
    'median' mode: First runs powerspec_peak in 'peak' mode to find peaks within gate.
        Then runs again in peak mode, except return median peak if no peak is detected.
    """
    power, period, phase = powerspec(data, axis, n, False)
    if mode=='max':
        gateidx = np.where((period>gate[0]) & (period < gate[1]))[0]
    elif mode == 'median':
        _, pperiod, _ = fc.powerspec_peak(data, axis, 'peak', gate, show=False)
        medval = np.nanmedian(pperiod)
        medidx = np.where(period<=medval)[0][0]
    elif mode=='constant':
        if cval is None:
            power_mean = power.mean(axis=0)
            constidx = np.argmax(power_mean[(period >= gate[0]) & (period < gate[1])])
        else:
            constidx = np.where(period<=cval)[0][0]
    peak_phase = np.zeros(len(data))
    peak_period = np.zeros(len(data))
    peak_power = np.zeros(len(data))
    for i, row in enumerate(power):
        if mode in ['peak', 'median']:
            peakinds = detect_peaks(row, mph=.1, show=False)
            peakvals = row[peakinds]
            idx = np.argsort(peakvals)[::-1]
            peakpers = period[peakinds][idx]
            if gate:
                peakpers = peakpers[(peakpers>gate[0]) & (peakpers<gate[1])]
            if peakpers.size:
                iperiod = peakpers[0]
                iphase = phase[i, period==iperiod]
                ipower = power[i, period==iperiod]
            elif mode == 'median':
                iperiod = period[medidx]
                iphase = phase[i, medidx]
                ipower = power[i, medidx]
            else:
                iperiod = np.nan
                iphase = np.nan
                ipower = np.nan
        elif mode == 'max':
            rowgated = row[gateidx]
            idx = np.argmax(rowgated)
            idx += gateidx[0]
            iperiod = period[idx]
            iphase = phase[i, idx]
            ipower = power[i, idx]
        elif mode == 'constant':
            iperiod = period[constidx]
            iphase = phase[i, constidx]
            ipower = power[i, constidx]
            
        peak_period[i] = iperiod
        peak_phase[i] = iphase
        peak_power[i] = ipower
    peak_phase = peak_phase*(180/np.pi)
    
    if show:
        # Setup Figure
        rcParams['xtick.direction'] = 'out'
        rcParams['ytick.direction'] = 'out'
        plt.figure(1, (7, 5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6])
        gs.update(wspace=0.05)

        # Plot period over time
        ax1 = plt.subplot(gs[1, 0])
        y = np.arange(len(peak_period)) if (t is None) else t
        ax1.scatter(peak_period, y, color='black', alpha=.8, lw=0, s=5, clip_on=True)
        ph.adjust_spines(ax1, ['left', 'bottom'])
        ax1.set_xlabel('Period (glomeruli)', labelpad=20)
        pmax = 18
        ax1.set_xlim(0, pmax)
        ax1.set_ylim(0, len(peak_period))
        ax1.invert_yaxis()
        plt.grid()

        # Plot histogram of period
        ax2 = plt.subplot(gs[0, 0])
        ph.adjust_spines(ax2, ['left'])
        binwidth = .1
        bins = np.arange(0, 18, binwidth)
        period_vals = peak_period[np.isnan(peak_period)==False]
        h, xedges = np.histogram(period_vals, bins=bins, density=True)
        ax2.bar(xedges[:-1], h, color='grey', lw=0, width=binwidth)
        ax2.set_xlim(0, pmax)
        ymax = np.round(h.max(), decimals=1)
        ax2.set_yticks([0, ymax])
        ax2.set_ylim(0, ymax)
        
        for ax in [ax1, ax2]:
            for line in gate:
                ax.axvline(line, alpha=.3, ls='--')
        
        if fig_filename:
            plt.savefig(fig_filename, bbox_inches='tight')
    else:
        return peak_phase, peak_period, peak_power


# Peaks
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
        """Plot results of the detect_peaks function, see its help."""
        
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def get_halfwidth_peak(xs, ys):
    x_wrap = wrap(xs)
    y_wrap = np.tile(ys, 3)
    interpf = interp1d(x_wrap, y_wrap, 'linear')
    x_interp = np.arange(-180, 180.1, .1)
    y_interp = interpf(x_interp)
    halfwidth_idxs = np.where(np.diff(y_interp>=ys.max()/2.)>0)[0]
    if halfwidth_idxs.size == 2:
        peak = circmean(x_interp[halfwidth_idxs])
        return peak
    else:
        return None


# Sampling data
def get_inds_start_stop(inds, keep_ends=True):
    diff = np.diff(inds.astype(np.int8))
    start_inds = np.where(diff>0)[0] + 1
    stop_inds = np.where(diff<0)[0] + 1


    if len(stop_inds) == 0:
        stop_inds = np.array([len(inds)])

    if len(start_inds) == 0:
        start_inds = np.array([0])

    if stop_inds[0] <= start_inds[0]:
        if keep_ends:
            start_inds = np.insert(start_inds, 0, 0)
        else:
            stop_inds = stop_inds[1:]

    if start_inds[-1] >= stop_inds[-1]:
        if keep_ends:
            stop_inds = np.append(stop_inds, len(inds))
        else:
            start_inds = start_inds[:-1]
    return start_inds, stop_inds

def get_contiguous_inds(inds, min_contig_len=5, trim_left=0, trim_right=0, **kwargs):
    """
    Splits a boolean array into a list of lists, each list containing the indices for contiguous True elements.
    """
    start_inds, stop_inds = get_inds_start_stop(inds, **kwargs)
    cont_inds = [np.arange(start+trim_left, stop-trim_right) for start, stop in zip(start_inds, stop_inds) if stop-start > min_contig_len]
    return cont_inds

def rising_trig(signal, trigger=3, mindist=0):
    if any([i for i in signal.shape]) == 1:
        signal = np.squeeze(signal)
    trigs = np.where(np.diff((signal>trigger).astype(int))>0)[0]
    trigs = trigs[np.insert(np.diff(trigs)>mindist, 0, 1)]
    return np.squeeze(trigs)

def sparse_sample(data, leftinds, indint, 
                     metric= lambda x: np.mean(x, axis = 1),
                     **kwargs):
    
    ind0 = np.arange(0, indint)
    inds = np.tile(ind0, (len(leftinds), 1)) + leftinds[:, None]
    # `inds` is a matrix of
    # n_frames rows
    # n_abf_measurements_per_frame columns
    # inds[x, 0] is the 1st index of the abf from  x'th imaging frame (corrected for lag)
    # which you want to average
    
    nrows, ncols = np.shape(inds)    
    for row in range(nrows):
        for val in inds[row, :]:
            if val >= np.shape(data)[0]:
                inds = inds[:row, :]
                break
        if not(nrows) == np.shape(inds)[0]:
            break
    
    datab = metric(data[inds], **kwargs)

    return datab

def dense_sample(data, indint, metric = np.linspace, **kwargs):
    
    """
    takes a sparse vector (data), a new size int (indint, must be >= len(data)),
    and a metric (fctn: linspace).
    outputs an upsampled vector (datab) of length indint
    whose values are the interpolated values of data
    Every point from "data" is included in "datab", unchanged.
    
    [1, 3, 5] , 5 --> [1, 2, 3, 4, 5]
    interpolation equal when possible but sometimes...[1, 3, 5, 7], 12.
    8 interpolations into 3 spaces...
    In such a case, makes sure everybody gets 
    at least the quotient of excess indices / original length (in this case 2),
    distributes the remainder (in this case 2) as evenly as possible
    (in this case to the first and third interpolation)
    [1, 3, 5, 7], 12 --> [1., 1.5, 2., 2.5, 3., 3.67, 4.33, 5., 5.5, 6., 6.5, 7.]
    """
    
    datab = np.zeros((indint))
    n_interp = len(data) - 1
    excess = indint - len(data) # excess indices
    base_interpol = excess / n_interp # minimum number of middle values in each interpolation
    remainder = excess % n_interp # extra middle values. (< n_interp). distribute evenly throughout datab.
    
    if remainder == 1:
        left_edge_tack_ons = np.array([np.median(range(n_interp))])
    else:
        left_edge_tack_ons = np.linspace(0, n_interp-1, remainder) 
    
    datab_ind_counter = 0# keep track of inds of output vector
    for i, item in enumerate(data[:-1]):
        if np.size(left_edge_tack_ons) > 0:
            if i >= left_edge_tack_ons[0]:
                tack_on = 1
                left_edge_tack_ons = left_edge_tack_ons[1:]
            else:
                tack_on = 0
        else:
            tack_on = 0
        
        datab[datab_ind_counter:(datab_ind_counter+base_interpol+tack_on+1)] = metric(data[i], data[i+1], base_interpol+tack_on+2)[:-1]
        datab_ind_counter += (base_interpol+tack_on+1)
    datab[-1] = data[-1]
    return datab

def dense_sample_deprecated(data, edges, metric=np.mean, **kwargs):    # not used
    
    datab = np.zeros(len(edges)-1)
    for i in xrange(len(edges)-1):
        idx = np.arange(edges[i], edges[i+1])
        datab[i] = metric(data[idx], **kwargs)
    return datab

def sparse_sample_deprecated(data, leftinds, indint, metric=np.mean, **kwargs):
    datab = np.zeros_like(leftinds)
    for i, leftind in enumerate(leftinds):
        idx = np.arange(leftind, leftind+indint)
        datab[i] = metric(data[idx], **kwargs)
    return datab

def binyfromx(x, y, xbins, metric=np.median, nmin=0, **kwargs):
    binidx = np.digitize(x, xbins)
    yshape = list(y.shape)
    yshape[0] = len(xbins)-1
    ystat = np.zeros(yshape)
    for ibin in range(1, len(xbins)):
        yi = y[ibin==binidx]
        ystat[ibin-1] = metric(yi, **kwargs) if (len(yi) > nmin) else np.nan
    return ystat

def butterworth(input_signal, cutoff, passside='low', N=8, sampling_freq=100):
    Nyquist_freq = sampling_freq / 2.
    Wn = cutoff/Nyquist_freq
    b, a = signal.butter(N, Wn, passside)
    output_signal = signal.filtfilt(b, a, input_signal)
    return output_signal


# Manipulating Circular data
def unwrap(signal, period=360):
    if any([i for i in signal.shape]) == 1:
        signal = np.squeeze(signal)
    unwrapped = np.unwrap(signal*2*np.pi/period)*period/np.pi/2
    return unwrapped

def wrap(arr, cmin=-180, cmax=180):
    period = cmax - cmin
    arr = arr%period
    arr[arr>=cmax] = arr[arr>=cmax] - period
    arr[arr<cmin] = arr[arr<cmin] + period
    return arr

def circdiff(a, b, period=360, cmin=-180, cmax=180):
    """
    returns the circular difference of a-b
    wrapped with period `period`
    with mins and maxes as specified
    """
    diff = a-b
    diff = diff%period
    if diff > cmax:
        diff -= period
    elif diff <cmin:
        diff += period
    return diff

def circgrad(signal, method=np.gradient, **kwargs):
    if any([i for i in signal.shape]) == 1:
        signal = np.squeeze(signal)
    signaluw = unwrap(signal, **kwargs)
    dsignal = method(signaluw)
    return dsignal

def circmean(arr, low=-180, high=180, axis=None):
    return stats.circmean(arr, high, low, axis)

def circstd(arr, low=-180, high=180, axis=None):
    return stats.circstd(arr, high, low, axis)

def circ2lin(xs, ys, period=360):
    def shift2line(xs, ys, b, period=360):
        """
        Shift points greater than period/2 y distance away from diagonal + b closer to the line.
        """
        yline = xs + b
        ydist = ys - yline
        ys[ydist > period/2.] -= period
        ys[ydist < -period/2.] += period
        return ys
    
    # Take 1st guess at line with slope 1, shift wrapped values closer to the line
    y0 = ys[(xs>0) & (xs<10)].mean()
    x0 = 5
    b0 = y0 - x0
    ys = shift2line(xs, ys, b0, period)
    
    # Fit data to line with slope 1 (find b=y-intercept)
    def fitfunc(x, b):
        return x+b
    params, _ = curve_fit(fitfunc, xs, ys)
    b = params[0]
    # Re-shift to best fit line
    ys = shift2line(xs, ys, b, period)
    
    return ys, b

def tile(arr, period=360):
        arr2 = np.tile(arr, 3)
        arr2[:arr.size] -= period
        arr2[-arr.size:] += period
        return arr2

def moving_average(x, N):
    ma_vector = np.zeros(len(x))
    ma_vector[:] = np.nan
    if N%2 == 0: # if N is even
        shoulder_l = N/2
        shoulder_r = -(shoulder_l-1)
    else: # if N is odd
        shoulder_l = (N-1) / 2
        shoulder_r = -shoulder_l
    
    cumsum = np.cumsum(np.insert(x, 0, 0))
    ma_vector[shoulder_l:shoulder_r] = (cumsum[N:] - cumsum[:-N]) / float(N)
    ma_vector[:shoulder_l] = x[:shoulder_l]
    ma_vector[shoulder_r:] = x[shoulder_r:]
    return ma_vector

def circ_moving_average(a, n=3, low=-180, high=180):
    assert len(a) > n # ensure that the array is long enough
    assert n%2 != 0 # make sure moving average is odd, or this screws up time points
    shoulder = (n-1) / 2
    ma = np.zeros(len(a))
    ind0 = np.arange(-shoulder, shoulder+1)
    inds = np.tile(ind0, (len(a)-2, 1)) + np.arange(1, len(a)-1)[:, None]
    ma[1:-1] = circmean(a[inds], low, high, axis=1)
    ma[[0, -1]] = a[[0, -1]]
    return ma

def periodic_corr(x, y):
    """Periodic correlation, implemented using the FFT.

    x and y must be real sequences with the same length.
    """
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(y).conj()).real



# Miscellaneous
def contains(list1, list2):
    for el in list1:
        if el in list2:
            return el
    return False

def xcorr(a, v, sampling_period=1, norm=True):
    """
    Computes normalized ross-correlation between a and v.
    :param a: 1-d vector.
    :param v: 1-d vector of same length as a.
    :param sampling_period: time steps associated with indices in a and v.
    :return: t, delay a_t - v_t (if t>0, a comes after v)
            xc, full cross-correlation between a and v.
    """
    if not len(a)==len(v):
        print 'len(a) must equal len(v).'
    if norm:
        a = (a - np.mean(a)) / np.std(a)
        v = (v - np.mean(v)) /  np.std(v)
    l = len(a)
    a = a / (len(a) - 1)
    xc = np.correlate(a, v, 'full')
    t = np.arange(-(l-1), (l-1)+1)*sampling_period
    return t, xc

def neighbouring_xcorr(img, save=True, vmin=.8, vmax=1):    # not used
    def xcorr_square(y, x, img):
        a = np.zeros(8)
        m = 0
        c = img[:, y, x]
        c = (c - c.mean())/c.std()/c.size
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                if i==x and j==y: continue
                p = img[:, j, i]
                p = (p - p.mean())/p.std()
                a[m] = (c*p).sum()
                m += 1
        return a.mean()

    newimg = np.zeros((img.shape[-2], img.shape[-1]))
    for y in range(1, img.shape[-2]-1):
        for x in range(1, img.shape[-1]-1):
            newimg[y, x] = xcorr_square(y, x, img)
    
    if save:
        plt.imshow(newimg, vmin=vmin, vmax=vmax)
        plt.savefig(newimg)
        
    return newimg

def nan2zero(arr, make_copy=False):
    if make_copy:
        arr2 = copy.copy(arr)
        arr2[np.isnan(arr2)] = 0
        return arr2
    else:
        arr[np.isnan(arr)] = 0

def zero2nan(arr, make_copy=False):
    if make_copy:
        arr2 = copy.copy(arr)
        arr2[arr2 == 0] = np.nan
        return arr2
    else:
        arr[arr == 0] = np.nan

def nansem(arr, axis):
    std = np.nanstd(arr, axis=axis)
    n = (np.isnan(arr)==False).sum(axis=axis)
    sem = std / np.sqrt(n)
    return sem

def round(x, base=180, dir='up'):
    if dir == 'up':
        return int(base * np.ceil(float(x)/base))
    else:
        return int(base * np.floor(float(x)/base))

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        # linear interpolation of NaNs
        nans, x= nan_helper(y)
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate(x, y, mask, kind='linear'):
    """
    Replaces masked values with linearly interpolated values in x, y.
    """
    f = interp1d(x[~mask], y[~mask], kind, axis=-1)
    return f(x)

def flatten(listoflist):
    
    outlist = []
    for sub in listoflist:
        if type(sub) is list:
            if any([type(val) is list for val in sub]):
                sub = flatten(sub)
            outlist.extend(sub)    
        else:
            outlist.append(sub)
    
    return outlist

def center(boundaries):
    
    centers = [np.mean([a, b]) for (a, b) in zip(list(boundaries)[1:], list(boundaries)[:-1])]
    
    if type(boundaries) is np.ndarray:
        centers = np.array(centers)
    
    return centers
    
    
