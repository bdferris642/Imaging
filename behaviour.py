import functions as fc
from neo.io import AxonIO

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import plotting_help as ph

import numpy as np
from scipy import stats
import scipy.cluster as sc
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d

import sys, os, copy, math
from bisect import bisect

reload(fc)
reload(ph)

def read_abf(abf_filename):
    """
    Returns a dictionary. Keys are the channel names in lowercase, values are the channels.
    
    # Usage:
    abf = read_abf('2015_08_03_0000.abf'). abf = dict, abf_filename = string pointing to the abf file.
    
    # To show all channel keys:
    abf.keys()
    
    # Channels have useful attributes:
    temp = abf['temp']
    temp_array = np.array(temp)
    t = np.array(temp.times)
    
    # To plot:
    plt.plot(t, temp_array)
    
    # Can also get sampling rate and period:
    temp.sampling_rate
    temp.sampling_period
    """
    fh = AxonIO(filename=abf_filename)
    segments = fh.read_block().segments
        
    if len(segments) > 1:
        print 'More than one segment in file.'
        return 0
    
    analog_signals_ls = segments[0].analogsignals
    analog_signals_dict = {}
    for analog_signal in analog_signals_ls:
        analog_signals_dict[analog_signal.name.lower()] = analog_signal
    
    return analog_signals_dict

def volts2degs(volts, maxvolts=10):
    degs = volts*360/maxvolts
    degs[degs>360] = 360.
    degs[degs<0] = 0.
    return degs

#---------------------------------------------------------------------------#

class ABF():  
    def __init__(self, abffile, maxvolts=None):
        self.basename = ''.join(abffile.split('.')[:-1])
        self.abffile = abffile
        self.maxvolts = maxvolts
    
    def open(self, subsampling_rate = 20., enforce_subsampling_rate=20., delay_ms=15, nstims=None):
        
        
        abf = read_abf(self.abffile)
        # Collect subsampling indices based on camera triggers.
        camtriglabel = fc.contains(['cameratri', 'camtrig'], abf.keys())
        if camtriglabel:
            # Use cameratrigger to get framerate for wingbeat tracking.
            # Better to use if cameratrigger is available because it aligns the
            # wing beats to the camera acquisition.
            camtrig = abf[camtriglabel]
            self.camtrig = np.squeeze(np.array(camtrig))
            
            # these numbers are objects in camtrig, somehow
            self.sampling_period = float(camtrig.sampling_period)
            self.sampling_rate = float(camtrig.sampling_rate)
            
            # risingtrig is an array of where cameratrig goes from 0 --> 1
            risingtrig = np.where(np.diff((self.camtrig>1).astype(np.int8))>0)[0]
            if len(risingtrig):
                
                # subsampling_period_idx is the average # of indices b/w these risingtrig events
                # you multiply the camtrig-objects to get the others 
                subsampling_period_idx = np.diff(risingtrig).mean()
                self.subsampling_period = subsampling_period_idx*self.sampling_period
                self.subsampling_rate = 1./self.subsampling_period
                
                self.t_orig = np.array(camtrig.times)
                
                if enforce_subsampling_rate and (self.subsampling_rate >= enforce_subsampling_rate):
                    self.subsampling_rate = enforce_subsampling_rate
                    self.subsampling_period = 1. / enforce_subsampling_rate
                    
                    risingtrig_new = []
                    baseline_trig_time = self.t_orig[risingtrig[0]]
                    for i in risingtrig:
                        if self.t_orig[i] - baseline_trig_time >= self.subsampling_period:
                            risingtrig_new.append(i)
                            baseline_trig_time = self.t_orig[i]
                    risingtrig = np.array(risingtrig_new)                     
                        
                # inds to subsample. Take risingtrig and subtract a constant (idk why)
                # trim idx to make sure you never end up below 0
                subsampling_inds = np.round(risingtrig - 1.5*subsampling_period_idx).astype(int)
                trim_idx = np.sum(subsampling_inds<0)                    
                    
                self.subsampling_inds = subsampling_inds[trim_idx:]
                self.behaviour_inds = (risingtrig-10)[trim_idx:]
                self.t = self.t_orig[self.subsampling_inds]
            else:
                print 'Camera Triggers are not available.'
                camtriglabel = None

        else:
            self.subsampling_rate = subsampling_rate
            self.subsampling_period = 1.0/self.subsampling_rate
            example = abf[abf.keys()[0]]
            self.sampling_period = float(example.sampling_period)
            self.sampling_rate = float(example.sampling_rate)
            self.subsampling_step = int(example.sampling_rate/self.subsampling_rate)

            inds = np.arange(len(example))
            self.subsampling_inds = inds[::self.subsampling_step]
            self.behaviour_inds = self.subsampling_inds + int(round((delay_ms/1000.)*self.sampling_rate))
            trim_idx = np.sum(self.behaviour_inds>=len(example))
            if trim_idx:
                self.subsampling_inds = self.subsampling_inds[:-trim_idx]
                self.behaviour_inds = self.behaviour_inds[:-trim_idx]
            
            self.t_orig = np.array(example.times)
            self.t = self.t_orig[self.subsampling_inds]
        
        # Other channels to be subsampled.
        templabel = fc.contains(['temp'], abf.keys())
        if templabel:
            self.temp = np.array(abf['temp'])[self.subsampling_inds]
        
        xstimlabel = fc.contains(['vstim_x18', 'vstim_x', 'stim_x', 'xstim', 'x_ch'], abf.keys())
        if xstimlabel:
            self.xstim_orig = np.array(abf[xstimlabel])
            self.xstim = np.array(abf[xstimlabel])[self.subsampling_inds]
            if self.maxvolts:
                self.xstim = volts2degs(self.xstim, self.maxvolts) + self.angle_offset
            self.xstim = fc.wrap(self.xstim)
            self.dxstim_orig = fc.circgrad(self.xstim_orig)*self.sampling_rate
            self.dxstim = fc.circgrad(self.xstim)*self.subsampling_rate
                
        ystimlabel = fc.contains(['vstim_y', 'stim_y', 'ystim', 'y_ch'], abf.keys())
        if ystimlabel:
            # self.ystim_orig = np.array(abf[ystimlabel])
            self.ystim = np.array(abf[ystimlabel])[self.subsampling_inds]

        stimidlabel = fc.contains(['stimid', 'patid', 'patname'], abf.keys())
        if stimidlabel:
            self.stimid_raw_orig = np.array(abf[stimidlabel])
            self.stimid_raw = self.stimid_raw_orig[self.subsampling_inds]
            if nstims:
                self.stimid = self.get_stimid(nstims)
                # self.stimid = self.stimid_orig[self.subsampling_inds]
        
        pressurelabel = fc.contains(['pressure', 'pippuff'], abf.keys())
        if pressurelabel:
            self.pressure_orig = np.array(abf[pressurelabel])
            self.pressure = self.pressure_orig[self.subsampling_inds]
        
        vmlabel = fc.contains(['vm_b', 'vm'], abf.keys())
        if vmlabel:
            self.vm_orig = np.array(abf[vmlabel])
            self.vm = self.vm_orig[self.subsampling_inds]

        imlabel = fc.contains(['im_b', 'im'], abf.keys())
        if imlabel:
            self.im_orig = np.array(abf[imlabel])
            self.im = self.im_orig[self.subsampling_inds]

        pufflabel = fc.contains(['puff'], abf.keys())
        if pufflabel:
            self.puff = np.array(abf[pufflabel])[self.subsampling_inds]
        
        # Channels that are not subsampled.
        frametriglabel = fc.contains(['frame tri', 'frametri', '2pframest'], abf.keys())
        if frametriglabel:
            self.frametrig = np.squeeze(np.array(abf[frametriglabel]))
        
        if 'pockels' in abf.keys():
            self.pockels = np.array(abf['pockels'])
        
        aolabel = fc.contains(['ao', 'arena_sti'], abf.keys())    
        if aolabel:
            self.arena_stim = np.array(abf[aolabel])
        
        # tags = AxonIO(filename=self.abffile).read_header()['listTag']
        # self.tags = np.array([tag['lTagTime'] for tag in tags])
        # if len(self.tags):
        #     self.t_tags = self.t_orig[self.tags]
        
        # general purpose dictionary
        self.stim_inds = {}
        
        return abf
    
    def get_temp_transition(self, transition_temp):
        if not hasattr(self, 'temp'):
            return 0
        
        t_heating_transitions = self.t[np.where(np.diff(self.temp > transition_temp) > 0)]
        #cooling_transitions = self.t[np.where(np.diff(self.temp > transition_temp) < 0)]
        if len(t_heating_transitions) > 0:
            return t_heating_transitions[0]
        else:
            return 0

    def get_stimid(self, nstims):
        if nstims==2:
            guess = np.array([0, .63/5., .63])
        else:
            guess = np.arange(nstims+1).astype(float)/nstims*.63
        centroids, _ = sc.vq.kmeans(self.stimid_raw, guess)
        centroids = np.sort(centroids)
        stimid, _ = sc.vq.vq(self.stimid_raw, centroids)
        return np.array(stimid)

    def _is_hot(self, trial_inds, hot_min=30, hot_max=40):
        local_temp = self.temp[trial_inds]
        local_max_temp = max(local_temp)
        local_min_temp = min(local_temp)
        is_hot = (local_min_temp > hot_min) and (local_max_temp < hot_max)
        return is_hot.astype(int)
    
    def _is_walking(self, stasis_thresh = 1, walking_thresh = 2,
               filt = gaussian_filter1d, sigma_s = .5,
              freq = 50):
    
        if stasis_thresh >= walking_thresh:
            print 'walking_thresh must be greater than stasis_thresh'
            quit()
        
        dforw_filt = filt(self.dforw, freq*sigma_s)
        
        walking_array = []
        
        def_walking = dforw_filt >= walking_thresh
        def_static = dforw_filt < stasis_thresh
        
        for ival in range(len(dforw_filt)):
            if def_walking[ival]:
                walking_array.append(True)
            elif def_static[ival]:
                walking_array.append(False)
            elif ival == 0:
                walking_array.append(False)
            else:
                walking_array.append(walking_array[ival-1])
        
        walking_array.append(walking_array[-1])
        walking_array = np.array(walking_array)
        return walking_array
    
    def _is_static(self, dforw_thresh = 1, dhead_thresh = 50,
                filt = gaussian_filter1d, sigma_s = .1,
                freq = 50):
        
        dforw_filt = filt(self.dforw, freq*sigma_s)
        dhead_filt = filt(self.dhead, freq*sigma_s)
        
        def_static = np.logical_and((dforw_filt < dforw_thresh), (abs(dhead_filt) < dhead_thresh))
        
        static_array = np.zeros((len(dforw_filt)), dtype=bool)
        static_array[def_static] = True
        
        return static_array
    

#---------------------------------------------------------------------------#

class Parsed():
    def __init__(self, abf, channels, stimdict=None, window_lens=[], shoulder_lens=[], parse_window=[], baseline_window=[],
                 turn_channel=None, turn_thresh=8, trig_channel='default', trig_thresh=20, min_trig_dist_s=5,
                 parse_channel='default', template=None, tif=False):
        self.abf = abf
        if template:
            self.stimdict = template.stimdict
            self.stims = template.stims
            self.nstims = template.nstims
            self.window_lens = template.window_lens
            self.shoulder_lens = template.shoulder_lens
            self.parse_window = template.parse_window
            self.baseline_window = template.baseline_window
            self.turn_channel = template.turn_channel
            self.turn_thresh = template.turn_thresh
            self.trig_channel = template.trig_channel
            self.trig_thresh = template.trig_thresh
            self.min_trig_dist_s = template.min_trig_dist_s
            self.parse_channel = template.parse_channel
            self.trial_ts = template.trial_ts
            self.stimids = template.stimids
            self.nruns = template.nruns
            self.ts = self._get_ts(tif)
            self.t = self.ts.values()[0]
            self.active_bytrial = template.active_bytrial
        else:
            self.stimdict = stimdict
            self.stims = stimdict.values()
            self.nstims = len(self.stims)
            self.window_lens = window_lens
            self.shoulder_lens = shoulder_lens
            self.parse_window = parse_window
            self.baseline_window = baseline_window
            self.turn_channel = turn_channel
            self.turn_thresh = turn_thresh
            self.trig_channel = trig_channel
            self.trig_thresh = trig_thresh
            self.min_trig_dist_s = min_trig_dist_s
            self.parse_channel = parse_channel
            self.trial_ts, self.stimids = self._get_trial_info(self.nstims, trig_channel, trig_thresh, min_trig_dist_s, parse_channel)
            
            if hasattr(abf, 'temp'):
                self.hot_thresh = 27
                self.t_heating_transition = abf.get_temp_transition(transition_temp=self.hot_thresh)
                if self.t_heating_transition > 0:
                    self.nruns = 2
                else:
                    self.nruns = 1
            else:
                self.nruns = 1
            
            self.ts = self._get_ts(tif)
            self.t = self.ts.values()[0]
            self.active_bytrial = np.zeros(len(self.stimids)).astype(bool)
        
        self.rundict = {0: 'cool', 1: 'hot'}
        self.runs = self.rundict.values()
        self.turndirs = ['left', 'straight', 'right', 'all']
        bt = abf.t
        self.parsed = {channel: self._parse_signal(bt, getattr(abf, channel), delta=channel[0]=='d') for channel in channels}

    def mean(self):
        means = {}
        for signal in self.keys():
            means[signal] = {}
            for stim in self.stimdict.values():
                means[signal][stim] = {}
                for nrun in self.rundict.values():
                    means[signal][stim][nrun] = {}
                    for turndir in turndirs:
                        means[signal][stim][nrun][turndir] = self.parsed[signal][stim][nrun][turndir].mean(axis=0)
        return means
    
    def __getitem__(self, key):
        return self.parsed[key]
    
    def keys(self):
        return self.parsed.keys()

    def _get_stimids(self, trial_ts, stim, nstims, window):
        t = self.abf.t_orig
        trial_means = np.array([stim[(t>=start) & (t<start+window)].mean() for start in trial_ts])
        centroids, _ = sc.vq.kmeans(trial_means, nstims)
        centroids = np.sort(centroids)
        stimids, _ = sc.vq.vq(trial_means, centroids)
        stimids = np.array(stimids)
        return stimids
    
    def _get_trial_info(self, nstims, trig_channel='default', trig_thresh=20, min_trig_dist_s=5, parse_channel='default', window=1):
        t = self.abf.t_orig
        if trig_channel == 'default':
            if hasattr(self, 'arena_stim'):
                trig = self.abf.arena_stim
                stim = self.abf.arena_stim
                
                # Get trial starts
                trigdiff = np.abs(np.diff((trig>.02).astype(np.int8)))>0
                trial_ts = t[trigdiff]
                trial_ts_diff = np.diff(trial_ts)
                trial_ts = trial_ts[:-1][trial_ts_diff>.1]
                stimids = self._get_stimids(trial_ts, stim, nstims, window)
                
            else:
                trig = self.abf.ystim_orig
                stim = self.abf.xstim_orig
                
                # Get trial starts
                trigdiff = np.diff(trig)
                trial_ts = t[np.where(trigdiff > 0.7)]
                trial_ts_diff = np.append(np.diff(trial_ts), t[-1]) #append last t value in order to keep the last trial
                trial_ts = trial_ts[np.where(trial_ts_diff > 0.1)]
                
                ol_ts = trial_ts[1::2]
                
                ol_stimids = self._get_stimids(ol_ts, stim, nstims-1, window)
                ol_stimids += 1
                stimids = np.zeros(len(trial_ts)).astype(int)
                stimids[1::2] = ol_stimids
        else:
            trig = getattr(self.abf, trig_channel)
            if parse_channel == 'default':
                stim = trig
            else:
                stim = getattr(self.abf, parse_channel)
            trig_inds =fc.rising_trig(trig, trig_thresh)
            trig_select = np.ones(len(trig_inds)).astype(bool)
            trig_select[1:] = np.diff(trig_inds)*self.abf.sampling_period > min_trig_dist_s
            trig_inds = trig_inds[trig_select]
            trial_ts = t[trig_inds]
            stimids = self._get_stimids(trial_ts, stim, nstims, window)
        
        
        return trial_ts, stimids

    def _get_nrun(self, start_t, end_t):
        t = self.abf.t
        start_ind = np.where(t>=start_t)[0][0]
        end_ind = np.where(t>=end_t)[0][0]
        if hasattr(self.abf, 'temp'):
            trial_inds = np.arange(start_ind, end_ind)
            nrun = self.abf._is_hot(trial_inds)
        else:
            nrun = 0
        nrun = self.rundict[nrun]
        return nrun
    
    def _get_turndir(self, trial_start):
        turn_thresh = self.turn_thresh
        t = self.abf.t
        turning = getattr(self.abf, self.turn_channel)
        parse_inds = (t>=trial_start+self.parse_window[0]) & (t<trial_start+self.parse_window[1])
        baseline_inds = (t>=trial_start+self.baseline_window[0]) & (t<trial_start+self.baseline_window[1])
        dlmr = turning[parse_inds].mean() - turning[baseline_inds].mean()
        if dlmr > turn_thresh: return 'right'
        elif dlmr < -turn_thresh: return 'left'
        else: return 'straight'
    
    def _gen_parse_object(self):
        parse_obj = {
            stim: {
                nrun: { turndir: [] for turndir in self.turndirs}
                    for nrun in self.runs}
                for stim in self.stims}
        return parse_obj
    
    def _get_ts(self, tif=False):
        extra = 0
        if tif: extra = self.abf.subsampling_period
        ts = {stim: np.arange(-self.shoulder_lens[istim],
                             self.window_lens[istim] + self.shoulder_lens[istim] + extra,
                             self.abf.subsampling_period)
                   for istim, stim in enumerate(self.stimdict.values())}
        return ts
    
    def _parse_signal(self, t, signal, delta=False):
        self.turndir_bytrial = []
        self.stim_bytrial = []
        parsed_signal = self._gen_parse_object()
        sampling_rate = 1./np.diff(t).mean()
        for istim, stimid in enumerate(self.stimids):
            trial_start = self.trial_ts[istim]
            window_len = self.window_lens[stimid]
            shoulder_len = self.shoulder_lens[stimid]
            start_t = trial_start - shoulder_len
            end_t = trial_start + window_len + shoulder_len
            
            # For aligning with tif recording
            if hasattr(self, 't_last'):
                if (start_t < 0) or (end_t > self.abf.t_last): continue
                
            stim = self.stimdict[stimid]
            tstim = self.ts[stim]
            ninds = len(tstim)
            
            # Only add block if fly is flying > 95% of the time.
            if True:#self.active_bytrial[istim] or self.abf._is_active(start_t, end_t, 0.95):
                start_ind = np.where(t >= start_t)[0][0]
                end_ind = start_ind + ninds
                
                if end_ind < len(t):
                    nrun = self._get_nrun(start_t, end_t)
                    if self.turn_channel is None:
                        turndir = 'all'
                    else:
                        turndir = self._get_turndir(trial_start)
                    self.turndir_bytrial.append(turndir)
                    self.stim_bytrial.append(stim)
                    
                    for turnd in set(['all', turndir]):
                        s = signal[start_ind:end_ind]
                        if delta:
                            baseline_inds = (tstim>=self.baseline_window[0]) & (tstim<self.baseline_window[1])
                            s = s - s[baseline_inds].mean()
                        parsed_signal[stim][nrun][turnd].append(s)
                    
                    self.active_bytrial[istim] = True
        
        # Convert list of blocks into a 2D array for each stimulus in this run.
        for stim in self.stims:
            for nrun in self.runs:
                for turndir in self.turndirs:
                    parsed_signal[stim][nrun][turndir] = np.array(parsed_signal[stim][nrun][turndir])
                        
        return parsed_signal

#---------------------------------------------------------------------------#
# Walking

class Walk(ABF):
    angle_offset = 86  #used to be -274 deg
    arena_edge = 135

    def __init__(self, abffile, nstims=None, ball_diam_mm=6.34, **kwargs):
        ABF.__init__(self, abffile, **kwargs)
        self.ball_diam_mm = ball_diam_mm
        self.open(nstims=nstims)
    
    def open(self, nstims=None):
        # ojo: enforce_subsampling_rate normally == 50
        abf = ABF.open(self, enforce_subsampling_rate=20, delay_ms=13.6, nstims=nstims)
        headlabel = fc.contains(['ball_head', 'ball_headi'], abf.keys())
        
        self.head = volts2degs(np.array(abf[headlabel])[self.behaviour_inds], maxvolts=2)
        self.head = fc.wrap(self.head)
        self.headuw = fc.unwrap(self.head)
        self.dhead = fc.circgrad(self.head)*self.subsampling_rate
        self.dhead_abs = np.abs(self.dhead)
        self.d_dhead = np.diff(self.dhead)
        
        sidelabel = fc.contains(['ball_side', 'int_posy'], abf.keys())
        self.side = volts2degs(np.array(abf[sidelabel])[self.behaviour_inds], maxvolts=2)
        self.side = fc.wrap(self.side)
        self.dside = self.deg2mm(fc.circgrad(self.side))*self.subsampling_rate
        self.sideuw = self.deg2mm(fc.unwrap(self.side))
        
        forwlabel = fc.contains(['ball_forw', 'int_posx'], abf.keys())
        self.forw = volts2degs(np.array(abf[forwlabel])[self.behaviour_inds], maxvolts=2)
        self.forw = fc.wrap(self.forw)
        self.dforw = self.deg2mm(fc.circgrad(self.forw))*self.subsampling_rate
        self.d_dforw = np.diff(self.dforw)
        self.forwuw = self.deg2mm(fc.unwrap(self.forw))

        self.speed = np.hypot(self.dside, self.dforw)
        
        self.x, self.y = self.get_trajectory()
        
        aolabel = fc.contains(['ao', 'arena_sti', 'patname', 'stimid'], abf.keys())
        if aolabel:
            self.ao = np.array(abf[aolabel])[self.behaviour_inds]
            
            # ojo: should this be the case?
            self.stimid = self.ao
        
        # attribute for whether the fly is walking
        self.is_walking = self._is_walking(stasis_thresh = .5, walking_thresh = 1)
        self.is_static = self._is_static(dforw_thresh = 1, dhead_thresh = 50)
    
    def parse(self):
        pass
    
    def _is_active(self, start_t, end_t, time_thresh=0.95):
        # Check that fly is walking forward for at least 95% of the trial
        #t = self.t
        #if end_t > t.max(): return False
        #flight_trial = self.flight[(t>=start_t) & (t<end_t)]
        #flying = np.sum(flight_trial)/len(flight_trial) > time_thresh
        return True
    
    def deg2mm(self, deg):
        circumference_mm = self.ball_diam_mm*np.pi
        mm = deg*(circumference_mm/360)
        return mm
    
    def get_dxdy(self, head_rad, side_mm, forw_mm):
        theta0 = math.atan2(forw_mm, side_mm)
        thetaf = theta0 + head_rad
        r = np.hypot(side_mm, forw_mm)
        dx = r*math.cos(thetaf)
        dy = r*math.sin(thetaf)
        return dx, dy

    def get_xiyi(self, x0, y0, head_rad, side_mm, forw_mm):
        dx, dy = self.get_dxdy(head_rad, forw_mm, side_mm)
        xi = x0 + dx
        yi = y0 + dy
        return xi, yi
    
    def total_forw_distance(self):
        dist = self.deg2mm(fc.circgrad(self.forw)).sum()
        return dist
    
    def get_trajectory(self, proj_forw=False):
        x = np.zeros(len(self.head))
        y = np.zeros(len(self.head))
        head_rads = np.radians(self.head)
        dforw_mms = self.deg2mm(fc.circgrad(self.forw, np.diff))
        if proj_forw:
            dside_mms = np.zeros(len(self.dside))
        else:
            dside_mms = self.deg2mm(fc.circgrad(self.side, np.diff))
        
        for i in xrange(1, len(x)):
            headi = head_rads[i-1:i+1].mean()   # avg heading between i-1 and i
            dside = dside_mms[i-1]              # diff between i-1 and i
            dforw = dforw_mms[i-1]              # diff between i-1 and i
            
            xi, yi = self.get_xiyi(x[i-1], y[i-1], headi, dside, dforw)
            x[i] = xi
            y[i] = yi
        return x, y


def plot_trajectory(self, proj_forw=False, tlim = [0., 900.], cmap=plt.cm.jet,
                    pad=10, tticks=[0., 300., 600., 900.], ticklen=100, color_z = None, fig_filename = ''):
    if proj_forw:
        x, y = self.get_trajectory(proj_forw)
    else:
        x, y = self.x, self.y
    if tlim:
        ind0 = np.where(self.t >= tlim[0])[0][0]
        ind1 = np.where(self.t >= tlim[1])[0][0]
    else:
        ind0 = 0
        ind1 = len(x)
        tlim = [0., self.t[-1]]
    x = x[ind0:ind1]-x[ind0]
    y = y[ind0:ind1]-y[ind0]
    color_z = color_z[ind0:ind1]
    
    
    f = plt.figure(1, (10, 10))
    cf = f.gca()
    cf.set_aspect("equal")
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    colours = np.arange(len(x)) / float(len(x))
    ax = plt.subplot(111)
    # ttick_dist = np.ceil(self.t.max()/60/10/10)*10*60
    # tticks = np.arange(0, self.t.max(), ttick_dist)
    if tticks:
        tticks = [ttick for ttick in tticks if ttick <= tlim[1]]
        labels = [str(ind + 1) for ind in range(len(tticks))]
    else:
        tticks = tlim
        labels = ['start', 'end']
    tinds = np.array([0]+[np.where(np.diff(self.t<ttick))[0][0]-ind0 for ttick in tticks if ttick>tlim[0]])
    
    for label, xl, yl in zip(labels, x[tinds], y[tinds]):
        plt.annotate(
            label, 
            xy = (xl, yl), xytext = (20, -20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    
    ph.colorline(x, y, color_z, cmap, lw=2)
    
    plt.xticks(np.arange(-10000, 10000, ticklen))
    plt.yticks(np.arange(-10000, 10000, ticklen))
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    max_range = max(x_range, y_range)
    x_center = (x.max() + x.min())/2
    y_center = (y.max() + y.min())/2
    plt.gca().set_xlim((x_center-max_range/2-pad, x_center+max_range/2+pad))
    plt.gca().set_ylim((y_center-max_range/2-pad, y_center+max_range/2+pad))
#             add_scalebar(ax, sizex=1000, labelx='mm')
    plt.grid(c='black', alpha=.6)
#         rcParams['xtick.direction'] = 'out'
#         rcParams['ytick.direction'] = 'out'
    plt.suptitle(self.basename.split(os.path.sep)[-1])
    plt.xlabel('Distance (mm)', size=12, labelpad=20)
    if fig_filename:
        plt.savefig(fig_filename)
    
    return ax

    def plot_head_hist(self, binsize=5, forw_speed_thresh=1, tao=.06, tmin_min=20):
        inds = self.t > tmin_min*60
        forw_speed = gaussian_filter1d(self.forw_vel, sigma=tao*self.subsampling_rate)
        h, x = np.histogram(self.head[inds][forw_speed[inds]>forw_speed_thresh], bins=np.arange(-180, 181, binsize), range=[-180, 180], density=True)
        plt.bar(x[:-1], h/binsize, width=binsize, color=blue, lw=0)
        plt.xlim(-180, 180)
        plt.xticks(np.arange(-180, 181, 45))

def plot_total_forw_dist(genotypes):
    plt.figure(1, (8, 4))
    ax = plt.subplot(111)
    label_pos = np.arange(len(genotypes))
    labels = [genotype.label for genotype in genotypes]
    plt.xticks(label_pos, labels, rotation=45, ha='right')
    plt.xlim(-1, len(genotypes))
    plt.ylabel('Total forward distance travelled (mm)')
    plt.grid(ls='--', c='black', alpha=0.3)

    for igenotype, genotype in enumerate(genotypes):
        nflies = len(genotype)
        xpos = igenotype
        distances = [fly.total_forw_distance() for fly in genotype]
        
        x = np.random.normal(xpos, .01, nflies)
        plt.scatter(x, distances, lw=0)

def plot_heading_v_walkingdir(self, binsize=5, tlim=None, save=True):
    t = self.t
    if tlim is None: tlim=[t[0], t[-1]]
    idx = (t>=tlim[0]) & (t<tlim[-1])
    speed = np.sqrt(self.side_vel**2 + self.forw_vel**2)
    idx = idx & (speed > .5)
    angle = np.arctan2(self.side_vel[idx], self.forw_vel[idx])*180/np.pi
    angle = circ(angle)
    bins = np.arange(-180, 181, binsize)
    plt.figure(1, (7, 5))
    ph.set_tickdir('in')
    ax = plt.subplot(111)
    plt.suptitle(self.basename.split(os.path.sep)[0])
    a = plt.hist(self.xstim[idx], bins=bins, alpha=.7, color=blue, width=binsize, lw=0, normed=True, label='Bar')
    b = plt.hist(angle, bins=bins, alpha=.7, color=green, width=binsize, lw=0, normed=True, label='Walking Direction')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', frameon=False, fontsize=12)
    plt.xlabel('Azimuthal Orientation (deg)', labelpad=20)
    plt.ylabel('Density', rotation='horizontal', ha='right', labelpad=20)
    plt.xlim(-180, 180)
    plt.grid()
    
    if save:
        plt.savefig('%s_heading_v_walkingdir_hist_%.0f-%.0fs.png' %(self.basename, tlim[0], tlim[1]), bbox_inches='tight')

def plot_vel_v_xpos(self, tlim=None, save=True, sf=1, antifixation=False, binsize=10, velaxis='head', velmax=None):
    
    if not tlim:
        tlim = [self.t[0], self.t[-1]]
    t = self.t
    idx = (t>tlim[0]) & (t<tlim[1])
    
    # Setup Figure
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    plt.figure(1, (sf*7, sf*9))
    plt.suptitle(self.basename.split(os.path.sep)[0])
    gs = gridspec.GridSpec(2, 1,  height_ratios=[2, 15])
    gs.update(hspace=.1)
    
    
    # Plot histogram of xposition, if can find center, draw line through it
    xbins = np.arange(-180, 181, binsize)
    xb = xbins[:-1]+binsize/2   # Bin centers
    xstim = copy.copy(self.xstim)
    xstim = xstim[idx]
    h, _ = np.histogram(xstim, bins=xbins, density=True)
    ymax = np.ceil(h.max()*1000)/1000
    axhist = plt.subplot(gs[0, 0])
    if antifixation:
        xlim = [0, 360]
        xb[xb<0] = xb[xb<0] + 360
    else:
        xlim = [-180, 180]
    ph.draw_axes(axhist, ['left'], xlim=xlim, ylim=[0, ymax], yticks=[0, ymax])
    axhist.fill_between(xb, h, facecolor=dgrey, edgecolor=black, lw=1, alpha=.6)
    axhist.set_ylabel('Bar Position\nDistribution', rotation='horizontal', ha='right', labelpad=30)
    peak = fc.get_halfwidth_peak(xb, h)
    # axhist.axvline(peak, c='black', ls='--')
    
    
    # Scatter plot xspeed vs. xpos
    xvel = getattr(self, 'd'+velaxis)[idx]
    xspeed = np.abs(xvel)
    # Setup axis
    ax = plt.subplot(gs[1, 0])
    plt.grid()
    ax.set_xlabel('Bar Position (deg)', labelpad=20)
    ax.set_ylabel('Bar Speed (deg/s)', rotation='horizontal', ha='right', labelpad=20)
    # ax.axvline(peak, c='black', ls='--')
    ymin = xspeed.min()
    if velmax:
        ymax = velmax
    else:
        ymax = min(1000, xspeed.max())
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xlim)
    y = np.arange(ymin, ymax)
    if antifixation:
        xstim[xstim<0] = xstim[xstim<0] + 360
        ax.set_xticks(range(0, 361, 45))
        ax.fill_betweenx(y, x1=self.arena_edge, x2=360-self.arena_edge, color='black', alpha=.08)
    else:
        xlim = [-180, 180]
        ax.set_xticks(range(-180, 181, 45))
        ax.fill_betweenx(y, x1=-180, x2=-self.arena_edge, color='black', alpha=.08)
        ax.fill_betweenx(y, x1=self.arena_edge, x2=180, color='black', alpha=.08)
        
    ax.scatter(xstim, xspeed, c=ph.blue, lw=0, s=5)
    
    xvelmean = fc.binyfromx(xstim, xspeed, xbins, np.mean, nmin=10)
    xvelstd = fc.binyfromx(xstim, xspeed, xbins, stats.sem, nmin=10)
    ax.scatter(xb, xvelmean, s=20, c='black')
    ax.errorbar(xb, xvelmean, yerr=xvelstd, linestyle="None", c='black')
    
    # xaccmean = fc.binyfromx(xstim, np.abs(np.gradient(xvel)), xbins, np.mean)
    # ax.fill_between(xb, xaccmean, facecolor=bblue, lw=1, alpha=.5)
    # ax.plot(xb, xaccmean, color=dgrey, lw=1)
    
    
    if save:
        plt.savefig('%s_xspeed_v_xpos_%s_%.0f-%.0fs.png' %(self.basename, velaxis, tlim[0], tlim[1]), bbox_inches='tight')
 
def plot_speed_hist(self, bins=100, range=[0, 40], save=True):
    speed = np.sqrt(self.forw_vel**2 + self.side_vel**2)
    a = plt.hist(speed, bins=bins, range=range, lw=0, color=blue, normed=True)
    plt.grid()
    # plt.ylabel('Density', rotation='horizontal', ha='right', labelpad=20)
    # plt.xlabel('Speed (mm/s)', labelpad=20)
    plt.suptitle(self.basename.split(os.path.sep)[0])
    if save:
        plt.savefig('%s_speed_hist.png' %self.basename, bbox_inches='tight')


# Electrophysiology plotting functions.
def plot_vm_v_xstim(self, xbinsize=2, save=True):
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    xbinsize = 2
    xbins = np.arange(-180, 180, xbinsize)
    vmbinned = fc.binyfromx(self.xstim, self.vm, xbins, metric=np.mean)
    xb = xbins[:-1] + xbinsize
    plt.figure(1, (7, 6))
    plt.scatter(xb, vmbinned, c='none')
    plt.xlim(-180, 180)
    plt.xticks(np.arange(-180, 181, 45))
    plt.xlabel('Bar position (deg)')
    plt.ylabel('Vm (mV)')
    plt.grid()
    if save:
        plt.savefig('%s_vm_v_xstim.png' %self.basename, bbox_inches='tight')

class P2X2_Walk(Walk):
    def parse(self):
        channels = ['pressure', 'head', 'dhead', 'headint', 'side', 'forw']
        stimdict = {0: 'pulse'}
        window_lens = [22]
        shoulder_lens = [2]
        parse_window = [1, 2]
        baseline_window = [-.2, 0]
        args = [stimdict, window_lens, shoulder_lens, parse_window, baseline_window]
        kwargs = {
            'trig_channel': 'pressure_orig',
            'trig_thresh': 20
        }
        self.parsed = Parsed(self, channels, *args, **kwargs)

# Class for handling multiple flies in one genotype or condition.

class Genotype():
    def __init__(self, fnames, bvtype, label='', **kwargs):
        if type(fnames) is str:
            genotype = fnames
            fnames = [genotype + fname for fname in os.listdir(genotype) if fname.endswith('.abf')]

        self.fnames = fnames
        self.bvtype = bvtype
        self.label = label
        self.nflies = len(fnames)
        self.flies = []

        for fname in self.fnames:
            fly = bvtype(fname, **kwargs)
            fly.parse()
            self.flies.append(fly)
            print 'Finished parsing %s.' %fname

        if hasattr(self.flies[0], 'nstims'):
            self.nstims = self.flies[0].nstims
        self.subsampling_rate = self.flies[0].subsampling_rate

    def __getitem__(self, key):
        # if key is of invalid type or value, the list values will raise the error
        return self.flies[key]

    def __len__(self):
        return len(self.flies)

    def __iter__(self):
        return iter(self.flies)



#---------------------------------------------------------------------------#
# Flight

class Flight(ABF):
    angle_offset = -103.5
    arena_edge = 108
    def __init__(self, abffile, tracktype='strokelitude', nstims=None):
        ABF.__init__(self, abffile)
        self.tracktype = tracktype
        self.open(nstims=nstims)
    
    def open(self, nstims=None):
        if self.tracktype == 'strokelitude':
            abf = ABF.open(self, delay_ms=12)
            lwalabel = fc.contains(['lwa', 'wba_l'], abf.keys())
            self.lwa = np.array(abf[lwalabel])[self.subsampling_inds] * 135 / 4. - 45

            rwalabel = fc.contains(['rwa', 'wba_r'], abf.keys())
            self.rwa = np.array(abf[rwalabel])[self.subsampling_inds] * 135 / 4. - 45

            self.dhead = self.lwa - self.rwa

            tachlabel = fc.contains(['tach'], abf.keys())
            if tachlabel:
                self.tach = np.array(abf[tachlabel])[self.subsampling_inds]
            # self.flight = self.get_flight_inds()
        elif self.tracktype == 'wingbeatanalyzer':
            abf = read_abf(self.abffile)
            angle_offset=-157.5
            xstim = abf['stim_x']
            self.xstim = volts2degs(np.array(xstim), maxvolts=5)
            self.xstim = (self.xstim + angle_offset) % 360
            self.xstim[self.xstim>180] = self.xstim[self.xstim>180] - 360
            self.t = np.array(xstim.times)
            self.subsampling_period = float(xstim.sampling_period)
            self.subsampling_rate = float(xstim.sampling_rate)
            self.ystim = np.array(abf['stim_y'])
            self.lwa = np.array(abf['l_wba'])
            self.rwa = np.array(abf['r_wba'])
            self.lmr = self.lwa - self.rwa
            self.lpr = self.lwa + self.rwa
            self.wbf = np.array(abf['wbf'])*100   # 1V = 100 Hz
            self.ao = np.array(abf['ao'])
            self.arena_edge = 165
    
    def get_flight_inds(self, trim_start_s=0, trim_stop_s=0, wba_thresh=0, min_flight_len_s=1):
        min_ind_len = self.subsampling_rate*min_flight_len_s
        flying_inds = np.concatenate(
           fc.get_contiguous_inds((self.lwa>wba_thresh) & (self.rwa>wba_thresh),
            min_contig_len=min_ind_len, keep_ends=True,
            trim_left=trim_start_s*self.subsampling_rate,
            trim_right=trim_stop_s*self.subsampling_rate))
        flight = np.zeros(len(self.lwa)).astype(bool)
        flight[flying_inds] = True
        return flight
  
    def _is_active(self, start_t, end_t, time_thresh=0.95):
        # Check that fly is flying for at least 95% of the trial
        t = self.t
        if end_t > t.max(): return False
        flight_trial = self.flight[(t>=start_t) & (t<end_t)]
        flying = np.sum(flight_trial)/len(flight_trial) > time_thresh
        return flying

    def fix_errors(self, wba, lodiff_thresh=35, hidiff_thresh=-100, lowba_thresh=30, hiwba_thresh=70):
        
        # Find errors. Tend to occur (1) as large positive deviations with low wbas and (2) as large negative deviations with high wbas.
        wba_diff = np.diff(wba)
        lo_jumps = np.where((wba[:-1] < lowba_thresh) & (wba_diff > lodiff_thresh))[0]+1
        hi_jumps = np.where((wba[:-1] > hiwba_thresh) & (wba_diff < hidiff_thresh))[0]+1
        #jumps = np.where(lo_jumps | hi_jumps)[0]+1
        
        errors = []
        for jump in lo_jumps:
            idx = jump
            baseline = wba[idx-1]
            if baseline > -44:
                while (wba[idx] - baseline) > lodiff_thresh:
                    errors.append(idx)
                    idx += 1
                    if idx > len(wba)-1:
                        break
        
        #for jump in hi_jumps:
        #    idx = jump
        #    baseline = wba[idx-1]
        #    while (wba[idx] - baseline) < hidiff_thresh:
        #        errors.append(idx)
        #        idx += 1
        #        if idx > len(wba)-1:
        #            break
        #
        #errors.sort()

        
        # Find the borders of each continuous stretch of errors.
        errors_bool = np.zeros(len(wba))
        errors_bool[errors] = 1
        errors_diff = np.diff(errors_bool)
        errors_borders = [np.where(errors_diff==1)[0], np.where(errors_diff==-1)[0]+1]
        
        # Replace each stretch of errors with a line joining either end (before and after the errors occurred).
        wba_fixed = wba.copy()
        for left, right in zip(errors_borders[0], errors_borders[1]):
            filler = np.linspace(wba[left], wba[right], right-left+1)
            wba_fixed[left:right] = filler[:-1]
        
        return wba_fixed
    
    def plot_wba_fixes(self, xlim=[]):
        fig = plt.figure(1, (18, 6))
        ax1 = plt.subplot(211)
        plt.plot(self.t, self.lwa, c=ph.red, lw=1.5)
        plt.plot(self.t, self.lwa_fixed, c=ph.blue, lw=1.5)
        keep_axes(ax1, ['left'])
        plt.ylim([-60, 90])
        plt.ylabel('LWA')
        if xlim:
            plt.xlim(xlim)
        #ax1.axhline(70, color=grey, ls='--')
        
        ax2 = plt.subplot(212)
        plt.plot(self.t, self.rwa, c=ph.red, lw=1.5)
        plt.plot(self.t, self.rwa_fixed, c=ph.blue, lw=1.5)
        keep_axes(ax2, ['left', 'bottom'])
        plt.ylabel('RWA (deg)')
        plt.ylim([-60, 90])
        plt.xlabel('Time (s)')
        #ax2.axhline(70, color=grey, ls='--')
        if xlim:
            plt.xlim(xlim)


class GratingExpandingDisk_Flight(Flight):
    angle_offset = 122.27
    def parse(self):
        channels = ['lmr', 'dlmr', 'xstim']
        stimdict = {0: 'left', 1: 'grating', 2: 'center', 3: 'right'}
        window_lens = [0.7]*4
        shoulder_lens = [2]*4
        parse_window = [.4, .6]
        baseline_window = [0, .2]
        args = [stimdict, window_lens, shoulder_lens, parse_window, baseline_window]
        kwargs = {
            'turn_thresh': 6
        }
        self.parsed = Parsed(self, channels, *args, **kwargs)

class ExpandingDisk_Flight(Flight):
    angle_offset=122.27
    def parse(self):
        channels = ['dlmr', 'xstim', 'ystim']
        stimdict = {0: 'left', 1: 'center', 2: 'right'}
        window_lens = [0.7]*3
        shoulder_lens = [1.4]*3
        parse_window = [.3, .4]
        baseline_window = [0, .2]
        args = [stimdict, window_lens, shoulder_lens, parse_window, baseline_window]
        kwargs = {
            'turn_thresh': 6
        }
        self.parsed = Parsed(self, channels, *args, **kwargs)    

class BarsLRCL_Flight(Flight):
    def parse(self):
        channels = ['dlmr', 'lmr', 'xstim']
        stimdict = {0: 'cl', 1: 'left', 2: 'right'}
        window_lens = [3]*3
        shoulder_lens = [3]*3
        parse_window = [0, 3]
        baseline_window = [-1, 0]
        args = [stimdict, window_lens, shoulder_lens, parse_window, baseline_window]
        kwargs = {
            'turn_thresh': 6
        }
        self.parsed = Parsed(self, channels, *args, **kwargs)
    
class BarsLCRGratingsCL_Flight(Flight):
    def parse(self):
        channels = ['dlmr', 'lmr', 'xstim']
        stimdict = {0: 'cl', 1: 'left', 2: 'grating', 3: 'center', 4: 'right'}
        window_lens = [3]*5
        shoulder_lens = [0]*5
        parse_window = [0, 3]
        baseline_window = [-1, 0]
        args = [stimdict, window_lens, shoulder_lens, parse_window, baseline_window]
        kwargs = {
            'turn_thresh': 6
        }
        self.parsed = Parsed(self, channels, *args, **kwargs)
    
class ClosedLoop_Flight(Flight):
    def parse(self):
        try:
            self.xstim
        except AttributeError:
            self.open_abf()
        
        #Remove sections where there is no flight
        if self.tracktype == 'strokelitude':
            inds = np.where((self.lwa > 0) & (self.rwa > 0))
        elif self.tracktype == 'wingbeatanalyzer':
            inds = np.where((self.wbf > 50))
        self.xstim_active = self.xstim[inds]

    def bin_xstim(self, twindow=2000, tshift=100):
        msec2inds_ratio = self.subsampling_rate/1000.
        iwindow = twindow*msec2inds_ratio # Convert twindow from msec to number of indices.
        ishift = tshift*msec2inds_ratio   # Convert tshift from msec to number of indices.
        theta_offset = 90 # Zero deg = front of arena, but 0 deg in unit circle is on the right, so rotate 90 CC.
        theta = np.deg2rad((self.xstim_active + theta_offset)%360)
        x = np.cos(theta)
        x_binned = np.array([np.mean(x[i : i+iwindow]) for i in np.arange(0, len(x), ishift)])
        y = np.sin(theta)
        y_binned = np.array([np.mean(y[i : i+iwindow]) for i in np.arange(0, len(y), ishift)])
        
        return x_binned, y_binned

    def polar_plot(self, vmax=None, binsize=3, fig_filename=''):
        
        # Generate polar bins.
        nbins=360/binsize
        r = [7, 10]
        theta = np.linspace(0, 2*np.pi, nbins)
        
        # "Grid" r and theta into 2D arrays
        r, theta = np.meshgrid(r, theta)
        
        # Compute histogram for stimulus position.
        hist, bins = np.histogram(self.xstim_active, bins=nbins, range=(0, 360), density=True)
        hist = np.array([[h, h] for h in hist])
        
        # Plot polar histogram.
        fig = plt.figure(1, (10, 10))
        ax = fig.add_subplot(111, projection='polar')
        cmap = cm.get_cmap('RdBu_r')
        if vmax:
            cax = ax.pcolormesh(theta, r, hist, antialiased=True, cmap=cmap, vmin=0, vmax=vmax)
        else:
            cax = ax.pcolormesh(theta, r, hist, antialiased=True, cmap=cmap)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi/2) # Zero deg = front of arena, but 0 deg in unit circle is on the right, so rotate 90 CC.
        ax.set_xticks(np.pi/180. * np.linspace(180,  -180, 8, endpoint=False))
        plt.ylim([0, 10])
        plt.yticks([])
        fig.colorbar(cax, pad=.15, shrink=.7)
        plt.show()
        
        if fig_filename:
            plt.savefig(fig_filename)#, bbox_inches='tight')
        
        return fig, ax

    def polar_hexplot_wvar(self, fig_filename='', twindow=2000, tshift=100, vmax=.05, show_circle=True,
                        gridsize=35, gs=None, gridind=None):

        # Ensure that self data has been processed before plotting
        try:
            self.xstim_active
        except AttributeError:
            try:
                self.xstim
            except AttributeError:
                self.open_abf()
            self.process_xstim()
        
        # Bin data in time
        
        x_binned, y_binned = self.bin_xstim(twindow=twindow, tshift=tshift)
        
        # Create Figure
        plotsize = 6
        if gs and gridind:
            ax1 = plt.subplot(gs[gridind[0], gridind[1]], axisbg=plt.cm.RdBu_r(0))
        elif bool(gs) != bool(gridind):
            sys.exit('Need both gridspec handle and index.')
        else:
            fig = plt.figure(1, (plotsize, plotsize))
            ax1 = fig.add_subplot(111, axisbg=plt.cm.RdBu_r(0))
    #     gridsize = (1, plotsize+1)
    #     ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=plotsize, axisbg=plt.cm.RdBu_r(0))
        
        
        # Plot histogram for stimulus position.
        c = np.ones(len(x_binned))/len(x_binned)   # Scales each point by the total number of points.
        hexax = plt.hexbin(x_binned, y_binned, c, cmap=plt.cm.RdBu_r, gridsize=gridsize, extent=[-1, 1, -1, 1], 
                         reduce_C_function=np.sum, vmin=0, vmax=vmax)
        # Show circle outline, radius=1
        if show_circle:
            circle1=plt.Circle((0,0), 1.05, fill=False)
            ax1.add_artist(circle1)
        keep_axes(ax1, [])
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])

    #     ax2 = plt.subplot2grid(gridsize, (0, plotsize))
    #     fig.colorbar(hexax, cax=ax2)
        
        if fig_filename:
            plt.savefig(fig_filename, bbox_inches='tight')
        
        #plt.show()
        
        return x_binned, y_binned

    def time_plot(self, xlim=[], gs=None, gs_idx=None, axes_to_keep=[], fig_filename=''):
        
        # Ensure that self data has been processed before plotting
        try:
            self.xstim
        except AttributeError:
            self.open_abf()
        
        fig = plt.figure(1, (20, 5))
        if gs:
            ax = plt.subplot(gs[gs_idx[0], gs_idx[1]])
            keep_axes(ax, axes_to_keep)
        else:
            ax = plt.subplot(111)
            axes_to_keep = ['left', 'bottom']
        if axes_to_keep:
            plt.ylabel('Stimulus position (deg)\n0=center')
            plt.xlabel('Time (s)')
            plt.yticks([-180, -90, 0, 90, 180])
        keep_axes(ax, axes_to_keep)
        plt.fill_between(self.t, -self.arena_edge, self.arena_edge, facecolor=(0.95, 0.95, 0.95), edgecolor=grey)
        ax.axhline(c=red)
        plt.plot(self.t, self.xstim, c=ph.blue)
        plt.ylim(-180, 180)
        
        if xlim:
            plt.xlim(xlim)
        
        
        if fig_filename:
            plt.savefig(fig_filename, bbox_inches='tight')


# Helper functions for plotting multiple ClosedLoopWingBeats.
def plot_closedloop_polar_hist(flies, cols=5, twindow=2000, tshift=100, vmax=.02, fig_filename=''):
    nflies = len(flies)
    rows = int(np.ceil((nflies+1)/float(cols)))
    sf=2.5
    fig = plt.figure(1, (cols*sf, rows*sf), facecolor=plt.cm.RdBu_r(0))
    
    cls = [fly.bin_xstim() for fly in flies]
    
    gs = gridspec.GridSpec(rows, cols)
    
    for i, cl in enumerate(cls):
        ax = plt.subplot(gs[i/cols, i%cols], axisbg=plt.cm.RdBu_r(0))
        x_binned, y_binned = cl
        h2d, xedges, yedges = np.histogram2d(y_binned, x_binned, bins=40, normed=True, range=[[-1.2, 1.2], [-1.2, 1.2]])
        h2d /= np.sum(h2d)
        plt.pcolormesh(yedges, xedges, h2d, cmap=plt.cm.RdBu_r, vmax=vmax)
        circle1=plt.Circle((0,0), 1.05, fill=False)
        ax.add_artist(circle1)
        keep_axes(ax, [])
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
    

    
    i += 1
    axsum = plt.subplot(gs[i/cols, i%cols], axisbg=plt.cm.RdBu_r(0))
    keep_axes(axsum, [])
    xsum_binned = np.concatenate([x for x, _ in cls]).flatten()
    ysum_binned = np.concatenate([y for _, y in cls]).flatten()
    h2d, xedges, yedges = np.histogram2d(ysum_binned, xsum_binned, bins=40, normed=True, range=[[-1.2, 1.2], [-1.2, 1.2]])
    h2d /= np.sum(h2d)
    plt.pcolormesh(yedges, xedges, h2d, cmap=plt.cm.RdBu_r, vmax=vmax)
    circle1=plt.Circle((0,0), 1.05, fill=False, edgecolor=black, lw=2)
    axsum.add_artist(circle1)
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    
    if fig_filename:
        plt.savefig(fig_filename, facecolor=plt.cm.RdBu_r(0), bbox_inches='tight')

def plot_closedloop_polar_hist2(genotypes, genotype_labels, cols=5, twindow=2000, tshift=100, vmax=.02, fig_filename=''):
    ngenotypes = len(genotypes)
    rows = int(np.ceil((ngenotypes+1)/float(cols)))
    sf=2.5
    fig = plt.figure(1, (cols*sf, rows*sf*.9), facecolor=plt.cm.RdBu_r(0))
    gs = gridspec.GridSpec(rows, cols)
    gs.update(hspace=0.8)
    
    for i, genotype in enumerate(genotypes):
        ax = plt.subplot(gs[i/cols, i%cols], axisbg=plt.cm.RdBu_r(0))
        cls = [fly.bin_xstim() for fly in genotype]
        xsum_binned = np.concatenate([x for x, _ in cls]).flatten()
        ysum_binned = np.concatenate([y for _, y in cls]).flatten()
        h2d, xedges, yedges = np.histogram2d(ysum_binned, xsum_binned, bins=40, normed=True, range=[[-1.2, 1.2], [-1.2, 1.2]])
        h2d /= np.sum(h2d)
        plt.pcolormesh(yedges, xedges, h2d, cmap=plt.cm.RdBu_r, vmax=vmax)
        circle1=plt.Circle((0,0), 1.05, fill=False, edgecolor=black)
        ax.add_artist(circle1)
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        keep_axes(ax, [])
        plt.xlabel(genotype_labels[i])
        ax.xaxis.label.set_color('white')
    
    if fig_filename:
        plt.savefig(fig_filename, facecolor=plt.cm.RdBu_r(0), dpi=100, bbox_inches='tight')

def plot_closedloop_time_hist(flies, plot_yeq0=True, gridsize=100, vmax=30, fig_filename=''):
    
    # Parse arguments.
    if type(flies) is not list:
        flies = [flies]
    
    nflies = len(flies)
    nrows = 1
    ngenotypes = 1
    
    
    # Setup Figure.
    scalefactor = 3
    fig = plt.figure(1, (7, 4), facecolor=plt.cm.RdBu_r(0))
    gs = gridspec.GridSpec(nrows, ngenotypes)
    
    genotype = 0
    
    # Plot all flies on same plot.
    axsum = plt.subplot(gs[0, genotype], axisbg=plt.cm.RdBu_r(0))
    if genotype==0:
        keep_axes(axsum, ['left', 'bottom'], 'white')
        plt.xticks([0, 60, 120, 180, 240, 300, 360])
        
        plt.ylabel('All Flies\nStimulus Position (deg)')
        plt.yticks([-90, 0, 90])
    else:
        keep_axes(axsum, [])
    plt.ylim([-180, 180])
    
    plt.xlabel(('Time (s)   %i flies') %nflies)
    # Plot all flies together
    cl_secs = min([max(fly.t) for fly in flies])
    cl_secs = 200
    cl_inds = cl_secs*100
    plt.xlim([0, cl_secs])
    xstims = [fly.xstim[:cl_inds] for fly in flies]
    xstims = np.concatenate(xstims)
    tcl = fly.t[:cl_inds]
    tcls = list(tcl)*nflies
    hexax = plt.hexbin(tcls, xstims, cmap=plt.cm.RdBu_r, gridsize=gridsize, extent=[0, cl_secs, -180, 180], 
                             reduce_C_function=np.sum, vmax=vmax*nflies)
    if plot_yeq0:
                yeq0 = np.zeros(cl_inds)
                plt.plot(tcl, yeq0, color=green, linestyle='--')
        
    if fig_filename:
        plt.savefig(fig_filename, bbox_inches='tight', facecolor=plt.cm.RdBu_r(0))
    plt.show()

def plot_closedloop_timeplots(flies, fig_filename=''):
    
    sf = 1.2
    nrows = len(flies)
    fig = plt.figure(1, (12*sf, len(flies)*sf))
    gs = gridspec.GridSpec(nrows, 1)
    gs.update(hspace=0.1)
    for ifly, fly in enumerate(flies[:10]):
        if ifly == 9:
            axes_to_keep = ['left', 'bottom']
        else:
            axes_to_keep = []
        fly.time_plot(gs=gs, gs_idx=(ifly, 0), axes_to_keep=axes_to_keep)
        plt.xlim([0, 360])
        if ifly == 9:
            plt.xticks([0, 60, 120, 180, 240, 300, 360])
    if fig_filename:
        plt.savefig(fig_filename, bbox_inches='tight')
    plt.show()


# Helper functions for plotting Flight objects.

class Plot():
    def __init__(self, flies):
        self.flies = flies
    
    def set_axes(self, ax, signal, show_axes=[], color='white', xlabel='default', ylabel='Time (s)',
                 tlim=None):
        if 'lmr' in signal:
            self.xlimits = 50
            xticks = [-25, 0, 25]
            if xlabel == 'default':
                xlabel = 'L-R WBA (deg)'
                if signal[0]=='d': xlabel = 'd' + xlabel
        elif signal == 'xstim':
            self.xlimits = 180
            xticks = [-90, 0, 90]
            if xlabel == 'default': xlabel = 'Stimulus\nPosition (deg)'
            ax.axvline(color=grey, linestyle='--')
        elif signal == 'ystim':
            xticks = 'auto'
            if xlabel == 'default': xlabel = 'Stimulus\nPosition (V)'
        
        fly = self.flies[0]
        t = fly.tol
        window_len = fly.window_lens[0]
        yticks = np.arange(0, fly.window_lens[0]+1)
        ph.draw_axes(ax, show_axes, xlabel, ylabel, xticks, yticks, color=color)
        if signal in ['lmr', 'xsitm']:
            ax.set_xlim([-self.xlimits, self.xlimits])
        if tlim:
            ax.set_ylim(tlim)
        else:
            ax.set_ylim([t[0], t[-1]])
    
    def hist2d(self, ax, signal, stim, nrun='cool', turndir='all', show_axes=[], color='white',
               cmap=plt.cm.RdBu_r, vmax=.015, normed=True, xlabel='default',
               tlim=None, first_nblocks=200):
        fly = self.flies[0]
        t = fly.tol
        
        runs = []
        for fly in self.flies:
            trials = fly.parsed[signal][stim][nrun][turndir][:first_nblocks]
            if trials.shape[0]:
                runs.append(trials)
        if len(runs) == 0: return 0
        runs = np.concatenate(runs)
        ntrials = len(runs)
        
        runs = runs.flatten()
        ntrials = len(runs)/len(t)
        ts = list(t)*ntrials
        self.set_axes(ax, signal, show_axes, color, xlabel)
        xbins = np.linspace(-self.xlimits, self.xlimits, 100)
        ybinsize = .02
        ybins = np.arange(t[0], t[-1], ybinsize)
        h2d, xedges, yedges, _ = ax.hist2d(runs, ts, bins=[xbins, ybins], normed=normed,
                                           cmap=cmap, vmax=vmax)
        ax.set_ylabel('%i trials' %ntrials, rotation='horizontal', ha='center')
        if tlim:
            ax.set_ylim(tlim)
        ax.invert_yaxis()
        return h2d, xbins, ybins
    
    def hist(self):
        pass
    
    def lines(self, ax, signal, stim, nrun='cool', turndir='all', show_axes=[], color='black',
              tlim=None):
        self.set_axes(ax, signal, show_axes, color, tlim=tlim)
        t = self.flies[0].tol
        for fly in self.flies:
            for trial in fly.parsed[signal][stim][nrun][turndir]:
                if trial.shape[0]:
                    ax.plot(trial, t, c=color)
        ax.invert_yaxis()
    
def plot_lmr_each_fly(flies, stimid=None, lmr_center=True, fig_filename=''):
    
    # Parse arguments.
    if type(flies) is not list:
        flies = [flies]
    
    nstims = flies[0].nstims
    nflies = len(flies)
    tol = flies[0].tol
    
    if stimid is None:
        stimids = range(nstims)
    else:
        stimids = [stimid]
    
    # Setup Figure.
    fig = plt.figure(1, (nstims*2, (nflies+3)))
    gs = gridspec.GridSpec(nflies+3, nstims)
    
    nrun=0
    for stimid in stimids:
        for nfly, fly in enumerate(flies):
            
            if lmr_center:
                # Calculate mean L-R WBA for bar oscillating around the center and use this value to draw a dashed line.
                lmr_center_mean = np.mean(fly.parsed['lmr'][1][0])
                yeq0 = np.ones(len(tol))*lmr_center_mean
            else:
                # Use y=0 as the dashed line.
                yeq0 = np.zeros(len(tol))
                
            # Plot lmr wba.
            ax1 = plt.subplot(gs[nfly, stimid])
            plt.ylim([-40, 40])
            plt.yticks([-20, 0, 20])
            if nfly == 0 and stimid==0:
                keep_axes(ax1, ['left'])
                plt.ylabel('L-R WBA')
            else:
                keep_axes(ax1, [])
            for block in range(len(fly.parsed['lmr'][stimid][nrun])):
                plt.plot(tol, fly.parsed['lmr'][stimid][nrun][block], color=grey)
            lmr_mean = np.mean(fly.parsed['lmr'][stimid][nrun], axis=0)
            plt.plot(tol, lmr_mean, color=blue, linewidth=1.5)
            plt.plot(tol, yeq0, color=grey, linestyle='--')
        
        # Plot stimulus position.
        ax2 = plt.subplot(gs[nflies, stimid])
        if stimid == 0:
            keep_axes(ax2, ['left', 'bottom'])
            plt.ylabel('Stimulus\nPosition (deg)')
            plt.xlabel('Time (s)')
            plt.yticks([-90, 0, 90])
        else:
            keep_axes(ax2, ['bottom'])
        
        for fly in flies:
            nblocks = len(fly.parsed['xstim'][stimid][nrun])
            for block in range(nblocks):
                xstim = fly.parsed['xstim'][stimid][nrun][block]
                plt.plot(tol, xstim, color=(.3, .3, .3))
            plt.plot(tol, yeq0, color=grey, linestyle='--')
            plt.ylim([-180, 180])
            plt.xticks([0, 1, 2, 3])
    if fig_filename:
        plt.savefig(fig_filename, bbox_inches='tight')

def plot_lmr_mean_ax_flies(flies, stimid, show_std=False, show_stim=False, fig_filename=''):
    tol = flies[0].tol
    tlim = [tol[0], tol[-1]]
    fig = plt.figure(1, (20, 10))
    lmr_means_all_flies = np.array([fly.get_lmr_means() for fly in flies])
    lmr_means_ax_flies = np.mean(lmr_means_all_flies, axis=0)
    lmr_stds_ax_flies = np.std(lmr_means_all_flies, axis=0)
    
    
    nrows = 2
    ncols = lmr_means_ax_flies.shape[1]
    gs = gridspec.GridSpec(nrows, ncols, height_ratios=[2,1])
    
    xstim_cools = []
    xstim_hots = []
    for fly in flies:
        temps = fly.parsed['temp']['openloop'][stimid][0].mean(axis=1)
        xstim = fly.parsed['xstim']['openloop'][stimid][0]
        xstim_cools.append(xstim[temps<27])
        xstim_hots.append(xstim[temps>27])
    xstim_cools = np.concatenate(xstim_cools)
    xstim_hots = np.concatenate(xstim_hots)
    xstims = [xstim_cools, xstim_hots]
    
    colours = [ph.blue, ph.orange]
    for nrun in range(ncols):
        # Plot WBA
        axwba = plt.subplot(gs[0, nrun])
        axwba.set_ylim([-45, 45])
        axwba.set_xlim(tlim)
        xticks = np.arange(0, tol[-1])
        xlabel = 'Time (s)'
        if nrun == 0:
            ph.draw_axes(axwba,
                         ['left'],
                         yticks=[-25, 0, 25])
            if not show_stim:
                ph.draw_axes(axwba,
                             ['left', 'bottom'],
                             xlabel=xlabel,
                             xticks=np.arange(0, tlim[-1]),
                             yticks=[-25, 0, 25])
            axwba.set_ylabel('L-R WBA\n(deg)', rotation='horizontal', ha='right')
        else:
            ph.draw_axes(axwba, [])
        
        lmr_mean = lmr_means_ax_flies[stimid, nrun]
        axwba.plot(tol, lmr_means_ax_flies[stimid, nrun], color=colours[nrun], linewidth=2)
        axwba.axhline(color=grey, linestyle='--')
        
        if show_std:
            lmr_std = lmr_stds_ax_flies[stimid, nrun]
            std_bottom = lmr_mean - lmr_std
            std_top = lmr_mean + lmr_std
            axwba.fill_between(tol, std_bottom, std_top, facecolor=(.9, .9, .9), edgecolor='white')
        
        
        # Plot Stimulus
        if show_stim:
            axstim = plt.subplot(gs[1, nrun])
            axstim.set_ylim([-180, 180])
            axstim.set_xlim(tlim)
            if nrun == 0:
                ph.draw_axes(axstim,
                          ['left', 'bottom'],
                          xlabel=xlabel,
                          xticks=xticks,
                          yticks=[-90, 0, 90])
                axstim.set_ylabel('Stimulus\nPosition (deg)', rotation='horizontal', ha='right')
            else:
                ph.draw_axes(axstim,
                             ['bottom'],
                             xticks=np.arange(0, tol[-1]))
            
            xstim = xstims[nrun]
            for trial in xstim:
                axstim.plot(tol, trial, c='black')

            axstim.axhline(color=grey, linestyle='--')
            plt.ylim([-180, 180])
            plt.xticks([0, 1, 2, 3])
                
    
    if fig_filename:
        plt.savefig(fig_filename, bbox_inches='tight')
    
def plot_openloop_hist(flies, show_all_flies=False, nrun=0, stimids=None, plot_lmr_center=False, plot_yeq0=False,
                       plot_lmr_mean=False, gridsize=100, vmax=12, fig_filename=''):
    
    # Parse arguments.
    if type(flies) is not list:
        flies = [flies]
    
    nflies = len(flies)
    if show_all_flies:
        nfly_rows = nflies
    else:
        nfly_rows = 0
    nstims = flies[0].nstims
    tol = flies[0].tol
    
    if stimids is None:
        stimids = range(nstims)
    elif type(stimids) is not list:
        stimids = [stimids]

    
    # Setup Figure.
    scalefactor = 3
    fig = plt.figure(1, (nstims*scalefactor*2, (nfly_rows+2)*scalefactor), facecolor=plt.cm.RdBu_r(0))
    gs = gridspec.GridSpec(nfly_rows+2, nstims)
    
    for stimid in stimids:
        if show_all_flies:
            for nfly, fly in enumerate(flies):
                
                if plot_lmr_center:
                    # Calculate mean L-R WBA for bar oscillating around the center and use this value to draw a dashed line.
                    lmr_center_mean = np.mean(fly.parsed['lmr'][1][0])
                    yeq0 = np.ones(len(tol))*lmr_center_mean
                else:
                    # Use y=0 as the dashed line.
                    yeq0 = np.zeros(len(tol))
                
                
                # Plot lmr wba.
                ax1 = plt.subplot(gs[nfly, stimid])
                plt.ylim([-50, 50])
                
                if nfly == 0 and stimid==0:
                    keep_axes(ax1, ['left'], 'white')
                    plt.yticks([-20, 0, 20])
                    plt.ylabel('L-R WBA')
                else:
                    keep_axes(ax1, [])
                
                if plot_lmr_mean:
                    lmr_mean = np.mean(fly.parsed['lmr'][stimid][nrun], axis=0)
                    plt.plot(tol, lmr_mean, color=green, linewidth=2)
                if plot_lmr_center or plot_yeq0:
                    plt.plot(tol, yeq0, color=grey, linestyle='--')
                
                run = fly.parsed['lmr'][stimid][nrun]
                tols = list(tol)*np.shape(run)[0]
                runs = run.flatten()
                try:
                    hexax = plt.hexbin(tols, runs, cmap=plt.cm.RdBu_r, gridsize=gridsize, extent=[-1, 4, -80, 80], 
                                         reduce_C_function=np.sum, vmax=vmax)
                except ValueError:
                    print 'Error in generating hexbin.'
        
        # Plot all flies on same plot.
        axsum = plt.subplot(gs[-2, stimid], axisbg=plt.cm.RdBu_r(0))
        if stimid==0:
            keep_axes(axsum, ['left'], 'white')
            plt.yticks([-25, 0, 25])
            plt.ylabel('All Flies\nL-R WBA (deg)')
        else:
            keep_axes(axsum, [])
        plt.ylim([-50, 50])
        # Calculate number of trials in order to normalize.
        ntrials = 0
        for fly in flies:
            itrials = np.shape(fly.parsed['lmr'][stimid][nrun])[0]
            ntrials += itrials
        runs = [fly.parsed['lmr'][stimid][nrun].flatten() for fly in flies]
        runs = np.concatenate(runs)
        ntrials = len(runs)/len(tol)
        tols = list(tol)*ntrials
        try:
            hexax = plt.hexbin(tols, runs, cmap=plt.cm.RdBu_r, gridsize=gridsize, extent=[-1, 4, -80, 80], 
                                 reduce_C_function=np.sum, vmax=vmax*ntrials/20)
        except ValueError:
            print shape(runs), shape(tols)
            
        
        
        # Plot stimulus position.
        ax2 = plt.subplot(gs[-1, stimid], axisbg=plt.cm.RdBu_r(0))
        if stimid == 0:
            keep_axes(ax2, ['left', 'bottom'], 'white')
            plt.ylabel('Stimulus\nPosition (deg)')
            plt.xlabel('Time (s)')
            plt.yticks([-90, 0, 90])
        else:
            keep_axes(ax2, ['bottom'], 'white')
        
        for fly in flies:
            nblocks = len(fly.parsed['xstim'][stimid][nrun])
            for block in range(nblocks):
                xstim = fly.parsed['xstim'][stimid][nrun][block]
                plt.plot(tol, xstim, color='white')
            yeq0stim = np.zeros(len(tol))
            plt.plot(tol, yeq0stim, color=grey, linestyle='--')
            plt.ylim([-180, 180])
            plt.xticks([0, 1, 2, 3])
    if fig_filename:
        plt.savefig(fig_filename, bbox_inches='tight', facecolor=plt.cm.RdBu_r(0))
    plt.show()

def plot_openloop_histv1(genotype, stims='all', nrun='cool', turndir='all', show_stimulus=True, show_all_flies=True, 
                         show_axes=True, plot_dlmr=False, vmaxind=60, vmaxsum=.01, cmap=plt.cm.RdBu_r, tlim=None,
                         color='white', first_nblocks=200, fig_filename=''):
    
    # Parse arguments.
    nflies = len(genotype)
    fly = genotype[0]
    if show_all_flies:
        ncols = nflies + 2
    else:
        ncols = 2
    nstims = genotype[0].nstims
    
    if stims == 'all':
        stims = fly.stimdict.values()
    elif type(stims) is not list:
        stims = [stims]

    
    # Setup Figure.
    tlen = genotype[0].tol.max() - genotype[0].tol.min()
    fig = plt.figure(1, (ncols*3, nstims*3), facecolor=cmap(0))
    gs = gridspec.GridSpec(nstims, ncols)
    
    for istim, stim in enumerate(stims):
        # Plot stimulus position.
        if show_stimulus:
            axstim = plt.subplot(gs[istim, 0], axisbg=plt.cm.RdBu_r(0))
            show_axes = []
            if istim == 0:
                show_axes.extend(['top', 'left'])
            p = Plot(genotype)
            p.lines(axstim, 'xstim', stim, nrun, turndir, show_axes, color, tlim=tlim)
    
        
        # Plot sum of all flies on same plot.
        axsum = plt.subplot(gs[istim, 1], axisbg=cmap(0))
        show_axes = []
        if istim == 0:
            show_axes.append('top')
            if not show_stimulus:
                show_axes.append('left')
        elif istim == len(stims)-1:
            axsum.set_xlabel(genotype.label)
        p = Plot(genotype)
        if plot_dlmr:
            signal = 'dlmr'
        else:
            signal = 'lmr'
        p.hist2d(axsum, signal, stim, nrun, turndir, show_axes,
                    color, cmap, vmaxsum,
                    tlim=tlim, first_nblocks=first_nblocks)
        
        # Plot all flies.
        if show_all_flies:
            for ifly, fly in enumerate(genotype):
                
                # Plot lmr wba.
                axsingle = plt.subplot(gs[istim, ifly+2], axisbg=plt.cm.RdBu_r(0))
                p = Plot([fly])
                
                p.hist2d(axsingle, signal, stim, nrun, turndir, show_axes,
                    color, cmap, vmaxind, normed=False,
                    xlabel=fly.basename.split(os.path.sep)[-1], tlim=tlim, first_nblocks=first_nblocks)
            
    if fig_filename:
        plt.savefig(fig_filename, bbox_inches='tight', facecolor=cmap(0))

def plot_openloop_histv2(genotypes, stims='all', nrun='cool', turndir='all', show_1d_hist=False, show_stimulus=False,
                         plot_dlmr=False, color='white', vmax=.01, tlim=None, cmap=plt.cm.RdBu_r, first_nblocks=200,
                         fig_filename=''):
    
    ngenotypes = len(genotypes)
    ncols = ngenotypes+2
    fly = genotypes[0][0]
    nstims = fly.nstims
    
    if stims == 'all':
        fly = genotypes[0][0]
        stims = fly.stimdict.values()
    elif type(stims) is not list:
        stims = [stims]

    # Setup Figure.
    fig = plt.figure(1, (ncols*3, nstims*2.5), facecolor=cmap(0))
    gs = gridspec.GridSpec(nstims, ncols)
    gs.update(wspace=0.3)

    for istim, stim in enumerate(stims):
        irow = istim        
        if show_1d_hist:
            axh1d = plt.subplot(gs[irow, ngenotypes+1], axisbg=cmap(0))
            
        for igenotype, genotype in enumerate(genotypes):
            p = Plot(genotype)
            
            # Plot Stimulus
            if show_stimulus:
                axstim = plt.subplot(gs[irow, 0], axisbg=cmap(0))
                show_axes = []
                if istim == 0:
                    show_axes.extend(['top', 'left'])
                p.lines(axstim, 'xstim', stim, nrun, turndir, show_axes, color, tlim=tlim)

            # Plot lmr 2D histogram.
            axh2d = plt.subplot(gs[irow, 1+igenotype], axisbg=cmap(0))
            show_axes = []
            if istim == 0 and igenotype == 0:
                show_axes.append('top')
                if not show_stimulus:
                    show_axes.append('left')
            elif istim == len(stims)-1:
                axh2d.set_xlabel(genotype.label + '\n%i flies' %len(genotype))
            h2d, xbins, ybins = p.hist2d(axh2d, 'lmr', stim, nrun, turndir, show_axes,
                                         color, plot_dlmr, cmap, vmax, tlim=tlim,
                                         first_nblocks=first_nblocks)

            
            if show_1d_hist:
                trange_ops = {1: [(1.15, 1.25), (2.15, 2.25)], 2: [(1.65, 1.75), (2.65, 2.75)]}
                tranges = trange_ops[show_1d_hist]
                # Plot 1d histogram from selected region between t0 and t1.
                h1ds = []
                for trange in tranges:
                    # Show region that is used for 1d histogram.
                    axh2d.fill_between(xbins[1:], trange[0], trange[1], facecolor=blue, edgecolor='white', alpha=0.2)
                    idx0 = bisect(ybins, trange[0]) - 1
                    idx1 = bisect(ybins, trange[1])
                    h1ds.append(np.mean(h2d[:, idx0:idx1], axis=1))
                h1d = np.mean(h1ds, axis=0)
                h1d /= np.sum(h1d)
                
                p.set_axes(axh1d, 'lmr', ['bottom'], 'white', xlabel='')
                if igenotype == 0:
                    colour = blue
                    h1d_outline = np.concatenate([np.ones(20)*h for h in h1d])
                    x_outline = np.linspace(-50, 50, len(h1d_outline))
                    axh1d.plot(x_outline, h1d_outline, c=plt.cm.RdBu_r(0), lw=0.5)
                else:
                    colour = green
                
                axh1d.bar(xbins[:-1], h1d, facecolor=colour, edgecolor=colour, alpha=1)
                
                if stim == 0:
                    axh1d.set_ylim([0, 0.15])
                else:
                    axh1d.set_ylim([0, 0.1])
                axh1d.set_xlim([-50, 50])
                axh1d.set_xticks([-25, 0, 25])
            
    if fig_filename:
        plt.savefig(fig_filename, bbox_inches='tight', facecolor=cmap(0))

def plot_openloop_hist_parsed(genotype, stim, nrun=0, plot_dlmr=False, cmap=plt.cm.RdBu_r, color='white',
                              vmaxstim=.015, vmaxlmr=.015, fig_filename=''):
    
    fly = genotype[0]
    t = fly.tol

    
    fig = plt.figure(1, (6, 10), facecolor=cmap(0))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 2, 1])
    
    ax3 = plt.subplot(gs[2, 0], axisbg=cmap(0))
    ph.draw_axes(ax3, ['bottom'], color=color)
    ax3.set_xticks([-90, 0, 90])
    ax3.set_xlim([-180, 180])
    
    p = Plot(genotype)
    
    for iturn, turndir in enumerate(['left', 'straight', 'right']):
        axstim = plt.subplot(gs[iturn, 0], axisbg=plt.cm.RdBu_r(0))
        show_axes=[]
        if iturn == 0: show_axes.extend(['top', 'left'])
        h2dstim, xedgesstim, yedgesstim = p.hist2d(
            axstim, 'xstim', stim, nrun, turndir, show_axes,
            color, cmap, normed=True, vmax=vmaxstim)

        
        axlmr = plt.subplot(gs[iturn, 1], axisbg=plt.cm.RdBu_r(0))
        show_axes = []
        if iturn == 0: show_axes.append('top')
        p.hist2d(axlmr, stim, nrun, 'lmr', show_axes, color, plot_dlmr,
                 cmap, normed=True, vmax=vmaxlmr, parse_bylmr=turn)

        
        h1d_idx = ((yedgesstim>=5) & (yedgesstim<6))[:-1]
        h1d = np.mean(h2dstim[:, h1d_idx], axis=1)
        h1d /= np.sum(h1d)
        h1d_outline = np.concatenate([np.ones(20)*h for h in h1d])
        x_outline = np.linspace(-180, 180, len(h1d_outline))
        if (iturn==0 and stim==2) or (iturn==1 and stim==3):
            ax3.fill_between(x_outline, h1d_outline, facecolor=blue, edgecolor='none', lw=0.5, alpha=1, zorder=1)
            ax3.plot(x_outline, h1d_outline, c=plt.cm.RdBu_r(0), lw=0.5, zorder=3)
        else:
            ax3.fill_between(x_outline, h1d_outline, facecolor=green, edgecolor='none', lw=0.5, alpha=1, zorder=2)
        
        axstim.fill_between(x_outline, 5, 6, facecolor=blue, edgecolor='white', alpha=0.3)

        
    if fig_filename:
        plt.savefig(fig_filename, bbox_inches='tight', facecolor=plt.cm.RdBu_r(0))

def plot_closedloop_intertrial_hist(flies, show_all_flies=False, plot_1d_fixation=True, gridsize=100, nrun=0, plot_yeq0=False, bins=100, vmax=.018, fig_filename=''):
    
    # Parse arguments.
    if type(flies) is not list:
        flies = [flies]
    
    nflies = len(flies)
    if show_all_flies:
        nrows = nflies + 1
    else:
        nrows = 1
    if plot_1d_fixation:
        nrows+=1
    ngenotypes = 1
    tcl = flies[0].t_closedloop
    yeq0 = np.zeros(len(tcl))
    
    # Setup Figure.
    scalefactor = 3
    fig = plt.figure(1, (5, 3*nrows), facecolor=plt.cm.RdBu_r(0))
    gs = gridspec.GridSpec(nrows, ngenotypes)
    genotype=0
    # Plot each fly separately.
    if show_all_flies:
        for nfly, fly in enumerate(flies):
                
            # Plot stimulus position.
            ax1 = plt.subplot(gs[nfly, genotype], axisbg=plt.cm.RdBu_r(0))
            plt.ylim([-180, 180])
            
            if nfly == 0:
                keep_axes(ax1, ['left'], 'white')
                plt.ylabel('Stimulus\nPosition (deg)')
                plt.yticks([-90, 0, 90])
            else:
                keep_axes(ax1, [])
            
            if plot_yeq0:
                plt.plot(tcl, yeq0, color=green, linestyle='--')
            
            irun = fly.xstim_closedloop[nrun]
            itcls = list(tcl)*irun.shape[0]
            iruns = irun.flatten()
            h2d, xedges, yedges = np.histogram2d(iruns, itcls, normed=True, bins=bins, range=[[-180, 180], [0, 3]])
            plt.pcolormesh(yedges, xedges, h2d, cmap=plt.cm.RdBu_r, vmax=.008)

    

    # Plot all flies on same plot.
    axsum = plt.subplot(gs[-1, genotype], axisbg=plt.cm.RdBu_r(0))
    keep_axes(axsum, ['left', 'bottom'], 'white')
    plt.xticks([0, 1, 2, 3])
    plt.ylabel('All Flies\nStimulus Position (deg)')
    plt.yticks([-90, 0, 90])
    plt.ylim([-180, 180])
    plt.xlim([0, 3])
    
    # Calculate number of trials in order to normalize.
    ntrials = 0
    for fly in flies:
        itrials = np.shape(fly.xstim_closedloop[nrun])[0]
        ntrials += itrials
    plt.xlabel(('Time (s)   %i trials') %ntrials)
    # Plot all flies together
    runs = [fly.xstim_closedloop[nrun].flatten() for fly in flies]
    runs = np.concatenate(runs)
    tcls = list(tcl)*ntrials

    h2d, xedges, yedges = np.histogram2d(runs, tcls, normed=True, bins=bins, range=[[-180, 180], [0, 3]])
    plt.pcolormesh(yedges, xedges, h2d, cmap=plt.cm.RdBu_r, vmax=.008)

    if plot_1d_fixation:
        # Show window that is plotted in 1d fixation on 2d hist.
        half_fixation_window = 30
        axsum.fill_between(tcl, -half_fixation_window, half_fixation_window, facecolor='none', edgecolor='black', lw=1.2)
        
        # Plot 1d fixation.
        ax1d = plt.subplot(gs[-2, genotype], axisbg=plt.cm.RdBu_r(0))
        keep_axes(ax1d, ['left', 'bottom'], 'white')
        plt.xlim([0, 3])
        plt.ylim([0, 1])
        idx0 = bisect(xedges, -half_fixation_window) - 1
        idx1 = bisect(xedges, half_fixation_window)
        h1d = np.sum(h2d[idx0:idx1, :], axis=0)/np.sum(h2d, axis=0)[0]
        ax1d.plot(yedges[:-1], h1d, color='white')
    if plot_yeq0:
                plt.plot(tcl, yeq0, color=green, linestyle='--')      
    if fig_filename:
        plt.savefig(fig_filename, bbox_inches='tight', facecolor=plt.cm.RdBu_r(0))
    plt.show()

def plot_closedloop_intertrial_hist2(genotypes, plot_1d_fixation=True, nrun=0, plot_yeq0=False, bins=100, vmax=.018, fig_filename=''):
    
    half_fixation_window = 30
    h2ds = []
    h1ds = []
    ntrials_ls = []
    for flies in genotypes:
        # Calculate number of trials.
        ntrials = 0
        for fly in flies:
            itrials = np.shape(fly.xstim_closedloop[nrun])[0]
            ntrials += itrials
        runs = [fly.xstim_closedloop[nrun].flatten() for fly in flies]
        runs = np.concatenate(runs)
        tcl = flies[0].t_closedloop
        tcls = list(tcl)*ntrials
        h2d, xedges, yedges = np.histogram2d(runs, tcls, normed=True, bins=bins, range=[[-180, 180], [0, 3]])
        
        idx0 = bisect(xedges, -half_fixation_window) - 1
        idx1 = bisect(xedges, half_fixation_window)
        h1d = np.sum(h2d[idx0:idx1, :], axis=0)/np.sum(h2d, axis=0)[0]
        
        h2ds.append(h2d)
        h1ds.append(h1d)
        ntrials_ls.append(ntrials)
    
    # Setup Figure.
    ngenotypes = len(genotypes)
    nrows = ngenotypes+1
    fig = plt.figure(1, (6, 3*nrows), facecolor=plt.cm.RdBu_r(0))
    gs = gridspec.GridSpec(nrows, 1)
    if plot_1d_fixation:    
        axh1d = plt.subplot(gs[0, 0], axisbg=plt.cm.RdBu_r(0))

    for igenotype in range(ngenotypes):
        axh2d = plt.subplot(gs[igenotype+1, 0], axisbg=plt.cm.RdBu_r(0))
        axh2d.pcolormesh(yedges, xedges, h2ds[igenotype], cmap=plt.cm.RdBu_r, vmax=.008)
        axh2d.set_xlim([0, 3])
        axh2d.set_ylim([-180, 180])
        axh2d.set_yticks([-90, 0, 90])
        if igenotype == ngenotypes-1:
            keep_axes(axh2d, ['left', 'bottom'], 'white')
            axh2d.set_xticks([0, 1, 2, 3])
            axh2d.set_ylabel('All Flies\nStimulus Position (deg)')
            axh2d.set_xlabel(('Time (s)   %i trials') %ntrials_ls[igenotype])
            
        else:
            keep_axes(axh2d, ['left'], 'white')
            axh2d.set_xlabel(('           %i trials') %ntrials_ls[igenotype])
        
        # Plot all flies together
        if plot_yeq0:
            axh2d.axhline(c=green, ls='--')
            
        if plot_1d_fixation:
            # Show window that is plotted in 1d fixation on 2d hist.
            axh2d.fill_between(tcl, -half_fixation_window, half_fixation_window, facecolor='none', edgecolor='black', lw=1.2)
            
            # Plot 1d fixation.
            if igenotype == 0:
                colour = blue
            elif igenotype == 1:
                colour = green
            axh1d.plot(yedges[:-1], h1ds[igenotype], color=colour, lw=2)
            keep_axes(axh1d, ['left', 'bottom'], 'white')
            axh1d.set_xlim([0, 3])
            axh1d.set_ylim([0, 1])
            axh1d.set_xticks([])
            
    if fig_filename:
        plt.savefig(fig_filename, bbox_inches='tight', facecolor=plt.cm.RdBu_r(0))
    plt.show()

def plot_closedloop_intertrial_histv2(genotypes, genotype_labels = [], plot_1d_fixation=True, nrun=0, plot_yeq0=False, bins=100, vmax=.018, fig_filename=''):
    
    half_fixation_window = 30
    h2ds = []
    h1ds = []
    ntrials_ls = []
    for flies in genotypes:
        # Calculate number of trials.
        ntrials = 0
        for fly in flies:
            itrials = np.shape(fly.xstim_closedloop[nrun])[0]
            ntrials += itrials
        runs = [fly.xstim_closedloop[nrun].flatten() for fly in flies]
        runs = np.concatenate(runs)
        tcl = flies[0].t_closedloop
        tcls = list(tcl)*ntrials
        h2d, xedges, yedges = np.histogram2d(tcls, runs, normed=True, bins=bins, range=[[0, flies[0].closedloop_len], [-180, 180]])
        
        idx0 = bisect(yedges, -half_fixation_window) - 1
        idx1 = bisect(yedges, half_fixation_window)
        h1d = np.sum(h2d[:, idx0:idx1], axis=1)/np.sum(h2d, axis=1)[0]
        
        h2ds.append(h2d)
        h1ds.append(h1d)
        ntrials_ls.append(ntrials)
    
    # Setup Figure.
    ngenotypes = len(genotypes)
    ncols = ngenotypes
    nrows = 2
    fig = plt.figure(1, (2.5*ncols, nrows*4.5), facecolor=plt.cm.RdBu_r(0))
    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(hspace=0.4)
    if plot_1d_fixation:    
        axh1d = plt.subplot(gs[0, 0:3], axisbg=plt.cm.RdBu_r(0))
        
    colours = [ph.bblue, ph.green, ph.yellow, ph.orange, ph.red, ph.purple]
    if not genotype_labels:
        genotype_labels = ['']*ngenotypes
    for igenotype in range(ngenotypes):
        axh2d = plt.subplot(gs[1:, igenotype], axisbg=plt.cm.RdBu_r(0))
        axh2d.pcolormesh(yedges, xedges, h2ds[igenotype], cmap=plt.cm.RdBu_r, vmax=.008)
        axh2d.set_ylim([0, 3])
        axh2d.set_xlim([-180, 180])
        axh2d.set_xticks([-90, 0, 90])
        axh2d.invert_yaxis()
        if igenotype == 0:
            keep_axes(axh2d, ['left', 'bottom'], 'white')
            axh2d.set_yticks([0, 1, 2, 3])
            axh2d.set_ylabel(('Time (s)'))
        else:
            keep_axes(axh2d, [], 'white')
        axh2d.set_xlabel('%s   %i trials' %(genotype_labels[igenotype], ntrials_ls[igenotype]))
        axh2d.xaxis.set_label_position('top')
        
        # Plot all flies together
        if plot_yeq0:
            axh2d.axhline(c=green, ls='--')
            
        if plot_1d_fixation:
            # Show window that is plotted in 1d fixation on 2d hist.
            #axh2d.fill_between(tcl, -half_fixation_window, half_fixation_window, facecolor='none', edgecolor='black', lw=1.2)
            axh2d.axvline(-half_fixation_window, c='black', ls='--')
            axh2d.axvline(half_fixation_window, c='black', ls='--')
            
            # Plot 1d fixation.
            
            axh1d.plot(xedges[:-1], h1ds[igenotype], color=colours[igenotype], label = genotype_labels[igenotype], lw=2)
            keep_axes(axh1d, ['left', 'bottom'], 'white')
            axh1d.set_xlim([0, 4])
            axh1d.set_ylim([0, 1])
            axh1d.set_xticks([0, 1, 2, 3])
            axh1d.set_xlabel('Time (s)')
            axh1d.set_ylabel('p')
    ## Draw dashed line for chance occupation between +- half_fixation_point.
    #chance = half_fixation_window/180.
    #chance_ts = np.array([0, flies[0].closedloop_len])
    #chance_ys = np.ones(2)*chance
    #axh1d.plot(chance_ts, chance_ys, ls='--', c=ph.blue, label='chance')
    ph.set_legend(axh1d, loc='center right', edgecolor='none', textcolor='white')
            
    if fig_filename:
        plt.savefig(fig_filename, bbox_inches='tight', facecolor=plt.cm.RdBu_r(0))
    plt.show()

def get_closedloop_xstim(flies, nrun):
    closedloop_xstim_all_flies = [fly.xstim_closedloop[nrun] for fly in flies]
    return closedloop_xstim_all_flies

def plot_lmr_mean_bytemp(genotype, trials_per_block=10, nblocks=3, stimorder=['left', 'center', 'right'],
                         tlim=None, ymax=30, lines=[], fig_filename=''):
    
    fly = genotype[0]
    
    nstims = 3
    tempdict = {'cool': 0, 'hot': 1}
    colours = {'hot': [ph.orange, ph.red, 'black'],
                'cool': [ph.bblue, ph.blue, ph.bgreen]}
    
    # Setup figure
    plt.figure(1, (18, 8))
    gs = gridspec.GridSpec(4, 2, height_ratios=[2, 4, 2, 1])
    
    
    rowdict = {'left':0, 'center':1, 'right':2}
    deg= u'\N{DEGREE SIGN}'
    titledict = {'cool': '19%sC' %deg, 'hot': '33%sC' %deg}
    turndirs = ['left', 'right']
    
    for itemp, temp in enumerate(['cool', 'hot']):
        for istim, stim in enumerate(stimorder):

            # Calculate mean L-R WBA across all flies
            means = {turndir: [[] for _ in range(nblocks)] for turndir in turndirs}
            
            for fly in genotype:
                t = fly.ttrial[stim]
                lmrs = fly.parsed['lmr'][stim][temp]['all']
                baseline_inds = (t>=fly.baseline_window[0]) & (t<fly.baseline_window[1])
                parse_inds = (t>=fly.parse_window[0]) & (t<fly.parse_window[1])
                baseline = lmrs[:, baseline_inds].mean(axis=1).reshape(len(lmrs), 1)
                lmrs -= baseline
                for block in range(nblocks):
                    start = block*trials_per_block
                    stop = (block+1)*trials_per_block
                    if start >= len(lmrs): continue # takes care of nan cases
                    lmrsblock = lmrs[start:stop]
                    
                    thresh = 0
                    right_bool = lmrsblock[:, parse_inds].mean(axis=1) > thresh
                    left_bool = lmrsblock[:, parse_inds].mean(axis=1) < -thresh
                    means['left'][block].append(lmrsblock[left_bool])
                    means['right'][block].append(lmrsblock[right_bool])
            
            for turndir in turndirs:
                for block in range(nblocks):
                    means[turndir][block] = np.concatenate(means[turndir][block], axis=0).mean(axis=0)
            
            
            # Plot L-R
            row = istim
            axlmr = plt.subplot(gs[row, itemp])
            if stim == 'left':
                turnds = ['right']
            elif stim == 'right':
                turnds = ['left']
            elif stim == 'center':
                turnds = ['left', 'right']
            for turndir in turnds:
                for i, block in enumerate(range(nblocks)):
                    colour = colours[temp][i]
                    axlmr.plot(t, means[turndir][block], c=colour, lw=2, label='block %i' %(i+1))
            for line in lines:
                axlmr.axvline(line, ls='--', c='black', alpha=0.5)
            
            if stim == 'left':
                ylim = [-5, ymax]
                yticks = np.arange(0, ymax+1, 10)
                axlmr.set_xlabel(titledict[temp], fontsize=18)
                axlmr.xaxis.set_label_position('top')
            elif stim == 'center':
                ylim = [-ymax, ymax]
                yticks = np.arange(-ymax, ymax+1, 10)
            elif stim == 'right':
                ylim = [-ymax, 5]
                yticks = np.arange(-ymax, 1, 10)
                
            ylabel = ''
            if itemp == 0:
                atk = ['left']
                if stim == 'left':
                    ylabel = 'Looming from Left'
                elif stim == 'right':
                    ylabel = 'Looming from Right'
                elif stim == 'center':
                    ylabel = 'Looming from Center\nL-R WBA (deg)'
            else:
                atk = []
            if stim == 'left':
                handles, labels = axlmr.get_legend_handles_labels()
                axlmr.legend(handles, labels, loc='upper left', frameon=False, fontsize=12)
                
            ph.draw_axes(axlmr, atk, ylim=ylim, yticks=yticks)
            if tlim:
                axlmr.set_xlim(tlim)
            axlmr.set_ylabel(ylabel, rotation='horizontal', ha='right', labelpad=20)
            
            # Plot stimulus
            ystim = fly.parsed['ystim'][stim][temp]['all'].mean(axis=0)
            axstim = plt.subplot(gs[nstims, itemp])
            axstim.plot(t, ystim, c='black', lw=1.5)
            if tlim:
                axstim.set_xlim(tlim)
            ph.draw_axes(axstim, ['bottom'])
    if fig_filename:
        plt.savefig(fig_filename, bbox_inches='tight')


# For Spontaneous turning in Flight
#---------------------------------------------------------------------------#

def get_baseline(t, signal, gradient_thresh=.08, d2_thresh=4, nd2_filt=5, gradient_n=2, lpf=30):
    """ 
    gradient_thresh and diff_thresh are used to calculate the baseline.
    1. relatively flat regions are isolated using gradient_thresh
    2. d2_thresh is used to remove flat regions that deviate too 
    much from the rest (ie. flat regions at the peak of saccades).
    This set is iterated nd2_filt times. d2 = 2nd derivative or diff n=2
    3. flat regions are connected by linear interpolation
    """
    
    # 1
    lp = fc.butterworth(signal, lpf)
    baseline_i = np.where(np.abs(np.gradient(lp, gradient_n))<gradient_thresh)[0]
    
    # 2
    for _ in range(nd2_filt):
        d2_inds = np.abs(np.diff(lp[baseline_i], n=2))<d2_thresh
        baseline_i = baseline_i[1:-1][d2_inds]
    
    # 3
    # First, insert endpoints
    if baseline_i[0] != 0:
        baseline_i = np.insert(baseline_i, 0, 0)
    if baseline_i[-1] != len(t)-1:
        baseline_i = np.append(baseline_i, len(t)-1)
    
    # Then, interpolate.
    bt = t[baseline_i]
    by = lp[baseline_i]
    b = interp1d(bt, by) # returns a function that generates baseline with input t, eg. b(t) = y vector
    baseline = b(t)
    
    return baseline


class Spontaneous_Flight(Flight):

    def find_saccades(self, amp_thresh=8, width_thresh_ms=50, gradient_thresh=.08, gradient_n=2,
                      d2_thresh=4, nd2_filt=4, gradient_lpf=30, trim_start_s=4, trim_stop_s=2,
                      plot_results=True, tlim=[], plot_subtracted_baseline=False, fig_filename=''):
        """
        1. lmr baseline is subtracted from the original signal
        2. signal from (1) is lowpass filtered
        3. signal from (2) is thresholded to find peaks
        4. peaks must cross the threshold for at least min_thresh_width_ms (milliseconds) in order to be retained 
        """
        
        lmr = self.lmr
        t = self.t_transition
        
        baseline = get_baseline(t, lmr, gradient_thresh=gradient_thresh, d2_thresh=d2_thresh,
                                nd2_filt=nd2_filt, gradient_n=gradient_n, lpf=gradient_lpf)
        baseline = fc.butterworth(baseline, .5)
        lmr_f = fc.butterworth(lmr-baseline, 5)
        
        turn_inds =fc.get_contiguous_inds(np.abs(lmr_f)>amp_thresh, min_contig_len=width_thresh_ms/1000.*self.subsampling_rate)
        peak_inds = np.zeros(len(turn_inds), dtype=np.uint32)
        for i, inds in enumerate(turn_inds):
            s = np.abs(lmr[inds])
            idx = np.where(s==s.max())[0][0]
            idx += inds[0]
            peak_inds[i] = int(idx)
        
        self.flight = self.get_flight_inds(trim_start_s, trim_stop_s)
        p = np.zeros(len(lmr), dtype=np.bool)
        p[peak_inds] = True
        p[self.flight==False] = False
        self.turns_bool = p
        peak_inds = np.where(p)[0]
        
        peaks_dtype = np.dtype([('indices', np.uint32), ('t', np.float), ('amplitude', np.float), ('lmr', np.float)])
        peaks = np.zeros((len(peak_inds),), dtype=peaks_dtype)
        peaks['t'] = t[peak_inds]
        peaks['indices'] = peak_inds
        peaks['amplitude'] = (lmr-baseline)[peak_inds]
        peaks['lmr'] = lmr[peak_inds]
        self.turns = peaks
        self.left_turns = peaks[peaks['amplitude']<0]
        self.right_turns = peaks[peaks['amplitude']>0]
        
        #Plot figure
        if plot_results:
            plt.figure(1, (20, 8))
            ax = plt.subplot(111)
            ph.draw_axes(ax, ['left', 'bottom'],
                      xlabel='Time (s)', ylabel='L-R WBA (deg)')
            plt.ylabel('L-R WBA (deg)', rotation='horizontal', ha='right')
            if plot_subtracted_baseline:
                s = lmr-baseline
                plt.plot(t, s, c='black', alpha=0.6, zorder=2)
                ax.axhline(c=black, ls='--')
                ax.axhline(amp_thresh, c=ph.grey, zorder=1)
                ax.axhline(-amp_thresh, c=ph.grey, zorder=1)
            else:
                s = lmr
                plt.plot(t, s, lw=1, c='black', alpha=0.5, zorder=2)
                plt.plot(t, baseline, c='black', lw=2)
            
            plt.scatter(self.left_turns['t'], self.left_turns['lmr'], edgecolor='none', color=blue, s=50, zorder=3)
            plt.scatter(self.right_turns['t'], self.right_turns['lmr'], edgecolor='none', color=red, s=50, zorder=3)
            if tlim:
                plt.xlim(tlim)
            plt.ylim([-60, 60])
            
            if fig_filename:
                plt.savefig(fig_filename, bbox_inches='tight')

    def parse(self, hot_thresh=26):
        
        if not hasattr(self, 't'):
            self.open_abf()
        
        self.t_heating_transition = self.get_temp_transition(transition_temp=hot_thresh)
        self.t_transition = self.t - self.t_heating_transition
        self.find_saccades(plot_results=False) # be careful: find_saccades uses t_transition
        
        
        # For compatibility with other triggered averaging methods.
        if self.t_heating_transition > 0:
            self.nruns = 2
        else:
            self.nruns = 1
        self.stimdict = {0: 'spont'}
        self.stims = self.stimdict.values()
        self.nstims = 1
        self.window_lens = [0.7]
        self.shoulder_lens = [2]
        self.parse_window = [.3, .4]
        self.baseline_window = [0, .2]
        self.amp_thresh = 5
        self.ttrial = {'spont': np.arange(-self.shoulder_lens[0],
                         self.window_lens[0] + self.shoulder_lens[0],
                         self.subsampling_period)}
        self.tol = self.ttrial['spont']
        self.stimids = np.zeros(len(self.turns)).astype(int)
        self.trial_ts = self.turns['t'] - 0.35
        self.flight_bytrial = np.zeros(len(self.stimids)).astype(bool)
        self.parsed = { 'lmr':  self._parse_signal(self.t, self.lmr),
                       'dlmr': self._parse_signal(self.t, self.lmr, delta=True)}
    
    def get_wba_xcorr(self, maxlags=2):
        
        try:
            self.lwa
        except AttributeError:
            self.open_abf()
        
        # Retrieve cool and hot regions of the data.
        cool_trange = self.temp_ts[0][0]
        hot_trange = self.temp_ts[1][0]
        cool_inds = np.where((self.t >= cool_trange[0]) & (self.t < cool_trange[1]))
        lwa_cool = self.lwa[0:60000]
        rwa_cool = self.rwa[0:60000]
        
        hot_inds = np.where((self.t >= hot_trange[0]) & (self.t < hot_trange[1]))
        lwa_hot = self.rwa[70000:120000]
        rwa_hot = self.rwa[70000:120000]
        
        # Compute cross-correlation for each region of data separately.
        # Only return +- maxlags around 0.
        self.maxlag = maxlags/self.wba_subsampling_period
        
        xc_cool = fc.xcorr(lwa_cool, rwa_cool)
        mid = len(xc_cool)/2
        self.xc_cool = xc_cool[mid-self.maxlag:mid+self.maxlag]
        
        xc_hot = fc.xcorr(lwa_hot, rwa_hot)
        mid = len(xc_hot)/2
        self.xc_hot = xc_hot[mid-self.maxlag:mid+self.maxlag]
        
        self.txc = np.arange(-maxlags, maxlags, self.wba_subsampling_period)
        return self.txc, self.xc_cool, self.xc_hot
    
    def plot_wba_xcorr(self):
        try:
            self.txc
        except AttributeError:
            self.get_wba_xcorr()
        
        fig = plt.figure(1, (12, 4))
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        keep_axes(ax1, ['left', 'bottom'])
        plt.plot(self.txc, self.xc_cool, c=ph.blue)
        plt.ylabel('Cross-correlation\ncoefficient')
        plt.xlabel('Time (s)')
        
        ax2 = plt.subplot2grid((1, 2), (0, 1))
        keep_axes(ax2, ['left', 'bottom'])
        plt.plot(self.txc, self.xc_hot, c=ph.orange)


# Helper functions for plotting multiple SpontaneousWingBeats:
def plot_saccade_raster(genotypes, tlim=[-360, 360], binsize_s=30, ymax=40, show_temp=True, plot_median=False,
                        after_transition_t=120, before_transition_t=-60, lw=1, fig_filename=''):
    ngenotypes = len(genotypes)
    plt.figure(1, (22, 14))
    nrows = ngenotypes+2
    ncols = 2
    gs = gridspec.GridSpec(nrows, ncols , height_ratios=[10] + [genotype.nflies for genotype in genotypes] + [3],
                           width_ratios=[4, 1])
    gs.update(hspace=0.4)
    
    # Set up temperature axis.
    if show_temp:
        axtemp = plt.subplot(gs[-1, 0])
        ph.draw_axes(axtemp, ['left', 'bottom'],
                  xlabel='Time (s)', ylabel='Temperature\n(deg C)',
                  xticks=np.arange(tlim[0], tlim[1], 60), yticks=[20, 34],
                  xlim=tlim, ylim=[18, 36])
        axtemp.set_ylabel('Temperature\n(deg C)', rotation='horizontal', ha='right')
        
    # Set up turning rate axis.
    axrate = plt.subplot(gs[0, 0])
    axrate.axhline(ls='--', c=ph.grey)
    ph.draw_axes(axrate, ['bottom', 'left'],
              ylabel='Turning rate\n(turns per minute)',
              xticks=[], yticks=range(0, ymax, 10),
              xlim=tlim, ylim=[0, ymax])
    axrate.set_ylabel('Turning rate\n(turns per minute)', rotation='horizontal', ha='right')
    
    # Set up bar graph axis.
    axbar = plt.subplot(gs[0, 1])
    ph.draw_axes(axbar, ['bottom', 'right'],
              xticks=[], yticks=range(0, ymax, 10),
              ylim=[0, ymax])
 
    colours = [ph.blue, ph.red, ph.green, ph.orange, 'black', ph.purple]
    
    for igenotype, genotype in enumerate(genotypes):
        ar_len = (tlim[1]-tlim[0])*genotype.subsampling_rate
        turns = np.zeros((genotype.nflies, ar_len))
        flight = np.zeros((genotype.nflies, ar_len))
        for ifly, fly in enumerate(genotype):
            t = fly.t_transition
            rate_inds = np.where((t>=tlim[0]) & (t<tlim[1]))[0][:turns.shape[1]]
            t2 = t[rate_inds]
            dt = fly.subsampling_rate
            start_ind = np.floor((t2[0]-tlim[0])*dt)
            stop_ind = start_ind + len(rate_inds)
            turns[ifly, start_ind:stop_ind] = fly.turns_bool[rate_inds]
            flight[ifly, start_ind:stop_ind] = fly.flight[rate_inds]
        
        turns[flight==False] = 0
        
        # Plot turning rate over time.
        turn_sum = turns.sum(axis=0)
        flight_sum = flight.sum(axis=0)
        binsize = binsize_s*genotype.subsampling_rate
        turn_rate = np.array([turn_sum[start:start+binsize].sum() / flight_sum[start:start+binsize].sum() for start in xrange(0, len(turn_sum), binsize)])*genotype.subsampling_rate*60
        t_binned = np.arange(tlim[0]+binsize_s/2., tlim[1], binsize_s)
        axrate.plot(t_binned, turn_rate, color=colours[igenotype], lw=2)
        
        # Bar Plot
        if fly.t_heating_transition > 0:
            # Calculate turning rates before and after heating transition.
            t2 = np.arange(tlim[0], tlim[1], .01)
            cool_ind_upper = np.where(t2<before_transition_t)[0][-1]
            turn_rate_perfly_cool = turns[:, :cool_ind_upper].sum(axis=1)/flight[:, :cool_ind_upper].sum(axis=1)*genotype.subsampling_rate*60
            turns_mean_cool = turn_rate_perfly_cool.mean()
            turns_median_cool = np.median(turn_rate_perfly_cool)
            turns_stderr_cool = stats.sem(turn_rate_perfly_cool)
            hot_ind_lower = np.where(t2>after_transition_t)[0][0]
            turn_rate_perfly_hot = turns[:, hot_ind_lower:].sum(axis=1)/flight[:, hot_ind_lower:].sum(axis=1)*genotype.subsampling_rate*60
            turns_mean_hot = turn_rate_perfly_hot.mean()
            turns_median_hot = np.median(turn_rate_perfly_hot)
            turns_stderr_hot = stats.sem(turn_rate_perfly_hot)
            
            # Plot turning rates.
            xpos = igenotype*2.5
            if plot_median:
                hcool = turns_median_cool
                hhot = turns_median_hot
            else:
                hcool = turns_mean_cool
                hhot = turns_mean_hot
            axbar.bar(xpos, hcool, yerr=turns_stderr_cool, color=colours[igenotype], ecolor=black, capsize=0)
            x_jitter_cool = (np.random.normal(xpos+0.4, .06, genotype.nflies))
            axbar.scatter(x_jitter_cool, turn_rate_perfly_cool, color=black, zorder=5, s=3)
            axbar.bar(xpos+1, hhot, yerr=turns_stderr_hot, color=colours[igenotype], ecolor=black, capsize=0)
            x_jitter_hot = (np.random.normal(xpos+1.4, .06, genotype.nflies))
            axbar.scatter(x_jitter_hot, turn_rate_perfly_hot, color=black, zorder=5, s=3)
            xs = zip(x_jitter_cool, x_jitter_hot)
            ys = zip(turn_rate_perfly_cool, turn_rate_perfly_hot)
            for ifly in range(len(genotype)):
                axbar.plot(xs[ifly], ys[ifly], c='black', lw=0.5)
        else:
            turn_rate_perfly = turns.sum(axis=1)/flight.sum(axis=1)*genotype.subsampling_rate*60
            turns_mean = turn_rate_perfly.mean()
            turns_stderr = stats.sem(turn_rate_perfly)
            axbar.bar(igenotype, turns_mean, yerr=turns_stderr, color=colours[igenotype], ecolor=black, capsize=0)
            x_jitter = (np.random.normal(igenotype+0.4, .06, genotype.nflies))
            axbar.scatter(x_jitter, turn_rate_perfly, color=black, zorder=5, s=3)
        
        # Raster Plot  
        axraster = plt.subplot(gs[igenotype+1, 0])
        show_axes = ['left']
        xlabel=''
        xticks = []
        if not show_temp and igenotype==ngenotypes-1:
            show_axes.append('bottom')
            xlabel = 'Time (s)'
            xticks = np.arange(tlim[0], tlim[1], 60)

        ph.draw_axes(axraster, show_axes,
                  xlabel, ylabel=(genotype.label + '\nn=%i' %genotype.nflies),
                  xticks=xticks, yticks=range(1, genotype.nflies+1),
                  xlim=tlim, ylim=[0.5, genotype.nflies+.5])
        axraster.yaxis.label.set_color(colours[igenotype])
        axraster.invert_yaxis()
        axraster.set_ylabel(genotype.label + '\nn=%i' %genotype.nflies, rotation='horizontal', ha='right')
        
        nraster = genotype.nflies +.2
        for fly in genotype:
            # Plot temperature
            if show_temp:
                axtemp.plot(fly.t_transition, fly.temp, color=grey)
            
            # Raster plot saccades
            right_raster_ys = np.ones(len(fly.right_turns))*nraster
            axraster.scatter(fly.right_turns['t'], right_raster_ys, color=red, marker='|', lw=lw)
            nraster -= 0.25
            left_raster_ys = np.ones(len(fly.left_turns))*nraster
            axraster.scatter(fly.left_turns['t'], left_raster_ys, color=blue, marker='|', lw=lw)
            nraster -= 0.75
            
            
    
    if fig_filename:  
        plt.savefig(fig_filename, bbox_inches='tight', facecolor='none')
    
def plot_saccade_bars(genotypes, tlim=[-360, 360], binsize_s=30, ymax=40, plot_median=False,
                      after_transition_t=120, before_transition_t=-60, connect_dots=False, lw=1, fig_filename=''):
    # Set up bar graph axis.
    plt.figure(1, (len(genotypes)*3.5, 7))
    axbar = plt.subplot(111)
    
    if ymax > 100:
        ytick_spacer = 25
    else:
        ytick_spacer = 10
    ph.draw_axes(axbar, ['bottom', 'left'],
              yticks=range(0, ymax, ytick_spacer),
              ylim=[0, ymax])
    label_pos = np.arange(len(genotypes))
    labels = [genotype.label for genotype in genotypes]
    plt.xticks(label_pos, labels, rotation=45, ha='right')
    axbar.set_ylabel('Turning rate\n(turns per min)', rotation='horizontal', ha='right', labelpad=30)
    for igenotype, genotype in enumerate(genotypes):
        ar_len = (tlim[1]-tlim[0])*genotype.subsampling_rate
        turns = np.zeros((genotype.nflies, ar_len))
        flight = np.zeros((genotype.nflies, ar_len))
        for ifly, fly in enumerate(genotype):
            t = fly.t_transition
            rate_inds = np.where((t>=tlim[0]) & (t<tlim[1]))[0][:turns.shape[1]]
            t2 = t[rate_inds]
            dt = fly.subsampling_rate
            start_ind = np.floor((t2[0]-tlim[0])*dt)
            stop_ind = start_ind + len(rate_inds)
            turns[ifly, start_ind:stop_ind] = fly.turns_bool[rate_inds]
            flight[ifly, start_ind:stop_ind] = fly.flight[rate_inds]
        
        turns[flight==False] = 0
        
        if fly.t_heating_transition > 0:
            # Calculate turning rates before and after heating transition.
            t2 = np.arange(tlim[0], tlim[1], .01)
            
            cool_ind_upper = np.where(t2<before_transition_t)[0][-1]
            turn_rate_perfly_cool = turns[:, :cool_ind_upper].sum(axis=1)/flight[:, :cool_ind_upper].sum(axis=1)*genotype.subsampling_rate*60
            
            hot_ind_lower = np.where(t2>after_transition_t)[0][0]
            turn_rate_perfly_hot = turns[:, hot_ind_lower:].sum(axis=1)/flight[:, hot_ind_lower:].sum(axis=1)*genotype.subsampling_rate*60
            
            # Take care of NaNs
            cool_notnan = np.isnan(turn_rate_perfly_cool) == False
            hot_notnan = np.isnan(turn_rate_perfly_hot) == False
            notnan = cool_notnan & hot_notnan
            turn_rate_perfly_cool = turn_rate_perfly_cool[notnan]
            turn_rate_perfly_hot = turn_rate_perfly_hot[notnan]

            turns_mean_cool = turn_rate_perfly_cool.mean()
            turns_median_cool = np.median(turn_rate_perfly_cool)
            turns_stderr_cool = stats.sem(turn_rate_perfly_cool)
            
            turns_mean_hot = turn_rate_perfly_hot.mean()
            turns_median_hot = np.median(turn_rate_perfly_hot)
            turns_stderr_hot = stats.sem(turn_rate_perfly_hot)
            
            # Plot turning rates.
            xpos = igenotype
            if plot_median:
                hcool = turns_median_cool
                hhot = turns_median_hot
            else:
                hcool = turns_mean_cool
                hhot = turns_mean_hot
            barwidth=0.3
            nflies = len(turn_rate_perfly_cool)
            axbar.bar(xpos-barwidth, hcool, yerr=turns_stderr_cool, color=blue, ecolor=black, width=barwidth, capsize=0, label='20C')
            x_jitter_cool = (np.random.normal(xpos-barwidth/2, .03, nflies))
            axbar.scatter(x_jitter_cool, turn_rate_perfly_cool, color=black, zorder=5, s=3, alpha=0.5)
            axbar.bar(xpos, hhot, yerr=turns_stderr_hot, color=orange, ecolor=black, width=barwidth, capsize=0, label='34C')
            x_jitter_hot = (np.random.normal(xpos+barwidth/2, .03, nflies))
            axbar.scatter(x_jitter_hot, turn_rate_perfly_hot, color=black, zorder=5, s=3, alpha=0.5)
            xs = zip(x_jitter_cool, x_jitter_hot)
            ys = zip(turn_rate_perfly_cool, turn_rate_perfly_hot)
            if connect_dots:
                for ifly in range(nflies):
                    axbar.plot(xs[ifly], ys[ifly], c='black', lw=0.5, alpha=0.5)
            if igenotype == 0:
                handles, labels = axbar.get_legend_handles_labels()
                axbar.legend(handles, labels, loc='best', frameon=False)
        else:
            turn_rate_perfly = turns.sum(axis=1)/flight.sum(axis=1)*genotype.subsampling_rate*60
            turns_mean = turn_rate_perfly.mean()
            turns_stderr = stats.sem(turn_rate_perfly)
            axbar.bar(igenotype, turns_mean, yerr=turns_stderr, color=grey, ecolor=black, width=barwidth, capsize=0)
            x_jitter = (np.random.normal(igenotype+0.4, .06, genotype.nflies))
            axbar.scatter(x_jitter, turn_rate_perfly, color=black, zorder=5, s=3)
        
        
    if fig_filename:  
        plt.savefig(fig_filename, bbox_inches='tight', facecolor='none')
 
def plot_leftright_xcorr(flies):
    shibires = flies[:4]
    w1118s = flies[4:]
    
    def get_xcs(flies):
        # xcs = cross correlations
        xcs = [[] for _ in range(2)]
        for n, xc in enumerate(xcs):
            for fly in flies:
                tlims = fly.temp_ts[n][0]
                inds = np.where((fly.tsub >= tlims[0]) & (fly.tsub < tlims[1]))
                xc.append(np.correlate(fly.lwa[inds], fly.rwa[inds], mode='same'))
        return xcs
    
    shibire_xcs = get_xcs(shibires)
    w1118_xcs = get_xcs(w1118s)
    
    print shibire_xcs[0][0]
    
    fig = plt.figure(1, (6, 3))
    
    for row, genotype in enumerate([shibire_xcs, w1118_xcs]):
        for col, shibire_xc in enumerate(shibire_xcs):
            ax = plt.subplot2grid((2, 2), (row, col))
            for local_xc in shibire_xc:
                ax.plot(local_xc[0], color=grey)

def plot_lpr(flies, fig_filename='', change=False):
    fig = figure(1, (20, 5))
    gridsize = (3, 7)
    for nfly, fly in enumerate(flies):
        ax1 = subplot2grid(gridsize, (0, nfly), rowspan=2)
        lpr = fly.lwa + fly.rwa
        plot(fly.tsub, lpr, c=ph.blue)
        if nfly == 0:
            keep_axes(ax1, ['left'])
        else:
            keep_axes(ax1, [])
        ylim([10, 180])
        
        
        
        ax2 = subplot2grid(gridsize, (2, nfly), sharex=ax1)
        plot(fly.tsub, fly.temp, c=ph.grey)
        if nfly == 0:
            keep_axes(ax2, ['left', 'bottom'])
        else:
            keep_axes(ax2, ['bottom'])
        ylim([20, 36])
        
        if nfly==1 and change:
            xlim([840, 2040])
            xticks([840, 1440, 2040])
        else:
            xlim([0, 1200])
            xticks([0, 600, 1200])
    tight_layout()
    if fig_filename:
        plt.savefig(fig_filename, bbox_inches='tight')

def plot_lar(genotypes, genotype_labels=[], tlims=[], lowpass=10, nrows=4, sf=1, fig_filename=''):
    plt.figure(1, (18*sf, 10*sf))
    ncols = len(genotypes)
    gs = gridspec.GridSpec(nrows, ncols)
    
    for igenotype, genotype in enumerate(genotypes):
        for ifly, fly in enumerate(genotype[:nrows]):
            ax = plt.subplot(gs[ifly, igenotype])
            plt.plot(fly.t, fly.lwa, c=(.61, .77, .99))
            plt.plot(fly.t, fc.butterworth(fly.lwa, lowpass), c=ph.blue, lw=0.6, label='LWA')
            plt.plot(fly.t, fly.rwa, c=(1, .62, .52))
            plt.plot(fly.t, fc.butterworth(fly.rwa, lowpass), c=ph.red, lw=0.6, label='RWA')
            plt.plot(fly.t, fly.lmr, c=ph.grey)
            plt.plot(fly.t, fc.butterworth(fly.lmr, lowpass), c='black', lw=0.6, label = 'LMR')
            if ifly == nrows-1 and igenotype == ncols-1:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, loc=4)
            if ifly == nrows-1:
                if igenotype == 0:
                    keep_axes(ax, ['left', 'bottom'])
                    plt.ylabel('WBA (deg)')
                    plt.xlabel('Time (s)')
                else:
                    keep_axes(ax, ['bottom'])

            elif ifly == 0 and genotype_labels:
                keep_axes(ax, ['top'])
                plt.xticks([])
                plt.xlabel(genotype_labels[igenotype])
            else:
                keep_axes(ax, [])
            if tlims:
                plt.xlim(tlims[igenotype])
            plt.ylim([-40, 80])
            
            if fig_filename:
                plt.savefig(fig_filename, bbox_inches='tight', dpi=150)

def plot_lmr_around_temp_transition(flies, window=60, offset=60, fig_filename=''):
    nflies = len(flies)
    fig = plt.figure(1, (15, nflies*2))
    gs = gridspec.GridSpec(nflies, 2)

    
    for ifly, fly in enumerate(flies):

        t = fly.t_transition
        cool = (t>=-offset-window) & (t<-offset)
        hot = (t>=offset) & (t<offset+window)
        
        transition_idx = (t>=-window) & (t<window)
        colors = [blue, orange]
        for itemp, temp in enumerate([cool, hot]):
            ax1 = plt.subplot(gs[ifly, itemp])
            ax1.axhline(c=grey, ls='--')
            tlocal = t[temp]
            ax1.plot(tlocal, fly.lmr[temp], c=(.8, .8, .8))
            lmr_10hzlp = fc.butterworth(fly.lmr, 10)
            ax1.plot(tlocal, lmr_10hzlp[temp], c=colors[itemp])
            #ax1.scatter(fly.left_turns['t'], fly.left_turns['lmr'], edgecolor='none', color=blue, s=50, zorder=3)
            #ax1.scatter(fly.right_turns['t'], fly.right_turns['lmr'], edgecolor='none', color=red, s=50, zorder=3)

            xlabel = ''
            ylabel = ''
            yticks = []
            atk = []
            if ifly == nflies-1:
                atk.append('bottom')
                if itemp==0:
                    atk.append('left')
                    yticks = [-25, 0, 25]
                    ylabel = 'L-R WBA\n(deg)'
                    xlabel = 'Time (s)'
            else:
                ph.draw_axes(ax1, [])
            ph.draw_axes(ax1, atk,
                         xlabel, ylabel,
                         xticks=None, yticks=yticks,
                         xlim=[tlocal.min(), tlocal.max()], ylim=[-50, 50])
            ax1.set_ylabel(ylabel, rotation='horizontal', ha='right')
            #ax1.spines['left'].set_color('black')
    if fig_filename:
            plt.savefig(fig_filename, bbox_inches='tight')

#---------------------------------------------------------------------------#

# Generalized plotting function
def plot(self, tlim=None, channels=['head', 'pressure'], hr=None, c='black', lw=1, lines=[], trigger=None, trigthresh=100):
    if not type(c) is list: c = [c]*len(channels)
    if not type(lw) is list: lw = [lw]*len(channels)
    if not hr: hr=[1]*len(channels)
    if not tlim: tlim = [self.t[0], self.t[-1]]
    idx = (self.t>=tlim[0]) & (self.t<tlim[1])
        
    plt.figure(1, (18, 5))
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    gs = gridspec.GridSpec(len(channels), 1, height_ratios=hr)
    axs = []
    for i, channel_name in enumerate(channels):
        channel = getattr(self, channel_name)
        ax = plt.subplot(gs[i, 0])
        if fc.fc.contains(['head', 'side', 'forw', 'xstim'], [channel_name]):
            ax.set_ylim(-180, 180)
            ax.set_yticks([-180, 0, 180])
            ax.axhline(c='black', ls='--', alpha=.3)
            ph.circplot(self.t, channel, circ='y', c=c[i], lw=lw[i])
        else:
            ymin = np.floor(sorted(channel[idx])[20]/10)*10
            ymax = np.ceil(sorted(channel[idx])[-20]/10)*10
            ax.set_ylim(ymin, ymax)
            ax.set_yticks([ymin, ymax])
            ax.plot(self.t, channel, c=c[i], lw=lw[i])
        spines = ['left', 'bottom'] if i == len(channels)-1 else ['left']
        ph.adjust_spines(ax, spines, xlim=tlim)
        if i == len(channels)-1: ax.set_xlabel('Time (s)')
        ax.set_ylabel(channel_name)
        axs.append(ax)
    if trigger:
        trig_inds =fc.rising_trig(getattr(self, trigger), trigger=trigthresh)
        lines.extend(list(self.t[trig_inds]))
    for ax in axs:
        for line in lines:
            ax.axvline(line, c=ph.orange, ls='--')


