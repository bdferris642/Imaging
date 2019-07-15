import behaviour as bh
import tiffile as tiff
import functions as fc



import xml.etree.ElementTree as ET
import copy, os, glob, csv
from bisect import bisect

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.collections import LineCollection
import plotting_help as ph
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mp

import numpy as np
from scipy import ndimage, stats
import scipy.cluster as sc
from scipy.interpolate import interp1d
from skimage import transform
import circstats as cs
from scipy.ndimage.filters import gaussian_filter

from random import randint
from matplotlib import patches

reload(bh)
reload(fc)
reload(ph)

ph.set_tickdir('out')

# The classes

class Channel():
    """
    Class for handling one channel in a TIF recording.
    """

    def __init__(self, file, celltype):
        self.file = file
        self.celltype = celltype


class GCaMP():
    """
    Parent Class for handling TIF files.
    """
    def __init__(self, folder, celltype, celltype2=None, xmlfile = True):
        #self.basename = folder + folder.split(os.path.sep)[-2]
        self.basename = folder
        #print self.basename
        
        # TIF file
               
        greenfile = \
        (glob.glob(folder + os.path.sep + '*reg_gauss.tif') +
         glob.glob(folder + os.path.sep + '*gauss.tif') +
         glob.glob(folder + os.path.sep + 'C1*reg.tif') +
         glob.glob(folder + os.path.sep + '*_reg.tif') +
         glob.glob(folder + os.path.sep + '*_reg_interleaved.tif') +
         glob.glob(folder + os.path.sep + 'C1*.tif') +
         glob.glob(self.basename + '.tif')
         )[0]

        redfile = \
        (glob.glob(folder + os.path.sep + 'C2*reg.tif') +
        glob.glob(folder + os.path.sep + 'C2*.tif') +
        glob.glob(folder + os.path.sep + '*_red.tif')
        )
        
        # ojo: redfile does not include reg files...

        if redfile and celltype2:
            redfile = redfile[0]
        else:
            redfile = None

        self.c1 = Channel(greenfile, celltype)
        self.c2 = Channel(redfile, celltype2)

        self.folder = folder
        self.basename = folder + folder.split(os.path.sep)[-2]
        self.xmlfile = self.basename + '.xml'

        self.info = {}
        self.info['filename'] = folder
        self.info['basename'] = self.basename
        if xmlfile:
            self.info['pockels'] = self._get_pockels(self.xmlfile)
            self.info['laserpower_mW'] = self._get_laser_power(self.xmlfile)


        transformfile = glob.glob(self.basename + '_transform.csv')
        if transformfile:
            self.transformfile = transformfile[0]
            self.reg_header, self.reg_xys, self.reg_dist = self._get_reg_xys(self.transformfile)

        ctlmaskfile = glob.glob(folder + os.path.sep + '*_ctlmask.tif')
        if ctlmaskfile:
            self.c1.ctlmaskfile = ctlmaskfile[0]
            self.c1.ctlmask = tiff.imread(self.c1.ctlmaskfile) < 50
        
        # OJO: bf edit jan 16th, 2017
        areamaskfile = glob.glob(folder + os.path.sep + '*_areamask.tif')
        if areamaskfile:
            self.c1.areamaskfile = areamaskfile[0]
            self.c1.areamask = tiff.imread(self.c1.areamaskfile)

    def open(self, tiffile):
        # Open tif file.
        tif = tiff.imread(tiffile)

        # Reshape tif so that multi-z and single-z tifs have the same shape dimension
        if len(tif.shape) == 3:
            newshape = list(tif.shape)
            newshape.insert(1, 1)
            newshape = tuple(newshape)
            tif = tif.reshape(newshape)

        # Extract acquisition info.
        self.len = tif.shape[0]
        self.zlen = tif.shape[1]
        self.info['dimensions_pixels'] = tif.shape[-2:]
        self.info['zlen'] = self.zlen

        return tif

    def roi(self, tif, mask, metric=np.mean):
        mask = np.tile(mask, (self.len, 1, 1, 1))
        masked = tif[mask]
        masked = masked.reshape((self.len, masked.size / self.len))
        return metric(masked, axis=-1)

    def norm_over_time(self, signal, whole_image=None, mode='dF/F0'):
        if mode == 'zscore':
            newsignal = signal - signal.mean(axis=0)
            div = newsignal.std(axis=0)
            div[div == 0] = 1
            newsignal /= div   
            f0=None
        elif mode == 'dF/F0':
            # in this mode, F0 is calculated from dimmest IMAGING pixels (i.e. inside glomeruli)
            signal_sorted = np.sort(signal, axis=0)
            
            f0 = np.mean(signal_sorted[:int(signal.shape[0]*.05), :], axis=0)
            # std = signal.std(axis=0)
            # std[std==0] = np.nan
            f0[f0 == 0] = np.nan
            newsignal = (signal - f0) / f0
        elif mode == 'linear':
            signal_sorted = np.sort(signal, axis=0)
            f0 = np.mean(signal_sorted[:int(signal.shape[0]*.05), :], axis=0)
            fm = np.mean(signal_sorted[:int(signal.shape[0]*.95), :], axis=0)
            newsignal = (signal - f0) / (fm - f0)
        elif mode == 'dF/baseline':
            f0_calc_thresh = 40
            f0_frac = 0.05
            # in this mode, F0 is calculated from dimmest pixels from WHOLE image (e.g. inside glomeruli)
            time_avg_image = np.nanmean(whole_image, axis=0)
            time_avg_image_1d = time_avg_image.flatten()
            # get threshold to consider a pixel `f0`: bottom 5% of pixels above avg fluo of 40
            # concern is not dim-ness necessarily. It is invariance...look at this later...
            time_avg_image_1d = time_avg_image_1d[time_avg_image_1d>f0_calc_thresh]
            npixels = len(time_avg_image_1d)
            time_avg_image_1d_sorted = np.sort(time_avg_image_1d)           
            f0_thresh = time_avg_image_1d_sorted[int(npixels*f0_frac)]
            
            # to get f0 pixels, make a mask of pixels whose avg fluo is <= the threshold,
            # tile this mask the number of time measurements, apply to whole image
            f0_mask = np.tile((time_avg_image < f0_thresh) * (time_avg_image > f0_calc_thresh),
                             (np.shape(whole_image)[0], 1, 1, 1))
            f0_whole_image=np.where(f0_mask, whole_image, np.nan)
            #for i in range(time_avg_image.shape[0]):            
             #   plt.imshow(f0_mask[1, i, :, :])
              #  plt.show()
            
            # average all f0 pixels to put 1d in time dimension
            f0_1d = np.nanmean(np.nanmean(np.nanmean(f0_whole_image, axis=3), axis=2), axis=1) 
            # make f0 same shape as `signal`
            f0 = np.transpose(np.tile(f0_1d, (signal.shape[1], 1)))   
            newsignal = (signal - f0) / f0   
        return newsignal, f0

    def norm_per_row(self, signal):
        signal = signal - signal.mean(axis=-1).reshape((len(signal), 1))
        signal = signal / signal.std(axis=-1).reshape((len(signal), 1))
        return signal

    def plot_masks(self, show_skel=True, show_mask=True, fig_filename=''):
        ax = plt.subplot(111)
        if show_mask:
            img = self.tifm
        else:
            img = self.tif
        plt.pcolormesh(img.var(axis=0).max(axis=0))
        ny, nx = self.tif.shape[-2:]
        plt.xlim(0, nx)
        plt.ylim(0, ny)
        ax.invert_yaxis()

        if show_skel:
            xs = [x for x, _ in self.points]
            ys = [y for _, y in self.points]
            plt.scatter(xs, ys, c=np.arange(len(xs)), marker=',', lw=0, cmap=plt.cm.jet)

        if fig_filename:
            plt.savefig(fig_filename, bbox_inches='tight', dpi=150)

    def plot(self, data='skel', cmap=plt.cm.CMRmap, vmin=None, vmax=None):
        if hasattr(self, data):
            data = getattr(self, data)
        else:
            print 'Undefined data type.'
            return

        plt.figure(1, (6, 20))
        ax1 = plt.subplot(111)
        xedges = np.arange(0, data.shape[1] + 1)
        img = plt.pcolormesh(xedges, range(self.len), data, cmap=cmap)
        if vmin:
            img.set_clim(vmin=vmin)
        if vmax:
            img.set_clim(vmax=vmax)
        ax1.invert_yaxis()
        plt.xlim([0, dxedges[-1]])

    def _get_pv_times(self, xmlfile):
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        times = []
        for sequence in root.getchildren():
            for frame in sequence:
                if 'relativeTime' in frame.attrib:
                    time = float(frame.attrib['relativeTime'])
                    times.append(time)
        times = np.array(times)
        return times

    def _get_pockels(self, xmlfile):
        with open(xmlfile) as xml:
            for line in xml:
                if 'laserPower_0' in line:
                    value = int(line.split()[-2].split('"')[-2])
                    return value

    def _get_laser_power(self, xmlfile):
        with open(xmlfile) as xml:
            for line in xml:
                if 'twophotonLaserPower_0' in line:
                    value = int(line.split()[-2].split('"')[-2])
                    return value

    def _get_reg_xys(self, transformfile):
        with open(transformfile) as fh:
            v = csv.reader(fh)
            table = []
            for line in v:
                try:
                    table.append(map(float, line))
                except ValueError:
                    header = line
        table = np.array(table).T
        table[0] -= 1
        dimlens = np.max(table[:3], axis=-1).astype(int) + 1
        xys = np.zeros(list(dimlens) + [2]).astype(float)
        for (c, z, t, x, y) in table.T:
            c, z, t = map(int, [c, z, t])
            xys[c, z, t] = [x, y]
        header = [h[0].lower() for h in header]

        def f(ar):
            return np.hypot(*ar)
        dist = np.apply_along_axis(f, 0, xys.T).T
        dist = dist[0].max(axis=0)
        return header, xys, dist

    def get_subregdist_mask(self, thresh=5, regdist=None):
        if regdist is None:
            regdist = self.reg_dist
        mask = (regdist - regdist.mean()) < thresh
        mask = ndimage.binary_erosion(mask, iterations=1)
        return mask
    
    def process_channel(self, channel):
        if channel.celltype is None: return
        channel.ppg = 1.
        # Parse images into glomerular representation and normalize
        channel.a = self._get_gloms(channel.tif, channel.mask, channel.celltype)
        channel.al = self.norm_over_time(channel.a, mode='linear')[0]
        channel.an = self.norm_over_time(channel.a, mode='dF/F0')[0]
        channel.az = self.norm_over_time(channel.a, mode='zscore')[0]
        #ojo
        #for i in range(self.c1.mask.shape[0]):
         #   plt.imshow(self.c1.mask[1, :, :])
        #  plt.show()
        channel.ab, channel.f_baseline = self.norm_over_time(channel.a, 
                                                             whole_image = channel.tif,
                                                             mode = 'dF/baseline')
    
        # Measure phase from glomerular representation
        channel.acn = self.get_continuous_bridge(channel.an, channel.celltype)
        channel.acz = self.get_continuous_bridge(channel.az, channel.celltype)
        
        # extract the phase with a fourier transform
        channel.phase, channel.period = self.get_fftphase(channel.acn, div=1,
                                                          mode='constant', cval=8)
        # extract the phase by treating every glomerulus as a vector
        channel.pva_n, channel.pva_n_amplitude  = self.get_pva(channel.acn,
                                                               angles_deg = np.tile(np.linspace(-180, 135., 8), 2),
                                                               window=1)
        channel.pva_z, channel.pva_z_amplitude  = self.get_pva(channel.acz,
                                                               angles_deg = np.tile(np.linspace(-180, 135., 8), 2),
                                                               window=1)
        
        channel.pva_n_f, channel.pva_n_f_amplitude  = self.get_pva(channel.acn,
                                                               angles_deg = np.tile(np.linspace(-180, 135., 8), 2),
                                                               window=3)
        channel.pva_z_f, channel.pva_z_f_amplitude  = self.get_pva(channel.acz,
                                                               angles_deg = np.tile(np.linspace(-180, 135., 8), 2),
                                                               window=3)
        
        if channel.celltype == 'EIP':
            channel.phase = fc.wrap(channel.phase + 45)
        channel.phasef = fc.circ_moving_average(channel.phase)
        channel.power = self.get_fftpower_at_peak(channel.acn, channel.period)
        channel.peak = np.nanmax(channel.an, axis=1)
        channel.xedges = np.arange(19)
    
        # Interpolate glomeruli and cancel phase
        channel.xedges_interp, channel.an_cubspl = self.interpolate_glomeruli(channel.an, 
                                                                              channel.celltype)
        channel.an_nophase = self.cancel_phase(channel.an_cubspl, channel.phase, 10,
                                               channel.celltype)
    
        ind0 = np.arange(0, 10)
        idx = np.tile(ind0, (18, 1)) + 10*np.arange(18)[:, None]
        channel.an_nophase_sub = np.nanmean(channel.an_nophase[:, idx], axis=-1)
    
        _, channel.az_cubspl = self.interpolate_glomeruli(channel.az, channel.celltype)
        channel.az_nophase = self.cancel_phase(channel.az_cubspl, 
                                               channel.phase, 10, channel.celltype)
        channel.az_nophase_sub = np.nanmean(channel.az_nophase[:, idx], axis=-1)
    
        # Calculate Right - Left Bridge using zscore
        x = channel.xedges[:-1]
        channel.rightz = np.nanmean(channel.az[:, x >= 9], axis=1)
        channel.leftz = np.nanmean(channel.az[:, x < 9], axis=1)
        channel.rmlz = channel.rightz - channel.leftz
        channel.rplz = channel.rightz + channel.leftz
    
        channel.rightn = np.nanmean(channel.an[:, x >= 9], axis=1)
        channel.leftn = np.nanmean(channel.an[:, x < 9], axis=1)
        channel.rmln = channel.rightn - channel.leftn
        channel.rpln = channel.rightn + channel.leftn
    
        # Calculate the expected projection onto the ellipsoid body given known anatomy.
        channel.an_ebproj = self.get_ebproj(channel.an.T, channel.celltype).T

    def get_continuous_bridge(self, bridge, celltype):
        """
        Since glomeruli L9 and R1 are adjacent and map to the same ellipsoid body wedge,
        these are averaged into a single glomerulus in the bridge for the purpose of extracting
        a phase and period from the signal.
        """
        if celltype == 'EIP':
            width = 16
            bridgec = np.zeros((len(bridge), width))
            bridgec[:, :8] = bridge[:, 1:9]
            bridgec[:, -8:] = bridge[:, 9:17]
            # bridgec[:, 7] = bridge[:, 8:10].mean(axis=-1)
        elif celltype == 'PEN_alt':
            bridgec = np.zeros_like(bridge)
            bridgec[:] = bridge[:]
            bridgec[:, 8:10] = np.mean([bridge[:, :2], bridge[:, -2:]], axis=0)
        elif celltype == 'PEN' or celltype == 'SPSP':
            width = 16
            bridgec = np.zeros((len(bridge), width))
            bridgec[:, :8] = bridge[:, :8]
            bridgec[:, 8:] = bridge[:, 10:]
            #bridgec[0:6] = bridge[1:7]
            #bridgec[7] = bridge[0]
            #bridgec[8] = bridge[-1]
            #bridgec[9:] = bridge[7:-1]
        else:
            print 'error: no celltype!'
    
        nans = np.isnan(bridgec[0]).sum()
        if nans > 0:
            if nans == 1:
                print '%i glomerulus is missing.' %nans
            else:
                print '%i glomeruli are missing.' %nans
            fc.nan2zero(bridgec)
        return bridgec



class Bridge(GCaMP):
    def __init__(self, folder, celltype, celltype2=None, xmlfile=True):
        """
        self.gctypes = []

        maskfile = glob.glob(self.folder + os.path.sep + '*_mask.tif')
        if maskfile:
            self.maskfile = maskfile[0]
            self.mask = tiff.imread(self.maskfile) > 0

        skelfile = glob.glob(self.folder + os.path.sep + '*_skel.tif')
        if skelfile:
            self.skelfile = skelfile[0]
            self.skelmask = tiff.imread(self.skelfile) > 0
        """
        GCaMP.__init__(self, folder, celltype, celltype2, xmlfile)

        segfile = (glob.glob(self.folder + os.path.sep + 'C1*_seg.tif') +
                    glob.glob(self.folder + os.path.sep + '*seg.tif') +
                   glob.glob(self.folder + os.path.sep + '*pbmask.tif'))

        if segfile:
            self.c1.maskfile = self.c2.maskfile = segfile[0]
            self.c1.mask = self.c2.mask = tiff.imread(self.c1.maskfile)
            if len(np.shape(self.c1.mask)) == 5:
                self.c1.mask = self.c2.mask = tiff.imread(self.c1.maskfile)[:, :, 0, :, :]
                

        self.open()

        if hasattr(self.c1, 'ctlmask'):
            self.c1.ctl = self.roi(self.c1.tif, self.c1.ctlmask)

    def open(self):
        tif = GCaMP.open(self, self.c1.file)
        print self.c1.file
        if len(tif.shape) == 4:
            self.c1.tif = tif
            self.c2.tif = GCaMP.open(self, self.c2.file) if self.c2.file else None
        elif len(tif.shape) == 5:
            self.c1.tif = tif[:, :, 0]
            self.c2.tif = tif[:, :, 1]

        if not self.c1.tif.shape[1] == self.c1.mask.shape[0]:            
            mask = np.zeros(list(self.c1.tif.shape)[-3:])
            mask[:self.c1.mask.shape[0]] = self.c1.mask
            self.c1.mask = self.c2.mask = mask
             

        if not self.c2.tif is None:
            self.process_channel(self.c1, n_cha = 0)
            self.process_channel(self.c2, n_cha = 1)
        else:
            self.process_channel(self.c1, n_cha = None)

    def process_channel(self, channel, n_cha = None):
        if channel.celltype is None: return
        channel.ppg = 1.
        # Parse images into glomerular representation and normalize
        
        if n_cha is None:
            mask = channel.mask
        else:
            mask = channel.mask[:, n_cha, :, :]
        print mask.shape, channel.tif.shape
        channel.a = self._get_gloms(channel.tif, mask, channel.celltype)
        #channel.al = self.norm_over_time(channel.a, mode='linear')[0]
        channel.an = self.norm_over_time(channel.a, mode='dF/F0')[0]
        channel.az = self.norm_over_time(channel.a, mode='zscore')[0]
        #ojo
        #for i in range(self.c1.mask.shape[0]):
         #   plt.imshow(self.c1.mask[1, :, :])
          #  plt.show()
        channel.ab, channel.f_baseline = self.norm_over_time(channel.a, whole_image = channel.tif, mode = 'dF/baseline')

        # Measure phase from glomerular representation
        channel.acn = self.get_continuous_bridge(channel.an, channel.celltype)
        channel.acz = self.get_continuous_bridge(channel.az, channel.celltype)
        
        # extract the phase with a fourier transform
        channel.phase, channel.period = self.get_fftphase(channel.acn, div=1,
                                                          mode='constant', cval=8)
        # extract the phase by treating every glomerulus as a vector
        channel.pva_n, channel.pva_n_amplitude  = self.get_pva(channel.acn,
                                                               angles_deg = np.tile(np.linspace(-180, 135., 8), 2),
                                                               window=1)
        channel.pva_z, channel.pva_z_amplitude  = self.get_pva(channel.acz,
                                                               angles_deg = np.tile(np.linspace(-180, 135., 8), 2),
                                                               window=1)
        
        channel.pva_n_f, channel.pva_n_f_amplitude  = self.get_pva(channel.acn,
                                                               angles_deg = np.tile(np.linspace(-180, 135., 8), 2),
                                                               window=3)
        channel.pva_z_f, channel.pva_z_f_amplitude  = self.get_pva(channel.acz,
                                                               angles_deg = np.tile(np.linspace(-180, 135., 8), 2),
                                                               window=3)
        
        
        channel.phase, channel.period = self.get_fftphase(channel.acn, div=1, mode='constant', cval=8)
        if channel.celltype == 'EIP':
            channel.phase = fc.wrap(channel.phase + 45)
        channel.phasef = fc.circ_moving_average(channel.phase)
        channel.power = self.get_fftpower_at_peak(channel.acn, channel.period)
        channel.peak = np.nanmax(channel.an, axis=1)
        channel.xedges = np.arange(19)

        # Interpolate glomeruli and cancel phase
        channel.xedges_interp, channel.an_cubspl = self.interpolate_glomeruli(channel.an, channel.celltype)
        channel.an_nophase = self.cancel_phase(channel.an_cubspl, channel.phase, 10, channel.celltype)

        ind0 = np.arange(0, 10)
        idx = np.tile(ind0, (18, 1)) + 10*np.arange(18)[:, None]
        channel.an_nophase_sub = np.nanmean(channel.an_nophase[:, idx], axis=-1)

        _, channel.az_cubspl = self.interpolate_glomeruli(channel.az, channel.celltype)
        channel.az_nophase = self.cancel_phase(channel.az_cubspl, channel.phase, 10, channel.celltype)
        channel.az_nophase_sub = np.nanmean(channel.az_nophase[:, idx], axis=-1)

        # Calculate Right - Left Bridge using zscore
        x = channel.xedges[:-1]
        channel.rightz = np.nanmean(channel.az[:, x >= 9], axis=1)
        channel.leftz = np.nanmean(channel.az[:, x < 9], axis=1)
        channel.rmlz = channel.rightz - channel.leftz
        channel.rplz = channel.rightz + channel.leftz

        channel.rightn = np.nanmean(channel.an[:, x >= 9], axis=1)
        channel.leftn = np.nanmean(channel.an[:, x < 9], axis=1)
        channel.rmln = channel.rightn - channel.leftn
        channel.rpln = channel.rightn + channel.leftn

        # Calculate the expected projection onto the ellipsoid body given known anatomy.
        channel.an_ebproj = self.get_ebproj(channel.an.T, channel.celltype).T

    def get_continuous_bridge(self, bridge, celltype):
        """
        Since glomeruli L9 and R1 are adjacent and map to the same ellipsoid body wedge,
        these are averaged into a single glomerulus in the bridge for the purpose of extracting
        a phase and period from the signal.
        """
        if celltype == 'EIP':
            width = 16
            bridgec = np.zeros((len(bridge), width))
            bridgec[:, :8] = bridge[:, 1:9]
            bridgec[:, -8:] = bridge[:, 9:17]
            # bridgec[:, 7] = bridge[:, 8:10].mean(axis=-1)
        elif celltype == 'PEN_alt':
            bridgec = np.zeros_like(bridge)
            bridgec[:] = bridge[:]
            bridgec[:, 8:10] = np.mean([bridge[:, :2], bridge[:, -2:]], axis=0)
        
        elif celltype == 'PEN'  or celltype == 'SPSP':
            width = 16
            bridgec = np.zeros((len(bridge), width))
            bridgec[:, 0:7] = bridge[:, 1:8]
            bridgec[:, 7] = bridge[:, 0]
            bridgec[:, 8] = bridge[:, -1]
            bridgec[:, 9:] = bridge[:, 10:-1]
            
        nans = np.isnan(bridgec[0]).sum()
        if nans > 0:
            if nans == 1:
                print '%i glomerulus is missing.' %nans
            else:
                print '%i glomeruli are missing.' %nans
            fc.nan2zero(bridgec)
        return bridgec

    def get_ebproj(self, pb, celltype, proj='preserve', mergefunc=np.nansum):
        """
        Projects a matrix with protecerebral bridge as its first or last (0 or -1) axis to
        an ellipsoid body axis. Assumes 18 glomeruli with no interpolation.
        :param pb: PB fluorescent signal matrix. Axis 0 or -1 must be the protocerebral bridge axis (ie. len=18)
        :param celltype: 'EIP' or 'PEN', designating the celltype.
        :return: eb matrix.
        """

        def map_eb2pb(ebcol, celltype):
            if celltype in ['PEN', 'PEN_alt']:
                col1 = int(((ebcol - 1) % (8))) # Left bridge
                col2 = int((ebcol % (8) + 10))  # Right bridge
                pbcols = [col1, col2]
            elif celltype == 'EIP':
                if ebcol % 2 == 0:  # If eb wedge is even, then the pb glomerulus is on the left side
                    if ebcol == 0:
                        pbcols = [8]
                    else:
                        pbcols = [ebcol / 2]
                else:  # If eb wedge is odd, then the pb glomerulus is on the right side
                    pbcols = [ebcol / 2 + 9]
            return np.array(pbcols)

        nwedges_dict = {'EIP': 16, 'PEN': 8, 'PEN_alt': 8}
        ncols = nwedges_dict[celltype]

        if pb.shape[-1] == 18:
            pbdata = pb.T
        else:
            pbdata = pb

        if proj == 'merge':
            shape = list(pbdata.shape)
            shape = [ncols] + shape[1:]
            eb = np.zeros(shape)
            eb[:] = np.nan
            for ebcol in range(ncols):
                pbcols = map_eb2pb(ebcol, celltype)
                ebi = mergefunc(pbdata[pbcols], axis=0)
                eb[ebcol] = ebi

            if pb.shape[-1] == 18:
                return eb.T
            else:
                return eb

        elif proj == 'preserve':
            shape = list(pbdata.shape)
            shape = [2, ncols] + shape[1:]
            eb = np.zeros(shape)
            eb[:] = np.nan
            for ebcol in range(ncols):
                pbcols = map_eb2pb(ebcol, celltype)
                if celltype=='EIP':
                    eb[ebcol%2, ebcol] = pbdata[pbcols]
                elif celltype == 'PEN':
                    eb[:, ebcol] = pbdata[pbcols]

            if pb.shape[-1] == 18:
                return np.array([ebi.T for ebi in eb])
            else:
                return eb

    def get_xcphase(self, data, period):
        nx = data.shape[1]
        xs = np.arange(nx * 2)
        f = 1. / period
        siny = np.sin(2 * np.pi * f * xs) * 2
        phases = np.zeros(len(data))
        for irow, row in enumerate(data):
            #     row = bh.fc.butterworth(row, 5)  # Uncomment to lowpass filter each row before cross-correlation. Much slower, no real difference so far.
            xc = np.correlate(row, siny, mode='full')
            xc = xc[xc.size / 2:]
            phase = xs[np.argmax(xc)]
            phase *= 360 / period
            phases[irow] = phase
        xcphases = fc.wrap(phases)
        return xcphases
    
    def get_pva(self, ac,
                angles_deg = np.tile(np.linspace(-180., 135., 8), 2), window = 1):
        """
        computes the phase of a fluorescent signal (`ac`) by treating every glomerlus as a vector
        inputs: a sanitized matrix of fluoresence values n_timepoints x nglomeruli (channel.acn or channel.acz)
        Assumes the following column labels
        In EIPs, channel.ac(n/z) colums are [8L, 7L, 6L, 5L, 4L, 3L, 2L, 1L, 1R, 2R, 3R, 4R, 5R, 6R, 7R, 8R]
        In PENs, channel.ac(n/z) colums are [8L, 7L, 6L, 5L, 4L, 3L, 2L, 9L, 9R, 2R, 3R, 4R, 5R, 6R, 7R, 8R
    
        angles_deg: an array of angles representing the direction each glomerulus is pointing toward
        same length as n_columns in `ac`
        
        window: 
        
        outputs:
        1: an array representing the average angle of that vector (in degrees)
        2: the magnitude of the resultant vector
        """
        
        # create a vector of angles represented as complex numbers
        # the bridge is a double tiling of the angles from -180 --> 135
        
        angles_rad = np.rad2deg(angles_deg)
        angles_complex = np.array([np.cos(phi) + np.sin(phi)*1j for phi in angles_rad])
        
        # multiply fluorescent signal by this vector
        ac_complex = ac*angles_complex
        
        # extract the angle of the average vector at every timepoint and the avg vector amplitude
        pva = np.angle(np.mean(ac_complex, axis=-1), deg=True)
        pva_amplitude = np.array([abs(c) for c in np.mean(ac_complex, axis=-1)])
        
        if window > 1:
            pva = fc.circ_moving_average(pva, n=window, low=-180, high=180)
            pva_amplitude = fc.moving_average(pva_amplitude, N=window)
        
        return pva, pva_amplitude
        
        
    def get_fftphase(self, data, div=1, **kwargs):
        phase, period, _ = fc.powerspec_peak(data, **kwargs)
        return -phase, period[0] / div

    def get_fftpower_at_peak(self, gcn, peak_period):
        power, period, phase = fc.powerspec(gcn, norm=False)
        return power[:, np.where(period >= peak_period)[0][0]]

    def _get_gloms(self, signal, segmask, celltype):
        centroids = [13, 25, 38, 51, 64, 76, 89, 102, 115, 128, 140, 154, 166, 179, 192, 205, 217, 230]
        gloms = np.zeros((self.len, len(centroids)))
        gloms.fill(np.nan)
        for glomid, val in enumerate(centroids):  # glomid, mask in enumerate(glomask):
            mask = (segmask == val)
            if mask.sum():
                glom = signal * mask
                gloms[:, glomid] = np.nansum( glom.reshape(self.len, signal[0].size) , axis=1) / mask.sum()

        if celltype == 'PEN':
            gloms[:, 8:10] = np.nan
        elif celltype == 'EIP':
            gloms[:, [0, 17]] = np.nan

        return gloms

    def interpolate_glomeruli(self, glom, celltype, kind='cubic', dinterp=.1):
        tlen, glomlen = glom.shape[:2]
        row_interp = np.zeros((tlen, int(glomlen/dinterp)))
        if celltype == 'EIP':
            nans, x = fc.nan_helper(np.nanmean(glom, axis=0))
            wrap = np.zeros_like(glom)
            wrap[:, 1:17] = glom[:, 1:17]
            wrap[:, [0, 17]] = glom[:, [8, 9]]
            x = np.arange(0, 18, 1)
            f = interp1d(x, wrap, kind, axis=-1)
            # f = np.interp(x, wrap, kind, axis=-1)
            x_interp = np.arange(1, 17, dinterp)
            row_interp[:, int(1./dinterp) : int(17./dinterp)] = f(x_interp)
        elif celltype == 'PEN' or celltype == 'SPSP':
            wrap_left = np.zeros((tlen, 10))
            wrap_left[:, 1:9] = glom[:, :8]
            wrap_left[:, [0, 9]] = glom[:, [7, 0]]
            x = np.arange(0, 10, 1)
            fleft = interp1d(x, wrap_left, kind, axis=-1)

            wrap_right = np.zeros((tlen, 10))
            wrap_right[:, 1:9] = glom[:, -8:]
            wrap_right[:, [0, 9]] = glom[:, [17, 10]]
            fright = interp1d(x, wrap_right, kind, axis=-1)

            x_interp = np.arange(1, 9, dinterp)
            row_interp[:, :int(8/dinterp)] = fleft(x_interp)
            row_interp[:, -int(8/dinterp):] = fright(x_interp)

        fc.zero2nan(row_interp)
        x_interp = np.arange(0, 18, dinterp)
        return x_interp, row_interp

    def cancel_phase(self, gc, phase, ppg, celltype, offset=0):
        period = 8 #if celltype=='PEN' else 9
        # period_inds = int(self.period * ppg)
        period_inds = int(period*ppg)
        offset = int(offset * period_inds / 360)
        gc_nophase = np.zeros_like(gc)
        x = np.arange(0, 18, 1./ppg)
        for i in xrange(len(gc)):
            shift = int(np.round((-phase[i] + 180) * period_inds / 360)) + offset
            row = np.zeros(len(x))
            row[:] = np.nan

            if celltype == 'EIP':
                left_ind = (x < 9) & (x >= 1)
                right_ind = (x >= 9) & (x < 17)
            elif celltype in ['PEN', 'SPSP', 'PEN_alt']:
                left_ind = (x < 8)
                right_ind = (x >= 10)
            row[left_ind] = np.roll(gc[i, left_ind], shift)
            row[right_ind] = np.roll(gc[i, right_ind], shift)

            # row = np.array([gc[i][(j - shift) % period_inds] for j in range(gc.shape[1])])
            gc_nophase[i] = row
        return gc_nophase

    def _get_skel(self, signal, points):
        skel = []
        for x, y in points:
            datapoint = signal[:, y, x]
            if datapoint.sum() > 0:
                skel.append(datapoint)
        skel = np.array(skel).transpose()
        return skel

    def get_transform(self, skelmask, segmask=None, length=25, xmax=180, spacing=8, offset=1, plot=False, save=False):
        # Get control points orthogonal to skeleton
        strings = fc.get_strings(skelmask)
        assert len(strings) == 1
        points = strings[0]

        if segmask is None:
            # Subsample points by spacing, offset optional
            spoints = points[offset::spacing]
        else:
            # Get points where segmask crosses skelmask
            segskel = skelmask & segmask
            segx, segy = np.where(segskel.T)
            segpoints = zip(segx, segy)

            # Sort segmentation points along the skeleton
            segpoints_mask = np.array(map(lambda pt: (pt[0], pt[1]) in segpoints, points))
            spoints = points[segpoints_mask]

        slope = fc.get_slope(spoints)
        v = fc.get_vector(slope, spoints, length, orthogonal=True)
        dst = np.concatenate([v[:, 0], spoints, v[:, 1]])

        # Compute corresponding control points as a rectangular lattice
        ny, nx = skelmask.shape
        xs = np.linspace(0, xmax, len(spoints))
        ys = np.ones(len(spoints))
        src = np.concatenate([np.array([xs, ys * i * length / 2.]).T for i in range(3)])

        # get transform of image
        tform = transform.PiecewiseAffineTransform()
        tform.estimate(src, dst)
        output_shape = (length, xmax)

        if plot:
            plt.figure(1, (nx / 20., ny / 20.))
            rcParams['xtick.direction'] = 'in'
            rcParams['ytick.direction'] = 'in'
            lc = LineCollection(v, array=np.zeros(len(spoints)), cmap=plt.cm.Greys)
            ax = plt.subplot(111, axisbg=plt.cm.RdBu_r(0))
            ax.add_collection(lc)
            ax.set_xlim(0, nx)
            ax.set_ylim(0, ny)
            ax.invert_yaxis()

            mx, my = np.where(self.mask.T)
            plt.scatter(mx, my, c=ph.bgreen, s=35, lw=0, marker=',')

            x, y = points.T
            plt.plot(x, y, c='red', lw=3)

            sx, sy = spoints.T
            plt.scatter(sx, sy, c='black', s=40, zorder=10)

            if save:
                plt.savefig('%s_bridge_transform.png' % (self.basename), bbox_inches='tight')

        return tform, output_shape

    def get_bridge(self, signal, skelmask, mask, splice=False, fill=False, **kwargs):
        # Get transform with output shape
        if hasattr(self, 'tform'):
            tform = self.tform
            output_shape = self.tform_output_shape
        else:
            tform, output_shape = self.get_transform(skelmask, **kwargs)
            self.tform, self.tform_output_shape = tform, output_shape

        # Signal cannot exceed 1
        scale = np.abs(signal).max()
        s = signal / scale  # scale down - signal cannot exceed 1.
        maskw = transform.warp(mask, tform,
                               output_shape=output_shape) < .99  # Define warped mask in order to take mean.
        z = 0  # Only supports z-projection so far.
        data = np.zeros([len(s), output_shape[1]])  # Initialize data array.
        for it in xrange(len(s)):
            d = transform.warp(s[it, z], tform, output_shape=output_shape)
            dm = np.ma.masked_array(d, maskw)
            data[it] = dm.mean(axis=0)

        data = data * scale  # scale back up to original range

        xmax = output_shape[1]
        ppg = xmax / 18
        if 'segmask' in kwargs.keys():
            if splice:  # For PENs: see what the signal looks like it you splice out the two unrepresented center glomeruli
                spliced_data = np.zeros((len(data), xmax - ppg * 2))
                spliced_data[:, :8 * ppg] = data[:, :8 * ppg]
                spliced_data[:, -8 * ppg:] = data[:, -8 * ppg:]
                data = spliced_data
            else:
                splidx = np.arange(8 * ppg, 10 * ppg + 1)
                if fill:  # for PENs: fill in the two unrepresented center glomeruli
                    print 'Filling in center glomeruli.'
                    period = 8 * ppg
                    data[:, splidx] = (data[:, (splidx + period) % (18 * ppg)] + data[:, splidx - period]) / 2
        fc.nan2zero(data)
        return data

    def plot_warped_bridge(self, save=False):
        orig = self.tifm.mean(axis=0).mean(axis=0)
        orig = orig / np.abs(orig).max()
        warped = transform.warp(orig, self.tform, output_shape=self.tform_output_shape)

        ny, nx = orig.shape
        plt.figure(1, (nx / 20., ny / 20.))
        ax = plt.subplot(111)
        plt.pcolormesh(orig)
        plt.xlim(0, self.skelmask.shape[1])
        plt.ylim(0, self.skelmask.shape[0])
        ax.invert_yaxis()

        if save:
            plt.savefig('%s_bridge.png' % (self.basename), bbox_inches='tight')

        ny, nx = warped.shape
        plt.figure(2, (nx / 20., ny / 20.))
        ax = plt.subplot(111)
        plt.pcolormesh(warped)
        ax.invert_yaxis()
        plt.xlim(0, self.tform_output_shape[1])
        plt.ylim(0, self.tform_output_shape[0])

        if save:
            plt.savefig('%s_warped_bridge.png' % (self.basename), bbox_inches='tight')


class EllipsoidBody(GCaMP):
    def __init__(self, folder, celltype, celltype2=None, xmlfile=True):
        GCaMP.__init__(self, folder, celltype, celltype2, xmlfile)

        segfile_c1 = glob.glob(self.folder + os.path.sep + 'C1*_seg.tif')
        segfile_c2 = glob.glob(self.folder + os.path.sep + 'C2*_seg.tif')
        if segfile_c1 and segfile_c2:
            self.c1.maskfile = segfile_c1[0]
            self.c2.maskfile = segfile_c2[0]
            self.c1.mask = tiff.imread(self.c1.maskfile) == 0
            self.c2.mask = tiff.imread(self.c2.maskfile) == 0
        else:
            segfile = (glob.glob(self.folder + os.path.sep + '*_seg.tif') +
                    glob.glob(self.folder + os.path.sep + '*ebmask.tif'))
            if segfile:
                self.c1.maskfile = self.c2.maskfile = segfile[0]
                mask = tiff.imread(self.c1.maskfile)
                if len(mask.shape) == 5:
                    mask = mask[:, :, 1]
                if len(mask.shape) == 4:
                    self.c1.mask = mask[:, 0,] == 0
                    self.c2.mask = mask[:, 1] == 0
                elif len(mask.shape) == 3:
                    self.c1.mask = mask == 0
                elif len(mask.shape) == 2:
                    self.c1.mask = mask[None, :] == 0
        if len(self.c1.mask.shape) == 2:
            self.c1.mask = self.c1.mask[None, :, :]

        if hasattr(self.c2, 'mask'):
            if len(self.c2.mask.shape) == 2:
                self.c2.mask = self.c1.mask[None, :, :]
        else:
            self.c2.mask = None

        centerfile = glob.glob(self.folder + os.path.sep + '*center.tif')
        if centerfile:
            self.centerfile = centerfile[0]
            self.centermask = tiff.imread(self.centerfile) == 255

        self.nwedges_dict = {'EIP': 16, 'PEN': 16}

        self.open()

    def open(self):
        tif = GCaMP.open(self, self.c1.file)
        if len(tif.shape) == 4:
            self.c1.tif = tif
            self.c2.tif = GCaMP.open(self, self.c2.file) if self.c2.file else None
        elif len(tif.shape) == 5:
            self.c1.tif = tif[:, :, 0]
            self.c2.tif = tif[:, :, 1]
        #
        # print self.c1.tif.shape
        # print self.c1.mask.shape
        if not self.c1.tif.shape[1] == self.c1.mask.shape[0]:
            mask0 = np.zeros(list(self.c1.tif.shape)[-3:])
            c1mask = copy.copy(mask0)
            c1mask[-self.c1.mask.shape[0]:] = self.c1.mask
            self.c1.mask = c1mask
            if self.c2.mask:
                c2mask = copy.copy(mask0)
                c2mask[-self.rmask.shape[0]:] = self.c2.mask
                self.c2.mask = c2mask

        self.process_channel(self.c1)
        self.process_channel(self.c2)

        if not self.c2.mask is None:
            self.c2.rmlz = self.get_eb_rml('c1.phase', 'c2.az')
            self.c2.rmln = self.get_eb_rml('c1.phase', 'c2.an')
        self.c1.rmlz = self.get_eb_rml('c1.phase', 'c1.az')
        self.c1.rmln = self.get_eb_rml('c1.phase', 'c1.an')

    def process_channel(self, channel):
        if channel.celltype is None: return
        channel.xedges, channel.wedge_rois, channel.theta, channel.a = \
            self.get_wedges(channel.tif, channel.mask, channel.celltype)
        channel.an = self.norm_over_time(channel.a, mode='dF/F0')
        channel.phase = self.get_phase(channel.an, channel.theta)
    
        channel.phasef = fc.circ_moving_average(channel.phase, 3)
        channel.period = self.nwedges_dict[channel.celltype]

        channel.xedges_interp, channel.an_cubspl = self.interpolate_wedges(channel.an, channel.xedges)
        channel.an_nophase = self.cancel_phase(channel.an_cubspl, channel.phase)

        channel.az = self.norm_over_time(channel.a, mode='zscore')[0]
        _, channel.az_cubspl = self.interpolate_wedges(channel.az, channel.xedges)
        channel.az_nophase = self.cancel_phase(channel.az_cubspl, channel.phase)
    
    def get_phase(self, r, theta):
        phase = np.zeros(len(r))
        for i in xrange(len(r)):
            xi, yi = fc.polar2cart(r[i], theta)
            ri, thetai = fc.cart2polar(xi.mean(), yi.mean())
            phase[i] = thetai
        phase *= 180 / np.pi
        return phase

    def get_wedge_rois(self, center, nwedges, offset=0):
        thetas = np.arange(0, np.pi, np.pi/8)
        ms = np.tan(thetas)
        bs = center[1] - ms*center[0]
        dims = self.info['dimensions_pixels']

        # use the center of each pixel as a threshold for segmenting into each wedge
        points = np.array([(i, j) for j in np.arange(dims[0]) for i in np.arange(dims[1])]).T + .5

        def greater_than(points, m, b):
            return points[1] >= (m*points[0] + b)

        def less_than(points, m, b):
            return points[1] < (m*points[0] + b)

        rois16 = np.zeros((16, points.shape[1])).astype(bool)
        for i in range(16):
            j = i%8
            if i < 4:
                roi = greater_than(points, ms[j], bs[j]) & less_than(points, ms[j+1], bs[j+1])
            elif i == 4:
                roi = ( (points[0]) < center[0] ) & greater_than(points, ms[j+1], bs[j+1])
            elif i > 4 and i < 7:
                roi = less_than(points, ms[j], bs[j]) & greater_than(points, ms[j+1], bs[j+1])
            elif i == 7:
                roi = less_than(points, ms[j], bs[j]) & ( (points[1]) >= center[1] )
            elif i > 7 and i < 12:
                roi = less_than(points, ms[j], bs[j]) & greater_than(points, ms[j+1], bs[j+1])
            elif i == 12:
                roi = ( points[0] >= center[0] ) & less_than(points, ms[j+1], bs[j+1])
            elif i > 12:
                roi = greater_than(points, ms[j], bs[j]) & less_than(points, ms[(i+1)%8], bs[(i+1)%8])
            rois16[i] = roi

        rois16 = rois16.reshape((16, dims[0], dims[1]))
        rois16 = np.flipud(rois16) # for some reason need this to match the tiffile orientation
        if nwedges == 8:
            rois8 = np.zeros((8, dims[0], dims[1]))
            for i in range(8):
                rois8[i] = rois16[2*i] | rois16[2*i-1]
            rois = np.roll(rois8[::-1], -1, axis=0)
            xedges = np.arange(0, 17, 2)
        elif nwedges == 16:
            rois = np.roll(rois16[::-1], -3, axis=0)
            xedges = np.arange(0, 17)

        return xedges, rois

    def get_wedges(self, tif, mask, celltype):
        # Compute center of ellipsoid body from centermask
        x, y = np.where(self.centermask.T)
        self.center = (x[0], y[0])
        if self.center == (0, 0):
            self.centermask = self.centermask == False
            x, y = np.where(self.centermask.T)
            self.center = (x[0], y[0])
        self.center = np.array(self.center) + .5


        # Apply mask to each z-plane of tif, and take man across z
        tifm = copy.copy(tif).astype(float)
        tifm[:, mask==False] = np.nan
        tifmz = np.nanmean(tifm, axis=1)

        # Compute ROIs using center coordinate
        xedges, rois = self.get_wedge_rois(self.center, self.nwedges_dict[celltype])

        # Compute mean value in each wedge ROI for each time frame
        wedges = np.zeros((self.len, self.nwedges_dict[celltype]))
        for i, roi in enumerate(rois):
            wedge = tifmz * roi[None, :, :]
            wedges[:, i] = np.nansum( wedge.reshape((self.len, roi.size)), axis=1) / roi.sum()

        # Thetas are the central angle of each wedge, and is used to compute the phase,
        # with zero pointing down (-pi/2).
        theta = np.arange(0, 2*np.pi, 2*np.pi/self.nwedges_dict[celltype])
        if self.nwedges_dict[celltype] == 16:
            theta -= 2*np.pi / 32

        return xedges, rois, theta, wedges

    def get_polar_warping_rois(self, channel, mask, center, celltype):
        def reproject_image_into_polar(self, data, origin, nbins):
            """Reprojects a 3D numpy array ("data") into a polar coordinate system.
            "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
            def index_coords(data, origin=None):
                """Creates x & y coords for the indicies in a numpy array "data".
                "origin" defaults to the center of the image. Specify origin=(0,0)
                to set the origin to the lower left corner of the image."""
                ny, nx = data.shape[:2]
                if origin is None:
                    origin_x, origin_y = nx // 2, ny // 2
                else:
                    origin_x, origin_y = origin
                x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                x -= origin_x
                y -= origin_y
                return x, y

            ny, nx = data.shape[:2]
            if origin is None:
                origin = (nx // 2, ny // 2)

            # Determine that the min and max r and theta coords will be...
            x, y = index_coords(data, origin=origin)
            r, theta = fc.cart2polar(x, y)

            # Make a regular (in polar space) grid based on the min and max r & theta
            r_i = np.linspace(r.min(), r.max(), nx)
            theta_i = np.linspace(-np.pi, np.pi, nbins)
            theta_grid, r_grid = np.meshgrid(theta_i, r_i)

            # Project the r and theta grid back into pixel coordinates
            xi, yi = fc.polar2cart(r_grid, theta_grid)
            xi += origin[0]  # We need to shift the origin back to
            yi += origin[1]  # back to the lower-left corner...
            xi, yi = xi.flatten(), yi.flatten()
            coords = np.vstack((xi, yi))  # (map_coordinates requires a 2xn array)

            # Reproject data into polar coordinates
            output = ndimage.map_coordinates(data.T, coords, order=1).reshape((nx, nbins))

            return output, r_i, theta_i

        channelm = copy.copy(channel).astype(float)
        channelm[:, ~mask] = np.nan
        channelmz = np.nanmean(channelm, axis=1)
        # channelmzn = self.norm_over_time(channelmz, mode='dF/F0')
        nbins = 16 if celltype == 'EIP' else 16
        polar_mask, r, theta = self.reproject_image_into_polar(mask.max(axis=0), center, nbins)
        channelp = np.zeros((len(channelmz), theta.size))
        for i, cart in enumerate(channelmz):
            polar_grid, r, theta = reproject_image_into_polar(cart, center, nbins)
            polar_grid[polar_mask < .99] = np.nan
            channelp[i] = np.nanmean(polar_grid, axis=0)
        return theta, channelp

    def interpolate_wedges(self, eb, xedges, kind='cubic', dinterp=.1):
        period_inds = eb.shape[1]
        tlen, glomlen = eb.shape[:2]
        row_interp = np.zeros((tlen, int(glomlen/dinterp)))
        wrap = np.zeros((eb.shape[0], eb.shape[1]+2))
        wrap[:, 1:period_inds+1] = eb[:, :period_inds]
        wrap[:, [0, period_inds+1]] = eb[:, [period_inds-1, 0]]
        x = np.arange(0, period_inds+2, 1)
        f = interp1d(x, wrap, kind, axis=-1)
        x_interp = np.arange(1, period_inds+1, dinterp)
        row_interp = f(x_interp)

        x_interp -= 1
        return x_interp, row_interp

    def cancel_phase(self, gc, phase, ppg=None, celltype=None, offset=180):
        period_inds = gc.shape[1]
        offset = int(offset * period_inds / 360)
        gc_nophase = np.zeros_like(gc)
        for i in xrange(len(gc)):
            shift = int(np.round((-phase[i] + 180) * period_inds / 360)) + offset
            row = np.roll(gc[i], shift)
            # row = np.zeros(gc.shape[1])
            # row = np.array([gc[i][(j - shift) % period_inds] for j in range(period_inds)])
            gc_nophase[i] = row
        return gc_nophase

    def get_eb_rml(self, phase_label='c1.phase', asym_label='c2.az', subdiv=100):
        phasec = getattr(self, phase_label.split('.')[0])
        phase = getattr(phasec, phase_label.split('.')[-1])
        ac = getattr(self, asym_label.split('.')[0])
        a = getattr(ac, asym_label.split('.')[-1])
        period = phasec.period
        phase = phase * period / 360
        a3 = np.tile(a, (1, 3))
        a3 = np.repeat(a3, subdiv, axis=1)
        phase = (np.round(phase*subdiv) + period*subdiv).astype(int)
        rml = np.zeros(len(a))
        rml[:] = np.nan
        halfperiod = int(period*subdiv/2.)
        for i in xrange(len(a)):
            left = a3[i, phase[i]-halfperiod:phase[i]].mean()
            right = a3[i, phase[i]:phase[i]+halfperiod].mean()
            rml[i] = right-left

        return rml


class Noduli(GCaMP):
    def __init__(self, greenfile, redfile=None, celltype='PEN'):
        GCaMP.__init__(self, greenfile, redfile, celltype)

        noLfile = glob.glob(self.folder + os.path.sep + '*_noL.tif')
        if noLfile: self.noLfile = noLfile[0]

        noRfile = glob.glob(self.folder + os.path.sep + '*_noR.tif')
        if noRfile: self.noRfile = noRfile[0]
        green = self.open()
        if hasattr(self, 'ctlmask'): self.greenctl = self.roi(green, self.ctlmask)

    def open(self):
        green = GCaMP.open(self, self.tiffile)
        self.no = np.zeros((self.len, 2))
        self.noLmask = tiff.imread(self.noLfile) > 0
        self.no[:, 0] = self.mask(green, self.noLmask)
        self.noRmask = tiff.imread(self.noRfile) > 0
        self.no[:, 1] = self.mask(green, self.noRmask)

        self.no_diff = self.no[:, 1] - self.no[:, 0]
        self.no_ratio = self.no[:, 1] / self.no[:, 0]

        return green


class GCaMP_ABF():
    """
    Class to handle an imaging experiment, both TIF and ABF together.
    """

    def __init__(self, folder, celltype, celltype2=None, bhtype=bh.Walk, imtype=Bridge,
                 nstims=None, **kwargs):
        self.folder = folder

        # ABF file
        self.abffile = glob.glob(folder + os.path.sep + '*.abf')[0]
        self.abf = bh.Walk(self.abffile, nstims, **kwargs)
        self.basename = self.abf.basename
        self.name = self.basename.split(os.path.sep)[-1]

        # Imaging Files
        self._get_frame_ts()
        zlen = self.abf.zlen
        imlabel_dict = {Bridge: 'pb', EllipsoidBody: 'eb'}

        def get_zslice(imtype, mask_zlen):
            if imtype == Bridge:
                return slice(0, mask_zlen)
            elif imtype == EllipsoidBody:
                return slice(zlen - mask_zlen, zlen)

        if not type(imtype) is tuple: imtype = [imtype]
        self.ims = []
        self.imlabels = []
        for i_imtype in imtype:
            label = imlabel_dict[i_imtype]
            setattr(self, label, i_imtype(folder, celltype, celltype2))
            tif = getattr(self, label)
            mask_len = (tif.c1.mask.sum(axis=2).sum(axis=1) > 0).sum()
            zslice = get_zslice(i_imtype, mask_len)
            self._get_tif_ts(tif, zslice)

            tif.c1.dphasef = fc.circgrad(tif.c1.phasef)*tif.sampling_rate
            tif.c1.dphase = fc.circgrad(tif.c1.phase)*tif.sampling_rate
            tif.c1.dpva_n = fc.circgrad(tif.c1.pva_n)*tif.sampling_rate
            tif.c1.dpva_z = fc.circgrad(tif.c1.pva_z)*tif.sampling_rate
            tif.c1.dpva_n_f = fc.circgrad(tif.c1.pva_n_f)*tif.sampling_rate
            tif.c1.dpva_z_f = fc.circgrad(tif.c1.pva_z_f)*tif.sampling_rate
            if tif.c2.celltype:
                tif.c2.dphase = fc.circgrad(tif.c2.phase)*tif.sampling_rate
                tif.c2.dphasef = fc.circgrad(tif.c2.phasef)*tif.sampling_rate
                tif.c2.dpva_n = fc.circgrad(tif.c2.pva_n)*tif.sampling_rate
                tif.c2.dpva_z = fc.circgrad(tif.c2.pva_z)*tif.sampling_rate
                tif.c2.dpva_n_f = fc.circgrad(tif.c2.pva_n_f)*tif.sampling_rate
                tif.c2.dpva_z_f = fc.circgrad(tif.c2.pva_z_f)*tif.sampling_rate
                
                
            # OJO! does not work with dual color for some reason
            #tif.xstim = self.subsample('abf.xstim', lag_ms=0, tiflabel=tif,
            #                            metric = lambda y: \
            #                                np.apply_along_axis(lambda x: \
            #                                   stats.circmean(x, low=-180, high=180), axis=1, arr=y))
            #tif.dxstim = fc.circgrad(tif.xstim) * tif.sampling_rate
            self.ims.append(tif)
            self.imlabels.append(label)

        # if hasattr(self.tif, 'ctlmask'):
        #     stimon_tifidx = self._get_stimon_tifidx()
        #     self.tif.greenctldiff = self.tif.greenctl[stimon_tifidx].mean() - self.tif.greenctl[
        #         stimon_tifidx == False].mean()

        print 'Finished opening %s.' % self.folder

    def get_pockels_on(self, trigs, pockel_thresh=.01, plot=False, tlim=None):
        pockels = self.abf.pockels
        t = self.abf.t_orig
        trig_period = np.diff(trigs[:100]).mean()
        pockel_ind = np.arange(trig_period * .1, trig_period * .9, 1).round().astype(int)
        
        pockel_inds = np.tile(pockel_ind, (len(trigs), 1))
        
        pockel_inds += trigs.reshape((len(trigs), 1))
        
        pishape = pockel_inds.shape
        
        pockels_on = pockels[pockel_inds.flatten()].reshape(pishape).mean(axis=1) > pockel_thresh

        if plot:
            plt.figure(1, (15, 5))
            plt.plot(t, self.abf.frametrig, c='black', alpha=.8)
            plt.plot(t, self.abf.pockels, c=ph.blue, alpha=.4)
            plt.scatter(t[trigs], np.ones(len(trigs)), c=pockels_on, lw=0, s=40, zorder=10, cmap=plt.cm.bwr)
            if tlim:
                plt.xlim(tlim)
            else:
                plt.xlim(-.1, 1.2)

        return pockels_on

    def _get_zlen(self, pockels_on):
        if pockels_on.all():
            return 1
        else:
            start, stop = fc.get_inds_start_stop(pockels_on)
            zlens = np.diff(np.vstack([start, stop]), axis=0)[0]
            #assert (zlens[0] == zlens).all()
            return zlens[0]

    def _get_frame_ts(self):
        # Extract frame starts from abffile.
        trigs = fc.rising_trig(self.abf.frametrig, 3)
        # if you stopped in the middle of frame acq, get rid of final frame. BF Dec 20, 2016
        if self.abf.frametrig[-1] > 3: # 3 seems to be the threshold
            trigs = trigs[:-1]

        # Set t=0 to when 2p acquisition begins.
        t_acq_start = self.abf.t_orig[trigs[0]]
        self.abf.t -= t_acq_start
        self.abf.tc = self.abf.t
        self.abf.t_orig -= t_acq_start

        # Get frame triggers times
        self.abf.ft = self.abf.t_orig[trigs]  # Frame triggers
        self.abf.dft = np.diff(self.abf.ft[:100]).mean()  # Frame trigger period

        self.abf.pockels_on = self.get_pockels_on(trigs)  # Need to take into account lost frame triggers due to flip-back of piezo z-motor.
        self.abf.tz = self.abf.ft[self.abf.pockels_on]  # t is not shifted by half a frame because they act as edges when plotted using pcolormesh.
        self.abf.zlen = self._get_zlen(self.abf.pockels_on)

    def _get_tif_ts(self, tif, zslice=None):
        if zslice is None:
            zslice = slice(0, self.abf.zlen)

        tif.zlen = zslice.stop - zslice.start

        # OJO! Previous version (changed April 18, 2018 -- BF)
        #tif.int = self.abf.dft * tif.zlen  # Integration period for one z-stack (includes z=1)
        tif.int = self.abf.dft * self.abf.zlen  # Integration period for one z-stack (includes z=1)
        total_zlen = self.abf.zlen
        
        # times when z slice acq began, relative to the start of imaging 
        tif.t = self.abf.tz[zslice.start::total_zlen][:tif.len]
        # Print warning if TIF images corresponds to number of frame triggers.
        if len(tif.t) != tif.len:
            print 'Number of frames and triggers do not match: \nNumber of frames: %i \nNumber of triggers: %i.' % (tif.len, len(tif.t))
        tif.tc = tif.t + tif.int / 2.

        # the indices of the abf corresponding to beginning of z slice acq
        tif.ind = np.array([np.where(self.abf.t >= t)[0][0] for t in tif.t])
        
        # an estimate of how many abf data you collect during the integration time of the frame
        tif.indint = int(np.round(tif.int / self.abf.subsampling_period))

        # Set sampling rate and period
        tif.sampling_period = np.diff(tif.t).mean()
        tif.subsampling_period = tif.sampling_period
        tif.sampling_rate = 1. / tif.sampling_period
        tif.info['sampling_period'] = tif.sampling_period
        tif.info['sampling_rate'] = tif.sampling_rate
        tif.info['recording_len'] = tif.t[-1] + tif.sampling_period

    def _get_stimon_tifidx(self):
        xstim = copy.copy(self.abf.xstim)
        degperpix = 360 / (24 * 8)
        # 6-pixel wide bar, angle of bar is defined at its center, therefore 3 pixels past the 'edge'
        # the bar is no longer visible, use 2.5 pixels as threshold to compensate for noise
        xstim[xstim < (135 + 2.5 * degperpix)] += 360
        stimon_abfidx = xstim > (225 - 2.5 * degperpix)

        stimon_tifidx = np.zeros(len(self.tif.t)).astype(bool)
        for i, idx in enumerate(self.tif.ind):
            inds = np.zeros(len(xstim)).astype(bool)
            inds[range(idx, idx + self.tif.indint)] = True
            stimon = (inds * stimon_abfidx).sum() > 0
            stimon_tifidx[i] = stimon

        return stimon_tifidx

    def rebin_tif(self, tbin=0.05):
        gcbinneds = []
        nrun = 0
        window_len = self.abf.window_lens[0]
        shoulder_len = self.abf.shoulder_lens[0]
        t0 = -shoulder_len
        tf = window_len + shoulder_len
        ti = np.arange(t0, tf, tbin)
        for stimid in range(self.abf.nstims):
            t = self.tif.tparsed[stimid][nrun]
            gc = self.tif.parsed['normed'][stimid][nrun]
            rows = [[] for _ in ti]
            for itrial, gctrial in enumerate(gc):
                for iline, tline in enumerate(t[itrial]):
                    bi = bisect(ti, tline) - 1
                    rows[bi].append(gctrial[:, iline])

            rows = [np.array(row) for row in rows]
            binned = np.array([row.mean(axis=0) for row in rows])
            gcbinneds.append(binned)

        gcbinneds = np.array(gcbinneds)

        return ti, gcbinneds

    def check_timing(self, tlim=[-1, 2]):
        plt.figure(1, (15, 5))
        plt.plot(self.abf.t_orig, self.abf.frametrig)
        plt.plot(self.abf.t_orig, self.abf.pockels)
        plt.scatter(self.tif.t, np.zeros_like(self.tif.t), zorder=3)
        plt.xlim(tlim)

    def subsample(self, signal, lag_ms=0, tiflabel=None, metric = '', **kwargs):
        """
        Subsample abf channel so that it matches the 2P scanning sampling frequency and integration time.
        Signal is averaged over the integration time of scanning, which is less than the full scanning period.
        """

        if type(signal) is str:
            if not metric:
                if signal.split('.')[-1] in ['forw', 'head', 'side', 'xstim']:

                    metric = lambda y: \
                                np.apply_along_axis(lambda x: \
                                    stats.circmean(x, low=-180, high=180), axis=1, arr=y)
                else:
                    metric = lambda x: np.mean(x, axis = 1)
            
            if 'abf' in signal:
                signal = signal.split('abf.')[1]
            signal = getattr(self.abf, signal)

        if tiflabel is None:
            tif = self.ims[0]
        elif type(tiflabel) is str:
            tif = self.get_objects(tiflabel)[-1]
        else:
            tif = tiflabel            

        indlag = int(np.round(lag_ms / 1000. / self.abf.subsampling_period))
        inds = tif.ind - indlag
        
        signal_subsampled = fc.sparse_sample(signal, inds, tif.indint, metric = metric, **kwargs)
        # args to sparse_sample are abf, tif inds, number of tif inds
        #t_subsampled = self.abf.t[inds] + self.tif.int / 2.
        return signal_subsampled
    
    def upsample(self, tiflabel):
        
        if tiflabel is None:
            tif = self.ims[0]
        elif type(tiflabel) is str:
            tif = getattr(self, tiflabel)
        else:
            tif = tiflabel

        #indint is the number of behavioral inds that take place during imaging        
        ind_imstart = np.where(self.abf.t >= self.ims[0].t[0])[0][0]
        ind_imstop = np.where(self.abf.t >= self.ims[0].t[-1])[0][0]
        
        indint = ind_imstop - ind_imstart
    
        imsignal_upsampled = fc.dense_sample(tif, indint)       
        return imsignal_upsampled
        
    
    
    def bin_gcamp(self, imlabel, binlabel, bins, lag_ms=0, stim=[], abs_sig = False,
                  metric=np.nanmean):
        """
        Bins gc time frames as a function of the binsignal during each time frame.
        Takes mean of time frames within same binsignal bin.
        Lag is positive if the bridge signal comes after the binsignal.
        Returns a 2D matrix: len(bins)-1 x bridge.shape[1]
        """
        # Collect signals
        if type(imlabel) is str:
            _, a = self.get_signal(imlabel)
        else:
            a = imlabel

        if type(binlabel) is str:
            _, x = self.get_signal(binlabel)
        else:
            x = binlabel

        x = self.subsample(binlabel, lag_ms=lag_ms, metric=metric)  # subsample binsignal to match tif sampling

        if stim:
            stimid = np.round(self.subsample('abf.stimid_raw', lag_ms=lag_ms, metric=metric), 1)
            stimidx = np.zeros(len(stimid))
            for s in stim:
                stimidx += np.array(s == stimid)
            stimidx = stimidx.astype(bool)
                                
        ab = np.zeros((len(bins) - 1, a.shape[1]))
        
        ab[:] = np.nan
        digitized = np.digitize(x, bins)
        for i in range(1, len(bins)):
            idx = (digitized == i)
            if stim:
                idx = idx & stimidx
            if idx.sum() > 0:
                ab[i-1] = np.mean(a[idx], axis=0)
        return ab

    def get_offset(self, toffset=None, phase_label='phase'):
        im, c, phase = self.get_objects(phase_label)
        if type(toffset) is tuple or type(toffset) is list:
            idx = (im.tc>=toffset[0]) & (im.tc<toffset[1])
            offset = fc.circmean(phase[idx] - im.xstim[idx])
            return offset
        else:
            if toffset is None:
                toffset = im.tc[np.where((im.tc >= 2) & (im.xstim > -135) & (im.xstim < 135))[0][0]]
            idx = np.where(im.tc >= toffset)[0]
            if len(idx) > 0:
                offset = phase[idx[0]+1] - im.xstim[idx[0]]
                return offset
            else:
                return None

    def get_delay_matrix(self, signal, max_delay=5, metric=fc.circmean, **kwargs):
        if type(signal) is str: signal = getattr(self.abf, signal)
        t = self.abf.t
        sr = self.abf.subsampling_rate
        sp = self.abf.subsampling_period
        subind = self.tif.ind
        subint = self.tif.indint
        max_delay_ind = int(np.round(max_delay * sr))
        delay_matrix = np.zeros((2 * max_delay_ind + 1, len(subind)))
        for i, dind in enumerate(xrange(-max_delay_ind, max_delay_ind + 1)):
            row = bh.fc.sparse_sample(signal, subind + dind, subint, metric, **kwargs)
            delay_matrix[i] = row
        delays = np.arange(-delay, delay + sp, sp)

        self.delays, self.delay_matrix = delays, delay_matrix
        return delays, delay_matrix

    def interpolate_phase(self, phase, power, thresh):
        phase_interp = fc.interpolate(self.tif.t, phase, power>thresh)
        return phase_interp

    def get_labels(self, label):
        """
        :param label: String pointing to an object. If parent objects are not specified,
         then they will be filled in with defaults.
         Defaults: tif > abf. c1 > c2. c = None if signal label is not found in c.

         Examples: 'pb.c1.phase', or 'abf.head' are complete.
         'phase' will be completed to tif.c1.phase (tif = self.imlabels[0]).
         'head' will be completed to tif.head (to be subsampled on the fly, if necessary),
         unless label = 'abf.head', in which case it is left as is.
        :return:
        """
        labels = label.split('.')
        if labels[0] == 'abf':
            return labels
        else:
            c = self.ims[0].c1
            if len(labels) < 2:
                if labels[-1] in dir(c):
                    clabel = 'c1'
                else:
                    clabel = self.imlabels[0]
                labels = [clabel] + labels
            if len(labels) < 3 and labels[-1] in dir(c):
                labels = [self.imlabels[0]] + labels
        return labels

    def get_objects(self, label, lag_ms=0):
        """
        :param label: String pointing to a signal. If parent objects are not specified,
        they are are filled in with defaults (see get_labels).
        :param lag_ms: Time lag in ms to subsample signal if not already subsampled.
        :return:
        """
        labels = self.get_labels(label)
        if labels[0] == 'abf':
            abf = getattr(self, labels[0])
            signal = getattr(abf, labels[-1])
            return abf, None, signal
        else:
            tif = getattr(self, labels[0])
            if len(labels) < 3:
                c = None
                
                if labels[-1] in ['forw', 'xstim', 'head', 'side']:
                    this_metric = lambda y: \
                                    np.apply_along_axis(lambda x: \
                                        stats.circmean(x, low=-180, high=180), axis=1, arr=y)
                    
                else:
                    this_metric = lambda x: np.mean(x, axis = 1)
                signal = self.subsample(labels[-1], lag_ms, tif, metric = this_metric)
            else:
                c = getattr(tif, labels[1])
                signal = getattr(c, labels[-1])

            return tif, c, signal

    def get_signal(self, label, lag_ms=0):
        labels = self.get_labels(label)
        par, c, signal = self.get_objects(label, lag_ms)
        if len(labels) < 3:
            if 'orig' in labels[-1]:
                return par.t_orig, signal
            else:
                return par.tc, signal
        elif labels[-1][0] == 'a':
            if signal.shape[1] > 18:
                return (par.t, c.xedges_interp), signal
            else:
                return (par.t, c.xedges), signal
        else:
            return par.tc, signal

    def get_dark_gain(self, tlim, phaselabel='pb.c1.dphase'):
        head = self.subsample('head', lag_ms=250, metric=fc.circmean)
        head = fc.circ_moving_average(head, n=3)
        im, _, _ = self.get_objects(phaselabel)
        dhead = fc.circgrad(head) * im.sampling_rate
        # head = fc.unwrap(head)
        t, dphase = self.get_signal(phaselabel)
        idx = (t>=tlim[0]) & (t<tlim[1])
        a, b, r, p, e = stats.linregress(dhead[idx], dphase[idx])
        return -a
    
    def testfuc(self, ):
        pass

    def get_headint(self, gain, tiflabel=None, lag_ms=0):
        dhead = fc.circgrad(self.abf.head)
        headint = np.cumsum(dhead * gain)
        return fc.wrap(headint)
        
    def get_epochs(rec, min_t_s = 5, max_t_s = 180, round_to = 0.1,
                   epochs = {'Dark':0.6, 'OL right':0.9, 'OL left':1.2, 'CL':1.5},
                   max_len = 100):
        
        """
        takes a recording, a minimum time in seconds, and a dictionary 
        whose values are floats rounded to nearest tenth representing the 
        value of rec.abf.stimid_raw during the epoch defined by the key. 
        
        outputs a dictionary, where each key has a tuple value.
        the first value is the stimid of the epoch, 
        the second is an np.array of start and end times 
        that the stimid_raw of the abf maintained this value for at least min_t_s
        """
        output_dict = {}
        if round_to:
            rounded_stimid = np.array([round(i / round_to) * round_to for i in rec.abf.ao])
            for (k, val) in epochs.items():
                epochs[k] = round(val / round_to) * round_to
        else:
            rounded_stimid = np.array([round(i, 1) for i in rec.abf.ao])
        t0 = np.where(rec.abf.t >= 0)[0][0]
        tmin = np.where(rec.abf.t >= min_t_s)[0][0]
        min_inds = tmin - t0
        
        # if no epochs given, you could generate automatically with something like:
        # unique_vals = list(set([round(i, 1) for i in r1.abf.stimid_raw]))
        # for i in range(len(unique_vals)):
        #     epochs[i] = unique_vals[i]
        
        for (k, val) in epochs.items():
            if (rounded_stimid == val).any():
                cont_inds = fc.get_contiguous_inds(inds = (rounded_stimid == val),
                                                   min_contig_len= min_inds)
                edge_ts = []
                for isub, sub in enumerate(cont_inds):
                    if isub < (max_len) and (rec.abf.t[sub[-1]] - rec.abf.t[sub[0]]) < max_t_s:
                        edge_ts.append((rec.abf.t[sub[0]], rec.abf.t[sub[-1]]))
                edge_ts = np.array(edge_ts)
                output_dict[k] = (val, edge_ts)
            else:
                output_dict[k] = (val, [])
        return output_dict 


def get_objects(self, label, subsample=False, lag_ms=0):
    """
    :param label: String pointing to a signal. If parent objects are not specified,
    they are are filled in with defaults (see get_labels).
    :param lag_ms: Time lag in ms to subsample signal if not already subsampled.
    :return:
    """
    labels = self.get_labels(label)
    if labels[0] == 'abf':
        abf = getattr(self, labels[0])
        signal = getattr(abf, labels[-1])
        return abf, None, signal
    else:
        tif = getattr(self, labels[0])
        if subsample:
            c = None
            signal = self.subsample(labels[-1], lag_ms, tif, metric=subsample)
        elif len(labels) ==2:
            c = None
            signal = getattr(tif, labels[-1])
        else:
            c = getattr(tif, labels[1])
            signal = getattr(c, labels[-1])

        return tif, c, signal


def get_signal(self, label, **kwargs):
    labels = self.get_labels(label)
    par, c, signal = get_objects(self, label, **kwargs)
    if len(labels) < 3:
        if 'orig' in labels[-1]:
            return par.t_orig, signal
        else:
            return par.tc, signal
    elif labels[-1][0] == 'a':
        if signal.shape[1] > 18:
            return (par.t, c.xedges_interp), signal
        else:
            return (par.t, c.xedges), signal
    else:
        return par.tc, signal


# Collects recordings into a list of flies, each of which is a tuple of GCaMP_ABF recordings.

def get_recs(recnamess, celltype, parent_folder='./', **kwargs):

    def get_trial_name(recname):
        year = recname[:4]
        month = recname[4:6]
        day = recname[6:8]
        recnum = recname[8:10]
        date = year + '_' + month + '_' + day
        return date + '/' + date + '-0' + recnum + '/'

    recs = [
        [GCaMP_ABF(parent_folder + get_trial_name(recname), celltype, **kwargs)
         for recname in recnames]
        for recnames in recnamess
    ]
    return recs


colours = {'EIP': ph.blue, 'PEN': ph.orange, 'SPSP': 'seagreen'}
dcolours = {'EIP': ph.nblue, 'PEN': ph.dorange, 'SPSP': 'darkgreen'}
cmaps = {'EIP': plt.cm.Blues, 'PEN': plt.cm.Oranges, 'SPSP': plt.cm.Greens}

def plot_rml_v_behav(rec, tlim = [0, 300], 
                      imaging = ['leftz', 'rightz'],
                      im_overlay = 'pb.c1.rmlz',
                      behav_overlay = 'abf.dhead',
                      channels = [],
                      cmap = plt.cm.get_cmap('jet'),
                      plot_nan_gloms = False, show_puffs = False, t_offset = None,
                      fig_filename = '', 
                      vmin = None, vmax = None, lw=1.5, cbar_anchor = (-.5, 1),
                      shade_sides = True, accumulated_rotation = []):
    
    # set up figure
    ncols = 0
    if imaging:
        ncols += 1
        im_width = sum([min(np.shape(np.matrix(rec.get_signal(cha)[-1]))) for cha in imaging])
            
    if im_overlay or behav_overlay:
        ncols += 1
        overlay_cols = 1
    else:
        overlay_cols = 0
    ncols += len(channels)
    
    if plot_nan_gloms:
        scale = .33
    else:
        scale = 1.
    
    ang_span = 1
    #if accumulated_rotation:
    #    ang_span = (accumulated_rotation[1] - accumulated_rotation[0])/360. 
    #    if ang_span <= 1.:
    #        ang_span = 1
    
    wr = [im_width * .35 * scale] + [1*ang_span]*overlay_cols + [.75]*(len(channels))
    plt.figure(1, (sum(wr)*3.5, 8))
    gs = gridspec.GridSpec(1, ncols, width_ratios = wr)
    ax_fluo = plt.subplot(gs[0, 0])
    ax_behav = plt.subplot(gs[0, 1])
    gs.update(wspace = .35, hspace = .15)

    ind0 = np.where(rec.ims[0].t >= tlim[0])[0][0]
    abf_ind0 = np.where(rec.abf.t >= tlim[0])[0][0]
    
    # take care of cases where the user is dumb and puts in too big a time
    if tlim[1] > rec.abf.t[-1] or tlim[1] > rec.ims[0].t[-1]:
        min_t = min([rec.abf.t[-1], rec.ims[0].t[-1]])
        ind1 = np.where(rec.ims[0].t >= min_t)[0][0]
        abf_ind1 = np.where(rec.abf.t >= min_t)[0][0] 
    else:
        ind1 = np.where(rec.ims[0].t >= tlim[1])[0][0]    
        abf_ind1 = np.where(rec.abf.t >= tlim[1])[0][0]    
    t0 = tlim[0]
    
    # set up colormesh imaging plot
    # which plots imaging intesity over time as a color
    cmaxes = []
    cmins = []
    
    # get puffs
    if show_puffs:
        puff_inds = fc.detect_peaks(np.squeeze(rec.abf.puff), mph=2.5, mpd=1, threshold=0, edge='rising')
        t_refractory_s = 2.0
        refractory_inds = np.where(rec.abf.t >= t_refractory_s)[0][0] - np.where(rec.abf.t >= 0)[0][0]
    
    for icol, label in enumerate(imaging):
        # Get imtype (pb or eb), channel, and gc
        imlabel, clabel, alabel = rec.get_labels(label)
        im, c, gc = rec.get_objects(label)
        
        # the gc is nmeas x glom in glomerular case (which we want)
        # and (nmeas,) in single channel case
        # if non glomerular, convert shape gc to matrix (nmeas, 1)
        if gc.ndim == 1:
            gc = np.matrix(gc)
            gc = gc.T
        gc_clean = np.zeros((np.shape(gc))) * np.nan
        t = im.t
        if np.shape(gc)[1]== 1: # if not glomerular, make a nans vector number of measurements long
            glomerular = False
            ncols = 1
        else: # if glomerular, make a nans matrix nmeasurements long and nglomeruli wide
            glomerular = True
            ncols = np.shape(gc)[-1]
    
        # delete nan columns
        clean_counter = 0
        ngloms = 1               
        for iglom in range(ncols):
            if not glomerular:
                all_nans = np.all(np.isnan(gc))
                column = gc[:, 0]
                column = np.squeeze(np.asarray(column))
            elif glomerular:
                all_nans = np.all(np.isnan(gc[:, iglom]))
                column = gc[:, iglom]
            if not all_nans:

                gc_clean[:, clean_counter] = column
                ngloms += 1
                clean_counter += 1
        
        gc_clean = gc_clean[:, 0:ngloms]
        xedges = np.arange(ngloms) + icol
        cm = cmap
        if plot_nan_gloms and glomerular:
            gc[np.isnan(gc)] = 0
            gc2plot = gc
            xedges = np.arange(np.shape(gc)[1]+1)
        else:
            gc2plot = gc_clean
        img = ax_fluo.pcolormesh(xedges, t[ind0:ind1], gc2plot[ind0:ind1, :], 
                                 cmap=cm)
        cmaxes.append(gc2plot[ind0:ind1].max())
        cmins.append(gc2plot[ind0:ind1].min())
        if not vmin is None:
            if not type(vmin) is tuple: vmin = (vmin,)
            if len(vmin)-1 >= icol:
                img.set_clim(vmin=vmin[icol])
        if not vmax is None:
            if not type(vmax) is tuple: vmax = (vmax,)
            if len(vmax)-1 >= icol: # about the body axis
                img.set_clim(vmax=vmax[icol])
    
    if show_puffs:
        for ip, p in enumerate(puff_inds):
            if (p-puff_inds[ip-1] >= refractory_inds) or (ip == 0):
                ax_fluo.axhline(rec.abf.t[p], ls = '--', lw = 1, color = 'gold')
                
    ax_fluo.invert_yaxis()
    ax_fluo.set_xticks(range(xedges[-1]+1))
    imtitle = ''
    for i in imaging:
        if not i == imaging[0]:
            imtitle += ', '
        imtitle += i
    ax_fluo.set_title(imtitle, y = 1.05)
    
    # set up colorbar
    if glomerular:
        cb = plt.colorbar(img, ax=ax_fluo, aspect=6, fraction = .2, shrink = .3, 
                          pad=.001,
                          anchor=cbar_anchor,
                          location = 'left',
                          use_gridspec=False)
    else:
        cb = plt.colorbar(img, ax=ax_fluo, aspect=6, fraction = .2, 
                          pad=.225,
                          anchor=cbar_anchor,
                          location = 'left',
                          use_gridspec=False)
    if vmax:
        cmax = vmax[icol]
    else:
        cmax = np.floor(max(cmaxes))
        #if cmax==0:
            #cmax = np.floor(gc[idx].max()*10)/10.
    if vmin:
        cmin = vmin[icol]
    else:
        cmin = np.ceil(min(cmins))
    cb.set_ticks(np.arange(np.ceil(cmin), np.floor(cmax + 1)))
    
    # Set up im / behav overlay plot
    # a reminder: these should both be 1d
    
    if behav_overlay:   
        # use subsampled behav overlay if  `channel` instead of `abf.channel`
        if rec.get_labels(behav_overlay)[0] == 'abf':
            subsampled_behav = False
            bo_t = rec.get_signal(behav_overlay)[0][abf_ind0:abf_ind1]
            behav_overlay_sig = rec.get_signal(behav_overlay)[-1][abf_ind0:abf_ind1]
        elif rec.get_labels(behav_overlay)[0] in ['pb', 'fb', 'eb']:
            subsampled_behav = True
            bo_t = rec.get_signal(behav_overlay)[0][ind0:ind1]
            
            
            behav_overlay_sig = rec.get_objects(behav_overlay)[-1][ind0:ind1]

        if accumulated_rotation:
            behav_overlay_sig = fc.unwrap(behav_overlay_sig)
            #behav_overlay_sig -= behav_overlay_sig[0] 
            print 'min behav', min(behav_overlay_sig), 'max behav', max(behav_overlay_sig)
            
    if im_overlay:
        #im_overlay_sig = fc.wrap(fc.unwrap(rec.get_signal(im_overlay)[-1])-180)
        im_overlay_sig = rec.get_signal(im_overlay)[-1]
        #if accumulated_rotation:
            #im_overlay_sig = fc.unwrap(im_overlay_sig)
            #im_overlay_sig -= im_overlay_sig[0]
         

        if t_offset and (rec.get_labels(im_overlay)[-1] in ['phase', 'phasef']):
            im_offset_ind = np.where(rec.ims[0].t >= t_offset)[0][0]
            if (rec.get_labels(behav_overlay)[-1] in ['xstim', 'head']):
                
                if subsampled_behav:
                    behav_offset_ind = np.where(rec.ims[0].t >= t_offset)[0][0]
                else:
                    behav_offset_ind = np.where(rec.abf.t >= t_offset)[0][0]
                
                if accumulated_rotation:
                    offset_amount = fc.unwrap(rec.get_signal(im_overlay)[-1])[im_offset_ind] - \
                                    fc.unwrap(rec.get_signal(behav_overlay)[-1])[behav_offset_ind]
                    print offset_amount
                else:
                    offset_amount = rec.get_signal(im_overlay)[-1][im_offset_ind] - \
                                    rec.get_signal(behav_overlay)[-1][behav_offset_ind]
            else:
                offset_ind = np.where(rec.ims[0].t >= t_offset)[0][0]
                offset_amount = rec.get_signal(im_overlay)[-1][offset_ind]            
            if accumulated_rotation and im_overlay:
                im_overlay_sig = fc.unwrap(im_overlay_sig) - offset_amount
                print 'min im', min(im_overlay_sig), 'max im', max(im_overlay_sig)
                #im_overlay_sig = im_overlay_sig - offset_amount
            else:
                im_overlay_sig = fc.wrap(im_overlay_sig - offset_amount)
        im_overlay_sig = im_overlay_sig[ind0:ind1]
        
    if accumulated_rotation:
        #ax_behav.plot(behav_overlay_sig-behav_overlay_sig[0], bo_t, c = 'k')
        ph.circplot(behav_overlay_sig-behav_overlay_sig[0], bo_t, circ='x', ax=ax_behav, c = 'k')
    else:
        #ax_behav.plot(behav_overlay_sig, bo_t, c = 'k')
        #ph.circplot(fc.wrap(fc.unwrap(behav_overlay_sig)-180), bo_t, circ='x', ax=ax_behav, c = 'k')
        ph.circplot(behav_overlay_sig, bo_t, circ='x', ax=ax_behav, c = 'k')
    if show_puffs:
        for ip, p in enumerate(puff_inds):
            if (p-puff_inds[ip-1] >= refractory_inds) or (ip == 0):
                ax_behav.axhline(rec.abf.t[p], ls = '--', lw = 1, color = 'gold')
    ax_behav.invert_yaxis()
    if accumulated_rotation:
        xmin_behav = accumulated_rotation[0]
        xmax_behav = accumulated_rotation[1]
        ax_behav.set_xlim([xmin_behav, xmax_behav])
        ax_behav.set_xticks(np.arange(xmin_behav, xmax_behav+1, 180))
    else:
        if rec.get_labels(behav_overlay)[-1] in ['xstim', 'side', 'head', 'forw', 'phase', 'phasef']:
            xmin_behav = -180
            xmax_behav = 180
        else:
            xmax_behav = np.ceil(behav_overlay_sig.max() / 100.0) * 100.0
            xmin_behav = np.floor(behav_overlay_sig.min() / 100.0) * 100.0
        ax_behav.set_xticks([xmin_behav, xmax_behav])  
    
    if rec.get_labels(behav_overlay)[-1] == 'xstim' and shade_sides and not(accumulated_rotation):
        # shade from -180 --> -150 deg, covering tlim[1] - tlim[0] sec on yaxis, left-cornered on (x = -180, y = tlim[1])
        ax_behav.add_patch(mp.Rectangle((-180, tlim[0]), 45, tlim[1]-tlim[0],
            facecolor="grey", alpha = 0.4, lw = 0))
        # shade from 150 --> 180 deg, covering tlim[1] - tlim[0] sec on yaxis, left-cornered on (x = 150, y = tlim[1])
        ax_behav.add_patch(mp.Rectangle((135, tlim[0]), 45, tlim[1]-tlim[0],
            facecolor="grey", alpha = 0.4, lw = 0))
    
    if accumulated_rotation:
        ax_behav_im = ax_behav.twiny() # a new axis for the imaging overlay
    else:
        ax_behav_im = ax_behav.twiny() # a new axis for the imaging overlay
    ax_behav.xaxis.set_ticks_position('top') # have to put this after the twiny for some reason...
    if im_overlay:
        if accumulated_rotation:
            #ax_behav.plot(im_overlay_sig-im_overlay_sig[0], t[ind0:ind1], c = 'g')
            ph.circplot(im_overlay_sig-im_overlay_sig[0], t[ind0:ind1], circ='x', ax=ax_behav, c = 'dodgerblue')
        else:
            #ax_behav_im.plot(im_overlay_sig, t[ind0:ind1], c = 'g')
            ph.circplot(im_overlay_sig, t[ind0:ind1], circ='x', ax=ax_behav_im, c = 'dodgerblue')
            ax_behav_im.axvline(0, c = 'dodgerblue', lw = 0.5, ls = '--')
    ax_behav_im.invert_yaxis()
    if accumulated_rotation:
        xmin_im = accumulated_rotation[0]
        xmax_im = accumulated_rotation[1]
    elif rec.get_labels(im_overlay)[-1] in ['phase', 'phasef']:
        xmin_im = -180
        xmax_im = 180
    elif im_overlay:
        xmax_im = np.ceil(im_overlay_sig.max() * 10) / 10.0
        xmin_im = np.floor(im_overlay_sig.min() * 10) / 10.0
    if im_overlay:
        if accumulated_rotation:
            ax_behav.set_xticks(np.linspace(xmin_im, xmax_im, 3))
            ax_behav_im.set_xticks(np.linspace(xmin_im, xmax_im, 3))
        
        else:
            ax_behav_im.set_xticks([xmin_im, xmax_im])
    ax_behav_im.xaxis.set_ticks_position('bottom')
    ax_behav_im.spines['bottom'].set_color('dodgerblue')
    ax_behav_im.xaxis.label.set_color('dodgerblue')
    ax_behav_im.tick_params(axis='x', colors='dodgerblue')
    
    ax_behav.set_title(behav_overlay + ' vs', y = 1.08, color='k')
    ax_behav_im.set_title(im_overlay, y = 1.05, color='dodgerblue')
    ax_fluo.set_ylim(tlim[::-1])
    ax_behav.set_ylim(tlim[::-1])
    ax_behav_im.set_ylim(tlim[::-1])
    
    # set up axes for additional channels
    # do not necessarily subsample
    if channels:
        colors = ['darkred', 'k', 'darkorange']
        for icha, cha in enumerate(channels):
            ax_cha = plt.subplot(gs[0, -1*len(channels) + icha])
            if rec.get_labels(cha)[0] in ['pb', 'eb', 'fb']: # if imaging label
                cha_t = rec.get_signal(cha)[0][ind0:ind1]
                cha_sig = rec.get_signal(cha)[-1]
                cha_min = np.floor(min(cha_sig))
                cha_max = np.ceil(max(cha_sig))
                cha_sig = cha_sig[ind0:ind1]
            elif rec.get_labels(cha)[0] == 'abf': # if unambiguous behaviour label
                cha_t = rec.get_signal(cha)[0][abf_ind0:abf_ind1]
                cha_sig = rec.get_signal(cha)[-1]
                cha_min = np.floor(min(cha_sig))
                cha_max = np.ceil(max(cha_sig))
                cha_sig = cha_sig[abf_ind0:abf_ind1]
            #ax_cha.plot(cha_sig, cha_t, label = cha, c = colors[icha])
            ph.circplot(cha_sig, cha_t, circ='x', ax=ax_cha, c = colors[icha])
            if rec.get_labels(cha)[-1] == 'xstim':
                # shade from -180 --> -150 deg, covering tlim[1] - tlim[0] sec on yaxis, left-cornered on (x = -180, y = tlim[1])
                ax_cha.add_patch(mp.Rectangle((-180, tlim[0]), 45, tlim[1]-tlim[0],
                    facecolor="grey", alpha = 0.4, lw = 0))
                # shade from 150 --> 180 deg, covering tlim[1] - tlim[0] sec on yaxis, left-cornered on (x = 150, y = tlim[1])
                ax_cha.add_patch(mp.Rectangle((135, tlim[0]), 45, tlim[1]-tlim[0],
                    facecolor="grey", alpha = 0.4, lw = 0))
            if show_puffs:
                for ip, p in enumerate(puff_inds):
                    if (p-puff_inds[ip-1] >= refractory_inds) or (ip == 0):
                        ax_cha.axhline(rec.abf.t[p], ls = '--', lw = 1, color = 'gold')
            ax_cha.set_ylim(tlim)
            ax_cha.set_title(cha, color = colors[icha], y=1.05)
            ax_cha.set_ylim(tlim[::-1])
            
            # assuming most channel signals will wrap around, just set at min and max of whole channel signal.
            # In most cases, -180 and 180. Feel free to change later.
            ax_cha.set_xticks([cha_min, cha_max])
            ax_cha.set_xlim([1.025*cha_min, 1.025*cha_max])
            ax_cha.spines['top'].set_color(colors[icha])
            ax_cha.xaxis.label.set_color(colors[icha])
            ax_cha.tick_params(axis='x', colors=colors[icha])
            ax_cha.xaxis.set_ticks_position('top')
    
    if fig_filename:
        plt.savefig(fig_filename)

# General plotting function

def plot(self, tlim=None, powerspec=[], imaging=['pb.c1.an'], phaseoverlay=True,
         overlay=['c1.phase', 'xstim'], channels=[], mask_powerspec=False, vmin=None,
         vmax=None, toffset=None, offset=None, cmap=None, lines=[], pmin=None, pmax=None,
         dark_gain=None, gs=None, row=0, trig_t=0, unwrap=False, cancel_phase=True,
         show_axes=True, sf=1, reset_t=False, suffix='', save=False, ar=2, size=None,
         highlight='arena', show_powerspec_cbar=False,
         cbar_anchor=(.5, .5), wspace=.03, lw=1.5, axlw=.25):
    """
    Most versatile plotting functions. Used to get explore datasets.
    :param self: GCaMP_ABF recording.
    :param tlim: Time window.
    :param powerspec: True/False. True to plot power spectrum.
    :param imaging: List of labels, starting with a. eg. 'an' = dF/F normalized.
    :param phaseoverlay: True/False. True to overlay phase on the imaging dataset.
    :param overlay: List of labels to plot together. Typically phase and bar or ball.
    :param channels: List of labels to plot under separate columns. Must be abf channels. eg. 'forw' or 'stimid'
    :param mask_powerspec: Do not plot power spectrum, but leave space for it.
    :param vmin: Minimum value in imaging colormap lookup table.
    :param vmax: Maximum value in imaging colormap lookup table.
    :param toffset: Time at which to cancel the offset between phase and bar. Can be an interval, in which case
    the circular mean offset will be computed.
    :param offset: Amount by which to offset the phase. If not defined (ie. None), will be computed using toffset.
    :param cmap: Colormap to plot imaging.
    :param lines: Times to plot horizontal lines across all columns.
    :param dark_gain: Gain with which to multiply the fly's heading. If None, it is computed as the slope between
    the phase and ball velocities in the tlim time window.
    :param gs: Used under plot_trials.
    :param row: Used under plot_trials.
    :param trig_t: Used under plot_trials.
    :param unwrap: True/False. True to unwrap circular data.
    :param show_axes: True to show_axes. Used under plot_trials.
    :param sf: Scale factor for figure size.
    :param reset_t: Sets the x-axis to start at 0.
    :param suffix: String appended to figure name.
    :param save: True to save figure.
    :param ar: Aspect ratio.
    :param filetype: eg.'png', 'svg', 'pdf', etc.
    :return:
    """

    if highlight == 'arena' and unwrap:
        highlight = 'all'

    color_dict = {
        plt.cm.Blues: ph.blue,
        plt.cm.Greens: ph.dgreen,
        plt.cm.Purples: ph.dpurple,
        plt.cm.Oranges: ph.orange,
        plt.cm.CMRmap: ph.dpurple,
        plt.cm.jet: ph.blue
    }

    if not tlim: tlim = [self.ims[0].t[0], self.ims[0].t[-1]]
    if type(tlim) is int:
        trial_times = self.abf.t[self.abf.stimid == tlim]
        tlim = (trial_times[0], trial_times[-1])
    if toffset is None: toffset = tlim[0]

    if reset_t:
        t0 = tlim[0]
        tlim = [0, tlim[1]-tlim[0]]
    else:
        t0 = trig_t

    tabf = self.abf.t  - t0
    tabf_orig = self.abf.t_orig - t0

    # Setup Figure if not triggereds
    if not gs:
        ncols = np.array([len(imaging), len(powerspec), len(channels), len(overlay) > 0]).sum()
        def wr_fcn(imaging_label):
            if 'eb' in imaging_label:
                return 3
            elif 'pb' in imaging_label:
                return 3
        wr = map(wr_fcn, imaging) + [1]*len(powerspec) + [3]*(len(overlay)>0) + [1]*(len(channels))  # width ratios
        hr = [3, 1]
        if size is None:
            size = (10*sf*np.sqrt(ar), 10*sf/np.sqrt(ar))
        fig = plt.figure(1, size)
        # plt.suptitle(self.folder, size=10)
        rcParams['xtick.direction'] = 'out'
        rcParams['ytick.direction'] = 'out'
        gs = gridspec.GridSpec(2, ncols, width_ratios=wr, height_ratios=hr)
        gs.update(wspace=wspace, hspace=0)
    axs = []

    if show_axes:
        stim = 'BAR' if self.abf.ystim.mean() < 5 else 'DARK'
        temp = int(np.round(self.abf.temp.mean()))
        print '%s, %iC' %(stim, temp)

    # Plot Imaging
    if trig_t > 0:
        red_pulse = self.ims[0].c2.an[(self.ims[0].t>0) & (self.ims[0].t<.3)].mean(axis=0)
        pipette = np.argmax(red_pulse) + .5

    irow = row * 2
    
    for icol, label in enumerate(imaging):
        # Get imtype (pb or eb), channel and gc
        imlabel, clabel, alabel = self.get_labels(label)
        im, c, gc = self.get_objects(label)
        #gc = np.matrix(gc)
        #gc = gc.transpose()
        #nmeas = np.shape(gc)[0]
        
        if type(self.get_signal(label)[0]) is tuple: # glomerular
            print 'glomerular'
            (t, xedges), gc = self.get_signal(label)
        else: # not golmerular, leftz or something like that
            t, column = self.get_signal(label)
            t = np.matrix(t)
            t = t.transpose()
            column = np.matrix(column)
            column = column.transpose()
            xedges = np.array(range(icol+2))
            if icol == 0:
                gc = column
            else:
                print np.shape(gc), np.shape(column)
                gc = np.hstack((gc, column))
        
        # Setup axis
        axgc = plt.subplot(gs[irow, icol])
        if trig_t > 0:
            axgc.arrow(pipette, -.4, 0, .2, head_width=0.5, head_length=0.1, fc='r', ec='r', lw=2)

        if show_axes:
            # axgc.set_xlabel(imlabel + ' ' + c.celltype, labelpad=20)
            if imlabel == 'pb':
                axgc.set_xticks(np.array([0, 9, 18]))
            elif imlabel == 'eb':
                xticks = np.array([0, 8, 16])
                axgc.set_xticks(xticks)
                axgc.set_xticklabels(xticks)
            spines = []
            if icol == 0:
                spines.append('left')
                axgc.set_ylabel('Time (s)', rotation='horizontal', labelpad=40)
        else:
            spines = []
        ph.adjust_spines(axgc, spines, ylim=tlim, xlim=[xedges[0], xedges[-1]])

        # Plot gc
        gc[np.isnan(gc)] = 0
        if cmap is None:
            cm = cmaps[c.celltype]
        elif type(cmap) is tuple:
            cm = cmap[icol]
        else:
            cm = cmap
        
        print xedges, np.shape(t-t0), np.shape(gc)
        img = plt.pcolormesh(xedges, t-t0, gc, cmap=cm)
        plt.show(img)
        if not vmin is None:
            if not type(vmin) is tuple: vmin = (vmin,)
            if len(vmin)-1 >= icol:
                img.set_clim(vmin=vmin[icol])
        if not vmax is None:
            if not type(vmax) is tuple: vmax = (vmax,)
            if len(vmax)-1 >= icol:
                img.set_clim(vmax=vmax[icol])
                
        
        # Overlay phase
        if phaseoverlay:
            # scale phase to periodicity of the pb or eb,
            # +.5 is to place it in the middle of the glomerulus
            # (since the glomerulus number maps to the left edges of the pcolormesh grid)
            phase = c.phasef * c.period / 360  +.5 
            phase[phase < 0] = phase[phase < 0] + c.period
            if c.celltype == 'EIP':
                phase[phase<1] = phase[phase<1]+c.period
            ph.circplot(phase, im.tc-t0, c='black', alpha=1.,
                        period=c.period, zorder=11, lw=lw)
        axgc.invert_yaxis()
        axs.append(axgc)

        # Draw colorbar
        botax = plt.subplot(gs[irow+1, icol])
        ph.adjust_spines(botax, [])
        cb = plt.colorbar(img, ax=axgc, aspect=6, pad=0, shrink=.5, anchor=cbar_anchor,
                          use_gridspec=False, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('bottom')
        cb.outline.set_linewidth(lw/2.)
        cb.ax.tick_params('both', length=2, width=lw/2., which='major')
        if vmax:
            cmax = vmax[icol]
        else:
            idx = (t>=tlim[0]) & (t<tlim[1])
            cmax = np.floor(gc[idx].max())
            if cmax==0:
                cmax = np.floor(gc[idx].max()*10)/10.
        if vmin:
            cmin = vmin[icol]
        else:
            cmin = 0
        cb.set_ticks([cmin, cmax])

        # sbsize=10
        # ph.add_scalebar(axgc, sizey=sbsize, sizex=0, width=.1, loc=2,
        #                 color=c, hidey=True)
        # axsb.text(ineuropil+ic, 0, sbsize, ha='center')
        
    

    if powerspec:
        for alabel in powerspec:
            icol += 1
            topax = plt.subplot(gs[irow, icol])
            im = self.ims[0]
            _, a = get_signal(self, alabel)
            power, period, phase = fc.powerspec(a, mask=False, cmap=plt.cm.Greys, show=False, n=10)


            # Plot power spectrum over time
            ph.adjust_spines(topax, ['top'], lw=axlw)
            # topax.set_ylabel('Time (s)', rotation='horizontal', ha='right', labelpad=20)
            if not mask_powerspec:
                yedges = im.t-t0
                xedges = period
                img = topax.pcolormesh(xedges, yedges, power, cmap=plt.cm.Greys)
            topax.set_xlabel('Period (glomeruli)', labelpad=4)
            # ax1.scatter(self.tif.t, periods, color='black', alpha=.8, lw=0, s=5, clip_on=True)
            topax.set_xlim(2, 32)
            topax.set_ylim(tlim)
            topax.invert_yaxis()
            topax.set_xscale('log', basex=2)
            topax.xaxis.set_major_formatter(ScalarFormatter())
            topax.set_xticks([2, 8, 32])

            if pmin is None: pmin=0
            if pmax is None: pmax=1
            img.set_clim(pmin, pmax)

            if show_powerspec_cbar:
                botax = plt.subplot(gs[irow+1, icol])
                ph.adjust_spines(botax, [])
                cb = plt.colorbar(img, ax=topax, aspect=6, shrink=.5, pad=0,
                                 anchor=cbar_anchor,
                          use_gridspec=False, orientation='horizontal')
                cb.ax.xaxis.set_ticks_position('bottom')
                cb.outline.set_linewidth(lw/2.)
                cb.ax.tick_params('both', length=2, width=lw/2., which='major')



                cb.set_clim(pmin, pmax)
                cb.set_ticks([pmin, pmax])

                # Draw colorbar
                # cb = plt.colorbar(img, ax=topax, aspect=4, shrink=.15, pad=-3, anchor=(-15, .1),
                #                   use_gridspec=False)
                #
                # cb.ax.yaxis.set_ticks_position('left')

            #Plot average power spectrum
            # botax = plt.subplot(gs[irow+1, icol])
            # ph.adjust_spines(botax, ['left'])
            # idx1 = (im.t-t0 >= tlim[0]) & (im.t-t0 < tlim[1])
            # idx2 = (period<=32)
            # h = power[:, idx2]
            # h = h[idx1].mean(axis=0)
            # ymax = fc.round(h.max()*10, 1, 'up')/10.
            # botax.set_ylim(0, ymax)
            # botax.set_yticks([0, ymax])
            # botax.plot(period[period<=32], h, c='black', lw=1)
            # botax.set_xlim(2, 32)
            # botax.set_xscale('log', basex=2)
            # ph.adjust_spines(botax, ['left'])

    # Plot overlayed circ plot  
    if overlay:
        icol += 1

        # Setup axis
        ax = plt.subplot(gs[irow, icol])

        # This is just to get the axes aligned (there is no colorbar for this panel)s
        cb = plt.colorbar(img, ax=ax, aspect=6, shrink=.5, pad=0,
                                 anchor=cbar_anchor,
                          use_gridspec=False, orientation='horizontal')
        cb.outline.set_linewidth(axlw)
        cb.ax.tick_params('both', length=2, width=lw/2., which='major')
        cb.set_clim(pmin, pmax)
        cb.set_ticks([pmin, pmax])

        if unwrap:
            xlim = None
            xlabel = r'Rotation ($\pi$ rad)'
        else:
            xlim = (-1, 1)
            xlabel = r'Rotation ($\pi$ rad)'

        if show_axes:
            ax.set_xlabel(xlabel, labelpad=4)
            ph.adjust_spines(ax, ['top'], lw=axlw, xlim=xlim, xticks=(-1, 0, 1), ylim=tlim)
        else:
            ph.adjust_spines(ax, [], lw=axlw, xlim=xlim, xticks=[], ylim=tlim)
        # ax.axvline(0, c='black', ls='--', alpha=0.4)


        intmin = -1.
        intmax = 1.
        for i, label in enumerate(overlay):
            t, a = get_signal(self, label)
            # a = copy.copy(a)
            color = 'grey'
            if 'head' in label:
                if dark_gain is None:
                    dark_gain = self.get_dark_gain((t0+tlim[0], t0+tlim[1]), 'pb.c1.dphase')
                print 'Dark gain: %.2f' %dark_gain
                head = -self.get_headint(dark_gain)
                head = fc.unwrap(head)
                head = self.subsample(head)
                t = self.ims[0].t
                if unwrap:

                    idx0 = np.where(t-t0>=tlim[0])[0][0]
                    head = head - head[idx0]
                a = head
                ph.circplot(head/180., t-t0, c=color, alpha=1, lw=lw, period=2, zorder=2)
            elif 'phase' in label:
                if unwrap and fc.contains(['abf.head'], overlay):
                    phase = fc.unwrap(a)
                    if cancel_phase:
                        idx_offset = np.where(t>=toffset)[0][0]
                        offset = phase[idx_offset] - head[idx_offset]
                        phase = phase - offset
                else:
                    phase = copy.copy(a)
                    if cancel_phase:
                        offset = self.get_offset(toffset, label)
                        print 'offset:', offset
                        phase = fc.wrap(phase - offset)

                _, c, _ = self.get_objects(label)
                color = colours[c.celltype] #if 'eb' in label else 'grey'
                a = phase
                ph.circplot(phase/180., t-t0, c=color, alpha=1, lw=lw,
                            period=2, zorder=2)
            else:
                # b = fc.circ_moving_average(a, n=3)
                ph.circplot(a/180., t-t0, c=color, alpha=1, lw=lw, period=2,
                            zorder=2)
                # plt.plot(a/180., t-t0, c=color, alpha=1, lw=2,
                #             zorder=2)
            a = a / 180.
            a_tlim = a[(t-t0 >= tlim[0]) & (t-t0 < tlim[1])]
            intmin = min(intmin, a_tlim.min())
            intmax = max(intmax, a_tlim.max())

        if unwrap:
            intmin = np.floor(intmin)
            intmax = np.ceil(intmax)
            # intmax = 3
            ax.set_xlim(intmin, intmax)
            xticks = np.arange(intmin, intmax+1, 1)
            ax.set_xticks(xticks)
        if highlight == 'arena':
            ax.fill_betweenx(tabf, x1=-1, x2=-self.abf.arena_edge/180.,
                             color=ph.grey15, edgecolor='none', zorder=1)
            ax.fill_betweenx(tabf, x1=self.abf.arena_edge/180., x2=1,
                             color=ph.grey15, edgecolor='none', zorder=1)
        elif highlight == 'all':
            if unwrap:
                ax.fill_betweenx(tabf, x1=intmin, x2=intmax,
                                 color=ph.grey15, edgecolor='none', zorder=1)
            else:
                ax.fill_betweenx(tabf, x1=-1, x2=1,
                                 color=ph.grey15, edgecolor='none', zorder=1)

        axs.append(ax)
        ax.invert_yaxis()

    # Plot channels individually
    for i, label in enumerate(channels):
        icol += 1
        ax = plt.subplot(gs[irow, icol])
        # This is just to get the axes aligned (there is no colorbar for this panel)s
        cb = plt.colorbar(img, ax=ax, aspect=6, shrink=.5, pad=0,
                                 anchor=cbar_anchor,
                          use_gridspec=False, orientation='horizontal')
        cb.outline.set_linewidth(axlw)
        cb.ax.tick_params('both', length=2, width=lw/2., which='major')
        cb.set_clim(pmin, pmax)
        cb.set_ticks([pmin, pmax])


        if type(label) is str:
            t, signal = self.get_signal(label)
        else:
            signal = label
            if len(signal) == len(self.abf.t):
                t = self.abf.t
            else:
                t = self.ims[0].tc


        if fc.contains(['head', 'side', 'forw', 'xstim'], [label]):
            ax.set_xlim(-180, 180)
            ax.set_xticks([-180, 0, 180])
            # ax.axvline(c='black', ls='--', alpha=.3)
            ph.circplot(signal, t, circ='x', c='black', lw=1)
        else:
            # idx = (t-t0 >= tlim[0]) & (t-t0 < tlim[1])
            # xmin = np.floor(signal.min())
            # xmax = np.ceil(signal.max())
            # ax.set_xlim(xmin, xmax)
            # ax.set_xticks([xmin, xmax])
            # ax.set_xlim(-.1, 1.1)
            ax.plot(signal, t-t0, c='black', lw=1)

        ph.adjust_spines(ax, [], ylim=tlim) #['top']
        if show_axes and type(label) is str:
            ax.set_xlabel(label.title(), labelpad=20)
        ax.invert_yaxis()
        axs.append(ax)

    if trig_t:
        lines.append(0)

    for ax in axs:
        for line in lines:
            ax.axhline(line-t0, c='black', ls='--', lw=lw, dashes=[2, 2])

    if save:
        figname = '%s_%i-%is' % (self.basename, tlim[0], tlim[1])
        if self.basename[0] == '/':
            figname = '.' + figname
        if save is True:
            exts=('png', 'pdf')
        else:
            if not type(save) is tuple:
                save = (save,)
            exts=save
        ph.save(figname.split('/')[-1] + suffix, exts=exts)


# Collecting trial data

def get_trial_ts(rec, trial_len=None):
    trials = np.diff(rec.abf.stimid) == 0
    trials = ndimage.binary_erosion(trials, iterations=25)
    trials = ndimage.binary_dilation(trials, iterations=25).astype(int)
    t0s = rec.abf.t[ np.where(np.diff( trials ) > 0)[0]+1 ]
    if trial_len is None:
        t1s = rec.abf.t[ np.where(np.diff( trials ) < 0)[0]+1 ]
    else:
        trial_starts = np.zeros(len(trial_len))
        trial_starts[1:] = np.cumsum(trial_len)[:-1]

        t1s = np.zeros(len(t0s))
        for i, t0 in enumerate(t0s):
            j = np.argmin(np.abs(trial_starts - t0))
            t1s[i] = t0 + trial_len[j]

    within_bounds = t1s < rec.ims[0].t.max()
    t0s = t0s[within_bounds]
    t1s = t1s[within_bounds]

    if len(t0s):
        if t0s[0] > t1s[0]:
            t1s = t1s[1:]
        return zip(t0s, t1s)
    else:
        return None


def get_trials_ts2(rec, tlim=(-2, 5), trigger='pressure', trigthresh=20, pulsethresh=.04):
    trig_inds = bh.fc.rising_trig(getattr(rec.abf, trigger), trigger=trigthresh,
                                mindist=.01 * rec.abf.subsampling_rate)
    if len(trig_inds) == 0:
        print 'No triggers detected.'
        return
    trig_ts = rec.abf.t[trig_inds]

    ttif = rec.tif.t + rec.tif.int / 2.

    ttrials = []
    greentrials = []
    redtrials = []
    phasetrials = []
    for trig_t in trig_ts:
        twindow = list(tlim + trig_t)
        inds = (ttif >= twindow[0]) & (ttif < twindow[1])
        baseline_inds = (ttif >= twindow[0] - tlim[0] - .5) & (ttif < twindow[0] - tlim[0] - .1)
        ttrial = ttif[inds] - trig_t
        redtrial = rec.tif.rcn[inds]
        red_pulse = redtrial[(ttrial > 0) & (ttrial < .5)]
        if len(red_pulse) and red_pulse.max() < pulsethresh:
            continue
        ttrials.append(ttrial)
        redtrials.append(redtrial)
        greentrials.append(rec.tif.gcn[inds])
        phasetrials.append(rec.tif.phase[inds])

    return zip(ttrials, greentrials, redtrials, phasetrials)


def get_trial_data(groups, metric, summary_fcn, stim_dict, axis=-1, barmintime=.5, minspeed=None,
                   lag_ms=300, temp_thresh=25, zero_offset=True, ntrials=2, maxregdist=4,
                   mintrialtime=10, maxdhead=None, minintensity=None):

    temps = ['cool', 'hot']
    labels = stim_dict.values()
    max_nflies = max(map(len, groups))
    data = [{temp: {label: [] for label in labels} for temp in temps} for _ in groups]

    for igroup, group in enumerate(groups):
        for ifly, fly in enumerate(group):
            flydata = {temp: {label: [] for label in labels} for temp in temps}
            for rec in fly:
                trial_ends = get_trial_ts(rec)
                if trial_ends is None: continue
                im = rec.ims[0]
                trial_tif_inds = [(im.t > start) & (im.t < end) for start, end in trial_ends]
                trial_abf_inds = [(rec.abf.t > start) & (rec.abf.t < end) for start, end in trial_ends]

                xstim = rec.subsample('xstim', lag_ms=lag_ms, metric=fc.circmean)
                barmask = (xstim > -135) & (xstim < 135)
                if barmintime != 0 and not barmintime is None:
                    iters = int(np.ceil(barmintime * im.sampling_rate))
                    barmask = ndimage.binary_erosion(barmask, iterations=iters)

                if not minspeed is None:
                    speedmask = rec.subsample('speed', lag_ms=lag_ms, metric=np.mean) > minspeed

                if not maxregdist is None:
                    regdistmask = im.get_subregdist_mask(thresh=maxregdist)

                if not maxdhead is None:
                    dheadmask = np.abs(rec.abf.dhead) < maxdhead
                    dheadmask = rec.subsample(dheadmask, lag_ms=lag_ms, metric=np.mean)
                    dheadmask = np.floor(dheadmask).astype(bool)
                    dheadmask = ndimage.binary_erosion(barmask, iterations=1)

                if not minintensity is None:
                    intensitymask = np.nanmax(im.c1.an, axis=1) > minintensity

                for itrial, (trial_abf_ind, trial_tif_ind) in enumerate(zip(trial_abf_inds, trial_tif_inds)):
                    temp = 'hot' if rec.abf.temp[trial_abf_ind].mean() > temp_thresh else 'cool'

                    # Get head and phase correctly sampled, with correct units
                    stimid = rec.abf.stimid[trial_abf_ind].mean()
                    if (stimid - np.floor(stimid)) > 0:
                        print rec.name
                        print rec.abf.t[trial_abf_ind].min(), rec.abf.t[trial_abf_ind].max()
                        print 'Trial range is incorrect.'
                    if stimid in stim_dict.keys():
                        label = stim_dict[stimid]
                    else:
                        continue
                    trial_ind = trial_tif_ind

                    if ('bar' in label.lower()) and (not barmintime is None):
                        trial_ind = trial_ind & barmask

                    if not minspeed is None:
                        trial_ind = trial_ind & speedmask

                    if not maxregdist is None:
                        trial_ind = trial_ind & regdistmask

                    if not 'bar' in label.lower() and not maxdhead is None:
                        trial_ind = trial_ind & dheadmask

                    if not minintensity is None:
                        trial_ind = trial_ind & intensitymask

                    if (trial_ind.sum() * im.sampling_period) < mintrialtime:
                        flydata[temp][label].append( np.nan )

                    else:
                        flydata[temp][label].append( metric(rec, trial_ind, label, lag_ms) )

            for temp in temps:
                for label in labels:
                    if len(flydata[temp][label]):
                        try:
                            w = summary_fcn(flydata[temp][label], axis=axis)
                        except ValueError:
                            print flydata[temp][label]
                        data[igroup][temp][label].append( w )

    return data


def get_trial_data2(group, stim_dict, channel, lag_ms=300, temp_thresh=25):
    """
    :param group: List of fly objects, each of which is a tuple of recs.
    :param stim_dict: Dictionary of stimulus ID to stimulus label.
    :param channel: String label of the channel to be parsed.
    :param lag_ms: Lag of channel with respect to tif triggers.
    :param temp_thresh: Threshold for parsing into hot and cool trials.
    :return: Numpy array with dimensions:
    len(group), len(stim_dict), len(temps), ntrials, max len of a trial
    """
    temps = [0, 1]
    rec = group[0][0]
    max_trial_len_tifinds = 0
    max_ntrials = 0
    for fly in group:
        for rec in fly:
            trial_ends = get_trial_ts(rec)
            if not trial_ends: continue
            imax_trial_len_sec = max(map(lambda x: x[1]-x[0], trial_ends))
            sr = rec.tif.sampling_rate
            imax_trial_len_tifinds = imax_trial_len_sec * sr
            max_trial_len_tifinds = max(max_trial_len_tifinds, imax_trial_len_tifinds)

            ntrials = len(trial_ends) / len(stim_dict)
            max_ntrials = max(max_ntrials, ntrials)

    data = np.empty((len(group), len(stim_dict), len(temps), max_ntrials, max_trial_len_tifinds))
    data.fill(np.nan)

    if hasattr(rec.tif, channel):
        channeltype = 'tif'
    elif hasattr(rec.abf, channel):
        channeltype = 'abf'

    for ifly, fly in enumerate(group):
        for rec in fly:
            trial_ends = get_trial_ts(rec)
            if trial_ends is None: continue
            sp = rec.tif.sampling_period
            tif_inds_set = [(rec.tif.t > (start + sp)) & (rec.tif.t < (end - sp)) for start, end in trial_ends]
            abf_inds_set = [(rec.abf.t > (start + sp)) & (rec.abf.t < (end - sp)) for start, end in trial_ends]
            stimids = [rec.abf.stimid[ind].mean() for ind in abf_inds_set]
            temps = [rec.abf.temp[ind].mean() > temp_thresh for ind in abf_inds_set]

            if channeltype == 'tif':
                rec_channel = getattr(rec.tif, channel)
            elif channeltype == 'abf':
                rec_channel = rec.subsample(channel, lag_ms)

            stim_counter = np.zeros(len(stim_dict))
            for itrial, tif_ind in enumerate(tif_inds_set):
                istimid = stimids[itrial] - 1 # stimids start at 1, need to go to zero-based indexing
                itemp = temps[itrial]
                istim = stim_counter[istimid] # the trial number of this specific stimid
                ichannel = rec_channel[tif_ind]

                try:
                    data[ifly, istimid, itemp, istim, :len(ichannel)] = ichannel
                except IndexError:
                    print rec.name

                stim_counter[istimid] += 1

    return data


def get_trial_inds(rec, abf_or_tif, trial_ts, tlag_s=2):
    subrec = getattr(rec, abf_or_tif[:3])
    t = subrec.t_orig if 'orig' in abf_or_tif else subrec.t
    peak_inds = np.array([np.where(t >= trial_t)[0][0] for trial_t in trial_ts])
    sr = subrec.sampling_rate if fc.contains(['orig', 'tif'], abf_or_tif) else subrec.subsampling_rate
    indlag = np.ceil(tlag_s * sr)
    abf_ind = np.arange(-indlag, indlag+1)
    abf_inds = np.tile(abf_ind, (len(peak_inds), 1)) + peak_inds[None, :].T
    return abf_inds.astype(int)


def get_abf_trials(rec, channel, trial_ts, tlag_s=2):
    t = rec.abf.t_orig if 'orig' in channel else rec.abf.t
    peak_inds = np.array([np.where(t >= peak_t)[0][0] for peak_t in trial_ts])
    sr = rec.abf.sampling_rate if 'orig' in channel else rec.abf.subsampling_rate
    indlag = tlag_s * sr
    ch = getattr(rec.abf, channel)

    ch_trials = [ch[ind - indlag : ind + indlag + 1]
                    for ind in peak_inds
                    if (ind-indlag>0) and (ind+indlag+1<len(ch))]

    t_trials = [t[ind - indlag : ind + indlag + 1] - t[ind]
                for ind in peak_inds
                if (ind-indlag>0) and (ind+indlag+1<len(t))]

    return t_trials, ch_trials


def get_tif_trials(rec, metric, trial_ts, tlag_s=2):
#     peak_ts = rec.abf.t[abf_peak_inds]
    gcz = rec.tif.gcz
    glomeruli = rec.tif.xedges[:-1]
    if type(metric) is str:
        metric_values = getattr(rec.tif, metric)
    else:
        metric_values = metric(rec)
    t = rec.tif.t
    metric_trials = []
    t_trials = []
    lag_ind = int(np.ceil(tlag_s*rec.tif.sampling_rate))
    indmax = len(t)-1
    for peak_t in trial_ts:
        if peak_t > t.max(): continue
        peak_ind = np.where(t >= peak_t)[0][0]
        idx = np.arange(peak_ind - lag_ind, peak_ind + lag_ind + 1)
        if (idx[0] > 0) and (idx[-1] < indmax):
            metric_trials.append(metric_values[idx])
            t_trials.append(t[idx] - peak_t)

#     t_trials = np.concatenate(t_trials)
#     rml_trials = np.concatenate(rml_trials)
    return t_trials, metric_trials


def get_group_data(group, label, filler_len=100, subsample=False, lag_ms=0):
    # Collect data
    nanfill = np.zeros((2, filler_len))
    nanfill[:] = np.nan
    if subsample:
        t_trigger = np.hstack(
            [np.hstack([
                        np.hstack([np.vstack([r.pb.t, r.subsample(label, lag_ms=lag_ms)]), nanfill])
                        for r in fly])
             for fly in group])
    else:
        t_trigger = np.hstack(
            [np.hstack([
                        np.hstack([r.get_signal(label), nanfill])
                        for r in fly])
             for fly in group])
    t, trigger = t_trigger
    dt = np.zeros_like(t)
    dt[:-1] = np.diff(t)
    dt[-1] = np.nan
    return dt, trigger


# Plotting trial data

def plot_trials(self, tlim=(-2, 5), imaging=('rcn', 'gcn'), gcoverlay=('phase'), overlay=('phase', 'xstimb'),
                channels=(), trigger='pressure', trigthresh=20, sf=1, save=True, **kwargs):
    """
    Plots all trials in a row. Essentially serially hijacks the plot function.
    :param self: GCaMP_ABF recording.
    :param tlim: Time window, 0 = stimulation time for each panel.
    :param imaging: See plot.
    :param gcoverlay: See plot.
    :param overlay: See plot.
    :param channels: See plot.
    :param trigger: The channel on which to trigger trials.
    :param trigthresh: Threshold with which to trigger each trial.
    :param sf: Scale factor for figure size.
    :param save: True to save figure.
    :param kwargs: More plot options.
    :return:
    """
    trig_inds = bh.fc.rising_trig(getattr(self.abf, trigger), trigger=trigthresh,
                                  mindist=.01 * self.abf.subsampling_rate)
    if len(trig_inds) == 0:
        print 'No triggers detected.'
        return

    trig_ts = self.abf.t[trig_inds]
    trig_ts = trig_ts[np.insert(np.diff(trig_ts) > 1, 0, 1)]
    nrows = len(trig_ts)
    ncols = np.array([len(imaging), powerspec, len(channels), len(overlay) > 0]).sum()
    def wr_fcn(imaging_label):
        if 'eb' in imaging_label:
            return 1
        elif 'pb' in imaging_label:
            return 2
    plt.figure(1, (3 * ncols * sf, 3 * nrows * sf))
    # plt.suptitle(self.folder, size=10)
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    wr = map(wr_fcn, imaging) + [1]*powerspec + [2]*(len(overlay)>0) + [1]*(len(channels))  # width ratios
    gs = gridspec.GridSpec(nrows*2, ncols, width_ratios=wr)
    ###
    plt.title(self.folder, size=10)
    gs.update(wspace=0.2)

    for row, trig_t in enumerate(trig_ts):
        show_axes = (row == 0)
        offset = self.get_offset(toffset=trig_t-1)
        plot(self, tlim, imaging, gcoverlay, overlay, channels, offset=offset, gs=gs, row=row,
            trig_t=trig_t, show_axes=show_axes, save=False, **kwargs)

    if save:
        plt.savefig('%s_%i-%is_plot.png' % (self.abf.basename, tlim[0], tlim[1]), bbox_inches='tight')

    plt.show()


def plot_trial_mean(self, tlim=(-2, 5), tbinsize=.2, t_collapse=(.2, .6), sf=1, save=True, **kwargs):
    """
    Plots mean across trials. Useful for stimulation experiments.
    :param self: GCaMP_ABF recording.
    :param tlim: Time window to plot.
    :param tbinsize: Time bins.
    :param t_collapse: Time window to average in 1D plot.
    :param sf: Scale factor for figure size.
    :param save: True to save figure.
    :param kwargs:
    :return:
    """

    ttrials, greentrials, redtrials, phasetrials = get_trials(self, tlim,  **kwargs)

    ntrials = len(ttrials)
    ttrials = np.concatenate(ttrials)
    greentrials = np.concatenate(greentrials)
    redtrials = np.concatenate(redtrials)

    tbins = np.arange(tlim[0], tlim[1] + tbinsize, tbinsize)
    green_mean = bh.fc.binyfromx(ttrials, greentrials, tbins, metric=np.median, axis=0)
    red_mean = bh.fc.binyfromx(ttrials, redtrials, tbins, metric=np.median, axis=0)

    plt.figure(1, (8 * sf, 5 * sf))
    rcParams['font.size'] = 9 * sf
    ph.set_tickdir('out')
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 10], width_ratios=[10, 10, 1])
    plt.ylabel('Time after stimulation (s)')
    plt.xlabel('Z-Score (stdev)')

    # Red Channel heatmap
    axred = plt.subplot(gs[1, 0])
    dx = np.diff(self.tif.xedges)[0] / 2.
    xedges = self.tif.xedges + dx
    xlim = [xedges[0], xedges[-1]]
    ph.adjust_spines(axred, ['left', 'bottom'], xticks=xedges[:-1:2] + dx)
    axred.set_ylabel('Time after stimulation (s)')
    axred.set_xlabel('Alexa568 (%i trials)' % ntrials)
    yedges = tbins
    axred.pcolormesh(xedges, yedges, red_mean, cmap=plt.cm.Reds)
    axred.axhline(c='black', ls='--', alpha=.8)
    axred.invert_yaxis()

    # Green channel heatmap (GCaMP)
    axgc = plt.subplot(gs[1, 1])
    ph.adjust_spines(axgc, ['bottom'], xticks=xedges[:-1:2] + dx)
    axgc.set_xlabel('GCaMP (%i trials)' % ntrials)
    yedges = tbins
    axgc.pcolormesh(xedges, yedges, green_mean, cmap=plt.cm.Blues, alpha=1)
    axgc.axhline(c='black', ls='--', alpha=.8)
    axgc.invert_yaxis()

    # 1D snapshot of green and red channels along bridge
    # Green channel
    ax = plt.subplot(gs[0, 1])
    ph.adjust_spines(ax, ['top'], xticks=xedges[:-1:2] + .5)
    g = xedges[1:-2] + dx
    tb = tbins[:-1] + tbinsize / 2.
    post_inds = (tb >= t_collapse[0]) & (tb < t_collapse[1])
    pre_inds = (tb >= -.4) & (tb < 0)
    green_tcollapse = green_mean[post_inds, 1:-1].mean(axis=0) - green_mean[pre_inds, 1:-1].mean(axis=0)
    ax.plot(g, green_tcollapse, c=ph.blue)
    # Red channel
    ax2 = ax.twinx()
    ph.adjust_spines(ax2, ['top'])
    redinds = (tb >= 0) & (tb < .4)
    red_tcollapse = red_mean[redinds, 1:-1].mean(axis=0)  # - red_mean[pre_inds, 1:-1].mean(axis=0)
    ax2.plot(g, red_tcollapse, c=ph.red)
    ax2.set_xlim(xlim)

    # show phases of green and red channels
    # green_peak_inds = fc.detect_peaks(green_tcollapse, mph=0)
    # red_peak_glom = np.argmax(red_tcollapse)+2 # +1 fora zero-based index, +1 for missing first glomerulus
    # green_peak_glom = green_peak_inds[np.argmin(np.abs(green_peak_inds - red_peak_glom))]+2   # +1 fora zero-based index, +1 for missing first glomerulus
    # print red_peak_glom
    # print green_peak_glom
    # ax.axvline(red_peak_glom, c='red') # +1 adjusts for missing first glomerulus
    # ax.axvline(green_peak_glom, c='blue') # +1 adjusts for missing first glomerulus
    # ax.set_xlim(xlim)

    # Draw an arrow where the pipette was positioned (maximum signal in red channel)
    red_maxglom = np.argmax(red_tcollapse) + 1
    axgc.arrow(red_maxglom + 1, -.4, 0, .2, head_width=0.5, head_length=0.1, fc='r', ec='r', lw=2)
    axred.arrow(red_maxglom + 1, -.4, 0, .2, head_width=0.5, head_length=0.1, fc='r', ec='r', lw=2)

    # 1D snapshot of peak glomerulus in green and red channels over time     
    # ax = plt.subplot(gs[1, 2])
    # ph.adjust_spines(ax, ['right'])
    # ax.axhline(c='black', ls='--', alpha=.8)
    # ax.plot(red_mean.mean(axis=1), tb, c=ph.red)
    # ax.invert_yaxis()
    # ax2 = ax.twiny()
    # ph.adjust_spines(ax2, ['right'])
    # ax2.plot(green_mean[:, np.argmax(green_tcollapse)], tb, c=ph.blue)

    if save:
        plt.savefig('%s_%i-%is_2d_median_trials.png' % (self.abf.basename, tlim[0], tlim[1]), bbox_inches='tight')



# Miscellaneous plots

def plot_offsets(group, color='black', nstims=2, mode='mean',
                 lw=.5, size=None, s=1, ncols=3, minbartime=10,
                 trial_len=None, figname=''):
    """
    Plots offsets under two modes. Mode 'mean' plots the circular mean and circular standard
    deviation for each fly. Mode 'all' plots the circular mean for each trial in each fly.
    :param group: List of fly objects (each a list of recs).
    :param color: Color to plot.
    :param nstims: Number of stimuli in each recording.
    :param mode: 'mean' or 'all'. See above.
    :param figname: Figure name to save.
    :return:
    """
    stim_dict = {1: 'Bar', 2: 'Dark'}
    ntrials_per_rec = 6

    trial_starts = np.zeros(len(trial_len))
    trial_starts[1:] = np.cumsum(trial_len)[:-1]

    offsets = []
    offset_std = []
    for ifly, fly in enumerate(group):

        fly_offsets = []
        fly_offset_std = []

        for rec in fly:
            ts = get_trial_ts(rec, trial_len=trial_len)
            stimid = rec.subsample('stimid')
            xstim = rec.subsample('abf.xstim', lag_ms=300)
            phase = rec.pb.c1.phase

            rec_offsets = np.zeros(len(trial_len))
            rec_offsets[:] = np.nan
            rec_offset_std = np.zeros(len(trial_len))
            rec_offset_std[:] = np.nan

            bar_cnt = 0
            for i, (t0, t1) in enumerate(ts):
                idx = (rec.pb.t >= t0) & (rec.pb.t < t1)
                istimid = int(stimid[idx].mean().round())
                if istimid==0:
                    print stimid[idx].mean()
                    print rec.name
                if stim_dict[istimid] == 'Bar':
                    idx = idx & (rec.pb.xstim > -135) & (rec.pb.xstim < 135)
                    if idx.sum()*rec.pb.sampling_period < minbartime:
                        ioffset = np.nan
                        ioffset_std = np.nan
                    else:
                        ioffset = fc.circmean(phase[idx] - xstim[idx])
                        ioffset_std = fc.circstd(phase[idx] - xstim[idx])

                    j = np.argmin(np.abs(trial_starts-t0))
                    rec_offsets[j] = ioffset /180.
                    rec_offset_std[j] = ioffset_std / 180.
                    bar_cnt += 1

            fly_offsets.extend(list(rec_offsets))
            fly_offset_std.extend(list(rec_offset_std))
        # fly_offsets = np.concatenate(fly_offsets)
        # fly_ntrials = (np.isnan(fly_offsets)==False).sum()
        offsets.append( fly_offsets )
        offset_std.append( fly_offset_std )


    maxlen = 0
    for fly in offsets:
        maxlen = max([maxlen, len(fly)])


    if mode == 'mean':
        # Plot median offset for each fly
        if size is None:
            size = (4, 4)
        plt.figure(1, size)
        ax = plt.subplot(111)

        stimlabels = ['Bar', 'Dark']
        plt.xticks(np.arange(len(group))+1)
        plt.xlim(0, len(group)+1.5)
        plt.ylim(-1, 1)
        plt.yticks([-1,0,1])
        ph.adjust_spines(ax, ['left', 'bottom'])

        mean = fc.circmean(offsets, axis=1)
        err = fc.circstd(offsets, axis=1)
        mean, err = zip(*sorted(zip(mean, err)))
        plt.errorbar(np.arange(len(group))+1, mean, err, fmt='_', c=color, lw=1)
    elif mode == 'all':
        # Plot offset for each trial in each fly
        nrows = int(np.ceil(len(group)/float(ncols)))
        if size is None:
            size = (nrows*3, ncols*3)
        plt.figure(1, size)
        gs = gridspec.GridSpec(nrows, ncols)
        gs.update(wspace=.2, hspace=.4)
        for ifly in range(len(group)):
            irow = ifly/ncols
            icol = ifly%ncols
            ax = plt.subplot(gs[irow, icol])

            if irow==nrows-1 and icol==0:
                ax.set_ylabel('Offset (pi rad)')
                ax.set_xlabel('Trial Number')
                spines = ['left', 'bottom']
                # ax.set_xticks(np.arange(0, maxlen, 3)+1)
            else:
                spines = []
            ph.adjust_spines(ax, spines, ylim=(-1, 1), yticks=[-1, 0, 1],
                             xlim=(.5, maxlen+.5), lw=lw/2.)
            # ax.grid(c=ph.grey4, ls='--', zorder=1)
            ax.axhline(c=ph.grey4, ls='--', zorder=1)
            x = np.arange(len(offsets[ifly]))+1
            ax.scatter(x, offsets[ifly], edgecolor=color, facecolor=color,
                       s=s, zorder=3)
            ax.errorbar(x, offsets[ifly], offset_std[ifly], fmt='_',
                     c=color, elinewidth=.5, capthick=.5,
                     mew=.5, capsize=1, ms=2)
            for i in range(1, ntrials_per_rec+1, ntrials_per_rec*2):
                ax.fill_between([i-.5, i+ntrials_per_rec-.5], -1, 1, facecolor=ph.grey2,
                                edgecolor='none')

    if figname:
        ph.save(figname, rec)


def get_offset_map(xstim, phase, bins):
    offsets = np.zeros(len(bins)-1)
    offsets[:] = np.nan

    if len(xstim) > 0:
        dig = np.digitize(xstim, bins)
        for i in xrange(1, len(bins)):
            offsets[i-1] = fc.circmean(phase[dig==i] - xstim[dig==i])

    return offsets

def merge_maps(map1, map2):
    nanmap1 = np.isnan(map1)
    newmap = copy.copy(map1)
    newmap[nanmap1] = map2[nanmap1]
    return newmap


def plot_offsets2(group, color='black', nstims=2, mode='mean',
                 lw=.5, size=None, s=1, ncols=3, minbartime=10,
                  binwidth=10, trial_len=None, figname=''):
    """
    Plots offsets under two modes. Mode 'mean' plots the circular mean and circular standard
    deviation for each fly. Mode 'all' plots the circular mean for each trial in each fly.
    :param group: List of fly objects (each a list of recs).
    :param color: Color to plot.
    :param nstims: Number of stimuli in each recording.
    :param mode: 'mean' or 'all'. See above.
    :param figname: Figure name to save.
    :return:
    """

    bins = np.arange(-180, 181, binwidth)
    ntrials_per_rec = 6

    stim_dict = {1: 'Bar', 2: 'Dark'}
    ntrials_per_stim = 6
    # offsets = np.zeros((len(group), ntrials_per_stim*2))
    # offsets[:] = np.nan
    offsets = []
    offset_std = []
    for ifly, fly in enumerate(group):
        fly_offsets = []
        fly_offset_std = []

        offset_map0 = np.zeros(len(bins)-1)
        offset_map0[:] = np.nan

        for irec, rec in enumerate(fly):
            ts = get_trial_ts(rec, trial_len=trial_len)
            stimid = rec.subsample('stimid')
            xstim = rec.subsample('abf.xstim', lag_ms=300)
            phase = rec.pb.c1.phase

            rec_offsets = []
            rec_offset_std = []
            for itrial, (t0, t1) in enumerate(ts):
                idx = (rec.pb.t >= t0) & (rec.pb.t < t1)
                istimid = int(stimid[idx].mean().round())
                if istimid==0:
                    print stimid[idx].mean()
                    print rec.name
                if stim_dict[istimid] == 'Bar':
                    idx = idx & (xstim > -135) & (xstim < 135)
                    min_exposure_inds = int(round(minbartime*rec.ims[0].sampling_rate))/2
                    idx = ndimage.binary_erosion(idx, iterations=min_exposure_inds)
                    idx = ndimage.binary_dilation(idx, iterations=min_exposure_inds)

                    ioffset_map = get_offset_map(xstim[idx], phase[idx], bins)
                    diff_idx = (np.isnan(ioffset_map)==False) & (np.isnan(offset_map0)==False)
                    diff = ioffset_map[diff_idx] - offset_map0[diff_idx]
                    ioffset_dist = fc.circmean(diff)
                    ioffset_dist_std = fc.circstd(diff)

                    # ioffset = fc.circmean(phase[idx] - xstim[idx])
                    # ioffset_std = fc.circstd(phase[idx] - xstim[idx])

                    rec_offsets.append( ioffset_dist /180. )
                    rec_offset_std.append( ioffset_dist_std / 180.)

                    offset_map0 = merge_maps(offset_map0, ioffset_map)

            fly_offsets.extend(rec_offsets)
            fly_offset_std.extend(rec_offset_std)
        # fly_offsets = np.concatenate(fly_offsets)
        # fly_ntrials = (np.isnan(fly_offsets)==False).sum()
        offsets.append( fly_offsets )
        offset_std.append( fly_offset_std )
    # offsets = offsets / 180.
    maxlen = 0
    for fly in offsets:
        maxlen = max([maxlen, len(fly)])


    if mode == 'mean':
        # Plot median offset for each fly
        if size is None:
            size = (4, 4)
        plt.figure(1, size)
        ax = plt.subplot(111)

        stimlabels = ['Bar', 'Dark']
        plt.xticks(np.arange(len(group))+1)
        plt.xlim(0, len(group)+1.5)
        plt.ylim(-1, 1)
        plt.yticks([-1,0,1])
        ph.adjust_spines(ax, ['left', 'bottom'])

        mean = fc.circmean(offsets, axis=1)
        err = fc.circstd(offsets, axis=1)
        mean, err = zip(*sorted(zip(mean, err)))
        plt.errorbar(np.arange(len(group))+1, mean, err, fmt='_', c=color, lw=1)
    elif mode == 'all':
        # Plot offset for each trial in each fly
        nrows = int(np.ceil(len(group)/float(ncols)))
        print nrows
        if size is None:
            size = (nrows*3, ncols*3)
        plt.figure(1, size)
        gs = gridspec.GridSpec(nrows, ncols)
        gs.update(wspace=.2, hspace=.4)
        for ifly in range(len(group)):
            irow = ifly/ncols
            icol = ifly%ncols
            ax = plt.subplot(gs[irow, icol])

            if irow==nrows-1 and icol==0:
                ax.set_ylabel('Offset (pi rad)')
                ax.set_xlabel('Trial Number')
                spines = ['left', 'bottom']
                ax.set_xticks(np.arange(0, maxlen, 3)+1)
            else:
                spines = []
            ph.adjust_spines(ax, spines, ylim=(-1, 1), yticks=[-1, 0, 1],
                             xlim=(.5, maxlen+.5), lw=lw/2.)
            # ax.grid(c=ph.grey4, ls='--', zorder=1)
            ax.axhline(c=ph.grey4, ls='--', zorder=1)
            x = np.arange(len(offsets[ifly]))+1
            ax.scatter(x, offsets[ifly], edgecolor=color, facecolor=color,
                       s=s, zorder=3)
            ax.errorbar(x, offsets[ifly], offset_std[ifly], fmt='_',
                     c=color, elinewidth=.5, capthick=.5,
                     mew=.5, capsize=1, ms=2)
            for i in range(1, ntrials_per_rec+1, ntrials_per_rec*2):
                ax.fill_between([i-.5, i+ntrials_per_rec-.5], -1, 1, facecolor=ph.grey2,
                                edgecolor='none')

    if figname:
        ph.save(figname, rec)


def plot_dark_bar_transition_offset_diff(group, nstims=2,
                                         bar_window_s=50, min_bar_len_s=10,
                                         min_bar_s=None, size=None, figname=''):
    stim_dict = {1: 'Bar', 2: 'Dark'}
    ntrials_per_stim = 3
    fts = []    # first transitions
    sts = []    # subsequent transitions
    for ifly, fly in enumerate(group):
        for irec, rec in enumerate(fly):
            ts = get_trial_ts(rec)
            stimid = rec.subsample('stimid')
            xstim = fc.circ_moving_average(rec.pb.xstim, n=3)
            phase = rec.pb.c1.phase
#             xstim = rec.pb.xstim
            idx_barall = (rec.pb.xstim > -135) & (rec.pb.xstim < 135)
            for i, (t0, t1) in enumerate(ts):
                idx_bartrial = (rec.pb.t >= t0) & (rec.pb.t < (t0 + bar_window_s)) & (rec.pb.t < t1)
                istimid = int(stimid[idx_bartrial].mean().round())
                if istimid==0:
                    print stimid[idx_bartrial].mean()
                    print rec.name
                if stim_dict[istimid] == 'Bar' and i!=0:
                    idx_bartrial = idx_bartrial & idx_barall
                    if not min_bar_s is None:
                        contigs = fc.get_contiguous_inds(idx_bartrial)
                        min_ninds = int(np.ceil(min_bar_s / rec.pb.sampling_period))
                        start_contig = np.where(np.array(map(len, contigs)) >= min_ninds)[0][0]
                        idx0 = contigs[start_contig][0]
                        t0 = rec.pb.t[idx0]

                        idx_bartrial = idx_barall & (rec.pb.t >= t0) & (rec.pb.t < (t0 + bar_window_s)) & (rec.pb.t < t1)

                    idx_darktrial = idx0 - 1
                    # if idx_bartrial.sum()*rec.pb.sampling_period < min(min_bar_len_s, bar_window_s): continue
                    baroffset = fc.circmean(phase[idx_bartrial] - xstim[idx_bartrial])
                    darkoffset = fc.circmean(phase[idx_darktrial] - xstim[idx_darktrial])
                    transition = fc.circmean(baroffset - darkoffset)
                    if irec == 0 and i == 1:
                        fts.append(transition)
                    else:
                        sts.append(transition)

    fts = np.array(fts) #/ 180.
    sts = np.array(sts) #/ 180.
    print 'Number of first transitions: %i' %len(fts)
    print 'Number of subsequent transitions: %i' %len(sts)

    binsize = 30
    xbins = np.arange(-180, 181, binsize)
    hft, x = np.histogram(fts, bins=xbins, density=True)
    sft, x = np.histogram(sts, bins=xbins, density=True)
    hft *= binsize
    sft *= binsize

    if size is None: size = (4, 3)
    plt.figure(1, size)
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'], lw=.25)
    ax.set_xlim(-180, 181)
    ax.set_xticks(xbins)
    ax.bar(x[:-1], hft, width=binsize, facecolor=ph.blue, edgecolor='black', lw=.25)
    ax.bar(x[:-1], sft, width=binsize, facecolor=ph.grey1, edgecolor='black', lw=.25)


    if figname:
        ph.save(figname, exts=['pdf'])


def plot_periods(group, color='black', show_labels=True,
                 size=None, lw=None, figname=''):
    periods = np.zeros((2, len(group)))
    for ifly, fly in enumerate(group):
        for rec in fly:
            if rec.pb.c1.celltype == 'PEN':
                a = rec.pb.c1.acn
            elif rec.pb.c1.celltype == 'EIP':
                a = rec.pb.c1.an[:, 1:-1]
            power, period, phase = fc.powerspec(a)
            window = (period > 2) & (period < 18)
            period = period[window]
            stimid = rec.subsample('stimid')
            clbar = (stimid == 1) & (rec.pb.xstim > -135) & (rec.pb.xstim < 135)
            dark = (stimid == 2)
            pbar = period[np.argmax(power[clbar].mean(axis=0)[window])]
            pdark = period[np.argmax(power[dark].mean(axis=0)[window])]
            if pbar >10 or pdark > 10:
                print pbar, pdark
                print rec.name
            periods[0, ifly] = period[np.argmax(power[clbar].mean(axis=0)[window])]
            periods[1, ifly] = period[np.argmax(power[dark].mean(axis=0)[window])]

    # Plot periods per stimulus
    if size is None:
        size = (3, 3)
    plt.figure(1, size)
    ax = plt.subplot(111)
    stimlabels = ['Bar', 'Dark']
    plt.xticks(range(len(group)), stimlabels)
    plt.xlim(-.5, 2-.5)
    plt.ylim(7, 9)
    plt.yticks([7, 8, 9])
    ph.adjust_spines(ax, ['left', 'bottom'], lw=.25)

    for istim, stimlabel in enumerate(stimlabels):
        x = np.random.normal(istim, .05, len(group))
        med = np.nanmedian(periods[istim])
        std = np.nanstd(periods[istim])
        plt.scatter(x, periods[istim], facecolor='none', edgecolor=color,
                    s=3, lw=.35)
        plt.errorbar(istim+.3, med, std, fmt='_',
                     c=color, elinewidth=.5, capthick=.5,
                     mew=.5, capsize=1, ms=2)

    if figname:
        ph.save(figname, rec)

def plot_correlations(group, color='black', show_labels=True, nstims=2,
                      lag_ms=300, mintrialtime_s=10, minspeed=0, mintop2=.8,
                      sigma_s=2, figname='', size=None):
    stim_dict = {1: 'Bar', 2: 'Dark'}
    rs = np.zeros((nstims, len(group)))
    rs[:] = np.nan
    for ifly, fly in enumerate(group):
        fly_rs = []
        for rec in fly:
            ts = get_trial_ts(rec)
            stimid = rec.subsample('abf.stimid')
            xstim = rec.subsample('abf.xstim', lag_ms=lag_ms)
            xstim = fc.circ_moving_average(xstim, n=3)
            head = -rec.subsample('abf.head', lag_ms=lag_ms)
            head = fc.circ_moving_average(head, n=3)
            dhead = fc.circgrad(head) * rec.pb.sampling_rate
            sigma = sigma_s / rec.pb.sampling_period
            dhead = ndimage.filters.gaussian_filter1d(dhead, sigma)
            dphase = ndimage.filters.gaussian_filter1d(rec.pb.c1.dphasef, sigma)
            speed = rec.subsample('abf.speed', lag_ms=lag_ms)
            top2 = np.nanmean(np.sort(rec.pb.c1.al, axis=1)[:, -4:])
            # dxstim = rec.subsample('abf.dxstim', lag_ms=lag_ms)

            rec_rs = np.zeros((nstims, len(ts)/nstims))
            rec_rs[:] = np.nan
            bar_cnt = 0
            dark_cnt = 0
            for i, (t0, t1) in enumerate(ts):
                idx = (rec.pb.t >= t0) & (rec.pb.t < t1)
                istimid = int(stimid[idx].mean().round())
                if istimid==0:
                    print stimid[idx].mean()
                    print rec.name
                if stim_dict[istimid] == 'Bar':
                    idx = idx & (rec.pb.xstim > -135) & (rec.pb.xstim < 135)
                    if not minspeed is None:
                        idx = idx & (speed > minspeed)
                    idx = idx & (top2 > mintop2)
                    if idx.sum()*rec.pb.sampling_period < mintrialtime_s: continue
                    ixstim = np.deg2rad(xstim[idx]+180)
                    iphase = np.deg2rad(rec.pb.c1.phasef[idx]+180)
                    ri = cs.corrcc(ixstim, iphase)
                    rec_rs[istimid-1, bar_cnt] = ri
                    bar_cnt += 1
                elif stim_dict[istimid] == 'Dark':
                    if not minspeed is None:
                        idx = idx & (speed > minspeed)
                    idx = idx & (top2 > mintop2)
                    if idx.sum()*rec.pb.sampling_period < mintrialtime_s: continue
                    idxstim = dhead[idx]
                    idphase = dphase[idx]
                    # a, b, ri, p, e = stats.linregress(idxstim, idphase)
                    ri = np.corrcoef(idxstim, idphase)[0, 1]
                    rec_rs[istimid-1, dark_cnt] = ri
                    dark_cnt += 1

            fly_rs.append(rec_rs)
        fly_rs = np.hstack(fly_rs)
        rs[:, ifly] = np.nanmean(fly_rs, axis=1)

    # Plot periods per stimulus
    if size is None:
        size = (3, 3)
    plt.figure(1, size)
    ax = plt.subplot(111)
    stimlabels = ['Bar', 'Dark']
    plt.xticks(range(len(group)), stimlabels)
    plt.xlim(-.5, 2-.5)
    plt.ylim(0, 1)
    plt.yticks([0, 1])
    ph.adjust_spines(ax, ['left', 'bottom'], lw=.25)
    for istim, stimlabel in enumerate(stimlabels):
        x = np.random.normal(istim, .02, len(group))
        med = np.nanmedian(rs[istim])
        std = np.nanstd(rs[istim])
        plt.scatter(x, rs[istim], facecolor='none', edgecolor=color, s=3, lw=.35)
        plt.errorbar(istim+.3, med, std, fmt='_',
                     c=color, elinewidth=.5, capthick=.5,
                     mew=.5, ms=2, capsize=1)

    if figname:
        ph.save(figname, rec)

def plot_frames(rec, ts, cmap=plt.cm.Blues, save=True, hspace=.1, mask=True):
    """
    Plots frames at different times, with a gaussian filter applied.
    :param rec: GCaMP_ABF object.
    :param ts: Time points to plot frames.
    :param cmap: Colormap for the frames.
    :param save: True will save the figure.
    :param hspace: How far the frames are spaced. Ranges (0, 1).
    :return:
    """
    plt.figure(1, (6, 6))
    gs = gridspec.GridSpec(len(ts), 1)
    gs.update(hspace=hspace)
    t = rec.eb.t
    if mask:
        tif = rec.ims[0].c1.tif*(rec.ims[0].c1.mask<254)
    else:
        tif = rec.ims[0].c1.tif
    tifmax = tif.max(axis=1)
    for i, ti in enumerate(ts):
        ax1 = plt.subplot(gs[i, 0])
        ph.adjust_spines(ax1, [])
        tidx = np.where(t>=ti)[0][0]
        ax1.set_ylabel('%.1f s' %(t[tidx] - ts[0]), rotation='horizontal', labelpad=30)
        frame = gaussian_filter(tifmax[tidx], 1.5)
        ax1.imshow(frame, cmap=cmap)#, vmin=10000)
    if save:
        ph.save(rec.basename.split('/')[-1] + '_frames', exts=['png', 'pdf'])


def plot_segmentations(rec, cmap=plt.cm.Blues, save=True, hspace=.1):
    """
    Plots the outline of the mask for each z-plane.
    :param rec: GCaMP_ABF object.
    :param cmap: Colormap for the frames.
    :param save: True to save figure.
    :param hspace: Spacing between frames.
    :return:
    """
    plt.figure(1, (6, 6))
    zs = [0, 9, 18]
    gs = gridspec.GridSpec(len(zs), 1)
    gs.update(hspace=hspace)
    tifmean = rec.pb.c1.tif.mean(axis=0)
    maskoutline = (rec.pb.c1.mask==0)

    for iz, z in enumerate(zs):
        ax1 = plt.subplot(gs[iz, 0])
        ph.adjust_spines(ax1, [])
        itifmean = copy.copy(tifmean[iz])
        itifmean[maskoutline[iz]] = 2**15
#         ax1.set_ylabel(r'%i $\u$m' %z, rotation='horizontal', labelpad=30)
        ax1.imshow(itifmean, cmap=cmap)
    if save:
        ph.save(rec.basename.split('/')[-1] + '_segments', exts=['png', 'pdf'])


def plot_stimulus_onoff(self, save=True):
    stimon_tifidx = self._get_stimon_tifidx()
    tifshape = self.tif.tif.shape
    newshape = (tifshape[0], tifshape[1], tifshape[2] * tifshape[3])
    tifctl = (self.tif.tif * self.tif.ctlmask).reshape(newshape).mean(axis=-1)
    tifbridge = (self.tif.tifm).reshape(newshape).mean(axis=-1)

    ctl_baron = tifctl[stimon_tifidx]
    ctl_baroff = tifctl[stimon_tifidx == False]
    print 'Control, Bar Hidden mean, std:', ctl_baroff.mean(), ctl_baroff.std()
    print 'Control, Bar Visible mean, std:', ctl_baron.mean(), ctl_baron.std()

    bridge_baron = tifbridge[stimon_tifidx]
    bridge_baroff = tifbridge[stimon_tifidx == False]
    print 'Bridge, Bar Hidden mean, std:', bridge_baroff.mean(), bridge_baroff.std()
    print 'Bridge, Bar Visible mean, std:', bridge_baron.mean(), bridge_baron.std()

    plt.figure(1, (8, 6))
    plt.suptitle(self.folder)
    ax = plt.subplot(111)
    plt.scatter(np.random.normal(0, .1, len(ctl_baroff)), ctl_baroff, c=ph.dgrey, lw=0, s=5)
    plt.scatter(np.random.normal(1, .1, len(ctl_baron)), ctl_baron, c=ph.blue, lw=0, s=5)
    plt.scatter(np.random.normal(2, .1, len(bridge_baroff)), bridge_baroff, c=ph.dgrey, lw=0, s=5)
    plt.scatter(np.random.normal(3, .1, len(bridge_baron)), bridge_baron, c=ph.blue, lw=0, s=5)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Control\nBar hidden', 'Control\nBar visible', 'Bridge\nBar hidden', 'Bridge\nBar visible'])
    plt.grid()

    if save:
        plt.savefig('%s_stimulus_onoff_intensities.png' % (self.abf.basename), bbox_inches='tight')


def plot_distribution(group, label, xbins=None, figname=''):
    dt, dforw = get_group_data(group, label)
    mask = np.isnan(dforw)
    if xbins is None: xbins = 100
    h, x = np.histogram(dforw[~mask], bins=xbins, density=True)
    celltype = group[0][0].pb.c1.celltype
    color = colours[celltype]

    plt.figure(1, (6, 4))
    ph.set_tickdir('out')
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    plt.bar(x[:-1], h, width=np.diff(x)[0], color=color, edgecolor='black')

    if figname:
        ph.save(figname)


def plot_intensity_vs_age(groups, stim_dict, ages):
    stim_map = {v: k for k, v in stim_dict.items()}

    plt.figure(1, (4, 10))
    gs = gridspec.GridSpec(len(groups), 1)
    gs.update(hspace=.5)

    for igroup, (age, group) in enumerate(zip(ages, groups)):
        intensities = np.zeros(len(group))
        intensities[:] = np.nan
        for ifly, fly in enumerate(group):
            fly_intensities = np.zeros((2, len(fly)/2))
            cnt = np.zeros(2)
            for rec in fly:
                itemp = rec.abf.temp.mean() > 26
                dark = rec.subsample('stimid') == stim_map['Dark']
                fly_intensities[itemp, cnt[itemp]] = np.nanmean(rec.pb.c1.an[dark])
                cnt[itemp] += 1
            fly_intensity = fly_intensities[1].mean() / fly_intensities[0].mean()
            intensities[ifly] = fly_intensity

        ax = plt.subplot(gs[igroup, 0])
        ax.axhline(1, c='grey', ls='--')
        ax.axhline(0, c='grey', ls='--')
        ax.set_ylim(0, 1.5)
        ax.set_yticks(np.arange(0, 1.6, .5))
        if igroup == len(groups)-1:
            ph.adjust_spines(ax, ['left', 'bottom'])
            ax.set_xlabel('Age (days)')
        else:
            ph.adjust_spines(ax, ['left'])
        ax.scatter(age, intensities)


def plot_regtransform_distribution(rec, cutoff=5):
    filename = '%s_transform.csv' %rec.pb.basename
    with open(filename) as fh:
        v = csv.reader(fh)
        ar = []
        for line in v:
            try:
                ar.append(map(float, line))
            except ValueError:
                pass
    def f(ar):
        return np.hypot(*ar)
    ar = np.array(ar).T
    dist = np.apply_along_axis(f, 0, ar[-2:])
    bins = np.arange(0, 10, .25)

    ax = plt.subplot(111)
    a = plt.hist(dist, bins=bins)


    print 'Mean = %.2f' %dist.mean()
    print 'No > %i = %i'%(cutoff, (np.abs(dist - dist.mean()) > cutoff).sum())
    ax.axvline(dist.mean()+cutoff)


def plot_steadystate_phase_histogram(group, maxvel=.2, wait_s=1, minlen_s=2, save=True):
    dt, phase = get_group_data(group, 'pb.c1.phase')
    phase = fc.wrap(phase)
    # dt, dphase = get_group_data(group, 'pb.c1.dphasef')
    dt, dhead = get_group_data(group, 'pb.dhead')
    dt, stimid = get_group_data(group, 'abf.stimid', subsample=True)
    idx = (dhead < maxvel) & (dhead > -maxvel) #& (stimid==1)
    wait_idx = int(np.ceil(wait_s / .15))
    minlen_idx = int(np.ceil(minlen_s / .15))
    inds = fc.get_contiguous_inds(idx, trim_left=wait_idx, keep_ends=True)
    phases = np.array([phase[ind].mean() for ind in inds])

    celltype = group[0][0].pb.c1.celltype
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    binwidth = 5
    bins = np.arange(-180, 180, binwidth)
    # h, x = np.histogram(phases, bins=bins)
    # ax.bar(x[:-1], h, facecolor=ph.blue, width=binwidth)
    plt.hist(phases, bins=bins, facecolor=colours[celltype])
    ax.set_xlim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.set_ylabel('Counts')
    ax.set_xlabel('Phase')
    ylim = ax.get_ylim()
    ax.set_yticks(np.arange(ylim[0], ylim[1]+1))

    if save:

        ph.save('%s_Steady_state_phase_histogram' %celltype)


# Plotting Right - Left (or Right, Left alone) Bridge vs phase or heading velocity

def plot_rml_v_dhead(groups, alabel='az', vel='abf.dhead', turn_bins = np.arange(-375, 405, 30),
                     binsize=30, binlim= 360, xticks = [],
                     lag_ms=0, ylim=None, show_all=True, stim=None, grid = True,
                     figname='', lw=2, zero_at_origin=False, size=None, pi_rad = False, signal = 'rml'):
    if not type(groups) is list: groups = [groups]
    if not type(groups[0]) is list: groups = [groups]
    #turn_bins = np.arange(-binlim-binsize/2., binlim+1.5*binsize, binsize)
    binsize = turn_bins[1] - turn_bins[0]

    # Setup Figure
    if size is None: size = (6, 4)
    plt.figure(1, size)
    ph.set_tickdir('out')
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'], lw=.25)
    if not(xticks == []):
        ax.set_xticks(xticks)
        ax.set_xlim([xticks[0], xticks[-1]])
    else:
        if pi_rad:
            ax.set_xticks(np.arange(-2, 2.5, 1))
        else:
            ax.set_xticks(np.arange(-360, 540, 180))
    # ax.grid()
    if pi_rad:
        ax.set_xlabel('Turning velocity (pi rad/s)', labelpad=2)
    else:
        ax.set_xlabel('Turning velocity (deg/s)', labelpad=2)
        
    if signal == 'rml':
        ax.set_ylabel('Right - Left\nBridge (stdev)', labelpad=2)
    elif signal == 'rpl':
        ax.set_ylabel('Right + Left\nBridge (stdev)', labelpad=2)
        
    if grid:
        ax.axhline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
        ax.axvline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])

    # Plot groups of flies
    colours = {'EIP': ph.blue, 'PEN': ph.orange, 'SPSP': 'gray', 'ISP':'firebrick'}
    tbc = turn_bins[:-1] + np.diff(turn_bins)[0] / 2.  # Turn bin centers
    glomeruli = groups[0][0][0].pb.c1.xedges[:-1]
    for group in groups:
        rmls = np.zeros((len(group), len(tbc)))
        for ifly, fly in enumerate(group):
            ab = np.nanmean([rec.bin_gcamp('pb.c1.' + alabel, vel, turn_bins, lag_ms, stim) for rec in fly], axis=0)
            if signal == 'rml':
                rml = np.nanmean(ab[:, glomeruli>=9], axis=1) - np.nanmean(ab[:, glomeruli<9], axis=1)
            elif signal == 'rpl':
                rml = np.nanmean(ab[:, glomeruli>=9], axis=1) + np.nanmean(ab[:, glomeruli<9], axis=1)
            if zero_at_origin: rml = rml - rml[tbc==0]
            rmls[ifly] = rml
            colour = colours[rec.pb.c1.celltype]
            if pi_rad:
                tbc = tbc / 180.
            if show_all:

                ax.plot(tbc, rml, c=colour, lw=lw/3., alpha=.5, zorder=2)

        med = np.nanmean(rmls, axis=0)
        n = (np.isnan(rmls)==False).sum(axis=0)
        sem = np.nanstd(rmls, axis=0) / np.sqrt(n)
        ax.plot(tbc, med, c=colour, lw=lw, zorder=2)
        if not show_all:
            plt.fill_between(tbc/180., med-sem, med+sem, facecolor=colour,
                             edgecolor='none', alpha=.4, zorder=2)
        # for rml in rmls:
                #     ax.plot(tbc, rml, c=colours[i], lw=1, alpha=.5)
    if ylim:
        ax.set_ylim(ylim)
        ax.set_yticks(np.arange(ylim[0], ylim[1]+ylim[1]/2, ylim[1]))
    if figname:
        plt.savefig(figname)


def plot_rml_dhead_slope(groups, alabel='az', vel='abf.dhead',
                         specific_glomeruli = [], oplabel = '-', abs_sig = False,
                         lags_ms=np.arange(-2000, 2001, 100), input_colours = [],
                         ylim=None, stim = [], figname='', turn_bins = np.arange(-200, 200.1, 25)):
    if not type(groups) is list: groups = [groups]
    if not type(groups[0]) is list: groups = [groups]
    

    # Setup Figure
    plt.figure(1, (8, 5))
    ph.set_tickdir('out')
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    # ax.grid()
    ax.axhline(c='grey', ls='--')
    ax.axvline(c='grey', ls ='--')
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Slope (std / (pi rad/s))', labelpad=20)
    ax.tick_params(axis='x', labelsize=12)

    # Extract slope for each time lag for each group of flies
    
    colours = {'EIP': ph.blue, 'PEN': ph.orange, 'SPSP': 'gray', 'ISP':'firebrick'}
    tbc = turn_bins[:-1] + np.diff(turn_bins)[0] / 2.  # Turn bin centers
    
    glomeruli = np.array(groups[0][0][0].pb.c1.xedges[:-1].tolist())

    if specific_glomeruli:
        cond_L = lambda g: ((g < 9) & ([ind in specific_glomeruli for ind in g]))
        cond_R = lambda g: ((g >= 9) & ([ind in specific_glomeruli for ind in g]))
    else:
        cond_L = lambda g: (g < 9)
        cond_R = lambda g: (g >= 9)
    print glomeruli[cond_R(glomeruli)], glomeruli[cond_L(glomeruli)]
        
    for igroup, group in enumerate(groups):
        slopes = np.zeros((len(group), len(lags_ms)))
        for ifly, fly in enumerate(group):
            for ilag, lag_ms in enumerate(lags_ms):
                
                ab = np.nanmean([rec.bin_gcamp('pb.c1.' + alabel, vel, turn_bins, lag_ms, stim, abs_sig) for rec in fly], axis=0)
                #whole thing rl
                if oplabel == '-':
                    rml = np.nanmean(ab[:, cond_R(glomeruli)], axis=1) - np.nanmean(ab[:, cond_L(glomeruli)], axis=1)
                elif oplabel == '+':
                    rml = np.nanmean(ab[:, cond_R(glomeruli)], axis=1) + np.nanmean(ab[:, cond_L(glomeruli)], axis=1)
                
                #outer glomeruli
                #rml = np.nanmean(ab[:, 12:], axis=1) - np.nanmean(ab[:, :6], axis=1)

                a, b, r, p, e = stats.linregress(tbc, rml)
                slopes[ifly, ilag] = a

        slopes *= 180.
        med = np.nanmean(slopes, axis=0)
        n = (np.isnan(slopes)==False).sum(axis=0)
        sem = np.nanstd(slopes, axis=0) / np.sqrt(n)
        if len(input_colours) == len(groups):
            colour = input_colours[igroup]
        else:
            colour = colours[rec.pb.c1.celltype]
        
        plt.plot(lags_ms, med, c=colour, lw=3)
        plt.fill_between(lags_ms, med-sem, med+sem, facecolor=colour, edgecolor='none', alpha=.2)

    # Print peak lag:
    print lags_ms[np.argmin(med)], np.min(med), lags_ms[np.argmax(med)], np.max(med)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlim(lags_ms[0], lags_ms[-1])

    if figname:
        binsize = int(np.diff(lags_ms)[0])
        plt.savefig(figname + ('_%s_%ims_bins' % (alabel, binsize)))

def plot_rml_v_behavior(self, tlim = [], imaging = ['pb.c1.leftz', 'pb.c1.rightz'],
                        im_overlay = ['c1.rmlz'], behav_overlay = ['abf.head'],
                        unwrap = True,
                        figname=''):
    
    colordict = {'EIP':ph.cm.blues,
                 'PEN':ph.cm.oranges,
                 'SPSP': ph.cm.greens,
                 'ISP': 'firebrick'}
    
    for icol, label in enumerate(imaging):
        # Get imtype (pb or eb), channel and gc
        imlabel, clabel, alabel = self.get_labels(label)
        im, c, gloms_over_time = self.get_objects(label)
        (t, xedges), gc = self.get_signal(label)
        
    ncols = 0
    overlay = 0
    if im_overlay or behav_overlay:
        overlay = 1
    ncols += 1*len(imaging)
    ncols += overlay
    wr = [1] * len(imaging) + [4 * overlay]
    
    
    gs = gridspec.GridSpec(1, ncols, width_ratios=wr)
    ax_fluo = gs[0, 0]
    ax_overlay = gs[0, 1]
    
    


def plot_isolated_turns(group, tlim=(-2, 2), save=True, overlay=True,
                        trigger='pb.c1.dphasef', lines=[],  **kwargs):
    """
    Plots phase and R-L triggered on a change in phase (using detect peaks on dphase).
    :param group: List of flies.
    :param tlim: Time window.
    :param save: True to save figure.
    :param overlay: Overlay mean and sem of phase and R-L on same plot.
    Otherwise, plot them separately with individual traces.
    :param kwargs: kwargs for get_isolated_peak_inds.s
    :return:
    """
    ph.set_tickdir('out')
    # Setup figure
    if overlay:
        plt.figure(1, (7, 4))
        gs = gridspec.GridSpec(1, 2)
    else:
        plt.figure(1, (7, 7))
        gs = gridspec.GridSpec(2, 2)
    celltype = group[0][0].ims[0].c1.celltype
    colour = colours[celltype]

    for col, direction in enumerate(['right', 'left']):
        # Find peaks
        # Collect data from all groups
        filler_len = max(100, int(mpd_s*10))
        dt, trigger = get_group_data(group, trigger, filler_len)

        # find peaks
        dt, peak_inds = get_isolated_peak_inds(trigger, dt,  direction=direction, **kwargs)
        window_ind0 = np.round(np.arange((tlim[0]-1)/dt[0], (tlim[1]+1)/dt[0]+1)).astype(int)
        window_inds = np.tile(window_ind0, (len(peak_inds), 1)) + peak_inds[:, None]
        t0 = np.where((window_ind0==0))[0][0]-1

        # Plot Phase
        phase_dt, phase = get_group_data(group, 'pb.c1.dphase')
        phase_turns = phase[window_inds]
        ax = plt.subplot(gs[0, col])
        if overlay:
            axes = ['bottom', 'left']
            if col == 0:
                ph.adjust_spines(ax, ['bottom', 'left'])
                ax.set_ylabel('Phase (pi rad)', color=colour)
                ax.set_xlabel('Time (s)')
            else:
                ph.adjust_spines(ax, ['bottom'])
            ax.set_ylim(-.5, .5)
            ax.set_yticks([-.5, 0, .5])
        else:
            if col == 0:
                ph.adjust_spines(ax, ['left'])
                ax.set_ylabel('Phase (pi rad)')
            else:
                ph.adjust_spines(ax, [])
            ax.set_ylim(-1, 1)
            ax.set_yticks([-1, 0, 1])

        ax.set_xlim(tlim)
        ax.axhline(ls='--', c='black', alpha=.5)
        ax.axvline(ls='--', c='black', alpha=.5)


        t_interp = np.arange(tlim[0], tlim[1], .1)
        phase_interp = np.zeros((len(phase_turns), len(t_interp)))
        for i, (dti, phasei) in enumerate(zip(phase_dt, phase_turns)):
            t = window_ind0.astype(float)*dti
            phasei = fc.unwrap(phasei) / 180.
            phasei = phasei - phasei[t0]

            if not overlay:
                plt.plot(t, phasei, c='grey', alpha=.07)

            f = interp1d(t, phasei)
            phase_interp[i] = f(t_interp)

        # Plot mean
        mean = np.nanmean(phase_interp, axis=0)
        if overlay:
            std = np.nanstd(phase_interp, axis=0)
            n = (~np.isnan(phase_interp)).sum(axis=0)
            sem = std / np.sqrt(n)
            ax.fill_between(t_interp, mean-sem, mean+sem, edgecolor='none', facecolor=colour, alpha=.3)
            ax.plot(t_interp, mean, lw=1, c=colour)
        else:
            ax.plot(t_interp, mean, lw=2, c=colour)


        # Plot R-L
        rml_dt, rml = get_group_data(group, 'pb.c1.rmlz')
        rml_turns = rml[window_inds]

        if overlay:
            ax = ax.twinx()
            if col==1:
                ph.adjust_spines(ax, ['bottom', 'right'])
                ax.set_ylabel('Right - Left (std)')
            else:
                ph.adjust_spines(ax, ['bottom'])
            ax.set_ylim(-.5, .5)
            ax.set_yticks(np.arange(-.5, .6, .5))
        else:
            ax = plt.subplot(gs[1, col])
            if col ==0:
                ph.adjust_spines(ax, ['left', 'bottom'])
                ax.set_ylabel('Right - Left (std)')
                ax.set_xlabel('Time (s)')
            elif col == 1:
                ph.adjust_spines(ax, ['bottom'])
            ax.set_ylim(-1, 1)
            ax.set_yticks(np.arange(-1, 1.1, .5))
        ax.set_xticks(np.arange(tlim[0], tlim[1]+.5, 1))
        ax.set_xlim(tlim)
        ax.axhline(ls='--', c='black', alpha=.5)
        ax.axvline(ls='--', c='black', alpha=.5)
        if lines:
            for line in lines:
                ax.axvline(line, ls='--', c='black', alpha=.5)

        rml_interp = np.zeros((len(rml_turns), len(t_interp)))
        for i, (dti, rmli) in enumerate(zip(rml_dt, rml_turns)):
            t = window_ind0.astype(float)*dti
            # rmli = rmli - rmli[t0]
            if not overlay:
                plt.plot(t, rmli, c='grey', alpha=.07)

            f = interp1d(t, rmli)
            rml_interp[i] = f(t_interp)

        # Plot mean
        mean = np.nanmean(rml_interp, axis=0)
        if overlay:
            std = np.nanstd(rml_interp, axis=0)
            n = (~np.isnan(rml_interp)).sum(axis=0)
            sem = std / np.sqrt(n)
            ax.fill_between(t_interp, mean-sem, mean+sem, edgecolor='none', facecolor='grey', alpha=.3)
            ax.plot(t_interp, mean, lw=1, c=colour)
        else:
            ax.plot(t_interp, mean, lw=2, c=colour)


        # Plot

    if save:
        figname = '%s_R-L_triggered_mean' %celltype
        if overlay:
            figname += '_overlay'
        ph.save(figname)


def get_isolated_peak_inds(trigger, dt, direction='right', edge='left',
                           mph=200, mph_neighbor=100, mpd_s=.5, minpw_s=None,
                           maxpw_s=None, minpw_thresh=10, gaussfilt_s=None, arg2use = 0):
    """
    Find peak inds that are separated from other peaks, and of a minimum width.
    :param group: List of flies.
    :param trigger: typically 'c1.dphase' or 'pb.dhead'
    str Label pointing to the signal.
    :param direction: 'left' or 'right' for direction of phase or fly turn.
    :param edge:    'peak' return output of detect_peaks
                    'left' return start of peak
                    'right' return end of peak
    :param mph: Minimum peak height (see detect peaks).
    :param mph_neighbor: Minimum neighboring peak height.
    :param mpd_s: Minimum neighboring peak distance in seconds.
    :param minpw_s: Minimum peak width in seconds.
    :param mpw_thresh: Threshold with which to calculate the peak width.
    ie. Peak edges are defined as the first value to cross mpw_thresh from the peak
    on either side. Peak width is defined as right edge - left edge.
    :return:
    """
    
    def pick_var(v, a=arg2use):
        if type(v) is list:
            v = v[a]
        return v
    
    mph = pick_var(mph)
    mph_neighbor = pick_var(mph_neighbor)
    mpd_s = pick_var(mpd_s)
    minpw_s = pick_var(minpw_s)
    maxpw_s = pick_var(maxpw_s)
    minpw_thresh = pick_var(minpw_thresh)
    
    if not gaussfilt_s is None:
        sigma = gaussfilt_s/dt
        trigger = ndimage.filters.gaussian_filter1d(trigger, sigma)

    if type(dt) in [np.float64, float]:
        a = np.zeros(len(trigger))
        a.fill(dt)
        dt = a

    # Detect peaks
    valley = False if direction == 'right' else True
    peak_inds = fc.detect_peaks(trigger, mph=mph, valley=valley, edge='rising')
    # print 'Raw peaks > mph:\t%s' %len(peak_inds)

    # Select peaks that are far away from neighboring peaks with height greater than mph_neighbor
    if not mph_neighbor is None:
        neighbor_peak_inds_right = fc.detect_peaks(trigger, mph=mph_neighbor, show=False)
        neighbor_peak_inds_left = fc.detect_peaks(trigger, mph=mph_neighbor, valley=True, show=False)
        neighbor_peak_inds = np.unique(np.concatenate([neighbor_peak_inds_right, neighbor_peak_inds_left]))

        neighbor_peak_tile = np.tile(neighbor_peak_inds, (len(peak_inds), 1))
        peak_dist = (neighbor_peak_tile - peak_inds[:, None]) * dt[peak_inds[:, None]]
        peak_dist[peak_dist==0] = np.nan
        peak_dist = np.nanmin(np.abs(peak_dist), axis=1)
        peak_inds = peak_inds[peak_dist > mpd_s]
    # print 'Separated peaks:\t%s' %len(peak_inds)

    # Select peaks that have width greater than minpw_s.
    if (not minpw_s is None) or (not maxpw_s is None) or (not edge == 'peak'):
        if direction == 'right':
            peak_edges = np.where(np.diff(trigger > minpw_thresh))[0]
        else:
            peak_edges = np.where(np.diff(trigger < -minpw_thresh))[0]

        peak_edges_tile = np.tile(peak_edges, (len(peak_inds), 1))
        peak_edges_dist = (peak_edges_tile - peak_inds[:, None]).astype(float)

        # Get left edge distance from peak (negative)
        peak_edges_left = copy.copy(peak_edges_dist)
        peak_edges_left[peak_edges_dist>=0] = np.nan
        peak_edge_left = np.nanmax(peak_edges_left, axis=1)

        left_edge_inds = peak_inds + peak_edge_left

        # Get right edge distance from peak (positive)
        peak_edges_right = copy.copy(peak_edges_dist)
        peak_edges_right[peak_edges_dist<=0] = np.nan
        peak_edge_right = np.nanmin(peak_edges_right, axis=1)

        right_edge_inds = peak_inds + peak_edge_right

        # get width from right-left
        peak_width = peak_edge_right - peak_edge_left * dt[peak_inds]

        # filter peaks based on peak_width
        filter = np.ones(len(peak_inds)).astype(bool)
        if minpw_s:
            filter = filter & (peak_width > minpw_s)
        if maxpw_s:
            filter = filter & (peak_width < maxpw_s)


    # print 'Wide peaks:\t\t%s' %len(peak_inds)

    if edge == 'left':
        notnan = ~np.isnan(left_edge_inds)
        idx = left_edge_inds[notnan & filter].astype(int)
    elif edge == 'right':
        notnan = ~np.isnan(right_edge_inds)
        idx = right_edge_inds[notnan & filter].astype(int)
    elif edge == 'peak':
        notnan = ~np.isnan(peak_inds)
        idx = peak_inds[notnan & filter].astype(int)

    
    return dt[idx], idx


def get_isolated_peak_trials(group, trigger, channels=('c1.dphase',), arg2use = 0,
                             window=(-4, 4), stim=None, specific_glomeruli = [], **kwargs):
    # buffer the window by 1s:
    window = list(window)
    window[0], window[1] = window[0]-1, window[1]+1
    window = tuple(window)

    mintifdt = .1
    tifinds = np.arange(window[0]/mintifdt, window[1]/mintifdt+1, 1).astype(int)

    minabfdt = .02
    abfinds = np.arange(window[0]/minabfdt, window[1]/minabfdt+1, 1).astype(int)


    trigger_label = copy.copy(trigger)
    data = [[] for _ in channels]

    for ifly, fly in enumerate(group):
        for irec, rec in enumerate(fly):
            ttrig = rec.get_signal(trigger_label)[0]
            trigger = rec.get_objects(trigger_label)[-1]
            dt = np.diff(ttrig).mean()
            
            if specific_glomeruli and trigger_label.split('.')[-1][:3] in ['rml', 'rpl']:
                    glomeruli = np.array(rec.pb.c1.xedges[:-1].tolist())
                    cond_R = lambda g: ((g >= 9) & \
                                        ([ind in specific_glomeruli for ind in g]))
                    cond_L = lambda g: ((g < 9) & \
                                        ([ind in specific_glomeruli for ind in g]))
                    if trigger_label.split('.')[-1][:3] == 'rml':
                        op = lambda x, y: x - y
                    elif trigger_label.split('.')[-1][:3] == 'rpl':
                        op = lambda x, y: np.nansum(np.vstack((x, y)), axis=0)                    
                    if trigger_label.split('.')[-1][-1] == 'n':
                        fluo_sig = rec.get_signal('pb.c1.an')[-1]
                    elif trigger_label.split('.')[-1][-1] == 'z':
                        fluo_sig = rec.get_signal('pb.c1.az')[-1]
                    trigger = op(np.nanmean(fluo_sig[:, cond_R(glomeruli)], axis=1), 
                                np.nanmean(fluo_sig[:, cond_L(glomeruli)], axis=1))
            
            dt, idx = get_isolated_peak_inds(trigger, dt, arg2use = arg2use, **kwargs)
            if not stim is None:
                if 'abf' in trigger_label:
                    _, stimid = rec.get_signal('abf.ao')
                else:
                    stimid = rec.subsample('abf.ao')
                stimid = np.array([round(i, 1) for i in stimid])
                idxbin = np.zeros_like(stimid).astype(bool)                                    
                idxbin[idx] = True
                if type(stim) is list or type(stim) is ndarray:
                    idxbin = idxbin & np.in1d(stimid, np.array(stim))
                else:
                    idxbin = idxbin & (stimid == stim)
                idx = np.where(idxbin)[0]
            tpeaks = ttrig[idx]

            # Remove peaks that are outside of the 2P scanning time range
            tmax = rec.ims[0].t.max()
            tpeaks = tpeaks[tpeaks >= 0]
            tpeaks = tpeaks[tpeaks < tmax]
            # print ifly, irec, len(tpeaks)
            for ichannel, channel in enumerate(channels):

                ts, signal = rec.get_signal(channel)
                if np.shape(ts) == (2,):
                    ts = ts[0]
                
                if specific_glomeruli and channel.split('.')[-1][:3] in ['rml', 'rpl']:
                    glomeruli = np.array(rec.pb.c1.xedges[:-1].tolist())
                    cond_R = lambda g: ((g >= 9) & \
                                        ([ind in specific_glomeruli for ind in g]))
                    cond_L = lambda g: ((g < 9) & \
                                        ([ind in specific_glomeruli for ind in g]))
                    if channel.split('.')[-1][:3] == 'rml':
                        op = lambda x, y: x - y
                    elif channel.split('.')[-1][:3] == 'rpl':
                        op = lambda x, y: np.nansum(np.vstack((x, y)), axis=0)                    
                    if channel.split('.')[-1][-1] == 'n':
                        fluo_sig = rec.get_signal('pb.c1.an')[-1]
                    elif channel.split('.')[-1][-1] == 'z':
                        fluo_sig = rec.get_signal('pb.c1.az')[-1]
                    signal = op(np.nanmean(fluo_sig[:, cond_R(glomeruli)], axis=1), 
                                np.nanmean(fluo_sig[:, cond_L(glomeruli)], axis=1))
                
                for tpeak in tpeaks:
                    if 'abf' in channel:
                        idxi = abfinds + np.where(ts>=tpeak)[0][0]
                    else:
                        idxi = tifinds + np.where(ts>=tpeak)[0][0]

                    if idxi[0] >= 0 and idxi[-1] < len(ts):
                        data[ichannel].append([ts[idxi]-tpeak, signal[idxi]])
    
    if type(data[0][0][1]) is np.ndarray: #i.e. if this is a glomerular signal
        pass
    else:
        for i in range(len(data)):
            data[i] = np.array(data[i])

    peaks = {channel: idata for channel, idata in zip(channels, data)}

    return peaks


def plot_isolated_turns2(groups, tlim=(-2, 2), fig_filename = '', save = False, overlay=True,
                        trigger='pb.c1.dphasef', lines=(), show_all=False,
                         channels=('pb.c1.dphase', 'pb.c1.rmlz'), directions = ['left', 'right'], ylims=(),
                         yticksteps=(), size=None, lw=1, ls = '-', suffix='', stim=None, scale_y = [[]],
                         colors=None, scatter = [False, False], baseline_t = [],
                         specific_glomeruli = [[]], **kwargs):
    """
    Plots phase and R-L triggered on a change in phase (using detect peaks on dphase).
    :param group: List of flies.
    :param tlim: Time window.
    :param save: True to save figure.
    :param overlay: Overlay mean and sem of phase and R-L on same plot.
    Otherwise, plot them separately with individual traces.
    :param kwargs: kwargs for get_isolated_peak_inds.s
    :return:
    """

    # Setup figure
    if size is None: size = (3, 3)
    
    #note: can only scale y with up to 2 channels
    # does not work yet
    if (np.where(scale_y == [])[0]) and ylims:
        print 'scaling to: ', scale_y
        if scale_y[0]:
            top_scale = scale_y[0][1] - scale_y[0][0]
        else:
            top_scale = ylims[0][1] - ylims[0][0]
        if scale_y[1]:
            bottom_scale = scale_y[1][1] - scale_y[1][0]
        else:
            bottom_scale = ylims[1][1] - ylims[1][0]
        size = (3, 3*(((ylims[0][1] - ylims[0][0]) / 2*top_scale) + ((ylims[1][1] - ylims[1][0]) / 2*bottom_scale)))
        hr = [((ylims[0][1] - ylims[0][0]) / 2*top_scale), ((ylims[1][1] - ylims[1][0]) / 2*bottom_scale)]
    else:
        hr = [.5, .5]
        
    plt.figure(1, size)
    
    gs = gridspec.GridSpec(len(channels), 2, height_ratios = hr)
    
    axs = []

    for igroup, group in enumerate(groups):
        celltype = group[0][0].ims[0].c1.celltype
        if colors:
            colour = colors[igroup]
        else:
            colour = colours[celltype]
        for col, direction in enumerate(directions):
            # Find peaks
            peak_dict = get_isolated_peak_trials(group, trigger, channels, window=tlim,
                    stim=stim, direction=direction, arg2use = igroup,
                    specific_glomeruli = specific_glomeruli[igroup], **kwargs)
            print '%i %s Turns' %(len(peak_dict.items()[0][1]), direction.capitalize())

            for ichannel, label in enumerate(channels):
                row = ichannel
                peaks = peak_dict[label]
                
                ax = plt.subplot(gs[row, col])
                axs.append(ax)
                show_axes = []
                if row == len(peak_dict)-1:
                    show_axes.append('bottom')
                if col == 0:
                    show_axes.append('left')
                    ax.set_ylabel(label, labelpad=2)
                ph.adjust_spines(ax, show_axes, lw=.25)
                ax.axhline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
                # ax.axvline(ls='--', c='black', alpha=.5)

                if lines:
                    for line in lines:
                        ax.axvline(line, ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])


                # Plot mean peak
                if 'abf' in label:
                    dt = .02
                    c = colour
                else:
                    dt = .1
                    c = colour


                t_interp = np.arange(tlim[0]-.5, tlim[1]+.5, dt)

                y_interp = np.zeros((len(peaks), len(t_interp)))
                if scatter[ichannel]:
                    y_begin = []
                    y_end = []
                    y_rand = []
                    for i, (t, y) in enumerate(peaks):
                        #y_avg = np.nanmean(y[np.where((t>=(tlim[0]/2.)) & (t<0.))])
                        #plt.scatter(t, y-y_avg, lw=0, c=c,  s = 2.5)
                        
                        y_begin.append(y[np.where(t< 0)[0][-1]])
                        y_end.append(y[np.where(t>= 0)[0][0]])
                        for i in range(10):                       
                            y_rand.append(y[randint(0, len(y)-1)])
                    
                    t_span = tlim[1] - tlim[0]
                    t1q = tlim[0] + t_span*.4
                    t3q = tlim[1] - t_span*.4
                
                    plt.scatter([t1q]*len(y_begin), y_begin, c = 'seagreen', lw=0, s=2.5)
                    plt.plot([t1q-t_span/20., t1q+t_span/20.], [np.nanmean(y_begin), np.nanmean(y_begin)], lw=2, c='k')
                    
                    plt.scatter([t3q]*len(y_end), y_end, c = 'darkred', lw=0, s=2.5)
                    plt.plot([t3q-t_span/20., t3q+t_span/20.], [np.nanmean(y_end), np.nanmean(y_end)], lw=2, c='k')
                    
                    plt.scatter([tlim[1]-t_span/20.]*len(y_rand), y_rand, c = 'orange', lw=0, s=2.5)
                    plt.plot([tlim[1]-t_span/10., tlim[1]], [np.nanmean(y_rand), np.nanmean(y_rand)], lw=2, c='k')    
                        
                else:
                    for i, (t, y) in enumerate(peaks):
                        if label.split('.')[-1] == 'phase':
                            y = fc.unwrap(y) - y[np.argmin(np.abs(t))]
                        
                        f = interp1d(t, y, kind='linear', bounds_error = False)
                        y_interp[i] = f(t_interp)
                    mean = np.nanmean(y_interp, axis=0)
                    
                
                    if len(baseline_t) == 2:
                        base_ind0 = np.where(t_interp >= baseline_t[0])[0][0]
                        base_ind1 = np.where(t_interp >= baseline_t[1])[0][0]
                        mean -= np.nanmean(mean[base_ind0:base_ind1])
                    elif len(baseline_t) == 1:
                        base_ind0 = np.where(t_interp >= baseline_t[0])[0][0]
                        mean -= mean[base_ind0]
                        
                        
                    # if label.split('.')[-1] == 'phasef':
                    #     t_offset = 0
                    # elif ichannel==0:
                    #     zero_thresh = 2
                    #     if direction == 'left':
                    #         zeros = np.where(np.diff(mean<-zero_thresh))[0].astype(float)
                    #     elif direction == 'right':
                    #         zeros = np.where(np.diff(mean>zero_thresh))[0].astype(float)
                        # t0_idx = np.argmin(np.abs(t_interp)).astype(float)
                        # zero_dist = zeros-t0_idx
                        # zero_dist[zero_dist>=0] = np.nan
                        # t0_idx_new = t0_idx + np.nanmax(zero_dist)
                        #
                        # t_offset = t_interp[t0_idx_new]
    
                    # t_interp -= t_offset
                    ax.plot(t_interp, mean, lw=lw, c=c, zorder=2, ls = ls)
                    # ax.scatter(t_interp, mean, lw=lw, c='black', zorder=3, s=.2)
                    if show_all:
                        for t, y in peaks:
                            ax.plot(t, y, c='grey', alpha=.2, zorder=1)
                    else:
                        sem = fc.nansem(y_interp, axis=0)
                        # sem = np.nanstd(y_interp, axis=0)
                        ax.fill_between(t_interp,
                                        mean-sem,
                                        mean+sem,
                                        edgecolor='none',
                                        facecolor=c,
                                        alpha=.4)
                ax.set_xlim(tlim)
                if tlim[0] - tlim[1] >= 1:
                    ax.set_xticks(np.arange(tlim[0], tlim[-1]+1))
                else:
                    ax.set_xticks([tlim[0], 0, tlim[1]])
                if ylims:
                    ax.set_ylim(ylims[row])
                if yticksteps:
                    ax.set_yticks(np.arange(ylims[row][0], ylims[row][1]+yticksteps[row]/2., yticksteps[row]))
    
    if fig_filename:
        plt.savefig(fig_filename)

    #if save:
     #   channel_labels = '_'.join([channel.split('.')[-1] for channel in channels])
      #  figname = '%s_%s_triggered_mean_%s' %(celltype, trigger.split('.')[-1], channel_labels)
       # if overlay:
        #    figname += '_overlay'
        #figname += suffix
        #ph.save(figname)

# Plotting time difference


def plot_tif_xcorr(group, metric, tlim=(-2, 2), ylim=None, stim=None, figname=''):
    """
    Plots cross-correlation between two arrays with the same tif time base.
    :param group: List of flies.
    :param tlim: Cross-correlation time-window.
    :param ylim: Y-axis limits.
    :param save: True to save figure.s
    :return:
    """
    plt.figure(1, (5, 5))
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    # ax.set_xticks(np.arange(tlim[0], tlim[1]+.5, .5))
    ax.axvline(c='black', ls='--', alpha=.3)
    ax.set_ylabel('Correlation Coeff')
    ax.set_xlabel('Lag (s)')

    xi = np.arange(tlim[0], tlim[1], .1)
    xcs = np.zeros((len(group), len(xi)))
    xcs[:] = np.nan
    for ifly, fly in enumerate(group):
        flyxc = []
        for rec in fly:
            tif = rec.ims[0]
            if not stim is None:
                stimid = rec.subsample('stimid')
                idx = stimid == stim
            else:
                idx = np.ones(tif.len).astype(bool)
            signal1, signal2 = metric(rec)
            t, xc = fc.xcorr(signal1[idx], signal2[idx], tif.sampling_period)

            c = 'grey'
            idx = (t>tlim[0]-1) & (t<tlim[1]+1)
            ax.plot(t[idx], xc[idx], c=c, alpha=.3)

            f = interp1d(t[idx], xc[idx], 'cubic')
            flyxc.append(f(xi))

        xcs[ifly] = np.nanmean(flyxc, axis=0)
    ax.plot(xi, np.nanmean(xcs, axis=0), c=c, lw=3)
    ax.set_xlim(tlim)
    ax.set_xticks(np.arange(tlim[0], tlim[1]+1, 1))
    if ylim:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()

    if figname:
        ph.save(figname)


def plot_rml_dphase_xcorr(groups, tlim=(-2, 2), ylim=None,
                          rmllabel='c1.rmlz', dphaselabel='c1.dphase',
                          acceleration=False, genotype='', stim=None,
                          save=False, size=None, lw=.5, colors=None):
    """
    Plot cross correlation between R-L bridge and phase velocity.
    :param groups: List of flies.
    :param alabel: Label of bridge array.
    :param tlim: Time window.
    :param figname: Figure name.
    :return:
    """
    if size is None: size=(3, 3)
    plt.figure(1, size)
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'], lw=.25)
    # ax.set_xticks(np.arange(tlim[0], tlim[1]+.5, .5))
    ax.axvline(c=ph.grey5, ls='--', alpha=1, zorder=1, lw=lw/2., dashes=[2, 2])
    # ax.axhline(c=ph.grey5, ls='--', alpha=1, zorder=1, lw=lw/2., dashes=[2, 2])
    ax.set_ylabel('Correlation Coeff')
    ax.set_xlabel('Lag (s)')
    xi = np.arange(tlim[0], tlim[1], .1)
    for igroup, group in enumerate(groups):
        xcs = np.zeros((len(group), len(xi)))
        xcs[:] = np.nan
        for ifly, fly in enumerate(group):
            flyxcinterp = []
            flyxc = []
            for rec in fly:
                im, c, rml = rec.get_objects(rmllabel)
                _, dphase = rec.get_signal(dphaselabel)
                sp = im.sampling_period
                if acceleration:
                    dphase = np.gradient(dphase)

                if not stim is None: # fix this to take out trial-trial discontinuity
                    stimid = rec.subsample('abf.stimid')
                    stimidx = stimid==stim
                    rml = rml[stimidx]
                    dphase = dphase[stimidx]

                t, xc = fc.xcorr(rml, dphase, sp)
                idx = (t>tlim[0]-1) & (t<tlim[1]+1)
                flyxc.append(xc[idx])

                f = interp1d(t[idx], xc[idx], 'linear')
                flyxcinterp.append(f(xi))

            xcs[ifly] = np.nanmean(flyxcinterp, axis=0)

            flyxcmean = np.nanmean(flyxc, axis=0)
            if colors:
                color = colors[igroup]
            else:
                color = colours[c.celltype]
            ax.plot(t[idx], flyxcmean, c=color, alpha=.5, lw=lw/5., zorder=2)
        mean = np.nanmean(xcs, axis=0)
        print '%s Peak delay (s): %.3f' %(c.celltype, xi[np.argmax(np.abs(mean))])
        ax.plot(xi, mean, c=color, lw=lw, zorder=2)
        # ax.scatter(xi, mean, c='black', zorder=3, s=.5)
        # sem = np.std
        # ax.fill_between()
    ax.set_xlim(tlim)
    if ylim:
        ax.set_ylim(ylim)
        # ax.set_yticks(np.arange(-.4, .2, .1))
        # ax.set_yticks(np.arange(ylim[0], ylim[1]+.1, .1))
    if save:
        ph.save('%s_%s_xcorr' %(rmllabel, dphaselabel) + genotype )


def plot_rpl_speed_xcorr(groups, alabel='az', tlim=(-2, 2), figname=''):
    """
    Plot cross correlation between R+L bridge and speed.
    :param groups: List of flies.
    :param alabel: Label of bridge array.
    :param tlim: Time window.
    :param figname: Figure name.
    :return:
    """
    plt.figure(1, (5, 5))
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    # ax.set_xticks(np.arange(tlim[0], tlim[1]+.5, .5))
    ax.axvline(c='black', ls='--', alpha=.3)
    ax.set_ylabel('Correlation Coeff')
    ax.set_xlabel('Lag (s)')
    xi = np.arange(tlim[0], tlim[1], .1)
    for group in groups:
        xcs = np.zeros((len(group), len(xi)))
        xcs[:] = np.nan
        for ifly, fly in enumerate(group):
            flyxc = []
            for rec in fly:
                az = getattr(rec.pb.c1, alabel)

                speed = subsample(rec, 'speed', lag_ms=300, metric=np.nanmean)
                rpl = np.nanmean(az, axis=1)
                t, xc = fc.xcorr(rpl, speed, rec.pb.sampling_period)

                c = colours[rec.pb.c1.celltype]
                idx = (t>tlim[0]-1) & (t<tlim[1]+1)
                ax.plot(t[idx], xc[idx], c=c, alpha=.3)

                f = interp1d(t[idx], xc[idx], 'cubic')
                flyxc.append(f(xi))

            xcs[ifly] = np.nanmean(flyxc, axis=0)
        ax.plot(xi, np.nanmean(xcs, axis=0), c=c, lw=3)
    ax.set_xlim(tlim)

    if figname:
        ph.save(figname)


def plot_phase_head_xcorr(groups, tlim=(-2, 2), vel=True, ylim=None, stim=None,
                          mintriallen=30, lines=[], lw=1, figname=''):
    """
    Plot cross correlation between phase and heading position or velocity.
    :param groups: List of flies.
    :param tlim: Time window tuple.
    :param vel: True/False. True to use velocity, otherwise position.
    :param ylim: Scale y-axis.
    :param stim: If vel, only use points under stim.
    :param figname: Figure name.
    :return:
    """
    plt.figure(1, (3, 3))
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    ax.set_ylabel('Correlation Coeff')
    ax.set_xlabel('Lag (s)')
    colours = {'EIP': ph.blue, 'PEN': ph.orange, 'SPSP': 'seagreen'}
    xi = np.arange(tlim[0], tlim[1], .1)
    if ylim: ax.set_ylim(ylim)

    for group in groups:
        xcs = np.zeros((len(group), len(xi)))
        xcs[:] = np.nan
        for ifly, fly in enumerate(group):
            flyxc_interp = []
            flyxc = []
            for rec in fly:

                c = colours[rec.pb.c1.celltype]
                dhead = rec.subsample('dhead', lag_ms=0)
                head = rec.subsample('head', lag_ms=0, metric=fc.circmean)
                xstim = rec.subsample('xstim', lag_ms=0, metric=fc.circmean)
                stimid = rec.subsample('stimid', lag_ms=0)
                baridx = (xstim > -135) & (xstim < 135) & (stimid == 1)

                trial_ts = get_trial_ts(rec)
                t = rec.pb.t

                stimidx = stimid==stim
                for t0, t1 in trial_ts:
                    idx = (t >= t0) & (t < t1)
                    if stim:
                        idx = idx & stimidx
                    if stim==1:
                        idx = idx & baridx

                    if vel:
                        p = rec.pb.c1.dphase[idx]
                        h = dhead[idx]
                    else:
                        p = fc.unwrap(rec.pb.c1.phase[idx])
                        h = fc.unwrap(head[idx])


                    if idx.sum()*rec.pb.sampling_period > mintriallen:
                        txc, xc = fc.xcorr(p, h, rec.pb.sampling_period)
                        xcidx = (txc>tlim[0]-1) & (txc<tlim[1]+1)
                        # ax.plot(txc[xcidx], xc[xcidx], c=c, alpha=.1)
                        flyxc.append(xc[xcidx])

                        f = interp1d(txc[xcidx], xc[xcidx], 'cubic')
                        flyxc_interp.append(f(xi))

            ax.plot(txc[xcidx], np.nanmean(flyxc, axis=0), c=c, alpha=.5, lw=lw/3.)

            xcs[ifly] = np.nanmean(flyxc_interp, axis=0)
        mean = np.nanmean(xcs, axis=0)
        print '%s Peak Lag (s): %.3fs' %(rec.pb.c1.celltype, xi[np.argmin(mean)])
        ax.plot(xi, mean, c=c, lw=lw)
    ax.set_xlim(tlim)

    ax.axvline(c=ph.grey5, ls='--')
    if lines:
        for line in lines:
            ax.axvline(line, c=ph.grey5, ls='--')

    if figname:
        ph.save(figname)


def plot_c1c2_phasediff_dphase1_xcorr(group, save=False, **kwargs):
    """
    Plots cross-correlation between phase difference between c1 and c2 and the phase
    velocity, measured by c1 (usually EIPs).
    :param group: List of flies.
    :param tlim: Cross-correlation time-window.
    :param ylim: Y-axis limits.
    :param save: True to save figure.s
    :return:
    """

    def metric(rec):
        tif = rec.ims[0]
        phase_diff = fc.wrap(tif.c2.phase - tif.c1.phase)
        signal1 = fc.wrap(phase_diff - fc.circmean(phase_diff))
        signal2 = tif.c1.dphase
        return signal1, signal2

    if save:
        tif = rec.ims[0]
        figname = 'Phasediff_dphase_xcorr_%s_%s' %(tif.c1.celltype, tif.c2.celltype)
    else:
        figname = ''

    plot_tif_xcorr(group, metric, figname=figname, **kwargs)


def plot_dphase1_dphase2_xcorr(group, save=True, **kwargs):
    """
    Plots cross-correlation between phase difference between c1 and c2 and the phase
    velocity, measured by c1 (usually EIPs).
    :param group: List of flies.
    :param tlim: Cross-correlation time-window.
    :param ylim: Y-axis limits.
    :param save: True to save figure.s
    :return:
    """
    def metric(rec):
        tif = rec.ims[0]
        return tif.c1.dphase, tif.c2.dphase

    if save:
        tif = group[0][0].ims[0]
        figname = 'dphase1_dphase2_xcorr_%s_%s' %(tif.c1.celltype, tif.c2.celltype)
    else:
        figname = ''

    plot_tif_xcorr(group, metric, figname=figname, **kwargs)


def plot_phase1_phase2_xcorr(group, save=True, **kwargs):
    """
    Plots cross-correlation between phase difference between c1 and c2 and the phase
    velocity, measured by c1 (usually EIPs).
    :param group: List of flies.
    :param tlim: Cross-correlation time-window.
    :param ylim: Y-axis limits.
    :param save: True to save figure.s
    :return:
    """
    def metric(rec):
        tif = rec.ims[0]
        return fc.unwrap(tif.c1.dphase), fc.unwrap(tif.c2.dphase)

    if save:
        tif = group[0][0].ims[0]
        figname = 'phase1_phase2_xcorr_%s_%s' %(tif.c1.celltype, tif.c2.celltype)
    else:
        figname = ''

    plot_tif_xcorr(group, metric, figname=figname, **kwargs)



# Plotting Correlations between phase and bar/ball position/velocity

def plot_scatter_phase_v_xstim(groups, stim_dict, genotype='', save=True,
                               size=None, **kwargs):
    labels = stim_dict.values()
    def metric(rec, trial_ind, label, lag_ms=200):
        im = rec.ims[0]
        xstim = rec.subsample(rec.abf.xstim, lag_ms=lag_ms, metric=fc.circmean)
        if 'bar' in label.lower():
            iphase = im.c1.phase[trial_ind]
            ixstim = xstim[trial_ind]
            offset = fc.circmean(fc.wrap(iphase-ixstim))
            data = np.vstack([ixstim, iphase-offset])
        else:
            dxstim = fc.circgrad(xstim)[trial_ind] * im.sampling_rate
            dphase = im.c1.dphasef[trial_ind]
            data = np.vstack([dxstim, dphase])
        return data

    phase_v_xstim = get_trial_data(groups, metric, np.concatenate, stim_dict, **kwargs)

    ################
    if size is None: size = (4*len(stim_dict)*2, 3*len(groups))
    plt.figure(1, size)
    gs = gridspec.GridSpec(len(groups), len(stim_dict)*3, width_ratios=[5, 5, 2]*len(stim_dict), height_ratios=[5]*len(groups))
    gs.update(wspace=.4, hspace=.4)
    color = [ph.black, ph.black]
    for igroup in range(len(groups)):
        for ilabel, label in enumerate(labels):
            for itemp, temp in enumerate(['cool', 'hot']):
                ax = plt.subplot(gs[igroup, ilabel*3+itemp])

                if igroup == len(groups)-1 and itemp==0 and (ilabel==0 or ilabel==2):
                    ph.adjust_spines(ax, ['left', 'bottom'], lw=.25)
                    ax.set_ylabel('Phase (deg)')
                    if 'bar' in label.lower():
                        ax.set_xlabel('Bar Position (deg)')
                        ax.set_ylabel('Phase (deg/s)')
                    else:
                        ax.set_xlabel('Turning Velocity (deg/s)')
                        ax.set_ylabel('Phase Velocity (deg/s)')
                else:
                    ph.adjust_spines(ax, ['bottom'], lw=.25)
                    # ax.set_xticklabels([])


                if 'bar' in label.lower():
                    ax.set_xticks(np.arange(-180, 181, 90))
                    ax.set_yticks(np.arange(-180, 181, 90))
                    ax.set_xlim(-180, 180)
                    ax.set_ylim(-180, 180)

                else:
                    ax.set_xticks(np.arange(-360, 361, 180))
                    ax.set_yticks(np.arange(-360, 361, 180))
                    ax.set_xlim(-360, 360)
                    ax.set_ylim(-360, 360)


                # ax.grid()
                ax.axvline(ls='--', c=ph.grey5, dashes=[2, 2], zorder=1, lw=.5)
                ax.axhline(ls='--', c=ph.grey5, dashes=[2, 2], zorder=1, lw=.5)

                data = np.hstack(phase_v_xstim[igroup][temp][label])

                ax.scatter(data[0], data[1], edgecolor='none', facecolor=ph.black,
                           s=.25, alpha=1., zorder=2)
                c = np.corrcoef(data[0], data[1])[0, 1]
                print '%s: %.2f' %(temp, c)

    if save:
        if genotype:
            genotype = '_' + genotype
        ph.save('Scatter_phase_v_xstim' + genotype)


def plot_R_phase_xstim(groups, stim_dict, ylim=(-.5, 1), yticks=None, genotype='',
                       show_pcnt=True, colors=(ph.blue, ph.orange), save=True, size=None,
                       lw=.5, show_r=True, **kwargs):

    labels = stim_dict.values()
    def metric(rec, trial_ind, label, lag_ms=300):
        xstim = rec.subsample('xstim', lag_ms=lag_ms, metric=fc.circmean)
        im = rec.ims[0]
        if 'bar' in label.lower():
            phase = im.c1.phase[trial_ind]
            ixstim = xstim[trial_ind]
            ri = cs.corrcc(np.deg2rad(ixstim+180), np.deg2rad(phase+180))
        else:
            dxstim = fc.circgrad(xstim)[trial_ind] * im.sampling_rate
            dphase = im.c1.dphasef[trial_ind]
            a, b, r, p, e = stats.linregress(dxstim, dphase)
            ri = r
        return ri

    r = get_trial_data(groups, metric, np.mean, stim_dict, **kwargs)
    ################

    # Setup Figure
    if size is None: size = (len(stim_dict)*8, 5)
    plt.figure(1, size)
    ph.set_tickdir('out')
    ax = plt.subplot(111)
    # ax.axhline(c=ph.grey5, lw=lw, ls='--', dashes=[2,2])
    ph.adjust_spines(ax, ['left', 'bottom'], xlim=[-.5, 3.5], lw=.25)
    plt.ylabel('R', rotation=0, labelpad=20, ha='right')
    xlabels = []
    color = list(colors) * len(labels)
    ngroups = len(groups)
    inter_group_dist = .2
    if show_pcnt:
        barwidth = (1-inter_group_dist) / (ngroups*2)
    else:
        barwidth = (1-inter_group_dist) / (ngroups*3)
    all_rs = []
    for igroup in range(ngroups):
        for ilabel, label in enumerate(labels):
            rs = []
            for itemp, temp in enumerate(['cool', 'hot']):
                pts = np.array(r[igroup][temp][label])
                rs.append(pts)
                all_rs.append(pts)
                if not show_pcnt:
                    left_edge = ilabel + barwidth*(-1.8*(ngroups-1) + igroup*3+(igroup%2)+ igroup*.75 + itemp)
                    center_gauss = np.random.normal(left_edge + barwidth / 2, 0.002, len(pts))
                    x = np.ones(len(pts))*(left_edge+barwidth/2.)
                    plt.scatter(x, pts, c='none', zorder=2, s=lw*8, lw=lw)

                    std = np.nanstd(pts)
                    n = (np.isnan(pts)==False).sum()
                    sem = std / np.sqrt(n)

                    plt.bar(left_edge, np.nanmean(pts), width=barwidth, facecolor=color[itemp],
                            ecolor='black', lw=lw)
                    plt.errorbar(left_edge+barwidth/2., np.nanmean(pts), sem, fmt='none',
                             ecolor=ph.black, elinewidth=lw*1.5, capthick=lw*1.5,
                             capsize=lw*4, ms=lw*4)
                    xlabels.append('%s\n%s' % (label, temp))

            if show_pcnt:
                rs = np.array(rs)
                ratios = rs[1] / rs[0]
                means = np.nanmean(ratios)
                left_edge = ilabel + barwidth*(-1.5*(ngroups-1)+2  + igroup*2 + (igroup%2))
                center_gauss = np.random.normal(left_edge + barwidth / 2, .003, len(ratios))
                x = np.ones(len(pts))*(left_edge+barwidth/2.)
                plt.scatter(x, ratios, c='none', zorder=2, s=lw*8)
                plt.bar(left_edge, means, width=barwidth, facecolor=color[0], ecolor='black')

    label_pos = np.arange(len(labels))
    plt.xticks(label_pos, labels, ha='center')
    plt.xlim([-.5, len(labels)-.4])
    if yticks:
        plt.yticks(yticks)
    else:
        plt.yticks(np.arange(-1, 1.1, .5))
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim([-1, 1])

    if save:
        stimlabels = '_'.join(stim_dict.values())
        if genotype:
            genotype = '_' + genotype
        if show_pcnt:
            stimlabels += '_ratios'
        ph.save('R_phase_xstim' + genotype + '_' + stimlabels)

    if show_r:

        rfs = []
        for i in range(len(all_rs)/2):
            cool = all_rs[i*2]
            hot = all_rs[i*2+1]
            notnan = (np.isnan(cool) | np.isnan(hot)) == False
            rfs.append( (cool[notnan] - hot[notnan]) )
        # print 't-test'
        # print len(rfs)
        # print 'VT3-Gal4: %.2g' %stats.ttest_ind(rfs[2], rfs[1])[1]
        # print 'VT2-Gal4: %.2g' %stats.ttest_ind(rfs[4], rfs[3])[1]
        # print '12D09:Gal4 %.2g' %stats.ttest_ind(rfs[6], rfs[5])[1]
        #
        #
        # print 'VT3-shi: %.2g' %stats.ttest_ind(rfs[2], rfs[0])[1]
        # print 'VT2-shi: %.2g' %stats.ttest_ind(rfs[4], rfs[0])[1]
        # print '12D09-shi: %.2g' %stats.ttest_ind(rfs[6], rfs[0])[1]
        #
        # print
        # pooledgal4shi = np.concatenate([rfs[2], rfs[4], rfs[6]])
        # pooledgal4 = np.concatenate([rfs[1], rfs[3], rfs[5]])
        # print 'pooled-Gal4: %.2g' %stats.ttest_ind(pooledgal4shi, pooledgal4)[1]
        # print 'pooled-shi: %.2g' %stats.ttest_ind(pooledgal4shi, rfs[0])[1]

        print 'rank-sum'
        print len(rfs)
        print 'VT3-Gal4: %.2g' %stats.ranksums(rfs[2], rfs[1])[1]
        print 'VT2-Gal4: %.2g' %stats.ranksums(rfs[4], rfs[3])[1]
        print '12D09:Gal4 %.2g' %stats.ranksums(rfs[6], rfs[5])[1]


        print 'VT3-shi: %.2g' %stats.ranksums(rfs[2], rfs[0])[1]
        print 'VT2-shi: %.2g' %stats.ranksums(rfs[4], rfs[0])[1]
        print '12D09-shi: %.2g' %stats.ranksums(rfs[6], rfs[0])[1]

        print
        pooledgal4shi = np.concatenate([rfs[2], rfs[4], rfs[6]])
        pooledgal4 = np.concatenate([rfs[1], rfs[3], rfs[5]])
        print 'pooled-Gal4: %.2g' %stats.ranksums(pooledgal4shi, pooledgal4)[1]
        print 'pooled-shi: %.2g' %stats.ranksums(pooledgal4shi, rfs[0])[1]


        # print
        # print 'VT3-Gal4-hot: %.2g' %stats.ttest_ind(all_rs[5], all_rs[3])[1]
        # print 'VT2-Gal4-hot: %.2g' %stats.ttest_ind(all_rs[9], all_rs[7])[1]
        # print '12D09-Gal4-hot: %.2g' %stats.ttest_ind(all_rs[13], all_rs[11])[1]
        #
        # print 'VT3-shi-hot: %.2g' %stats.ttest_ind(all_rs[5], all_rs[1])[1]
        # print 'VT2-shi-hot: %.2g' %stats.ttest_ind(all_rs[9], all_rs[1])[1]
        # print '12D09-shi-hot: %.2g' %stats.ttest_ind(all_rs[13], all_rs[1])[1]


def plot_R_v_powerthresh(groups, stim_dict, range=(0, 10, .2), thresh_channel='peak', temp_thresh=25):
    temps = [0, 1]
    stim_keys = stim_dict.keys()
    thresh_range = np.arange(range[0], range[1], range[2])

    plt.figure(1, (10, 10))
    gs = gridspec.GridSpec(len(groups)*3, len(stim_dict), height_ratios=[3, 1.5, 1]*len(groups))

    for igroup, group in enumerate(groups):
        phase = get_trial_data2(group, stim_dict, 'phase')
        xstim = get_trial_data2(group, stim_dict, 'xstim', lag_ms=300)
        power = get_trial_data2(group, stim_dict, thresh_channel)

        # sr = group[0][0].tif.sampling_rate

        for istim in xrange(len(stim_dict)):
            r = np.zeros((len(group), len(temps), len(thresh_range))).astype(float)
            r.fill(np.nan)
            frac_pts = np.zeros_like(r)
            for ifly in xrange(len(group)):
                for itemp in temps:
                    for ithresh, thresh in enumerate(thresh_range):
                        ipower = power[ifly, istim, itemp]
                        mask = ipower > thresh
                        frac_pts[ifly, itemp, ithresh] = float(mask.sum()) / (np.isnan(ipower)==False).sum()

                        ddt = True
                        if 'bar' in stim_dict[istim+1].lower():
                            ris = np.zeros(phase.shape[3])
                            for itrial in xrange(phase.shape[3]):
                                iphase = phase[ifly, istim, itemp, itrial]
                                ixstim = xstim[ifly, istim, itemp, itrial]
                                notnans = np.isnan(iphase)==False
                                bar_visible = (ixstim > -135) & (ixstim < 135)
                                idx = notnans & bar_visible & mask[itrial]
                                iphase = np.deg2rad(iphase+180)[idx]
                                ixstim = np.deg2rad(ixstim+180)[idx]

                                ris[itrial] = cs.corrcc(ixstim, iphase)
                            r[ifly, itemp, ithresh] = np.mean(ris)
                        else:
                            mask = mask.flatten()
                            iphase = phase[ifly, istim, itemp].flatten()
                            ixstim = xstim[ifly, istim, itemp].flatten()
                            inds_set = fc.get_contiguous_inds(mask, min_contig_len=3, keep_ends=True)

                            if inds_set:
                                dphase = np.concatenate([fc.circgrad(iphase[inds]) for inds in inds_set])
                                dxstim = np.concatenate([fc.circgrad(ixstim[inds]) for inds in inds_set])
                                notnans = np.isnan(dphase)==False
                                a, b, ri, p, e = stats.linregress(dxstim[notnans], dphase[notnans])
                                r[ifly, itemp, ithresh] = ri

            rmean = np.nanmean(r, axis=0)
            rstd = np.nanstd(r, axis=0)
            n = (np.isnan(r)==False).astype(int).sum(axis=0)
            rsem = rstd / np.sqrt(n)
            error = rsem

            frac_mean = np.nanmean(frac_pts, axis=0)
            frac_std = np.nanstd(frac_pts, axis=0)

            ax = plt.subplot(gs[igroup*3, istim])
            if igroup==len(groups)-1 and istim==0:
                ph.adjust_spines(ax, ['left', 'bottom'])
                ax.set_ylabel('R')
            else:
                ph.adjust_spines(ax, [])
            colors = [ph.blue, ph.orange]

            ax.plot(thresh_range, rmean[0], c=ph.blue)
            ax.fill_between(thresh_range, rmean[0]-error[0], rmean[0]+error[0], facecolor=ph.blue, edgecolor='none', alpha=.4)
            ax.plot(thresh_range, rmean[1], c=ph.orange)
            ax.fill_between(thresh_range, rmean[1]-error[1], rmean[1]+error[1], facecolor=ph.orange, edgecolor='none', alpha=.4)
            ax.set_ylim(-.505, 1)
            ax.set_xlim(range[0], range[1])

            ax.grid()
            ax.set_yticks(np.arange(-.5, 1.1, .5))
            ax.axhline(0, ls='--', lw=1, alpha=.6, c='black')


            ax2 = plt.subplot(gs[igroup*3+1, istim])
            if igroup==len(groups)-1 and istim==0:
                ph.adjust_spines(ax2, ['left', 'bottom'])
                ax2.set_xlabel('Power threshold')
                ax2.set_ylabel('Fraction of data\nabove threshold', labelpad=15)
            else:
                ph.adjust_spines(ax2, [])
            colors = [ph.blue, ph.orange]
            ax2.plot(thresh_range, frac_mean[0], c=ph.blue)
            ax2.plot(thresh_range, frac_mean[1], c=ph.orange)
            ax2.set_ylim(-.01, 1)
            ax2.set_xlim(range[0], range[1])

            ax2.grid()
            ax2.set_yticks(np.arange(0, 1.1, .5))
            ax2.axhline(0, ls='--', lw=1, alpha=.6, c='black')




def plot_R_distribution(groups, stimdict, colours, stimlabel='Dark', windowlen=10,
                        temp_thresh=27, metric='R', vel=True, **kwargs):
    plt.figure(1, (6, 6))
    ax = plt.subplot(111)
    ax.axvline(c='black', ls='--', alpha=.5)
    ph.adjust_spines(ax, ['left', 'bottom'])
    width = .05
    bins = np.arange(-1, 1+width, width)

    icolour = 0
    for igroup, group in enumerate(groups):
        if vel:
            dt, dphase = get_group_data(group, 'pb.c1.dphasef')
            dt, dhead = get_group_data(group, 'abf.dhead', subsample=True, lag_ms=250)
            dt, dxstim = get_group_data(group, 'abf.dxstim', subsample=True, lag_ms=250)
        else:
            dt, phase = get_group_data(group, 'pb.c1.phasef')
            dt, xstim = get_group_data(group, 'abf.xstim', subsample=True, lag_ms=250)

        dt, stimid = get_group_data(group, 'abf.stimid', subsample=True, lag_ms=0)
        dt, temp = get_group_data(group, 'abf.temp', subsample=True, lag_ms=0)

        dt, xstim = get_group_data(group, 'abf.xstim', subsample=True, lag_ms=250)

        cool = temp < temp_thresh
        hot = temp > temp_thresh
        if len(groups) == 1:
            temp_idxs = [cool, hot]
        else:
            temp_idxs = [hot]
        dictstim = {val: key for key, val in stimdict.items()}
        idx0 = (stimid == dictstim[stimlabel])

        if 'Bar' in stimlabel:
            xstim_on = (xstim > -135) & (xstim < 135)
            xstim_on = ndimage.binary_erosion(xstim_on, iterations=2)
            idx0 = idx0 & xstim_on

        for itemp, temp_idx in enumerate(temp_idxs):
            idx = idx0 & temp_idx
            contig_inds = fc.get_contiguous_inds(idx, keep_ends=True)

            corrs = []
            for contig in contig_inds:
                if len(contig) < windowlen + 2: continue
                icorrs = np.zeros(len(contig)-windowlen+1)
                for i, ind in enumerate(contig):
                    if ind > contig[-1] - windowlen+1: continue
                    window = np.arange(ind, ind + windowlen, 1)
                    if vel:
                        a, b, r, p, e = stats.linregress(-dhead[window], dphase[window])
                        c = np.corrcoef(dxstim[window], dphase[window])[0, 1]
                        if metric == 'R':
                            icorrs[i] = r
                        elif metric == 'C':
                            icorrs[i] = c
                        elif metric == 'slope':
                            icorrs[i] = a
                    else:
                        r = cs.corrcc(np.deg2rad(xstim[window]+180), np.deg2rad(phase[window]+180))
                        icorrs[i] = r
                corrs.append(icorrs)
            corrs = np.concatenate(corrs)


            a = plt.hist(corrs, bins=bins, normed=True, histtype='stepfilled',
                         facecolor='none', edgecolor=colours[icolour], lw=2)
            icolour += 1
            # h, x = np.histogram(corrs, bins=bins, density=True)
            # plt.bar(x[:-1], h, width=width, facecolor='none', edgecolor=colours[itemp],
            #         lw=2, alpha=1)
            ax.set_xlim(-1, 1)





# Plotting the phase-subtracted signal
arr18 = np.arange(1., 19.)
arr16 = np.arange(1., 17.)
x_dict = {'pb':     {'PEN': arr18,
                    'EIP': arr18},
          'ebproj': {'PEN': arr16[::2] + .5,  # +.5 in PENs is to place the points in the center of each tile, which is in between two wedges, hence the .5
                    'EIP': arr16},
          'eb':     {'PEN': arr16,
                     'EIP': arr16}
          }

def plot_phase_subtracted_mean(group, neuropil='pb', proj='preserve', a='an_cubspl',
                               turn_vels=(0,), lag_ms=300, binsize=60, channels=(1,),
                               right_dashed=True, save=True, separate_eip=True,
                               phase_label='c1.phase', c1_ylim=None, c2_ylim=None,
                               c1_yticks=None, c2_yticks=None, offset=0,
                               scalebar_size=None, size=None, lw=.5, stim=None):
    """
    Plot the phase subtracted mean signal across the bridge or ellipsoid body.
    :param group: List of flies.
    :param neuropil: 'pb', 'eb', or 'ebproj'. Type of signal to plot.
    :param proj: 'merge' or 'preserve'. Type of projection to apply to bridge signal
    to arrive at ellipsoid body signal.
    :param turn_vels: Center of turning velocity bin to plot.
    :param lag_ms: Lag to use for binning heading velocity.
    :param binsize: Binsize for turning bins.
    :param channels: tuple, channels to plot.
    :param right_dashed: Draw right bridge with dashed line.
    :param save: Save figure.
    :param separate_eip: If proj = 'preserve', plot left and right EIPs separately,
    with right dashed and left solid linestyle. Otherwise, connect them, alternating
    left, right points with solid and open circles.
    :param phase_channel: Will use the phase from this channel to 'cancel' the phase
    from both channels, if two exist.
    :param c1_ylim: ylim for channel 1.
    :param c2_ylim: ylim for channel 2.
    :param c1_yticks: yticks for channel 1.
    :param c2_yticks: yticks for channel 2.
    :return:
    """

    # Draw Figure
    ph.set_tickdir('out')
    if size is None:
        if neuropil[:2] == 'eb':
            size = (2, 3)
        elif neuropil[:2] == 'pb':
            size = (4, 3)
    plt.figure(1, size)
    gs = gridspec.GridSpec(2, 1, height_ratios=(2, 1))
    gs.update(hspace=.3)
    ax = plt.subplot(gs[0, 0])
    axsb = plt.subplot(gs[1, 0])
    ph.adjust_spines(ax, ['bottom', 'left'], lw=.25)
    ph.adjust_spines(axsb, [])

    neuropils = copy.copy(neuropil)
    if not type(neuropils) is tuple:
        neuropils = (neuropils,)
    ylims = [c1_ylim, c2_ylim]
    if (c1_yticks is None) and (c2_yticks is None):
        yticks = ylims
    else:
        yticks = [c1_yticks, c2_yticks]
    plot_count = 0
    mins, maxs = [], []
    for ineuropil, neuropil in enumerate(neuropils):
        if ineuropil == 1:
            ax = ax.twinx()
            ph.adjust_spines(ax, ['bottom', 'left'])

        imlabel_dict = {'pb': 'pb', 'eb': 'eb', 'ebproj': 'pb'}
        imlabel = imlabel_dict[neuropil]

        # Collect data from group of flies
        turn_bins = np.arange(-360-binsize/2., 360.1+binsize/2., binsize)
        turn_bin_idx = [np.where(turn_bins>=turn_vel)[0][0]-1 for turn_vel in turn_vels]

        r = group[0][0]
        _, _, x = r.get_objects('%s.c1.xedges_interp' %imlabel)
        abs = np.zeros((len(group), len(channels), len(turn_vels), len(x)))
        abs[:] = np.nan
        for ifly, fly in enumerate(group):
            flyab = np.zeros((len(fly), len(channels), len(turn_vels), len(x)))
            flyab[:] = np.nan
            for irec, rec in enumerate(fly):
                for ichannel, channel in enumerate(channels):
                    _, _, phase = rec.get_objects(phase_label)
                    im, ch, an_cubspl = rec.get_objects('%s.c%i.%s' %(imlabel, channel, a))
                    az_nophase = im.cancel_phase(an_cubspl, phase, 10, ch.celltype, offset=offset)
                    ab = rec.bin_gcamp(az_nophase, 'abf.dhead', turn_bins, lag_ms=lag_ms, stim=stim)
                    flyab[irec, ichannel] = ab[turn_bin_idx]
            abs[ifly] = np.nanmean(flyab, axis=0)

        # # Subsample back into glomerular / wedge resolution from cubic spline interpolation
        idx = np.arange(len(x)).reshape((len(x)/10, 10))
        ab_sub = np.nanmean(abs[:, :, :, idx], axis=-1)

        # Compute mean and standard error
        ab_mean = np.nanmean(ab_sub, axis=0)
        n = (np.isnan(ab_sub)==False).sum(axis=0)
        ab_sem = np.nanstd(ab_sub, axis=0) / np.sqrt(n)

        def error_propagation_add(arr, axis=0):
            err = np.sqrt(np.sum(arr**2, axis=axis))
            return err


        if neuropil == 'ebproj':
            eb_mean = []
            eb_sem = []
            for ichannel, channel in enumerate(channels):
                ch = getattr(rec.pb, 'c%i' %channel)
                eb_mean.append( rec.pb.get_ebproj(ab_mean[ichannel], ch.celltype, proj,
                                mergefunc=np.nansum) )
                eb_sem.append( rec.pb.get_ebproj(ab_sem[ichannel], ch.celltype, proj,
                                mergefunc=error_propagation_add) )
            ab_mean = eb_mean
            ab_sem = eb_sem

        # Now we plot
        rec = group[0][0]
        celltypes = []

        for ic, channel in enumerate(channels):
            if ic == 1:
                ax = ax.twinx()
                ph.adjust_spines(ax, ['right'])
            _, _, celltype = rec.get_objects('%s.c%i.celltype' %(imlabel, channel))
            ax.set_ylabel(celltype + ' ' + neuropil, labelpad=2)
            celltypes.append(celltype)
            colour = colours[celltype] if ineuropil == 0 else 'grey'
            for it in range(len(turn_vels)):
                if len(channels)==1 and len(turn_vels)>1 and turn_vels[it] == 0:
                    c = 'grey'
                elif len(channels)==2 and (rec.ims[0].c1.celltype==rec.ims[0].c2.celltype):
                    c = ph.green if ic == 0 else ph.red
                else:
                    c = colour

                if neuropil=='ebproj' and proj=='preserve':
                    x = x_dict[neuropil][celltype]
                    if celltype == 'PEN':
                        line = ab_mean[ic][:, it]
                        ax.plot(x, line[0], c=c, lw=lw)
                        ax.plot(x, line[1], c=c, lw=lw, ls='--', dashes=[2, 2])
                        sem = ab_sem[ic][:, it]
                        ax.fill_between(x,
                                        line[0] - sem[0],
                                        line[0] + sem[0],
                                        facecolor=c, edgecolor='none', alpha=.3)
                        ax.fill_between(x,
                                        line[1] - sem[1],
                                        line[1] + sem[1],
                                        facecolor=c, edgecolor='none', alpha=.3)
                    elif celltype == 'EIP':
                        x = x
                        if separate_eip:
                            ax.plot(x[::2], ab_mean[ic][0, it, ::2], c=c,  lw=lw)
                            ax.plot(x[::2]+1, ab_mean[ic][1, it, 1::2], c=c, lw=lw, ls='--', dashes=[2, 2])
                        else:
                            ax.scatter(x[::2], ab_mean[ic][0, it, ::2], c=c,  edgecolor=c, lw=lw)
                            ax.scatter(x[::2]+1, ab_mean[ic][1, it, 1::2], c='white', edgecolor=c, lw=lw, zorder=3)
                            line = np.nansum(ab_mean[ic][:, it], axis=0)
                            ax.plot(x, line, c=c, lw=lw)
                            sem = np.nansum(ab_sem[ic][:, it], axis=0)
                            ax.fill_between(x,
                                            line - sem,
                                            line + sem,
                                            facecolor=c, edgecolor='none', alpha=.3)
                    min = np.nanmin(ab_mean[ic][:, it])
                    max = np.nanmax(ab_mean[ic][:, it])
                else:
                    x = x_dict[neuropil][celltype]  #celltype='EIP'
                    if neuropil=='pb' and right_dashed:
                        ax.plot(x[x<10], ab_mean[ic][it, x<10], c=c, lw=lw)
                        ax.plot(x[x>=10], ab_mean[ic][it, x>=10], c=c, lw=lw, ls='--', dashes=[2, 2])

                    else:
                        ax.plot(x, ab_mean[ic][it], c=c, lw=lw)

                    ax.fill_between(x, ab_mean[ic][it]-ab_sem[ic][it],
                                    ab_mean[ic][it]+ab_sem[ic][it],
                                    facecolor=c, edgecolor='none', alpha=.3)

                    min = np.nanmin(ab_mean[ic][it])
                    max = np.nanmax(ab_mean[ic][it])

                plot_count += 1
            if ylims[channel-1] is None:
                meanrange = max - min
                margin = meanrange / 6.
                ylim = min-2*margin, max+2*margin
            else:
                ylim = ylims[channel-1]
            ax.set_ylim(ylim)
            ax.set_yticks(np.round(ylim, 4))

            if scalebar_size is None:
                yrange = ylim[1] - ylim[0]
                decimal = np.ceil(np.abs(np.log10(yrange)))
                decimal_sf = 10**(decimal)
                sbsize = np.round(yrange / 5. * decimal_sf) / decimal_sf
                if sbsize == 0:
                    decimal_sf = 10**(decimal+1)
                    sbsize = np.round(yrange / 5. * decimal_sf) / decimal_sf
                # print sbsize
            else:
                sbsize = scalebar_size[ic]


            ph.add_scalebar(ax, sizey=sbsize, sizex=0, width=.2/size[0], loc=3+(ineuropil+ic),
                        color=c, sbax=axsb, labely=sbsize)


        if neuropil[:2] == 'eb':
            ax.set_xticks(np.arange(1, 17), minor=True)
            ax.set_xticks([1, 16], minor=False)
            ax.set_xlim(1, 16)
        elif neuropil[:2] == 'pb':
            ax.set_xticks(np.arange(1, 19), minor=True)
            plt.xticks([1, 9, 10, 18], ['1', '9', '1', '9'])
            ax.set_xlim(1, 18)



    if save:
        celltype_str = '_'.join(['%s' %celltype for celltype in celltypes])
        neuropil_str = '_'.join(list(neuropils))
        if 'proj' in neuropil:
            neuropil_str = neuropil + '_' + proj
            # if proj=='preserve' and separate_eip:
            #     neuropil_str += '_eipsep'
        figname = celltype_str +  '_' + neuropil_str + ''.join(['_%i' %vel for vel in turn_vels])
        ph.save(figname)

    return ab_sub


def plot_phase_subtracted_mean_multistim(groups, stim_dict, normalize=False, genotype='', save=True, **kwargs):
    """
    Plot the average phase signal for multiple stimuli, cool and hot.
    Used for shibire experiments, where neurons are 'wildtype' (cool) and then inhibited
    (hot).
    :param groups: List of groups of flies (ie. experimental conditions).
    :param stim_dict: Dictionary mapping the stimid to a stimulus label. ie. 1: 'CL Bar'
    :param normalize: Normalize the hot condition to the maximum average response in the
    cool condition. Only plots this normalized hot condition.
    :param genotype: str, genotype label.
    :param save: True to save figure.
    :param kwargs: For get_trial_data.
    :return:
    """
    labels = stim_dict.values()
    def metric(rec, trial_ind, label):
        data = np.nanmean(rec.tif.gcn_nophase[trial_ind], axis=0)
        return data

    phase_cancelled = get_trial_data(groups, metric, np.mean, stim_dict, **kwargs)

    ################

    plt.figure(1, (15, 3))
    gs = gridspec.GridSpec(1, len(stim_dict))
    rec = groups[0][0][0]
    x = rec.tif.xedges_interp
    color = [ph.blue, ph.orange]
    linestyle = ['-', '--', ':']
    for ilabel, label in enumerate(labels):
        ax = plt.subplot(gs[0, ilabel])

        if ilabel == 0:
            ph.adjust_spines(ax, ['left', 'bottom'])
            ax.set_ylabel('$\Delta$F/F')

        else:
            ph.adjust_spines(ax, ['bottom'])
            ax.set_yticks([])
        ax.set_xlabel(label)

        xticks = np.array([0, 8, 10, 17])
        plt.xticks(xticks, (xticks % 9 + .5).astype(int))
        ax.set_xlim(0, 18)
        for igroup in range(len(groups)):
            if normalize:
                ax.axhline(1, ls='--', color='black', alpha=.4)
                ax.set_ylim([0, 1.25])
                if ilabel == 0: ax.set_yticks([0, 1])
                ipcs_cool = np.array(phase_cancelled[igroup]['cool'][label])
                ipc_cool_mean = np.nanmean(ipcs_cool, axis=0)
                ipcs_hot = np.array(phase_cancelled[igroup]['hot'][label])
                ipc_hot_mean = np.nanmean(ipcs_hot, axis=0)
                ipc_cool_mean_peak = np.max(ipc_cool_mean)
                ipc_hot_mean_normd = ipc_hot_mean / ipc_cool_mean_peak
                ax.plot(x, ipc_hot_mean_normd, c=color[1], ls=linestyle[igroup], lw=2)
            else:
                ax.set_ylim([0, 6])
                if ilabel == 0: ax.set_yticks(np.arange(0, 7, 2))
                for itemp, temp in enumerate(['cool', 'hot']):
                    ipcs = np.array(phase_cancelled[igroup][temp][label])
                    ipc_mean = np.nanmean(ipcs, axis=0)
                    ipc_sem = stats.sem(ipcs, axis=0)
                    ax.plot(x, ipc_mean, c=color[itemp], ls=linestyle[igroup], lw=2)
            #         ax.fill_between(x, ipc_mean - ipc_sem, ipc_mean + ipc_sem, facecolor=color, edgecolor='none', alpha=.2)

    if save:
        if genotype:
            genotype = '_' + genotype
        ph.save('Phase_subtracted_mean_multistim' + genotype)


def plot_period_v_dhead(group, binsize=60, lag_ms=400, ylim=None, save=True):
    dhead_bins = np.arange(-360-.5*binsize, 360+1.5*binsize, binsize)
    period_binned = np.zeros((len(group), len(dhead_bins)-1))
    for ifly, fly in enumerate(group):
        flyperiod = []
        flydhead = []
        for rec in fly:
            power, period, phase = fc.powerspec(rec.pb.c1.acn)
            gate = (period > 6) & (period < 8.5)
            peak_period = period[gate][np.argmax(power[:, gate], axis=1)]
            dhead = subsample(rec, 'dhead', lag_ms=lag_ms)
            flyperiod.append(peak_period)
            flydhead.append(dhead)
        flyperiod = np.concatenate(flyperiod)
        flydhead = np.concatenate(flydhead)
        period_binned[ifly] = fc.binyfromx(flydhead, flyperiod, dhead_bins)

    bin_centers = dhead_bins[:-1] + np.diff(dhead_bins)[0]/2.

    med = np.nanmedian(period_binned, axis=0)
    n = (np.isnan(period_binned)==False).sum(axis=0)
    sem = np.nanstd(period_binned, axis=0) / np.sqrt(n)
    c = colours[rec.pb.c1.celltype]

    plt.figure(1, (5, 5))
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    ax.plot(bin_centers, med, c=c, lw=2)
    ax.fill_between(bin_centers, med-sem, med+sem, facecolor=c, edgecolor='none', alpha=.3)
    if ylim:
        ax.set_ylim(ylim)
    if save:
        celltype = rec.pb.c1.celltype
        figname = '%s_period_v_dhead_%ims_lag' %(celltype, lag_ms)
        ph.save(figname)



# Plotting phase difference between two channels

def plot_c1c2_phasediff(group, save=True):
    phase_diff_means = []
    phase_diff_stds = []
    for fly in group:
        iphase_diffs = []
        for rec in fly:
            tif = rec.ims[0]
            phase_diff = fc.wrap(tif.c1.phase - tif.c2.phase)
            iphase_diffs.append( phase_diff )

        iphase_diffs = np.hstack(iphase_diffs)
        phase_diff_means.append( fc.circmean(iphase_diffs) )
        phase_diff_stds.append( fc.circstd(iphase_diffs) )

    phase_diff_means = np.array(phase_diff_means)
    phase_diff_stds = np.array(phase_diff_stds)

    plt.figure(1, (5, 5))
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    nflies = len(group)
    x = np.arange(nflies) + 1
    phase_diff_means = fc.wrap(phase_diff_means, cmin=-360, cmax=0)
    ax.scatter(x, phase_diff_means, marker='o', c='black')
    ax.errorbar(x, phase_diff_means, phase_diff_stds, c='black', linestyle='none')
    ax.set_xlim(.5, nflies + .5)
    ax.set_xticks(np.arange(len(group))+1, minor=True)
    ax.set_xticks([1, len(group)], minor=False)
    ax.set_ylim(-360, 0)
    ax.set_yticks(np.arange(-360, 1, 90))
    ax.set_xlabel('Fly')
    ax.set_ylabel('Phase offset, PEN - EIP (deg)')

    print phase_diff_means.mean()

    if save:
        ph.save('PEN-EIP_phase_diff')



# Analysis for phase/bar slope vs position - now believe due to bleaching

def plot_slope_bardensity(self, channel='green', phase='phase', tbinsize=30, xbinsize=15,
                          tlim=None, slope_lim=None, forw_lim=None, idx_offset=-1,
                          minspeed=.2, x_range=[-45, 45], period=360, alpha=1, lines=[],
                          save=True, sf=1):
    ch = getattr(self, channel)
    t = self.tif.t
    if not tlim: tlim = [t[0], t[-1]]
    colours = [ph.red, ph.purple, 'black']
    t = self.abf.t
    tbins = np.arange(t[0], t[-1], tbinsize)

    # Setup Figure
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    plt.figure(1, (sf * 9, sf * 7))
    plt.suptitle(self.folder, size=10)
    gs = gridspec.GridSpec(4, 1)
    axs = []

    # Plot slope
    t = self.tif.t
    xstimb = self.abf.xstimb
    phase = getattr(ch, phase)

    if not minspeed is None:
        speedb = self.subsample(np.abs(self.abf.dforw))
        walking = speedb > minspeed
    else:
        walking = np.ones(len(t)).astype(bool)

    in_x_range = (xstimb >= x_range[0]) & (xstimb < x_range[1])
    walking_and_in_x_range = walking & in_x_range

    inds = [(t >= tbins[i]) & (t < tbins[i + 1]) & walking_and_in_x_range for i in range(len(tbins) - 1)]

    slope = np.zeros(len(inds))
    std_err = np.zeros(len(inds))
    for i, ind in enumerate(inds):
        idx = np.where(ind)[0]
        ixstimb = xstimb[idx + idx_offset]
        if ixstimb.size:
            iphase, _ = fc.circ2lin(ixstimb, phase[idx]) if ixstimb.size else np.nan
            islope, intercept, r_value, p_value, istd_err = stats.linregress(ixstimb, iphase)
            slope[i] = islope
            std_err[i] = istd_err
        else:
            slope[i] = np.nan
            std_err[i] = np.nan
    axslope = plt.subplot(gs[0, 0])
    axs.append(axslope)
    ph.adjust_spines(axslope, ['left'], xlim=tlim, yticks=np.arange(-1, 4, .5))
    axslope.set_ylabel('Slope (phi/theta)', labelpad=10, rotation='horizontal', ha='right')
    if slope_lim:
        axslope.set_ylim(slope_lim)
    tb = tbins[:-1] + tbinsize / 2.
    notnan = np.isnan(slope) == False
    axslope.plot(tb[notnan], slope[notnan], c=ph.bgreen, lw=2, zorder=2)
    # axslope.scatter(tb[notnan], slope[notnan], c='none', zorder=4)
    axslope.errorbar(tb[notnan], slope[notnan], yerr=std_err[notnan], c='black', fmt='o', lw=1, zorder=3)
    # axslope.set_ylim(tlim)
    # axslope.invert_yaxis()

    # Plot bar density
    t = self.abf.t
    xstim = self.abf.xstim
    if not minspeed is None:
        speed = np.abs(self.abf.dforw)
        walking = speed > minspeed
    else:
        walking = np.ones(len(t)).astype(bool)
    inds = [(t >= tbins[i]) & (t < tbins[i + 1]) & walking for i in range(len(tbins) - 1)]
    xbins = np.arange(-180, 181, xbinsize)
    bardensity = np.zeros((len(inds), len(xbins) - 1))
    for i, ind in enumerate(inds):
        bardensity[i, :], _ = np.histogram(xstim[ind], bins=xbins, density=True)
    axdensity = plt.subplot(gs[1, 0])
    axs.append(axdensity)
    ph.adjust_spines(axdensity, ['left'], ylim=[-180, 180], yticks=[-180, 0, 180], xlim=tlim)
    axdensity.pcolormesh(tbins, xbins, bardensity.T, cmap=plt.cm.Blues)
    axdensity.set_ylabel('Bar density', labelpad=10, rotation='horizontal', ha='right')
    # axdensity.invert_yaxis()

    # Plot bar position
    axstim = plt.subplot(gs[2, 0])
    axs.append(axstim)
    axstim.invert_yaxis()
    ph.adjust_spines(axstim, ['left'], ylim=[-180, 180], yticks=[-180, 0, 180], xlim=tlim)
    axstim.set_ylabel('Bar Position (deg)', labelpad=10, rotation='horizontal', ha='right')
    axstim.fill_between(self.abf.t, y1=-180, y2=-self.abf.arena_edge, color='black', alpha=0.1, zorder=10)
    axstim.fill_between(self.abf.t, y1=self.abf.arena_edge, y2=180, color='black', alpha=0.1, zorder=10)
    axstim.axvline(0, c='black', ls='--', alpha=0.4)
    ph.circplot(self.abf.t, self.abf.xstim, circ='y', c='black', alpha=.8)
    # ph.circplot(xstimb, t, c='black', alpha=.8)
    # Plot phase
    # ph.circplot(phase, t, c=ph.nblue, alpha=1, lw=1)
    # axstim.invert_yaxis()

    # Plot forward rotation
    axforw = plt.subplot(gs[3, 0])
    axs.append(axforw)
    ph.adjust_spines(axforw, ['bottom', 'left'], xlim=tlim)
    axforw.set_ylabel('Forward rotation (deg)', labelpad=10, rotation='horizontal', ha='right')
    axforw.axvline(0, c='black', ls='--', alpha=0.4)
    forwint = self.abf.forwuw
    axforw.plot(self.abf.t, forwint, c='black', alpha=.8)
    if forw_lim:
        axforw.set_ylim(forw_lim)
        axforw.set_yticks(forw_lim)
    axdforw = axforw.twinx()
    axdforw.plot(self.abf.t, self.abf.dforw, c='black', alpha=.6)
    ph.adjust_spines(axdforw, ['bottom', 'right'], xlim=tlim, ylim=[-10, 50])

    if lines:
        for line in lines:
            for ax in axs:
                ax.axvline(line, alpha=.6, c=ph.nblue, ls='--')

    if save:
        suffix = ''
        if tlim: suffix += '_%is-%is' % (tlim[0], tlim[1])
        plt.savefig('%s_phase_v_xpos_%s.png' % (self.abf.basename, suffix), bbox_inches='tight')


def plot_slope_v_xposdensity(recs, save=True, sf=1, method='polyreg', suffix=''):
    # Setup Figure
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    plt.figure(1, (sf * 7, sf * 7))
    # plt.suptitle(self.folder, size=10)
    ax = plt.subplot(111)

    # Collect data
    def get_phase_xpos_slope_xposdensity(self, channel='green', phase='phase', tlim=None, idx_offset=-1, binsize=10,
                                         minspeed=None, method=method, period=360):
        ch = getattr(self, channel)
        if not tlim:
            tlim = [ch.t[0], ch.t[-1]]
        abft = self.abf.t
        abfidx = (abft >= tlim[0]) & (abft < tlim[-1])
        tift = self.tif.t
        tifidx = (tift >= tlim[0]) & (tift < tlim[-1])
        if not minspeed is None:
            speed = np.abs(self.abf.dforw)
            abfidx2 = abfidx & (speed > minspeed)
            abfidx = abfidx2

            speedb = self.subsample(speed)
            tifidx2 = tifidx & (speedb > minspeed)
            tifidx = tifidx2

        # Get histogram of xposition
        xbins = np.arange(-180, 181, binsize)
        xb = xbins[:-1] + binsize / 2  # Bin centers
        t = self.abf.t
        xstim = copy.copy(self.abf.xstim)
        xstim = xstim[abfidx]
        h, _ = np.histogram(xstim, bins=xbins, density=True)
        hinterpf = interp1d(fc.tile(xb), np.tile(h, 3), 'linear')

        # Get slope of phase vs binned xposition
        # Linearize phase, find best fit diagonal (ie. find b in y=x+b)
        xstimb = self.abf.xstimb
        phases = getattr(ch, phase)
        phases = fc.wrap(phases)
        t = ch.t
        idx = np.where(tifidx)[0]
        xstimb = xstimb[idx + idx_offset]
        phases = phases[idx]
        phases, b = fc.circ2lin(xstimb, phases)

        if method == 'smooth':
            x = [-180, 180]
            y = x + b

            # Convert xstim, phase coordinates into distance from diagonal
            def xy2diag(xs, ys, b):
                ts = np.zeros_like(xs)
                ds = np.zeros_like(xs)
                for i, (x0, y0) in enumerate(zip(xs, ys)):
                    # (x1, y1) = closest point on diagonal
                    x1 = (x0 + y0 - b) / 2.
                    y1 = x1 + b
                    d = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                    if y0 < y1: d *= -1
                    ds[i] = d
                    t = np.sqrt(x1 ** 2 + (y1 - b) ** 2)
                    if x1 < 0: t *= -1
                    ts[i] = t
                return ts, ds

            ts, ds = xy2diag(xstimb, phases, b)

            # Bin values along diagonal
            diaglen = np.sqrt(2 * (360 ** 2))
            tbins = np.arange(-diaglen / 2., diaglen / 2. + 1, binsize)
            tb = tbins[:-1] + binsize / 2.
            db = bh.fc.binyfromx(ts, ds, tbins, np.median)
            idx = np.isnan(db) == False
            db = db[idx]
            tb = tb[idx]

            def diag2xy(t, d, b):
                a = np.sqrt((t ** 2) / 2.)
                a[t < 0] *= -1
                e = np.sqrt((d ** 2) / 2.)
                ex = copy.copy(e)
                ex[d > 0] *= -1
                ey = copy.copy(e)
                ey[d < 0] *= -1
                x = a + ex
                y = a + b + ey
                return x, y

            step = 5
            # Convert binned distances from diagonal to x, y coordinates, take slope
            x, y = diag2xy(tb, db, b)
            ytile = fc.tile(y[::step])
            xtile = fc.tile(x[::step])
            dy = np.gradient(ytile, np.gradient(xtile))
            # Find density of xposition at points defined by binning along the diagonal
            xposdensity = hinterpf(x[::step])
            slope = dy[len(y[::step]):-len(y[::step])]
        elif method == 'polyreg':
            p = np.poly1d(np.polyfit(fc.tile(xstimb, period), fc.tile(phases, period), 25))
            x = np.arange(-180, 181)
            dy = np.gradient(p(x), np.gradient(x))
            slope = np.array([dy[x == xbin] for xbin in xb]).ravel()
            xposdensity = h
        return xposdensity, slope

    # Collect slopes and densities for all recordings
    slopes = []
    xposdensities = []
    for rec in recs:
        tlim = None
        period = 360
        if rec.folder == '2015_03_27-004/': tlim = [100, 240]
        if rec.folder == '2015_03_27-008/': period = 270
        if rec.folder == '2015_04_14-003/': tlim = [0, 180]
        xposdensity, slope = get_phase_xpos_slope_xposdensity(rec, tlim=tlim, minspeed=.1, period=period)
        slopes.append(slope)
        xposdensities.append(xposdensity)
    xposdensities = np.concatenate(xposdensities)
    slopes = np.concatenate(slopes)

    # Plot data
    print 'npoints:', len(slopes)
    ax.set_ylabel('Slope\n(phase deg/bar deg)', rotation='horizontal', labelpad=80, size=14)
    ax.set_xlabel('Bar Position Density', labelpad=20, size=14)
    ax.scatter(xposdensities, slopes, c='none', lw=1, edgecolor='black')
    xlim = [xposdensities.min() - .001, xposdensities.max() + .001]
    ylim = [slopes.min() - .5, slopes.max() + .5]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.grid()

    # Fit data to line
    slope, intercept, r_value, p_value, std_err = stats.linregress(xposdensities, slopes)
    x = np.arange(-1, 2)
    y = slope * x + intercept
    plt.plot(x, y, c='red')
    text = 'R = %.3f\nR2 = %.3f\np = %.3e' % (r_value, r_value ** 2, p_value)
    plt.text(.006, 0, text, fontsize=12)

    if save:
        if len(recs) == 1: prefix += recs[0].abf.basename
        plt.savefig('slope_v_xposdensity%s.png' % suffix, bbox_inches='tight')


def plot_intensity_v_xstim(self, tlim=None, tlims=None, metric=np.sum, save=True):
    colors = [ph.nblue, red]

    bridge_intensity = metric(self.tif.gcn, axis=1)
    theta = self.abf.xstimb
    print theta.min(), theta.max()
    dforwb = self.subsample(self.abf.dforw)

    t = self.tif.t
    if tlims is None: tlims = [tlim]
    if tlims[0] is None: tlims[0] = [t[0], t[-1]]

    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    plt.figure(1, (8, 6))
    plt.suptitle(self.abf.basename.split(os.path.sep)[-1])
    ax = plt.subplot(111)
    plt.xlim(-180, 180)
    plt.xticks(np.arange(-180, 181, 45))
    plt.xlabel('Bar position (deg)')
    plt.ylabel('Intensity (au)')
    xbinsize = 20
    xbins = np.arange(-180, 181, xbinsize)
    xb = xbins[:-1] + xbinsize / 2.
    for i, tlim in enumerate(tlims):
        idx = (t >= tlim[0]) & (t < tlim[1])
        # idx = idx & (dforwb>.000001)
        c = colors[i]
        plt.scatter(theta[idx], bridge_intensity[idx], c='none', edgecolor=colors[i], s=20, alpha=.4)
        sum_mean = bh.fc.binyfromx(theta[idx], bridge_intensity[idx], xbins, np.mean, nmin=4)
        sum_std = bh.fc.binyfromx(theta[idx], bridge_intensity[idx], xbins, stats.sem, nmin=4)
        plt.errorbar(xb, sum_mean, yerr=sum_std, linestyle="None", c=c, zorder=2)
        plt.scatter(xb, sum_mean, s=30, c=c, zorder=3, lw=0, label='%i-%is' % (tlim[0], tlim[1]))
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles, labels, loc='upper right')

    plt.grid()

    if save:
        suffix = 'max' if metric is np.max else 'summed'
        plt.savefig('%s_%i-%is_%s_intensity_v_theta.png' % (self.abf.basename, tlim[0], tlim[1], suffix),
                    bbox_inches='tight')

def auto_plot(group, stim_dict, celltype, 
              t_hop_s = 15, **kwargs):
    
    # on a group with consistent cell type
    # iteratively runs the following through whole epochs
    # and through `t_hop_s` second-long hops through those epochs:
    # plot_rml_v_behav coloring leftz and rightz, comparing rml with dhead, + additional channels
    # plot_rml_v_behav coloring all glomeruli, comparing phase + xstim, + additional channels
    # also, within each epoch plots: Xcorr of R-L x dHead, Xcorr of R+L x dForw 
     
    # we want to plot isolated turns
    # we want xcorr (1) rml v dhead, (2) rml v dbar, (3) rPl v dforw, (4) rPl vdhead
    # in each epoch, we want rml_dhead_slope
    # in each epoch, we want a trajectory (1) with the rpl , and (2) with the rml
    # in all of these we want a figure in the plot category and in a folder with the fly
    
    for igen, gen in enumerate(group):
        for ifly, fly in enumerate(gen):
            for irec, rec in enumerate(fly):
                print '\n', '\n', rec.abf.basename.split('/')[-1]
                
                dirlist = []
                recdir = '/'.join(rec.basename.split('/')[:-1])+'/figures/'
                dirlist.append(recdir)
                rmlbehavdir = '/'.join(rec.basename.split('/')[0:2])+'/figures/overlay/rml_v_dhead/'
                dirlist.append(rmlbehavdir)
                rplbehavdir = '/'.join(rec.basename.split('/')[0:2])+'/figures/overlay/rpl_v_dforw/'
                dirlist.append(rplbehavdir)
                rmlphasedir = '/'.join(rec.basename.split('/')[0:2])+'/figures/overlay/phase_v_dbar/'
                dirlist.append(rmlphasedir)
                isodir = '/'.join(rec.basename.split('/')[0:2])+'/figures/isolated/'
                dirlist.append(isodir)
                xcrmldhdir = '/'.join(rec.basename.split('/')[0:2])+'/figures/xcorr_rml_dhead/'
                dirlist.append(xcrmldhdir)
                xcrpldfdir = '/'.join(rec.basename.split('/')[0:2])+'/figures/xcorr_rpl_dforw/'
                dirlist.append(xcrpldfdir)
                trajrpldir = '/'.join(rec.basename.split('/')[0:2])+'/figures/traj_rpl/'
                dirlist.append(trajrpldir)
                trajrmldir = '/'.join(rec.basename.split('/')[0:2])+'/figures/traj_rml/'
                dirlist.append(trajrmldir)
                slopedir = '/'.join(rec.basename.split('/')[0:2])+'/figures/rml_dhead_slope/'
                dirlist.append(slopedir)
                for d in dirlist:
                    if not os.path.exists(d):
                        os.makedirs(d)

                colordict = {'EIP': 'b', 'PEN':'o', 'SPSP':'seagreen', 'ISP':'crimson'}    
                epoch_dict = rec.get_epochs(epochs = stim_dict, **kwargs) # rec.get_epochs eventually...
                counter_dict = {}
                for key in epoch_dict.keys():
                    counter_dict[key] = 0 
                for (key, (stim_val, t_array)) in epoch_dict.items():
                    print '\n', ifly, irec, key
                    
                    rml_v_dhead_list = []
                    rpl_v_dforw_list = []                   
                    for [start_t, stop_t] in t_array:
                        abf_ind0 = np.where(rec.abf.t >= start_t)[0][0]
                        abf_ind1 = np.where(rec.abf.t >= stop_t)[0][0]
                        im_ind0 = np.where(rec.ims[0].t >= start_t)[0][0]
                        im_ind1 = np.where(rec.ims[0].t >= stop_t)[0][0]

                        print ifly, irec, 'Big Overlay'
                        im.plot_rml_v_behav(rec, tlim = [start_t, stop_t], 
                                           imaging = ['pb.c1.leftz', 'pb.c1.rightz'],
                                           im_overlay = 'pb.c1.rmlz', behav_overlay = 'dhead', 
                                           channels = ['abf.forw', 'abf.head'], show_puffs = True,
                                           fig_filename = rmlbehavdir + \
                                           '_'.join(['rml_v_behav', 'fly', str(ifly), \
                                                    'rec', str(irec), key, str(counter_dict[key]), 'all']))
                        plt.savefig(recdir + '_'.join(['rml_v_behav', 'fly', str(ifly), \
                                                       'rec', str(irec), key, str(counter_dict[key]), 'all']))
                        plt.close()

                        if celltype in ['EIP', 'PEN', 'ISP']:
                            # head v phase
                            if bar_bool:
                                b_o = 'xstim'
                                channels = ['abf.forw', 'abf.head']
                            else:
                                b_o = 'head'
                                channels = ['abf.forw']
                            im.plot_rml_v_behav(rec, tlim = [start_t, stop_t], 
                                       imaging = ['pb.c1.az'],
                                       behav_overlay = b_o, im_overlay = 'phasef', t_offset = start_t,
                                       channels = channels, show_puffs = True,
                                       fig_filename = rmlphasedir + '_'.join(['fluo_v_phase', 'fly', 
                                                    str(ifly), 'rec', str(irec), key, \
                                                     str(counter_dict[key]), 'all']))
                            plt.savefig(rmlphasedir + '_'.join(['fluo_v_phase', 'fly', 
                                                    str(ifly), 'rec', str(irec), key, \
                                                     str(counter_dict[key]), 'all']))
                            plt.close()

                        rml = rec.get_signal('rmlz')[-1][im_ind0:im_ind1]
                        rpl = rec.get_signal('rplz')[-1][im_ind0:im_ind1]
                        dhead = rec.get_signal('dhead')[-1][im_ind0:im_ind1]
                        dbar = rec.get_signal('dxstim')[-1][im_ind0:im_ind1]
                        dforw = rec.get_signal('dforw')[-1][im_ind0:im_ind1]

                        # XC rml v dhead (and dbar)
                        t, xc = fc.xcorr(rml, dhead, sampling_period=rec.ims[0].sampling_period)
                        if ('OL' in key) or ('ol' in key) or ('Ol' in key):
                            t2, xc2 = fc.xcorr(rml, dbar, sampling_period=rec.ims[0].sampling_period)
                        else:
                            t2, xc2 = [], []
                        rml_v_dhead_list.append((t, xc, t2, xc2))

                        # rpl v dforw, abs(dhead)
                        tf, xcf = fc.xcorr(rpl, dforw, sampling_period=rec.ims[0].sampling_period)
                        tabs, xcabs = fc.xcorr(rpl, np.abs(dhead), sampling_period=rec.ims[0].sampling_period)
                        rpl_v_dforw_list.append((tf, xcf, tabs, xcabs))
                        

                        # plot things in bouts of 15 seconds:
                        print ifly, irec, 'overlay bouts'
                        start_bout_t = start_t
                        end_bout_t = start_t + 15
                        while end_bout_t < (stop_t + 1): # added extra 1s for weird missing ind or something
                            # rml v behavior
                            if bar_bool:
                                channels = ['abf.forw', 'abf.xstim']
                            else:
                                channels = ['abf.forw', 'abf.head']  
                            im.plot_rml_v_behav(rec, tlim = [start_bout_t, end_bout_t], 
                                           imaging = ['pb.c1.leftz', 'pb.c1.rightz'],
                                           im_overlay = 'pb.c1.rmlz', behav_overlay = 'dhead', 
                                           channels = channels, show_puffs = True,
                                           fig_filename = rmlbehavdir + '_'.join(['rml_v_behav', 'fly', \
                                                        str(ifly), 'rec', str(irec), \
                                                        key, 'start', str(int(round(start_bout_t)))]))
                            plt.savefig(rmlbehavdir + '_'.join(['rml_v_behav', 'fly', str(ifly), \
                                        'rec', str(irec), key, 'start', str(int(round(start_bout_t)))]))
                            plt.close()

                            if celltype in ['EIP', 'PEN', 'ISP']:
                                # head v phase
                                if bar_bool:
                                    b_o = 'xstim'
                                    channels = ['abf.forw', 'abf.head']
                                else:
                                    b_o = 'head'
                                    channels = ['abf.forw']
                                im.plot_rml_v_behav(rec, tlim = [start_bout_t, end_bout_t], 
                                           imaging = ['pb.c1.az'],
                                           behav_overlay = b_o, im_overlay = 'phasef', t_offset = start_bout_t,
                                           channels = channels, show_puffs = True,
                                           fig_filename = rmlphasedir + '_'.join(['fluo_v_phase', 'fly', 
                                                        str(ifly), 'rec', str(irec), key, \
                                                        'start', str(int(round(start_bout_t)))]))
                                plt.savefig(rmlphasedir + '_'.join(['fluo_v_phase', 'fly', str(ifly), \
                                                        'rec', str(irec), key, 'start', \
                                                        str(int(round(start_bout_t)))]))
                                plt.close()
                            
                            start_bout_t += t_hop_s
                            end_bout_t += t_hop_s
                        
                        counter_dict[key] += 1
                    
                    print 'XC: R-L v dHead'
                    l = ['R-L v dHead']
                    title = 'Cross-correlation: R-L v dHead'
                    if ('OL' in key) or ('ol' in key) or ('Ol' in key):
                        l.append('R-L v dBar')
                        title += ', dBar'
                    plot_xcorrs(rml_v_dhead_list, labels = l, tlim = [-2, 2], ylim = (-.7, .3),
                                title = title + ' -- ' + key)
                    plt.savefig(xcrmldhdir + '_'.join(['xc_rml_dHead', str(ifly), \
                                                   str(irec), key, str(counter_dict[key])]))
                    plt.savefig(recdir + '_'.join(['xc_rml_dHead', 'fly', str(ifly), \
                                'rec', str(irec), key]))
                    plt.close()

                    print 'XC: R+L v dForw'
                    plot_xcorrs(rpl_v_dforw_list, labels = ['R+L v dForw', 'R+L v abs(dHead)'],
                                tlim = [-2, 2], ylim = (-.8, .8),
                                title = 'Cross-correlation: R+L v dForw, abs(dHead) -- ' + key)
                    plt.savefig(xcrpldfdir+ '_'.join(['xc_rpl_dForw', 'fly', str(ifly), \
                                                   'rec', str(irec), key]))
                    plt.savefig(recdir + '_'.join(['xc_rpl_dForw', 'fly', str(ifly), 'rec', \
                                str(irec), key]))
                    plt.close()

def plot_ol_trials(gen, ch_str, quantify = False, unwrap=False, ch_str2='',
                sigfunc=lambda x, y: x-y,
                t_plot = [-.5, 2.5], baseline_t = [0], ntrials=12,
                specific_glomeruli = [], walking_thresh=0, stasis_thresh=0, across_flies=True,
                color_dict = {'Stasis': 'grey', 'Right Slow':'tomato', 'Left Slow':'lightskyblue',
                  'Right Medium' :'red',  'Left Medium': 'blue',
                  'Right Fast' :'firebrick', 'Left Fast': 'navy'},
                nstim_dict = {'Stasis': 0, 'Right Slow': 1, 'Left Slow':2,
                  'Right Medium' :3,  'Left Medium':4,
                  'Right Fast' :5, 'Left Fast': 6},
                stim_dict = {'Stasis': 0.6, 'Left Slow':0.9, 'Right Slow':1.2, 'Left Medium' : 1.5,
                    'Right Medium': 1.8, 'Left Fast' : 2.1, 'Right Fast': 2.4},
                ylim = (), fig_filename = ''):
    x = np.arange(t_plot[0], t_plot[1], .02)

    # nflies x nstims x ntrials x len(trial)
    all_flies_all_trials = np.zeros((len(gen),
                                    len(stim_dict.keys()),
                                    ntrials,
                                    len(x)))

    ctr_mat = np.zeros((len(gen), len(stim_dict.keys())))
    all_flies_all_trials[:] = np.nan
    
    for ifly, rec in enumerate(gen):
        signal = rec.get_signal(ch_str)[-1]

        if stasis_thresh:
            is_static = _is_static(rec, dforw_thresh = 1, dhead_thresh = 50,
                                    sigma_s = .1)

        if specific_glomeruli:
            glomeruli = np.array(rec.pb.c1.xedges[:-1].tolist())
            cond_R = lambda g: ((g >= 9) & \
                                ([ind in specific_glomeruli for ind in g]))
            cond_L = lambda g: ((g < 9) & \
                                ([ind in specific_glomeruli for ind in g]))
            if ch_str.split('.')[-1][:3] == 'rml':
                op = lambda x, y: x - y
            elif ch_str.split('.')[-1][:3] == 'rpl':
                op = lambda x, y: x + y                    
            if ch_str.split('.')[-1][-1] == 'n':
                fluo_sig = rec.get_signal('pb.c1.an')[-1]
            elif ch_str.split('.')[-1][-1] == 'z':
                fluo_sig = rec.get_signal('pb.c1.az')[-1]
            signal = op(np.nanmean(fluo_sig[:, cond_R(glomeruli)], axis=1), 
                        np.nanmean(fluo_sig[:, cond_L(glomeruli)], axis=1)) 

        if ch_str[:3] == 'abf':
            t = rec.abf.t
            if unwrap:
                signal = fc.unwrap(signal)

        else:
            ind_imstart = np.where(rec.abf.t >= rec.ims[0].t[0])[0][0]
            ind_imstop = np.where(rec.abf.t >= rec.ims[0].t[-1])[0][0]
            indint = ind_imstop - ind_imstart
            if unwrap:
                signal = fc.dense_sample(fc.unwrap(signal), indint)
            else:
                signal = fc.dense_sample(signal, indint)
            t = fc.dense_sample(rec.ims[0].t, indint)

        if ch_str2:
            signal2 = rec.get_signal(ch_str2)[-1]
            if ch_str2[:3] == 'abf':
                t = rec.abf.t
                if unwrap:
                    signal2 = fc.unwrap(signal2)

            else:
                ind_imstart = np.where(rec.abf.t >= rec.ims[0].t[0])[0][0]
                ind_imstop = np.where(rec.abf.t >= rec.ims[0].t[-1])[0][0]
                indint = ind_imstop - ind_imstart
                if unwrap:
                    signal2 = fc.dense_sample(fc.unwrap(signal2), indint)
                else:
                    signal2 = fc.dense_sample(signal2, indint)
                t = fc.dense_sample(rec.ims[0].t, indint)

            signal = sigfunc(signal, signal2)

        eps = get_epochs_transl(rec, min_t_s = .5, max_t_s = 4, 
                                round_to = 0.15, tempchange=False,
                                epochs_ao = stim_dict, max_len = 100)

        for (key, val) in stim_dict.items():
            trctr = 0
            istim = nstim_dict[key]
            this_stim_trial_starts = [tlim[0] for tlim in eps[key][1]]
            for istart, start_t in enumerate(this_stim_trial_starts):

                if start_t > t[-1]:
                    continue
                    
                ind0 = np.where(t >= start_t)[0][0]
                ind1 = ind0 + 50*t_plot[1]
                ind_base = ind0 + 50*t_plot[0]

                if baseline_t:
                    if len(baseline_t) == 1:
                        sub_ind = np.where(x >= baseline_t[0])[0][0]
                        trial = signal[ind_base:ind1] - signal[sub_ind]
                    elif len(baseline_t) == 2:
                        sub_ind0 = np.where(x >= baseline_t[0])[0][0]
                        sub_ind1 = np.where(x >= baseline_t[1])[0][0]
                        trial = signal[ind_base:ind1] - np.nanmean(signal[sub_ind0:sub_ind1])
                else:
                    trial = signal[ind_base:ind1]
                    
                    
                if not len(trial) == 50*(t_plot[1]-t_plot[0]):
                    continue

                if stasis_thresh:
                    if sum(is_static[ind0:ind1]) >= stasis_thresh*(ind1-ind0):
                        all_flies_all_trials[ifly, istim, istart, :] = trial
                    else:
                        continue

                elif walking_thresh:
                    if sum(rec.abf.is_walking[ind0:ind1]) >= walking_thresh*(ind1-ind0):
                        all_flies_all_trials[ifly, istim, istart, :] = trial
                    else:
                        continue                        
                else:
                    all_flies_all_trials[ifly, istim, istart, :] = trial
                trctr += 1

            ctr_mat[ifly, istim] = trctr

    if quantify:
        quant_mat = np.zeros((len(stim_dict.items())))
        quant_mat[:] = np.nan
    for (key, val) in stim_dict.items():
        this_color = color_dict[key]
        istim = nstim_dict[key]
        if across_flies:
            avg = np.nanmean(np.nanmean(all_flies_all_trials[:, istim, :, :], axis=1),
                        axis=0)
            sem = np.nanstd(np.nanmean(all_flies_all_trials[:, istim, :, :], 
                        axis=1), axis=0) / \
                            np.sqrt(len(gen))
           
        else:
            a = all_flies_all_trials[:, istim, :, :]        
            a = np.reshape(a, (len(gen)*ntrials, len(x)))
            n_trials = 0
            for i in range(a.shape[0]):
                if not all([np.isnan(val) for val in a[i, :]]):
                    n_trials += 1
            avg = np.nanmean(a, axis=0)
            sem = np.nanstd(a, axis=0) / np.sqrt(n_trials)

        plt.plot(x, avg,
                     color = this_color, lw = 1.5)
        plt.fill_between(x, avg-sem, avg+sem,
                        color=this_color, lw=0, alpha = .3)
        
        if quantify:
            stim_start_ind = np.where(x >= 0)[0][0]
            quant_mat[istim] = np.mean(avg[stim_start_ind:]) - \
                                np.mean(avg[:stim_start_ind])

    print sorted(nstim_dict.items(), key =lambda x: x[1])

    if across_flies:
        print ctr_mat
    else:
        print np.sum(ctr_mat, axis=0)
    if ylim:
        plt.ylim(ylim)
        
    if fig_filename:
        plt.savefig(fig_filename)
    plt.show()
    if quantify:
        return all_flies_all_trials
    
def get_epochs_transl(rec, min_t_s = .2, max_t_s = 180, round_to = 0.1, tempchange = False,
                      channel = 'abf.ao',
               epochs_ao = {'Dark':0.6, 'OL right':0.9, 'OL left':1.2, 'CL':1.5},
               max_len = 100):

    """
    takes a recording, a minimum time in seconds, max time in seconds, 
    max_len: an int that is the maximum number of epochs you are expecting
    round_to: a float, a float you would like to round fly.ao to,
    and a `epochs` dictionary 
    whose values are floats rounded to nearest tenth representing the 
    value of rec.stimid_raw during the epoch defined by the key. 

    outputs a dictionary, where each key has a tuple value.
    the first value is the stimid of the epoch, 
    the second is an np.array of start and end INDICES 
    that the stimid_raw of the abf maintained this value for at least min_t_s
    """
    triggering_signal = rec.get_signal(channel)[-1]
    
    if tempchange:
        t = rec.abf.t_transition
    else:
        t = rec.abf.t
    
    output_dict = {}
    if round_to:
        rounded_stimid = np.array([round((round(i / round_to) * round_to), 1) \
                                    for i in triggering_signal])
        for (k, val) in epochs_ao.items():
            output_dict[k] = round((round(val / round_to) * round_to), 1)
    else:
        rounded_stimid = np.array([round(i, 1) for i in triggering_signal])
    t0 = np.where(t >= 0)[0][0]
    tmin = np.where(t >= min_t_s)[0][0]
    min_inds = tmin - t0

    for (k, val) in epochs_ao.items():
        if (rounded_stimid == val).any():
            cont_inds = fc.get_contiguous_inds(inds = (rounded_stimid == val),
                                               min_contig_len= min_inds)
            edge_ts = []
            for isub, sub in enumerate(cont_inds):
                if isub < (max_len) and (t[sub[-1]] - t[sub[0]]) < max_t_s:
                    edge_ts.append((t[sub[0]], t[sub[-1]]))
            edge_ts = np.array(edge_ts)
            output_dict[k] = (val, edge_ts)
        else:
            output_dict[k] = (val, [])
    return output_dict

def xc_group_by_lag(groups, gen_labels, im_ch, behav_ch,
                    specific_glomeruli = [[], [], [], []], 
                    colors = ['dodgerblue', 'orange', 'g', 'r'],
                    correlation_type = 'linear',
                    plot_indiv = False, ao_val = [.6], lags = np.arange(-1000, 2001, 50),
                    title_dict = {0.6:', Dark', 1.5:', Closed Loop', 3.3:'Star Field CL'},
                   mode = 'min', title = '', fig_filename = '',
                   ylim = (-.8, .4)):
    
    avgs = []
    all_outputs = []
    for ig, g in enumerate(groups):
        gen_output = []

        corrs = np.zeros((len(g), len(lags)))
        print corrs.shape
        corrs[:] = np.nan

        for ifly, fly in enumerate(g):
            fly_corrs = np.zeros((len(fly), len(lags)))
            fly_corrs[:] = np.nan
            if ifly >= len(g): break

            for irec, rec in enumerate(fly):

                rounded_ao = np.array([round(i, 1) for i in rec.subsample('abf.ao')])
                if not any([i in rounded_ao for i in ao_val]): continue 
                ind0 = np.where([i in ao_val for i in rounded_ao])[0][0]
                ind1 = np.where([i in ao_val for i in rounded_ao])[0][-1]
                print rec.pb.t[ind0], rec.pb.t[ind1]

                im_sig = rec.get_signal(im_ch)[1]
                if specific_glomeruli[ig] and im_ch.split('.')[-1][:3] in ['rml', 'rpl']:  
                    if im_ch.split('.')[-1][:3] == 'rml':
                        op = lambda x, y: x - y
                    elif im_ch.split('.')[-1][:3] == 'rpl':
                        op = lambda x, y: x + y
                    else:
                        continue
                    if im_ch.split('.')[-1][-1] == 'n':
                        fluo_sig = rec.get_signal('pb.c1.an')[-1]
                    elif im_ch.split('.')[-1][-1] == 'z':
                        fluo_sig = rec.get_signal('pb.c1.az')[-1]
                    else:
                        continue

                    glomeruli = np.array(rec.pb.c1.xedges[:-1].tolist())
                    cond_R = lambda g: ((g >= 9) & \
                                        ([ind in specific_glomeruli[ig] for ind in g]))
                    cond_L = lambda g: ((g < 9) & \
                                        ([ind in specific_glomeruli[ig] for ind in g]))
                    im_sig = op(np.nanmean(fluo_sig[:, cond_R(glomeruli)], axis=1),
                                  np.nanmean(fluo_sig[:, cond_L(glomeruli)], axis=1))

                corr = np.zeros(len(lags))
                for ilag, lag in enumerate(lags):
                    if correlation_type == 'linear':
                        if not np.shape(im_sig[ind0:ind1]) == np.shape(rec.subsample(behav_ch, lag_ms=lag)[ind0:ind1]):
                            continue
                        corr[ilag] = np.corrcoef(im_sig[ind0:ind1], 
                                  rec.subsample(behav_ch, lag_ms=lag)[ind0:ind1])[0, 1]
                        
                    elif correlation_type == 'circular':
                        if not np.shape(im_sig[ind0:ind1]) == np.shape(rec.subsample(behav_ch, lag_ms=lag)[ind0:ind1]):
                            continue
                        corr[ilag] = cs.corrcc(np.deg2rad(rec.subsample(behav_ch, lag_ms=lag)[ind0:ind1]),
                                        np.deg2rad(im_sig[ind0:ind1]))

                fly_corrs[irec] = corr
                

            corrs[ifly] = np.nanmean(fly_corrs, axis=0)

            if plot_indiv:
                plt.plot(lags, np.nanmean(fly_corrs, axis=0), color=colors[ig], lw=.4)

            if mode == 'min':
                gen_output.append(round(np.nanmin(np.nanmean(fly_corrs, axis=0)), 3))
            elif mode == 'max':
                gen_output.append(round(np.nanmax(np.nanmean(fly_corrs, axis=0)), 3))
            elif mode == 'max':
                gen_output.append(np.nanmean(fly_corrs, axis=0))
            elif type(mode) in [int, float]:
                ind = np.where(lags >= mode)[0][0]
                gen_output.append((round(np.nanmean(fly_corrs, axis=0)[ind], 3)))

            
        mean = np.nanmean(corrs, axis=0)

        stderr = np.nanstd(corrs, axis=0) / np.sqrt(len(g))
        plt.plot(lags, mean, c=colors[ig], lw=3, label=gen_labels[ig])
        
        if mode == 'min':
            plt.gca().axvline(lags[mean==np.nanmin(mean, axis=0)], c = colors[ig], lw =.4)
            print lags[mean==np.nanmin(mean)]
        elif mode == 'max':
            plt.gca().axvline(lags[mean==np.nanmax(mean, axis=0)], c = colors[ig], lw =.4)
            print lags[mean==np.nanmax(mean)]
            
        
        if not plot_indiv:
            plt.fill_between(lags, mean-stderr, mean+stderr, color=colors[ig], 
                             lw=0, alpha=0.2)

        if mode == 'all':
            gen_output = corrs
        all_outputs.append(gen_output)
    plt.xlim(lags[0], lags[-1])
    plt.ylim(ylim)
    plt.legend(fontsize = 6.5, frameon=False, loc=4)
    plt.gca().axhline(0, c='k', ls='--', lw = .8)
    plt.gca().axvline(0, c='k', ls='--', lw = .8)
    plt.gca().set_xlabel('Lag (ms)', fontsize = 14)
    if correlation_type == 'linear':
        plt.gca().set_ylabel('Cross-Correlation', fontsize = 13)
    elif correlation_type == 'linear':
        plt.gca().set_ylabel('Circular Cross-Correlation', fontsize = 13)
    for v in ao_val:
        title += title_dict[v]
    plt.gca().set_title(title, fontsize = 14, y = 1.03)
    plt.show()
    return all_outputs
    
def gain_vs_lag(groups, ao_val, lags = np.arange(-1000, 2001, 50), behav_str = 'abf.dhead', im_str = 'pb.c1.dphase',
                title_dict = {0.6:'Dark, ', 1.5:'Closed Loop, ', 3.3:'Star Field Closed Loop, '},
                colors = ['k', 'r'], gen_labels = ['Shi / +, 32C', 'Ib.Sps-P > Shi, 32C'],
                filter_type = ndimage.filters.gaussian_filter1d, filter_behav=[],
                statistics= False, quantify = False, mode= 250):
    
    
    whole_expt_gains = []
    for ig, g in enumerate(groups):
        gen_gains = np.zeros((len(g[0]), 3, len(lags)))
        #gen_gains = np.zeros((len(g[0]), len(g[0][0]), len(lags)))
        gen_gains[:] = np.nan
        for ifly, fly in enumerate(g[0]):
            for irec, rec in enumerate(fly):
                if irec >= len(fly): break
                rounded_ao = np.array([round(i, 1) for i in rec.subsample('abf.ao')])
                for val in ao_val:
                    if not val in rounded_ao: continue
                    ind0 = np.where(rounded_ao == val)[0][0]
                    ind1 = np.where(rounded_ao == val)[0][-1]
                    print rec.pb.t[ind0], rec.pb.t[ind1]

                    im_signal = rec.get_signal(im_str)[-1]
                        
                    for ilag, lag in enumerate(lags):
                        if filter_behav:
                            gen_gains[ifly, irec, ilag] = \
                            stats.linregress(filter_type(rec.subsample(behav_str, lag_ms=lag),
                                                        rec.pb.sampling_rate*filter_behav[0])[ind0:ind1],
                                                im_signal[ind0:ind1])[0]
                        
                        else:
                            gen_gains[ifly, irec, ilag] = \
                            stats.linregress(rec.subsample(behav_str, lag_ms=lag)[ind0:ind1],
                                                im_signal[ind0:ind1])[0]
    
                    plt.plot(lags, gen_gains[ifly, irec, :], c = colors[ig], lw = .2)
        whole_expt_gains.append(gen_gains)
        mean_gain = np.nanmean(np.nanmean(gen_gains, axis=1), axis=0)
        plt.plot(lags, mean_gain, c=colors[ig], lw=2,
                label = gen_labels[ig])
    plt.legend(fontsize = 6.5, frameon=False, loc=4)
    plt.gca().axhline(0, c='gray', ls='--', lw = .4)
    plt.gca().axvline(0, c='gray', ls='--', lw = .4)
    plt.gca().set_xlabel('Lag (ms)', fontsize = 14)
    plt.gca().set_ylabel('dPhase / dHead Slope', fontsize = 13)
    title = ' '.join(['Gain vs lag,', title_dict[ao_val[0]], '\n'])
    title+=(', '.join(gen_labels))
    plt.gca().set_title(title, fontsize = 14, y = 1.03)
    plt.show()
    
    mins = []
    maxes = []
    t_mins = []
    t_maxes = []
    for g in whole_expt_gains:
        
        # calculate the min and max values, and the times at which they occur
        # if mode is 'min', 'tmin', 'max', or 'tmax', output these values
        # if the mode is an int or float, output the value at this lag
        
        this_min = np.nanmin(np.nanmean(g, axis = 1), axis = 1)
        mins.append(this_min)
        this_max = np.nanmax(np.nanmean(g, axis = 1), axis = 1)
        maxes.append(this_max)
        
        gen_t_mins = []
        gen_t_maxes = []
        for ifly in range(g.shape[0]):
            avg_fly = np.nanmean(g, axis=1)[ifly, :]
            this_t_min = lags[np.where(avg_fly == avg_fly.min())]
            this_t_max = lags[np.where(avg_fly == avg_fly.max())]
            gen_t_mins.append(this_t_min)
            gen_t_maxes.append(this_t_max)
        t_mins.append(gen_t_mins)
        t_maxes.append(gen_t_maxes)
        
    if quantify:
        return whole_expt_gains
    
    if statistics:
            
        print gen_labels
        print 'min val:', [round(np.mean(i), 3) for i in mins], 'p =', \
                            round(stats.mannwhitneyu(mins[0], mins[1])[1], 3)
        print 't min:', [round(np.mean(i), 3) for i in t_mins], 'p =', \
                            round(stats.mannwhitneyu(t_mins[0], t_mins[1])[1], 3)
        print 'max val:', [round(np.mean(i), 3) for i in maxes], 'p =',\
                            round(stats.mannwhitneyu(maxes[0], maxes[1])[1], 3)
        print 't max:', [round(np.mean(i), 3) for i in t_maxes], 'p =',\
                            round(stats.mannwhitneyu(t_maxes[0], t_maxes[1])[1], 3)
    

def glom_glom_xcorr(group, im_str, stim):

    correl_gloms = np.zeros((18, 18))
    for glom1 in range(18):
        for glom2 in range(18):
            r_list = []
            for fly in group[0]:
                for rec in fly:
                    _, gc = rec.get_signal(im_str)
                    if stim:
                        rounded_ao = np.array([round(i, 1) for i in rec.subsample('abf.ao')])
                        ind0 = np.where(rounded_ao == stim)[0][0]
                        ind1 = np.where(rounded_ao == stim)[0][-1]
                    else:
                        ind0 = 0
                        ind1 = gc.shape[0]                
                    r, p = stats.pearsonr(gc[ind0:ind1, glom1], gc[ind0:ind1, glom2])
                    r_list.append(r)
            correl_gloms[glom1, glom2] = np.nanmean(r_list)
    
    correl_gloms[np.isnan(correl_gloms)] = 0
    img = plt.pcolormesh(np.arange(np.shape(gc)[1]+1), np.arange(np.shape(gc)[1]+1),
                  correl_gloms, cmap = plt.cm.get_cmap('jet'),
                  vmin = -.5, vmax = 1.)
    cb = plt.colorbar(img, ax=plt.gca(), aspect=6, fraction = .075)
    cb.set_ticks([-1., -0.5, 0, 0.5, 1.0])
    plt.show()
    
def group_x_vs_y(group, x_cha, y_cha, lags_ms = [0, 0], 
                   separate_indiv_flies = True, avg_binned = True, regress = False,
                 plot_indiv=False,
                 bin_edges = [-1., 0., 1., 2., 4., 6., 8.],
                   abs_x = False, abs_y = False,
                 condition_func = None, activity_func = None, ao_vals = [],
                 specific_glomeruli = [],
                 ylim = (), xlim = (), ylabel = '', xlabel = '',
                 cmap = plt.cm.get_cmap('jet'),
                 color = '', output = False,
                fig_filename = []):

    if not xlabel: xlabel = x_cha
    if not ylabel: ylabel = y_cha
    all_y = []
    all_x = []
    
    all_y_binned = np.zeros((len(group[0]), len(bin_edges)-1))
    all_y_binned[:] = np.nan
    
    if abs_x: xlabel = 'absolute value of ' + xlabel
    if abs_y: ylabel = 'absolute value of ' + ylabel
        
    for ifly, fly in enumerate(group[0]):
        fly_y = []
        fly_x = []
        for irec, rec in enumerate(fly):
            x = rec.get_signal(x_cha)[-1]
            y = rec.get_signal(y_cha)[-1]
            
            # treat rml or rpl differently if you only care about specific glomeruli
            if specific_glomeruli and \
                any([cha.split('.')[-1][:3] in ['rml', 'rpl'] for cha in [x_cha, y_cha]]):
                for icha, cha in enumerate([x_cha, y_cha]):
                    if not cha.split('.')[-1][:3] in ['rml', 'rpl'] :continue

                    xy_list = [x, y]
                    
                    # glomeruli are in range of [0, 18]
                    # set up conditionals for determining left and right
                    # left glomeruli are < 9, right glomeruli are >= 9
                    glomeruli = np.array(group[0][0][0].pb.c1.xedges[:-1].tolist())
                    cond_L = lambda g: ((g < 9) & \
                                        ([ind in specific_glomeruli for ind in g]))
                    cond_R = lambda g: ((g >= 9) & 
                                        ([ind in specific_glomeruli for ind in g]))
                    
                    if ifly == 0 and irec == 0:
                        print cha.split('.')[-1][:3]
                        print glomeruli[cond_R(glomeruli)], glomeruli[cond_L(glomeruli)]
                    
                    # get signal from each individual glomerulus
                    if cha.split('.')[-1][-1] == 'n':
                        alabel = 'an'
                    elif cha.split('.')[-1][-1] == 'z':
                        alabel = 'az'
                    ab = rec.get_signal('pb.c1.' + alabel)[-1]
                    
                    # depending on whether rpl or rml, add or subtract right and left
                    if cha.split('.')[-1][:3] == 'rpl':
                        signal = np.nanmean(ab[:, cond_R(glomeruli)], axis=1) + \
                                    np.nanmean(ab[:, cond_L(glomeruli)], axis=1)                        
                    elif cha.split('.')[-1][:3] == 'rml':
                        signal = np.nanmean(ab[:, cond_R(glomeruli)], axis=1) - \
                                    np.nanmean(ab[:, cond_L(glomeruli)], axis=1)
                    
                    # output this signal to x or y
                    xy_list[icha] = signal
                
                x = xy_list[0]
                y = xy_list[1]
                    
            if x_cha[:3] == 'abf' and not(y_cha[:3] == 'abf'):
                x = rec.subsample(x_cha, lag_ms=lags_ms[0])
                
            if y_cha[:3] == 'abf' and not(x_cha[:3] == 'abf'):
                y = rec.subsample(y_cha, lag_ms=lags_ms[1])
            
            if abs_x: x = np.abs(x)
            if abs_y: y = np.abs(y)
            
            if condition_func:
                condition = condition_func(rec)
            else:
                condition = np.ones(len(x)).astype(bool)
            if ao_vals:        
                if x_cha[:3] == 'abf' and y_cha[:3] == 'abf':
                    rao = np.array([round(i, 1) for i in rec.abf.ao])
                else:
                    rao = np.array([round(i, 1) for i in rec.subsample('abf.ao')])
                ao_array = np.array([val in ao_vals for val in rao])
                condition = condition*ao_array
            if activity_func:
                condition = condition*activity_func(rec)
                
            
            all_x.extend(x[condition])
            all_y.extend(y[condition])
            
            fly_x.extend(x[condition])
            fly_y.extend(y[condition])
        fly_x = np.array(fly_x)
        fly_y = np.array(fly_y)
        if regress:
            plt.scatter(fly_x[is_active], fly_y[is_active], 
                        lw=0, c = 'grey', alpha = .2)
            
        elif separate_indiv_flies:
            x_binned = []
            fly_y_binned = []
            fly_y_binned_sem = []
            if color:
                color = color
            else:
                color = cmap((ifly+.5)/len(group[0]))
            for i, left_edge in enumerate(bin_edges[:-1]):
                right_edge = bin_edges[i+1]
                x_binned.append(np.mean([left_edge, right_edge]))
                fly_y_this_bin = fly_y[(right_edge > fly_x) * (fly_x >= left_edge)]
                fly_y_binned.append(np.nanmean(fly_y_this_bin))
                fly_y_binned_sem.append(np.nanstd(fly_y_this_bin) / \
                                        np.sqrt(len(fly_y_this_bin)))
            
            all_y_binned[ifly, :] = fly_y_binned
            
            if plot_indiv:
                plt.errorbar(x_binned, fly_y_binned, yerr = fly_y_binned_sem,
                         c = color, lw = 0.6, alpha = .8)
    if avg_binned:
        all_y_binned_sem = np.zeros((len(bin_edges) - 1))
        for i in range(len(bin_edges)-1):
            if [val for val in all_y_binned[:, i] if not(np.isnan(val))]:
                n_clean = len([val for val in all_y_binned[:, i] if not(np.isnan(val))])
                all_y_binned_sem[i] = np.nanstd(all_y_binned[:, i], axis = 0) / \
                                    np.sqrt(n_clean)
        
        if not output:
            if plot_indiv:
                plt.errorbar(x_binned, np.nanmean(all_y_binned, axis = 0),
                     yerr = all_y_binned_sem,
                         lw = 3.5, c = color)
            else:
                print x_binned
                print np.nanmean(all_y_binned, axis = 0)
                plt.plot(x_binned, np.nanmean(all_y_binned, axis = 0),
                         lw = 3.5, c = color)
                plt.gca().fill_between(x_binned,
                                    np.nanmean(all_y_binned, axis = 0)-all_y_binned_sem,
                                     np.nanmean(all_y_binned, axis = 0)+all_y_binned_sem,
                                      lw=0, facecolor=color, alpha=0.3)
      
    if not output:
        if xlim: plt.gca().set_xlim(xlim)
        if ylim: plt.gca().set_ylim(ylim)

        if regress:
            slope, intercept, r, p, stderr = sc.stats.linregress(all_x, all_y)
            plt.plot(x, slope*x + intercept, color = 'gold')

        plt.gca().set_xlabel(xlabel)
        plt.gca().set_ylabel(ylabel)
        if fig_filename:
            plt.savefig(fig_filename)
        plt.show()
    elif output:
        return np.nanmean(all_y_binned, axis=0), all_y_binned_sem
    
    
def nanclean_2d(mat, metric = lambda m: np.any(np.isnan(m))):
    outmat = np.zeros((np.shape(mat)))
    ctr = 0
    for irow in range(np.shape(mat)[0]):
        vec = mat[irow, :]
        if not metric(vec):
            outmat[ctr, :] = vec
            ctr += 1
    return outmat[:ctr, :]
    
    
def plot_binned_bar_jump(gen, t_plot, im_t_bins,
                         channels = ['abf.xstim', 'pb.c1.phase', 'abf.head', 'abf.forw'],
                         unwrap = [False, False, True, True],
                         colors = ['k', 'dodgerblue', 'firebrick', 'navy'],
                        all_trials = True, spaghetti=False, plot_mean = True,
                     cond = True, adh_thresh = 50., df_thresh = 1.5, thresh_t_s = [-1.5, 9],
                         degs = [5, 10, 20, 30, 45, 60, 90], baseline_t = [0],
                        ylims = [[-10, 40], [-30, 30], [-10, 60], [-50, 250]], figsize=(12, 6),
                        dark_times = [[-.1, 5.1], [9.9, 15.1]], jump_time_list = [[5, 10]], ytickses= [], xticks = [-5, 0, 5, 10, 15, 20],
                        abf_frame_rate = 20,
                         print_ctr=False, trials2exclude = {}, fig_filename = ''):
    
    # X-axes upon which abf and imaging data will be plotted
    x_abf = np.arange(t_plot[0], t_plot[1], 1./abf_frame_rate)
    x_im = np.array([np.mean([a, b]) for (a, b) in zip(im_t_bins[:-1], im_t_bins[1:])])

    # Get the indices in which to apply the threshold
    thresh0 = int(thresh_t_s[0]*abf_frame_rate)
    thresh1 = int(thresh_t_s[1]*abf_frame_rate)

    # avg_channels is a list of arrays. Each array is shape (flies, trials, timepoints, number of event categories)
    avg_channels = []
    ctrs = []
    for cha_str in channels:
        if cha_str[:3] == 'abf':
            cha_mat = np.zeros((len(gen), 30, (t_plot[1]-t_plot[0])*abf_frame_rate, len(degs)))
        else:
            cha_mat = np.zeros((len(gen), 30, len(im_t_bins)-1, len(degs)))
        cha_mat[:] = np.nan
        avg_channels.append(cha_mat)
    
        # Since the above arrays are padded with nans, these ctr arrays will carry counters of valid trials
        ctr = np.zeros((len(gen), len(degs)))
        ctrs.append(ctr)
    
    
    for irec, rec in enumerate(gen):

        # get time vectors
        t = rec.abf.t
        t_im = rec.pb.t
        
        # dForw and abs dHead for thresholding
        dforw = np.squeeze(rec.abf.dforw)
        dhead_abs = np.abs(np.squeeze(rec.abf.dhead))
        
        # Get behavior and imaging vectors. Store in channel_vectors
        channel_vectors = []
        for icha, cha_str in enumerate(channels):
            # Get the signal for the channel
            # If the signal is 'offset', calculate the offset between xstim and phase
            if cha_str.lower() == 'offset':
                phase_im = np.squeeze(rec.pb.c1.phase[:len(t_im)])
                xstim_im = np.squeeze(rec.subsample('abf.xstim', lag_ms=0)[:len(t_im)])
                if len(xstim_im) < len(phase_im):
                    phase_im = phase_im[:len(xstim_im)]
                if unwrap[icha]:
                    phase_im = fc.unwrap(phase_im_wr)
                    xstim_im = fc.unwrap(xstim_im_wr)
                    cha = phase_im - xstim_im
                else:
                    cha = fc.wrap(phase_im - xstim_im)
            else:
                cha = rec.get_signal(cha_str)[-1]
            
                # Unwrap the appropriate channels
                if unwrap[icha]:
                    cha = fc.unwrap(np.squeeze(cha))
            channel_vectors.append(cha)
            
        
        for ideg, deg in enumerate(degs):
            # Get Jump indices from stim_inds
            # Special handling for the 180 degree case
            if type(deg) == str:
                if deg[-3:] == 'neg':
                    jump_inds_pos = []
                    jump_inds_neg = rec.abf.stim_inds[deg]['inds']
                else:
                    jump_inds_neg = []
                    jump_inds_pos = rec.abf.stim_inds[deg]['inds']
            if deg == 180:
                jump_inds_neg = []
                jump_inds_pos = rec.abf.stim_inds[180]['inds']
            elif deg == -180:
                jump_inds_pos = []
                jump_inds_neg = rec.abf.stim_inds[180]['inds']
            else:
                jump_inds_pos = rec.abf.stim_inds[deg]['inds']
                jump_inds_neg = rec.abf.stim_inds[-1*deg]['inds']
            
            # jump_inds is a list of stimulus start indices, -/+   
            jump_inds = list(sorted(fc.flatten([jump_inds_neg, jump_inds_pos])))
            
            for ii, ind in enumerate(jump_inds):
                
                ind_im = np.where(rec.pb.t >= rec.abf.t[ind])[0][0]
                jump_t_im = t_im[ind_im]
                
                # EXCLUSION CONDITIONS
                # if imaging stopped before the bar reappeared, skip
                # if you're conditioning and the trial doesn't meet the conditions, skip
                if irec in trials2exclude.keys():
                    if any([(a <= jump_t_im < b) for (a, b) in trials2exclude[irec]]):
                        continue
                if np.ceil(ind_im+t_plot[-1]*rec.pb.sampling_rate) >= len(t_im):
                    continue
                if cond and \
                    not(all(dhead_abs[ind+thresh0 : ind+thresh1] < adh_thresh) and \
                    all(dforw[ind+thresh0 : ind+thresh1] < df_thresh)):
                    continue
                    
                ctr[irec, ideg] = ctr[irec, ideg] + 1
            
                for icha, cha_str in enumerate(channels):
                    cha = channel_vectors[icha]
                    if cha_str[:3] == 'abf':
                        cha_type = 'abf'
                    else:
                        cha_type = 'im'
                    
                    # flip trials in which bar jumped counter-clockwise
                    if ind in jump_inds_pos:
                        orient = lambda x: x
                    elif ind in jump_inds_neg:
                        orient = lambda x: -x
                        
                    # if the data are unwrapped, use a simple mean to determine baseline values. Otherwise, use the circular mean
                    if unwrap[icha]:
                        mean = lambda x: np.mean(x)
                    else:
                        mean = lambda x: stats.circmean(x, high=180, low=-180)
                    
                    # Get the baseline values to zero. These are the TRIAL INDICES.
                    if len(baseline_t) == 2:
                        if cha_type == 'abf':
                            base0, base1 = [np.where(x_abf >= tt)[0][0] for tt in baseline_t]
                        elif cha_type == 'im':
                            base0, base1 = [np.where(np.round(x_im, 1) <= tt)[0][-1] for tt in baseline_t]
                    elif len(baseline_t) == 1:
                        if cha_type == 'abf':
                            base0 = np.where(x_abf >= baseline_t[0])[0][0]
                            base1 = base0 + 1
                        if cha_type == 'im':
                            base0 = np.where(np.round(x_im, 1) <= baseline_t[0])[0][-1]
                            base1 = base0 + 1
                    else:
                        base0 = 0
                        base1 = 1  
                    baseline = slice(base0, base1)
                    
                    # get the actual trial
                    # abf trials are just slices of the original channel vector
                    if cha_type == 'abf':
                        ind0 = np.where(t >= t[ind]+t_plot[0])[0][0]
                        ind1 = ind0 + (t_plot[1]-t_plot[0])*abf_frame_rate
                        if ind1 > len(rec.abf.t):
                            continue
                        trial = cha[ind0:ind1]
                        
                    # imaging channels are binned averages (bin boundaries defined by im_t_binds)
                    # timepoints from imaging channels are moved to the right by half an imaging bin-width
                    elif cha_type == 'im':
                        
                        trial = np.zeros((len(im_t_bins)-1))
                        trial[:] = np.nan
                        for j in range(len(im_t_bins)-1):
                            left_t = im_t_bins[j]
                            right_t = im_t_bins[j+1]
                            if jump_t_im+right_t +rec.pb.subsampling_period > t_im[-1]:
                                continue
                     
                            left_ind = np.where(t_im >= jump_t_im + left_t - .5*rec.pb.subsampling_period)[0][0]
                            right_ind = np.where(t_im >= jump_t_im + right_t - .5*rec.pb.subsampling_period)[0][0]
                            trial[j] = mean(cha[left_ind:right_ind])
                                           
                    # Every trial has its baseline subtracted appropriately and is reoriented according to jump direction
                    trial = orient(trial - mean(trial[baseline]))
                    
                    # trials are placed in the matrices containing them
                    avg_channels[icha][irec, ii, :, ideg] = np.squeeze(trial)
    
    # Draw the figure
    plt.figure(1, figsize)
    gs = gridspec.GridSpec(len(channels), len(degs))
    gs.update(hspace = .4, wspace = .35)
    
    for icha, cha_str in enumerate(channels):
        print 'plotting', cha_str
        n = 0.
        pltctr = 0.
        avg_channel = avg_channels[icha]
        if cha_str[:3] == 'abf':
            cha_type = 'abf'
            x = copy.copy(x_abf)
        else:
            cha_type = 'im'
            x = copy.copy(x_im)
        
        # if the data are unwrapped, use a simple mean and SEM. Otherwise, use the circular mean and circular var
        if unwrap[icha]:
            mean = lambda x, a: np.nanmean(x, axis=a)
            var = lambda x, a, n: np.nanstd(x, axis=a) / np.sqrt(n)
        else:
            mean = lambda x, a: stats.circmean(nanclean_2d(x), high=180, low=-180, axis=a)
            var = lambda x, a, n: stats.circstd(nanclean_2d(x), high=180, low=-180, axis=a)
            
        # Now you have a list len(channels) long
        # each element is a matrix of shape (flies, trials, timepoints, number of event categories)
        # Reshape or take a mean within flies, depending on how you're averaging
        sh = np.shape(avg_channel)
        if all_trials:            
            newsh = (sh[0]*sh[1], sh[2], sh[3])
            avg_channel = avg_channel.reshape(newsh)
        else:
            avg_channel = np.nanmean(avg_channel, axis=1)
    
        for ideg, deg in enumerate(degs):
            # count non-nan
            if all_trials:
                n=0
                for irow in range(avg_channel.shape[0]):
                    if not all([np.isnan(val) for val in avg_channel[irow, :, ideg]]):
                        n+=1
            else:
                n = len(gen)
            
            ax = plt.subplot(gs[icha, ideg])
            gs.update(hspace=.3)
            
            # MASK XSTIM AND OFFSET DURING DARK WITH NANS
            if cha_str in ['xstim', 'offset', 'abf.xstim']:
                on = np.ones(x.shape, dtype=bool)
                for sublist in dark_times:
                    if sublist[0] < x[0]:
                        left_ind = 0
                    else:
                        left_ind = np.where(x>=sublist[0])[0][0]   
                    if sublist[1] >= x_abf[-1]:
                        right_ind = len(x) - 1
                    else:
                        right_ind = np.where(x>=sublist[1])[0][0]
                    on[left_ind:right_ind] = False
                    
                #MASK X
                x[~on] = np.nan
            
            # If spaghetti, plot every trial (use color map)
            # If spaghett and plot_mean, plot every trial + thick line for the mean
            # Otherwise, plot the mean and sem
            if spaghetti:
                # MASK XSTIM etc
                pltctr = 0.
                for itrial in range(avg_channel.shape[0]):
                    if plot_mean:
                        this_color = colors[icha]
                        this_lw = 0.8
                        this_alpha = 0.5
                    else:
                        this_color = (colors[icha])(pltctr/n)
                        this_lw = 1.2
                        this_alpha = 0.75
                    if cha_str in ['xstim', 'offset', 'abf.xstim']:
                        masked_trial = copy.copy(avg_channel[itrial, :, ideg])
                        masked_trial[~on] = np.nan
                        #ax.plot(x, masked_trial, c=this_color, lw=this_lw, alpha=this_alpha)
                        if pltctr > n:
                            print 'Counter Overflow'
                            
                        ph.circplot(x, masked_trial,
                                    circ='y', ax=ax, c=this_color, lw=this_lw, alpha=this_alpha)
                    
                    trial = avg_channel[itrial, :, ideg]
                    ph.circplot(x, trial,
                                    circ='y', ax=ax, c=this_color, lw=this_lw, alpha=this_alpha)
                    #ax.plot(x, trial, color = this_color, lw=this_lw, alpha=this_alpha)
                    
                    if not np.all(np.isnan(trial)):
                        pltctr += 1.
                        
                if plot_mean:
                    mean_cha = mean(avg_channel[:, :, ideg], 0)
                    # COPY AND MASK MEAN
                    if cha_str in ['xstim', 'offset', 'abf.xstim']:
                        mean_cha_c = copy.copy(mean_cha)
                        mean_cha_c[~on] = np.nan
                        ax.plot(x, mean_cha_c, color=colors[icha], lw=2)
                    else:
                        ax.plot(x, mean_cha, color=colors[icha], lw=2)
                    
                    if cha_str in ['phase', 'pb.c1.phase']:
                        for jts in jump_time_list:
                            ax.plot([jts[0], jts[1]], [np.abs(deg), np.abs(deg)], ls='--', color='r', lw=1.75, alpha=0.6)
           
            else:
                mean_cha = mean(avg_channel[:, :, ideg], 0)
                var_cha = var(avg_channel[:, :, ideg], 0, n)
                #var_cha = np.zeros((avg_channel.shape[1]))
                
                #for i in range(nanclean_2d(avg_channel).shape[1]):
                    #var_cha[i] = np.rad2deg(cs.confmean(np.deg2rad(nanclean_2d(avg_channel)[:, i, ideg]), ci=0.95))
                
                
                # COPY AND MASK MEAN and SEM XSTIM
                if cha_str in ['xstim', 'offset', 'abf.xstim']:
                    mean_cha_c = copy.copy(mean_cha)
                    var_cha_c = copy.copy(var_cha)
                    mean_cha_c[~on] = np.nan
                    var_cha_c[~on] = np.nan
                     
                    ax.plot(x, mean_cha_c, color=colors[icha])
                    ax.fill_between(x, (mean_cha_c-var_cha_c), (mean_cha_c+var_cha_c),
                                     color=colors[icha], lw=0, alpha=0.3)
                
                else:
                    ax.plot(x, mean_cha, color=colors[icha])
                    ax.fill_between(x, (mean_cha-var_cha), (mean_cha+var_cha),
                         color=colors[icha], lw=0, alpha=0.3)
                
            if ideg == 0:
                ax.set_ylabel(cha_str)
                if icha == len(channels)-1:
                    ax.set_xlabel('Time (s)')
                    ax.set_xticks(xticks)
                if icha < len(ytickses):
                    if ytickses[icha]:
                        ax.set_yticks(ytickses[icha])
                else:
                    ax.set_yticks(np.linspace(ylims[icha][0], ylims[icha][1], 5))
            else:
                labels = [item.get_text() for item in ax.get_yticklabels()]                
                empty_string_labels = ['']*len(labels)
                ax.set_yticklabels(empty_string_labels)
            if icha != len(channels)-1:
                labels = [item.get_text() for item in ax.get_xticklabels()]                
                empty_string_labels = ['']*len(labels)
                ax.set_xticklabels(empty_string_labels)
                        
            ax.axvline(0, ls='--', color='gray', lw=0.8)
            ax.axhline(0, ls='--', color='gray', lw=0.8)
            for dtl in dark_times:
                dt0 = dtl[0]
                dtwidth = dtl[1]-dtl[0]
                ax.add_patch(patches.Rectangle((dt0, ylims[icha][0]), dtwidth, ylims[icha][1] - ylims[icha][0],
                        facecolor='k', alpha=0.15, linewidth=0))
                
                #ax.add_patch(patches.Rectangle((dt0, ylims[icha][0]), 5.1, ylims[icha][1] - ylims[icha][0],
                #        facecolor='k', alpha=0.15, linewidth=0))
                #ax.add_patch(patches.Rectangle((10, ylims[icha][0]), 5.1, ylims[icha][1] - ylims[icha][0],
                #        facecolor='k', alpha=0.15, linewidth=0))
            ax.set_xlim(t_plot)
            ax.set_ylim(ylims[icha])
    
    if fig_filename:
        plt.savefig(fig_filename)
    plt.show()
    
    if print_ctr:
        print '\n'
        print np.sum(ctr, axis=0),'\n', ctr
    
    outs = {}
    for icha, cha_str in enumerate(channels):
        outs[cha_str] = avg_channels[icha]
    outs['counter'] = ctr
    outs['x_abf'] = x_abf
    outs['x_im'] = x_im
    return outs

    


        
