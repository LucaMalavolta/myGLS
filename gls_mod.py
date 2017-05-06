#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2012 Sebastian Schröter, Stefan Czesla, and Mathias Zechmeister

# The MIT License (MIT)

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Note : The software is also available as part of the PyAstronomy package.
#        See: http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/index.html

from __future__ import print_function, division
import numpy as np
from numpy import sum, pi, cos, sin, arctan2, exp, log, sqrt,\
                  dot, argmax, arange

__version__ = '2017-02-21'
__author__ = 'Mathias Zechmeister, Stefan Czesla'

class Gls:
    """
    Compute the Generalized Lomb-Scargle (GLS) periodogram.

    The *Gls* class computes the error-weighted Lomb-Scargle periodogram as
    developed by [ZK09]_ using various possible normalizations.

    The constructor of *Gls* takes a *TimeSeries* instance (i.e., a light curve)
    as first argument. The constructor allows to pass keywords to adjust the
    `freq` array, which will be used to calculate the periodogram.

    The main result of the calculation, i.e., the power, are stored in the
    class property `power`.

    Parameters
    ----------
    lc : TimeSeries object or tuple or list
        The light curve data either in the form of a TimeSeries object (or any
        object providing the attributes time, flux, and error) or a tuple or list
        providing time as first element, flux as second element, and optionally,
        the error as third element.
    fbeg, fend : float, optional
        The beginning and end frequencies for the periodogram
        (inverse units of time axis).
    Pbeg, Pend : float, optional
        The beginning and end periods for the periodogram
        (same units as for time axis).
    ofac : int
        Oversampling factor of frequency grid (default=10).
    hifac : float
        Maximum frequency `freq` = `hifac` * (average Nyquist frequency)
        (default=1).
    freq : array, optional
        Contains the frequencies at which to calculate the periodogram.
        If given, fast and verbose option are not available.
        If not given, a frequency array will be automatically generated.
    nfreq: integer, optional
        Specifiy explicitely the number or frequencies rather than using the ofac value
    norm : string, optional
        The normalization; either of "ZK", "Scargle", "HorneBaliunas", "Cumming", "wrms", "chisq".
        The default is unity ("ZK").
    ls : boolean, optional
        If True, the conventional Lomb-Scargle periodogram will be computed
        (default is False).
    fast : boolean, optional
        If True, recursive relations for trigonometric functions will be used
        leading to faster evaluation (default is False).
    verbose : boolean, optional
        Set True to obtain some statistical output (default is False).

    Attributes
    ----------
    power : array
        The normalized power of the GLS.
    freq : array
        The frequency array.
    ofac : int
        The oversampling factor of frequency grid.
    hifac : float
        The maximum frequency.
    t : array
        The abscissa data values.
    y : array
        The ordinate data values.
    yerr : array
        The errors of the data values.
    norm : string, {'ZK', 'Scargle', 'HorneBaliunas', 'Cumming', 'wrms', 'chisq'}
        The used normalization.

    Examples
    --------
    Create 1000 unevenly sampled data points with frequency=0.1,
    measurement error and Gaussian noise
    >>> time = np.random.uniform(54000., 56000., 1000)
    >>> flux = 0.15 * np.sin(2. * np.pi * time / 10.)

    Add some noise
    >>> error = 0.5 * np.ones(time.size)
    >>> flux += np.random.normal(0, error)

    Compute the full error-weighted Lomb-Periodogram
    in 'ZK' normalization and calculate the significance
    of the maximum peak.
    >>> gls = Gls((time, flux, error), verbose=True)

    >>> maxPower = gls.pmax
    >>> print("GLS maximum power: ", maxPower)
    >>> print("GLS statistics of maximum power peak: ", gls.stats(maxPower))
    >>> gls.plot(block=True)

    """

    # Available normalizations
    norms = ['ZK', 'Scargle', 'HorneBaliunas', 'Cumming', 'wrms', 'chisq']

    def __init__(self, lc, fbeg=None, fend=None, Pbeg=None, Pend=None, ofac=10, hifac=1, freq=None, nfreq=None, norm="ZK", ls=False, fast=False, veusz=False, verbose=False, **kwargs):

        self.nfreq = nfreq
        self.freq = freq
        self.fbeg = fbeg
        self.fend = fend
        self.Pbeg = Pbeg
        self.Pend = Pend
        self.ofac = ofac
        self.hifac = hifac
        self.ls = ls
        self.norm = norm
        self.fast = fast
        self.veusz = veusz
        self.label = {'title': 'Generalized Lomb Periodogram',
                      'xlabel': 'Frequency'}
        if "stats" in kwargs:
          print("Warning: 'stats' option is outdated. Please use 'verbose' instead.")
          verbose = kwargs["stats"]

        self._normcheck(norm)

        self._assignTimeSeries(lc)
        self._buildFreq()
        self._calcPeriodogram()
        self.pnorm(norm)
        self._peakPeriodogram()

        # Output statistics
        if verbose:
            self.info()

    def _assignTimeSeries(self, lc):
        """
        A container class that holds the observed light curve.

        Parameters
        ----------
        time : array
            The time array.
        flux : array
            The observed flux/data.
        error : array, optional
            The error of the data values.
        """
        if isinstance(lc, (tuple, list)):
            # t, y[, yerr] where given as list or tuple
            if len(lc) in (2, 3):
                self.t = np.array(lc[0])
                self.y = np.array(lc[1])
                self.yerr = None
                if len(lc) == 3 and lc[2] is not None:
                    # Error has been specified
                    self.yerr = np.array(lc[2])
            else:
                raise(ValueError("lc is a list or tuple with " + str(len(lc)) + " elements. Needs to have 2 or 3 elements." + \
                                   " solution=Use 2 or 3 elements (t, y[, yerr]) or an instance of TimeSeries"))
        else:
            # Assume lc is an instance of TimeSeries.
            self.t, self.y, self.yerr = lc.time, lc.flux, lc.error

        self.th = self.t - self.t.min()
        self.tbase = self.th.max()

        self.N = len(self.y)

        # Re-check array length compatibility
        if (len(self.th) != self.N) or ((self.yerr is not None) and (len(self.yerr) != self.N)):
            raise(ValueError("Incompatible dimensions of input data arrays (time and flux [and error]). Current shapes are: " + \
                             ', '.join(str(np.shape(x)) for x in (self.t, self.y, self.yerr))))

    def _buildFreq(self):
        """
        Build frequency array (`freq` attribute).

        Attributes
        ----------
        fnyq : float
            Half of the average sampling frequency of the time series.

        """

        self.fstep = 1 / self.tbase / self.ofac   # frequency sampling depends on the time span, default for start frequency
        self.fnyq = 0.5 / self.tbase * self.N     # Nyquist frequency
        self.f = self.freq

        if self.freq is None:
            # Build frequency array if not present.
            if self.fbeg is None:
                self.fbeg = self.fstep if self.Pend is None else 1 / self.Pend
            if self.fend is None:
                self.fend = self.fnyq * self.hifac if self.Pbeg is None else 1 / self.Pbeg
            if self.fend <= self.fbeg:
                raise(ValueError("fend is smaller than (or equal to) fbeg but it must be larger." + \
                               "Choose fbeg and fend so that fend > fbeg."))

            if self.nfreq is None:
                self.freq = arange(self.fbeg, self.fend, self.fstep)
            else:
                self.freq, self.fstep = np.linspace(self.fbeg, self.fend, self.nfreq, retstep=True)

        elif self.fast:
            raise(ValueError("freq and fast cannot be used together."))

        self.nf = len(self.freq)

        # An ad-hoc estimate of the number of independent frequencies (ZK_09 Eq. 24).
        self.M = (self.fend-self.fbeg) * self.tbase

    def _calcPeriodogram(self):

        if self.yerr is None:
            w = np.ones(self.N)
        else:
            w = 1 / (self.yerr * self.yerr)
        self.wsum = w.sum()
        w /= self.wsum

        self._Y = dot(w, self.y)       # Eq. (7)
        wy = self.y - self._Y          # Subtract weighted mean
        self._YY = dot(w, wy**2)       # Eq. (10)
        wy *= w                        # attach errors

        C, S, YC, YS, CC, CS = np.zeros((6, self.nf), dtype=np.double)

        WF = np.zeros(self.nf)
        if self.fast:
            # prepare trigonometric recurrences.
            eid = exp(2j * pi * self.fstep * self.th)  # cos(dx)+i sin(dx)

        for k, omega in enumerate(2.*pi*self.freq):
            # Circular frequencies.
            if self.fast:
                if k % 1000 == 0:
                    # init/refresh recurrences to stop error propagation
                    eix = exp(1j * omega * self.th)  # exp(ix) = cos(x) + i*sin(x)
                cosx = eix.real
                sinx = eix.imag
                eix *= eid              # increase freq for next loop
            else:
                x = omega * self.th
                cosx = cos(x)
                sinx = sin(x)

            C[k] = dot(w, cosx)         # Eq. (8)
            S[k] = dot(w, sinx)         # Eq. (9)

            YC[k] = dot(wy, cosx)       # Eq. (11)
            YS[k] = dot(wy, sinx)       # Eq. (12)
            wcosx = w * cosx
            CC[k] = dot(wcosx, cosx)    # Eq. (13)
            CS[k] = dot(wcosx, sinx)    # Eq. (15)

            # Added spectral window computation
            WF[k] = np.sum(cosx,dtype=np.double)**2 + np.sum(sinx,dtype=np.double)**2

        self.wf = WF / self.N**2

        SS = 1. - CC
        if not self.ls:
            CC -= C * C            # Eq. (13)
            SS -= S * S            # Eq. (14)
            CS -= C * S            # Eq. (15)
        D = CC*SS - CS*CS          # Eq. (6)

        self._a = (YC*SS-YS*CS) / D
        self._b = (YS*CC-YC*CS) / D
        self._off = -self._a*C - self._b*S

        # power
        self.p = (SS*YC*YC + CC*YS*YS - 2.*CS*YC*YS) / (self._YY*D)   # Eq. (5) in ZK09

    def _normcheck(self, norm):
        """
        Check normalization

        Parameters
        ----------
        norm : string
            Normalization string
        """
        if norm not in self.norms:
            raise(ValueError("Unknown norm: " + str(norm) + ". " + \
                "Use either of " + ', '.join(self.norms)))

    def pnorm(self, norm="ZK"):
        """
        Assign or modify normalization (can be done afterwards).

        Parameters
        ----------
        norm : string, optional
            The normalization to be used (default is 'ZK').

        Examples
        --------
        >>> gls.pnorm('wrms')
        """
        self._normcheck(norm)
        self.norm = norm
        p = self.p
        power = p   # default ZK
        self.label["ylabel"] = "Power ("+norm+")"

        if norm == "Scargle":
            popvar = input('pyTiming::gls - Input a priori known population variance:')
            power = p / float(popvar)
        elif norm == "HorneBaliunas":
            power = (self.N-1)/2. * p
        elif norm == "Cumming":
            power = (self.N-3)/2. * p / (1.-self.p.max())
        elif norm == "chisq":
            power = self._YY *self.wsum * (1.-p)
            self.label["ylabel"] = "chisq"
        elif norm == "wrms":
            power = sqrt(self._YY*(1.-p))
            self.label["ylabel"] = "wrms"

        self.power = power

    def _peakPeriodogram(self):
        """
        Analyze the highest periodogram peak.
        """
        # Index with maximum power
        k = argmax(self.p)
        # Maximum power
        self.pmax = pmax = self.p[k]
        self.rms = rms = sqrt(self._YY*(1.-pmax))

        # Statistics of highest peak
        self.hpstat = p = {}

        # Best parameters
        p["fbest"] = fbest = self.freq[k]
        p["amp"] = amp = sqrt(self._a[k]**2 + self._b[k]**2)
        p["ph"] = ph = arctan2(self._a[k], self._b[k]) / (2.*pi)
        p["T0"]  = self.t.min() - ph/fbest
        p["offset"] = self._off[k] + self._Y            # Re-add the mean.

        # Error estimates
        p["amp_err"] = sqrt(2./self.N) * rms
        p["ph_err"] = ph_err = sqrt(2./self.N) * rms/amp/(2.*pi)
        p["T0_err"] = ph_err / fbest
        p["offset_err"] = sqrt(1./self.N) * rms

        # Get the curvature in the power peak by fitting a parabola y=aa*x^2
        if 1 < k < self.nf-2:
            # Shift the parabola origin to power peak
            xh = (self.freq[k-1:k+2] - self.freq[k])**2
            yh = self.p[k-1:k+2] - pmax
            # Calculate the curvature (final equation from least square)
            aa = dot(yh, xh) / dot(xh, xh)
            p["f_err"] = e_f = sqrt(-2./self.N / aa * (1.-self.pmax))
            p["Psin_err"] = e_f / fbest**2
        else:
            self.hpstat["f_err"] = np.nan
            self.hpstat["Psin_err"] = np.nan
            print("WARNING: Highest peak is at the edge of the frequency range.\nNo output of frequency error.\nIncrease frequency range to sample the peak maximum.")

    def sinmod(self, t):
        """
        Calcuate best-fit sine curve.

        Parameters
        ----------
        t : array
            Time array at which to calculate the sine.

        Returns
        -------
        Sine curve : array
            The best-fit sine curve (i.e., that for which the
            power is maximal).
        """
        try:
            p = self.hpstat
            return p["amp"] * sin(2*np.pi*p["fbest"]*(t-p["T0"])) + p["offset"]
        except Exception as e:
            print("Failed to calcuate best-fit sine curve.")
            raise(e)

    def info(self):
        """
        Prints some basic statistical output screen.
        """
        print("Generalized LS - statistical output")
        print("-----------------------------------")
        print("Number of input points:     %6d" % self.N)
        print("Weighted mean of dataset:   %f"  % self._Y)
        print("Weighted rms of dataset:    %f"  % sqrt(self._YY))
        print("Time base:                  %f"  % self.tbase)
        print("Number of frequency points: %6d" % self.nf)
        print()
        print("Maximum power p [%s]: %f" % (self.norm, self.power.max()))
        print("RMS of residuals:     %f" % self.rms)
        if self.yerr is not None:
            print("  Mean weighted internal error:  %f" % (sqrt(self.N/sum(1./self.yerr**2))))
        print("Best sine frequency:  %f +/- %f" % (self.hpstat["fbest"], self.hpstat["f_err"]))
        print("Best sine period:     %f +/- %f" % (1./self.hpstat["fbest"], self.hpstat["Psin_err"]))
        print("Amplitude:            %f +/- %f" % (self.hpstat["amp"], self.hpstat["amp_err"]))
        print("Phase (ph):           %f +/- %f" % (self.hpstat["ph"], self.hpstat["ph_err"]))
        print("Phase (T0):           %f +/- %f" % (self.hpstat["T0"], self.hpstat["T0_err"]))
        print("Offset:               %f +/- %f" % (self.hpstat["offset"], self.hpstat["offset_err"]))
        print("-----------------------------------")

    def plot(self, block=False, period=False):
        """
        Create a plot.
        """
        try:
            import matplotlib
            if (matplotlib.get_backend() != "TkAgg"):
                matplotlib.use("TkAgg")
            import matplotlib.pylab as plt
            from matplotlib.ticker import FormatStrFormatter
        except ImportError:
            raise(ImportError("Could not import matplotlib.pylab."))


        fig = plt.figure()
        fig.subplots_adjust(hspace=0.15, wspace=0.08, right=0.97, top=0.95)
        ax = fig.add_subplot(4, 1, 1)
        ax.set_title("Normalized periodogram")

        ax.set_ylabel(self.label["ylabel"])
        if period: ax.set_xscale("log")
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.plot(1/self.freq if period else self.freq, self.power, 'b-')

        fig.subplots_adjust(hspace=0.50, wspace=0.08, right=0.97, top=0.95)
        ax5 = fig.add_subplot(7, 1, 3, sharex=ax)
        ax5.set_xlabel("Time")
        if period:
           ax5.set_xscale("log")
           ax5.set_xlabel("Period")
        else:
           ax5.set_xlabel("Frequency")
        ax5.plot(1/self.freq if period else self.freq, self.wf, 'b-')

        fig.subplots_adjust(hspace=0.15, wspace=0.08, right=0.97, top=0.95)

        fbest, T0 = self.hpstat["fbest"], self.hpstat["T0"]
        # Data and model
        datstyle = {'yerr':self.yerr, 'fmt':'r.', 'capsize':0}
        tt = arange(self.t.min(), self.t.max(), 0.01/fbest)
        ymod = self.sinmod(tt)
        yfit = self.sinmod(self.t)
        ax1 = fig.add_subplot(4, 2, 5)
        # ax1.set_xlabel("Time")
        ax1.set_ylabel("Data")
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.errorbar(self.t, self.y, **datstyle)
        ax1.plot(tt, ymod, 'b-')

        tt = arange(T0, T0+1/fbest, 0.01/fbest)
        yy = self.sinmod(tt)
        ax2 = fig.add_subplot(4, 2, 6, sharey=ax1)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        # ax2.set_xlabel("Time")
        # ax2.set_ylabel("Data")
        ax2.errorbar(self.t*fbest % 1, self.y, **datstyle)
        xx = tt*fbest % 1
        ii = np.argsort(xx)
        ax2.plot(xx[ii], yy[ii], 'b-')

        # Residuals
        yres = self.y - yfit
        ax3 = fig.add_subplot(4, 2, 7, sharex=ax1)
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Residuals")
        ax3.errorbar(self.t, yres, **datstyle)
        ax3.plot([self.t.min(), self.t.max()], [0,0], 'b-')

        ax4 = fig.add_subplot(4, 2, 8, sharex=ax2, sharey=ax3)
        # ax4.set_title("Data")
        ax4.set_xlabel("Phase")
        # ax4.set_ylabel("Data")
        plt.setp(ax4.get_yticklabels(), visible=False)
        ax4.errorbar(self.t*fbest % 1, yres, **datstyle)
        ax4.plot([0,1], [0,0], 'b-')



        if hasattr(plt.get_current_fig_manager(), 'toolbar'):
            # check seems not needed when "TkAgg" is set
            plt.get_current_fig_manager().toolbar.pan()
        #t = fig.canvas.toolbar
        #plt.ToggleTool(plt.wx_ids['Pan'], False)
        if block: print("Close the plot to continue.")

        #plt.tight_layout()
        plt.show(block=block)
        return plt

    def prob(self, Pn):
        """
        Probability of obtaining the given power.

        Calculate the probability to obtain a power higher than
        `Pn` from the noise, which is assumed to be Gaussian.

        .. note:: Normalization
          (see [ZK09]_ for further details).

          - `Scargle`:
          .. math::
            exp(-Pn)

          - `HorneBaliunas`:
          .. math::
            \\left(1 - 2 \\times \\frac{Pn}{N-1} \\right)^{(N-3)/2}

          - `Cumming`:
          .. math::
            \\left(1+2\\times \\frac{Pn}{N-3}\\right)^{-(N-3)/2}

        Parameters
        ----------
        Pn : float
            Power threshold.

        Returns
        -------
        Probability : float
            The probability to obtain a power equal or
            higher than the threshold from the noise.
        """
        self._normcheck(self.norm)
        if self.norm == "ZK": return (1.-Pn)**((self.N-3.)/2.)
        if self.norm == "Scargle": return exp(-Pn)
        if self.norm == "HorneBaliunas": return (1.-2.*Pn/(self.N-1.))**((self.N-3.)/2.)
        if self.norm == "Cumming": return (1.+2.*Pn/(self.N-3.))**(-(self.N-3.)/2.)
        if self.norm == "wrms": return (Pn**2/self._YY)**((self.N-3.)/2.)
        if self.norm == "chisq": return (Pn/self._YY/self.wsum)**((self.N-3.)/2.)

    def probInv(self, Prob):
        """
        Calculate minimum power for given probability.

        This function is the inverse of `Prob(Pn)`.
        Returns the minimum power for a given probability threshold `Prob`.

        Parameters
        ----------
        Prob : float
            Probability threshold.

        Returns
        -------
        Power threshold : float
            The minimum power for the given false-alarm probability threshold.
        """
        self._normcheck(self.norm)
        if self.norm == "ZK": return 1.-Prob**(2./(self.N-3.))
        if self.norm == "Scargle": return -log(Prob)
        if self.norm == "HorneBaliunas": return (self.N-1) / 2. * (1.-Prob**(2./(self.N-3)))
        if self.norm == "Cumming": return (self.N-3) / 2. * (Prob**(-2./(self.N-3.))-1.)
        if self.norm == "wrms": return sqrt(self._YY * Prob**(2./(self.N-3.)))
        if self.norm == "chisq": return self._YY * self.wsum * Prob**(2./(self.N-3.))

    def FAP(self, Pn):
        """
        Obtain the false-alarm probability (FAP).

        The FAP denotes the probability that at least one out of M independent
        power values in a prescribed search band of a power spectrum computed
        from a white-noise time series is as large as or larger than the
        threshold, `Pn`. It is assessed through

        .. math:: FAP(Pn) = 1 - (1-Prob(P>Pn))^M \\; ,

        where "Prob(P>Pn)" depends on the type of periodogram and normalization
        and is calculated by using the *prob* method; *M* is the number of
        independent power values and is computed internally.

        Parameters
        ----------
        Pn : float
            Power threshold.

        Returns
        -------
        FAP : float
            False alarm probability.
        """
        prob = self.M * self.prob(Pn)
        if prob > 0.01:
            return 1. - (1.-self.prob(Pn))**self.M
        return prob

    def powerLevel(self, FAPlevel):
        """
        Power threshold for FAP level.

        Parameters
        ----------
        FAPlevel : float or array_like
              "False Alarm Probability" threshold

        Returns
        -------
        Threshold : float or array
            The power threshold pertaining to a specified false-alarm
            probability (FAP). Powers exceeding this threshold have FAPs
            smaller than FAPlevel.

        """
        Prob = 1.-(1.-FAPlevel)**(1./self.M)
        return self.probInv(Prob)

    def stats(self, Pn):
        """
        Obtain basic statistics for power threshold.

        Parameters
        ----------
        Pn : float
            Power threshold.

        Returns
        -------
        Statistics : dictionary
            A dictionary containing {'Pn': *Pn*, 'Prob': *Prob(Pn)* ,
            'FAP': *FAP(Pn)*} for the specified power threshold, *Pn*.
        """
        return {'Pn': Pn, 'Prob': self.prob(Pn), 'FAP': self.FAP(Pn)}

    def toFile(self, ofile, header=True):
        # LM: Added new flag to save files in Veusz-compatible format
        """
        Write periodogram to file.

        Parameters
        ----------
        ofile : string
            Name of the output file.
        """
        ofile_gls = ofile + '_GLS.dat'
        with open(ofile_gls, 'w') as f:
            if header:
                f.write("# Generalized Lomb-Scargle periodogram\n")
                f.write("# Parameters:\n")
                f.write("#    Data file: %s\n" % self.df)
                f.write("#    ofac     : %s\n" % self.ofac)
                f.write("#    hifac    : %s\n" % self.hifac)
                f.write("#    norm     : %s\n" % self.norm)
                f.write("# 1) Frequency, 2) Normalized power 3) Window function\n")
            if self.veusz:
                f.write("descriptor freq pow win\n")
            for line in zip(self.freq, self.power, self.wf):
                f.write("%f  %f  %f\n" % line)

        fbest, T0 = self.hpstat["fbest"], self.hpstat["T0"]

        ymod = self.sinmod(self.t)
        yres = self.y - ymod
        ypha = self.t*fbest % 1
        ofile_res = ofile + '_res.dat'

        with open(ofile_res, 'w') as f:
            if self.veusz:
                f.write("descriptor BJD_obs Y_res,+- Y_mod Y_pha Y_obs,+-\n")
            else:
                f.write("# BJD Y_res Y_err Y_mod Y_pha Y_obs Y_res\n")
            for line in zip(self.t, yres, self.yerr, ymod, ypha, self.y, self.yerr):
                f.write("%f  %f  %f  %f  %f  %f  %f\n" % line)

        #tt = arange(self.t.min(), self.t.max(), 0.01/fbest)
        #ymod = self.sinmod(tt)
        #ofile_mod = ofile + '_mod.dat'
        #with open(ofile_mod, 'w') as f:
        #    f.write("# BJD mod\n")
        #    for line in zip(ttt, ymod):
        #        f.write("%f  %f\n" % line)

        tt = arange(T0, T0+1/fbest, 0.01/fbest)
        yy = self.sinmod(tt)
        ph = tt*fbest % 1
        ii = np.argsort(ph)
        ofile_mod = ofile + '_mod.dat'
        with open(ofile_mod, 'w') as f:
            if self.veusz:
                f.write("descriptor BJD pha mod pha_sorted mod_sorted \n")
            else:
                f.write("#BJD pha mod pha_sorted mod_sorted \n")
            for line in zip(tt, ph, yy, ph[ii], yy[ii]):
                f.write("%f  %f  %f     %f  %f\n" % line)

        print("Results have been written to file: ", ofile_gls)
        print("Residuals have been written to file: ", ofile_res)
        print("Model has been written to file: ", ofile_mod)

def example():
    # Run the example in the Gls class.
    print("--- EXAMPLE CALCULATION ---")
    import doctest
    exec(doctest.script_from_examples(Gls.__doc__))
    print("----------------------------------------------------")


if __name__ == "__main__":

  import argparse

  parser = argparse.ArgumentParser(description='Generalized Lomb-Scargle periodogram.', add_help=False)
  argadd = parser.add_argument   # function short cut
  argadd('-?', '-h', '-help', '--help', help='show this help message and exit', action='help')
  argadd('df', nargs='?',
                   help='Data file (three columns: time, data, error). If not specified example will be shown.')
  argadd('-fbeg', type=float, help="Starting frequency for periodogram.")
  argadd('-fend', type=float, help="Stopping frequency for periodogram.")
  argadd('-Pbeg', type=float, help="Starting period for periodogram.")
  argadd('-Pend', type=float, help="Stopping period for periodogram.")
  argadd('-ofac', type=float, help="Oversampling factor (default=10).", default=10)
  argadd('-hifac', type=float, help="Maximum frequency (default=1).", default=1)
  argadd('-iter', type=int, help="Iterate on residuals")
  argadd('-fast', help="Use trigonometric recurrences.", action='store_true')
  argadd('-norm', help="The normalization (default=ZK).", choices=Gls.norms, default='ZK')
  argadd('-ofile', type=str, help="Output file for results.")
  argadd('-nostat', help="Switch off statistical output on screen.", dest='verbose',
                 default=True, action='store_false')
  argadd('-noplot',  help="Suppress plots.", dest='plot', default=True, action='store_false')
  argadd('-lines' , type=np.double, nargs=3, required=False, default=[0, 1, 2], help='Specify a different column number for the data (Python notation)')
  argadd('-ldiff' , type=np.double, nargs=2, help='Use the difference of two columns for the data (Python notation)')
  argadd('-trades', type=np.int, nargs='?', required=False, default=False, help='Standard input from TRADES ')
  argadd('-nfreq', type=np.int, nargs='?', required=False, default=False, help='Number of frequencies ')
  argadd('-skipr', type=np.int, nargs='?', required=False, default=0, help='Number of frequencies ')

  argadd('-veusz',  help="Create Veusz-compatible lables.", dest='veusz', action='store_true')

  args = vars(parser.parse_args())
  df = args.pop('df')
  ofile = args.pop('ofile')
  plot = args.pop('plot')
  iterate = args.pop('iter')
  lines = args.pop('lines')
  ldiff = args.pop('ldiff')
  #nfreq = args.pop('nfreq')
  skipr = args.pop('skipr')

  if args.pop('trades') != False:
      lines = [0,1,2]
      ldiff = [1,4]

  if df is None:
    # No data file given. Show example:
    example()
    print("Available options:")
    parser.print_help()
    exit(0)

  # A data file has been given.
  try:
    # dat = np.loadtxt(df, usecols=(0,1,2))
    dat = np.loadtxt(df, unpack=True, skiprows=skipr)
    if ldiff is None:
       tye = dat[lines[0]], dat[lines[1]], dat[lines[2]] if len(dat) > 2 else None
    else:
       tye = dat[lines[0]], dat[ldiff[1]]-dat[ldiff[0]], dat[lines[2]] if len(dat) > 2 else None
  except Exception as e:
    print("An error occurred while trying to read data file: ")
    print("  " + str(e))
    exit(9)


  if iterate is None:
      gls = Gls(tye, **args)
      if plot:
          gls.plot(block=True)

      if ofile:
          gls.df = df
          gls.toFile(ofile)
  else:

      if ofile is None:
          print("Please specify output file: ")
          exit(9)

      for it in xrange(0,iterate+1):
          gls = Gls(tye, **args)

          if plot: gls.plot(block=True)

          gls.df = df
          ofile_it = ofile + '_it' + repr(it)
          gls.toFile(ofile_it)

          df = ofile_it + '_res.dat'
          if gls.veusz:
            dat = np.loadtxt(df, unpack=True, comments='descriptor')
          else:
            dat = np.loadtxt(df, unpack=True)
          tye = dat[0], dat[1], dat[2] if len(dat) > 2 else None
