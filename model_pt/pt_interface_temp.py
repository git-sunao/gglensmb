import sys,os,copy
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as rbs
from scipy import integrate
from scipy import special,ndimage
from scipy.integrate import simps
# from dark emu
from dark_emulator.pyfftlog_interface import fftlog
from dark_emulator.darkemu.cosmo_util import cosmo_class
from dark_emulator.darkemu import cosmo_util
from dark_emulator.darkemu.pklin import pklin_gp
# pyhalotift
import pyhalofit
# fastpt
import fastpt as fpt
from astropy import constants

rho_cr = 2.77536627e11

class pt:
    def __init__(self, config=None):
        self.config = dict()
        self.config['fft_num'] = 1
        self.config['fft_logrmin'] = -3.0
        self.config["fft_logrmax"] = 5.0
        self.config["pi_max"]=False
        self.config["rsd"]=False
        self.config['1stterm'] = 'nonlinear'
        if config is not None:
            print('update config with', config)
            self.config.update(config)
        
    
