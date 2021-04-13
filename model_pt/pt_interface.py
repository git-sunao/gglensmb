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
"""
import dark_emulator
sys.path.append(os.path.dirname(dark_emulator.__file__))
from pyfftlog_interface import pyfftlog_class
from darkemu.cosmo_util import cosmo_class
import darkemu.cosmo_util
from darkemu.pklin import pklin_gp
"""
# pyhalotift
import pyhalofit
# fastpt
import fastpt as fpt
from astropy import constants

rho_cr = 2.77536627e11

def PLextrap(x, y, x_new):
    d = np.log(x[1])-np.log(x[0])
    # low
    x_low = x_new[x_new<x.min()]
    if y[0] > 0:
        ns=(np.log(y[1])-np.log(y[0]))/d
        Amp=y[0]/x.min()**ns
        y_low=x_low**ns*Amp
    else:
        ns=(np.log(-y[1])-np.log(-y[0]))/d
        Amp=-y[0]/x.min()**ns
        y_low=-x_low**ns*Amp
    # low
    x_high = x_new[x_new>x.max()]
    if y[-1] > 0:
        ns=(np.log(y[-1])-np.log(y[-2]))/d
        Amp=y[-1]/x.max()**ns
        y_high=x_high**ns*Amp
    else:
        ns=(np.log(-y[-1])-np.log(-y[-2]))/d
        Amp=-y[-1]/x.max()**ns
        y_high=-x_high**ns*Amp
    # 
    x_ex = np.concatenate([ x_low,x,x_high ])
    y_ex = np.concatenate([ y_low,y,y_high ])
    #return x_ex, y_ex,ius(x_ex,y_ex)(x_new)
    return ius(x_ex,y_ex)(x_new)

def window_tophat(kR):
    return 3.*(np.sin(kR)-kR*np.cos(kR))/kR**3

class pt():
    def __init__(self, config = None):
        # set up default config
        self.config = dict()
        self.config["fft_num"] = 1
        self.config["fft_logrmin"] = -3.0#-3.5
        self.config["fft_logrmax"] = 5.0#3.5
        self.config["low_kcut"] = None
        self.config["high_kcut"] = 10.0 #hiMpc
        self.config["pi_max"]=False
        self.config["rsd"]=False
        self.config["high_kcut_only_loop"] = True
        self.config["1stterm"] = "nonlinear"
        self.config['pz'] = {'do':False, 'filename':None}
        # override if specified in input
        if config is not None:
            for key in list(config.keys()):
                self.config[key] = config[key]
            print("Using %s power spectrum as the 1st term in bias expansion fully."%(self.config["1stterm"]))

        if self.config['pz']['do']:
            print('init (zs, p(zs)) for the computetion of magnification bias term. loading %s'%(self.config['pz']['filename']))
            source_z, pz = np.loadtxt(self.config['pz']['filename']).T
            self.source_z, self.pz = source_z[1:], pz[1:] # remove the first, because the redshift of the first is too low.

        # flag
        self.fftlog_param_has_changed = False
        self.cosmology_has_changed = True
        self.bias_has_changed = True
        self.low_kcut_has_changed = True
        self.high_kcut_has_changed = True
        self.redshift_has_changed = True
        self.firstterm_has_changed = True
        self.pi_max_has_changed = True
        self.current_redshift = 0.

        # paramters & instances
        self.b1,self.b2,self.bs2,self.b3 = 1.,0.,0.,0.
        self.point_mass = 0.0

        # init fftlog, cosmo, halofit, linear pk
        self.init_fftlog()
        self.halofit = pyhalofit.halofit()
        self.cosmo = cosmo_class()
        self.pkL = pklin_gp()
        self.set_cosmology(self.cosmo.get_cosmology(),init=True)
        # fastpt
        to_do = ['one_loop_dd', 'dd_bias', 'one_loop_cleft_dd']
        pad_factor = 1 # padding the edges with zeros before Pk repeats
        n_pad = pad_factor*len(self.fftlog.k)
        low_extrap = -self.config["fft_logrmax"] # Extend Plin to this log10 value if necessary (power law)
        high_extrap = -self.config["fft_logrmin"] # Extend Plin to this log10 value if necessary (power law)
        self.fpt_narrow_k = np.logspace(-3.0, 2.0, 512)
        self.C_window = .75 # Smooth the Fourier coefficients of Plin to remove high-frequency noise.
        self.fpt_obj = fpt.FASTPT(self.fpt_narrow_k,to_do=to_do,low_extrap=low_extrap,high_extrap=high_extrap,n_pad=n_pad)

    # functions to set parameters from outside
    def set_fftlog_param(self, fft_num=None, fft_logrmin=None, fft_logrmax=None):
        if (fft_num is not None) and (fft_num != self.config["fft_num"]):
            self.config["fft_num"] = fft_num
            self.fftlog_param_has_changed = True
        if (fft_logrmin is not None) and (fft_logrmin != self.config["fft_logrmin"]):
            self.config["fft_logrmin"] = fft_logrmin
            self.fftlog_param_has_changed = True
        if (fft_logrmax is not None) and (fft_logrmax != self.config["fft_logrmax"]):
            self.config["fft_logrmax"] = fft_logrmax
            self.fftlog_param_has_changed = True
    def set_cosmology(self, cparams, init=False):
        if np.any(self.cosmo.cparam != cparams) or init:
            try:
                cosmo_util.test_cosm_range_linear(cparams)
                print("set up cosmology : ", cparams)
                self.cosmo.cparam = copy.deepcopy(cparams)
                self.rho_m = (1. - cparams[0][2])*rho_cr
                self.pkL.set_cosmology(self.cosmo)
                cosmo_dict = {"Omega_de0":cparams[0][2], "Omega_K0":0.0, "w0":cparams[0][5],"wa":0.0,"h":np.sqrt((0.00064+cparams[0][0]+cparams[0][1])/(1.0-cparams[0][1]))}
                self.halofit.set_cosmology(cosmo_dict)
                self.cosmology_has_changed = True
            except:
                print('cosmological parameter out of supported range!')
        else:
            print("Got same cosmological parameters. Keep quantities already computed")
    def set_bias(self, bparams):
        b1 = bparams.get("b1" , self.b1)
        b2 = bparams.get("b2" , self.b2)
        bs2= bparams.get("bs2", self.bs2)
        b3 = bparams.get("b3" , self.b3)
        if (b1,b2,bs2,b3) != (self.b1,self.b2,self.bs2,self.b3):
            print("set up bias")
            print(bparams)
            self.b1,self.b2,self.bs2,self.b3 = b1,b2,bs2,b3
            self.bias_has_changed = True
        else:
            print("Got same bias parameters. Keep quantities already computed.")
    def set_point_mass(self, point_mass):
        if self.point_mass != point_mass:
            print("set up point mass ")
            print(point_mass)
            self.point_mass = point_mass
            # no need to set flag for point mass changes. Because point mass term is computed at every call of observables.
        else:
            print("Got same point mass parameter. Keep quantity already computed.")
    def set_low_kcut(self,low_kcut):
        if low_kcut!=self.config["low_kcut"]:
            if low_kcut is not None:
                print(("set low kcut : %f" %(low_kcut)))
            else:
                print("set low kcut : None")
            self.config["low_kcut"]=low_kcut
            self.low_kcut_has_changed = True
        else:
            print("Got same low kcut.")
    def set_high_kcut(self, high_kcut):
        if high_kcut!=self.config["high_kcut"]:
            print(("set high k cut : %f" %high_kcut))
            self.config["high_kcut"]=high_kcut
            self.high_kcut_has_changed = True
        else:
            print("Got same high kcut.")
    def set_1stterm(self, firstterm):
        if firstterm!=self.config["1stterm"]:
            print(("set 1st term : %s"%firstterm))
            self.config["1stterm"] = firstterm
            self.firstterm_has_changed = True
        else:
            print("Got same first term.")
    def set_pimax(self, pimax):
        if pimax!=self.config["pi_max"]:
            print(("set pimax : %s"%pimax))
            self.config["pi_max"] = pimax
            self.pi_max_has_changed = True
        else:
            print("Got same pimax.")
    
    # functions to get paramters from outside
    def get_fftlog_param(self):
        return {"fft_num":self.config["fft_num"], "fft_logrmin":self.config["fft_logrmin"], "fft_logrmax":self.config["fft_logrmax"]}
    def get_cosmology(self):
        return self.cosmo.get_cosmology()
    def get_bias(self):
        return {"b1":self.b1,"b2":self.b2,"bs2":self.bs2,"b3":self.b3}
    def get_point_mass(self):
        return self.point_mass
    def get_kcut(self):
        return self.config["low_kcut"], self.config["high_kcut"]
    def get_1stterm(self):
        return self.config["1stterm"]

    # checking flags and redshift and recomputation
    def ask_to_compute(self,z):
        self.check_redshift(z)
        self.check_flags()
    def force_to_compute(self,z=None):
        self.fftlog_param_has_changed, self.cosmology_has_changed, self.redshift_has_changed, self.kcut_has_changed, self.bias_has_changed, self.high_kcut_has_changed, self.low_kcut_has_changed, self.firstterm_has_changed, self.pi_max_has_changed  = [True]*9
        if z is None:
            z = self.current_redshift
            print("compute powers at z = %f"%z)
        self.ask_to_compute(z)
    def check_redshift(self,z):
        if z != self.current_redshift:
            self.current_redshift = z
            self.redshift_has_changed = True
        else:
            self.redshift_has_changed = False
    def check_flags(self):
        flag0 = self.fftlog_param_has_changed
        flag1 = self.cosmology_has_changed or self.redshift_has_changed
        flag2 = self.low_kcut_has_changed # and isinstance(self.config["low_kcut"] ,float)
        flag3 = self.high_kcut_has_changed# and isinstance(self.config["high_kcut"],float)
        flag4 = self.bias_has_changed or self.firstterm_has_changed or self.pi_max_has_changed
        if flag0 or flag1:# or flag2 or flag3:
            self.compute_pklin()
            self.compute_pkhalofit()
            self.compute_galaxy_bias_pt_terms()
        if flag0 or flag1 or flag2 or flag3:
            self.spt_k_window = np.ones(len(self.fftlog.k))
            self.pklin_pkhalo_window = np.ones(len(self.fftlog.k))
            self.cut_power_at_low_kcut()
            self.cut_power_at_high_kcut()
        if flag0 or flag1 or flag2 or flag3 or flag4:
            self.compute_Pgg_Pgm()
        self.fftlog_param_has_changed, self.cosmology_has_changed, self.redshift_has_changed, self.kcut_has_changed, self.bias_has_changed, self.high_kcut_has_changed, self.low_kcut_has_changed, self.firstterm_has_changed, self.pi_max_has_changed  = [False]*9

    # parts of de_interface : instances in de_interface but not in pt_inteface are removed.
    def Dgrowth_from_z(self, z):
        return self.cosmo.Dgrowth_from_z(z)
    def f_from_z(self, z):
        return self.cosmo.f_from_z(z)
    def get_sigma8(self, logkmin=-4, logkmax=1, nint=100):
        R = 8.
        ks = np.logspace(logkmin, logkmax, nint)
        logks = np.log(ks)
        kR = ks * R
        integrant = ks**3*self.get_pklin(ks)*window_tophat(kR)**2
        return np.sqrt(integrate.trapz(integrant, logks)/(2.*np.pi**2))
    def get_pklin(self, k):
        return self.pkL.get(k)
    def get_pklin_from_z(self, k, z):
        Dp = self.Dgrowth_from_z(z)
        return Dp**2 * self.pkL.get(k)

    # computing power spectrum
    def init_fftlog(self):
        self.fftlog = fftlog(self.config['fft_num'], self.config['fft_logrmin'], self.config['fft_logrmax'])
    def compute_pklin(self):
        self.pklin = self.get_pklin_from_z(self.fftlog.k, self.current_redshift)
    def compute_pkhalofit(self):
        self.halofit.set_pklin(self.fftlog.k, self.pklin, self.current_redshift)
        self.pkhalofit = self.halofit.get_pkhalo()
    def compute_galaxy_bias_pt_terms(self):
        fpt_narrow_pklin = self.get_pklin_from_z(self.fpt_narrow_k, self.current_redshift)
        P_bias_E = self.fpt_obj.one_loop_dd_bias_b3nl(fpt_narrow_pklin, C_window=self.C_window)
        self.P_bias_E = [PLextrap(self.fpt_narrow_k, P_bias_E[i], self.fftlog.k)  for i in [0,1,2,3,4,5,6,8] ]
        self.P_bias_E.append(P_bias_E[7])
        self.pkspt = self.pklin+self.P_bias_E[0]
    def cut_power_at_low_kcut(self):
        if type(self.config["low_kcut"]) == float:
            self.spt_k_window[self.fftlog.k < self.config["low_kcut"] ] = 0.
            self.pklin_pkhalo_window[self.fftlog.k < self.config["low_kcut"] ] = 0.
    def cut_power_at_high_kcut(self):
        self.spt_k_window *= np.exp(-self.fftlog.k**2/self.config["high_kcut"]**2)
        self.pklin_pkhalo_window *= 1.0 if self.config["high_kcut_only_loop"] else np.exp(-self.fftlog.k**2/self.config["high_kcut"]**2) 
    def compute_Pgg_Pgm(self):
        b1,b2,bs2,b3 = self.b1,self.b2,self.bs2,self.b3
        Pd1d2,Pd2d2,Pd1s2,Pd2s2,Ps2s2,Pd1p3 = self.P_bias_E[2:8]
        Pdd  = {"nonlinear":self.pkhalofit*self.pklin_pkhalo_window,
                "linear":self.pklin*self.pklin_pkhalo_window,
                "SPT":self.pkspt*self.spt_k_window
                }[self.config["1stterm"]]
        Pgg = [
            b1**2*Pdd, b1*b2*Pd1d2*self.spt_k_window, b1*bs2*Pd1s2*self.spt_k_window, b1*b3*Pd1p3*self.spt_k_window,
            0.25*b2**2*Pd2d2*self.spt_k_window, 0.25*b2*bs2*Pd2s2*self.spt_k_window, 0.25*bs2**2*Ps2s2*self.spt_k_window
            ]
        Pgm = [
            b1*Pdd, 0.5*b2*Pd1d2*self.spt_k_window, 0.5*bs2*Pd1s2*self.spt_k_window, 0.5*b3*Pd1p3*self.spt_k_window
            ]
        Pmm = [Pdd]
        self.Power = {"Pgg":Pgg, "Pgm":Pgm, "Pmm":Pmm}
    def get_term_label(self,latex=False):
        if latex:
            return {"Pgg":[r"$b_1^2$",r"$b_1\times b_2$",r"$b_1\times b_{s_2}$",r"$b_1\times b_3$",r"$b_2^2$",r"$b_2\times b_{s_2}$",r"$b_{s_2}^2$"],"Pgm":[r"$b_1$",r"$b_2$",r"$b_{s_2}$",r"$b_3$"],"Pmm":[r"$\mathrm{m}$"]}
        else:
            return {"Pgg":["b1 2","b1 b2","b1 bs2","b1 b3","b2 2","b2 bs2","bs2 2"],"Pgm":["b1","b2","bs2","b3"],"Pmm":["m"]}

    # get observables on internal radial array (=self.fftlog.r)
    def get_xi_array(self,z):
        # condition : alpha-beta = 3, alpha+q = 3/2, beta+q = -3/2  
        self.ask_to_compute(z)
        self.xi_terms = list()
        self.xi_terms.extend([ self.fftlog.pk2xi(self.Power['Pgg'][i]) for i in range(0,7) ])
        self.xi_sum = np.sum(self.xi_terms, axis=0)
        return self.xi_sum
    def get_xigm_array(self, z):
        # condition : alpha-beta = 3, alpha+q = 3/2, beta+q = -3/2  
        self.ask_to_compute(z)
        self.xigm_array = list()
        self.xigm_array.extend([ self.fftlog.pk2xi(self.Power['Pgm'][i]) for i in range(0,4) ])
        self.xigm_sum = np.sum(self.xigm_array, axis=0)
        return self.xigm_sum
    def get_ximm_array(self, z):
        # condition : alpha-beta = 3, alpha+q = 3/2, beta+q = -3/2  
        self.ask_to_compute(z)
        self.ximm_array = list()
        self.ximm_array.extend([ self.fftlog.pk2xi(self.Power['Pmm'][0]) ])
        self.ximm_sum = np.sum(self.ximm_array, axis=0)
        return self.ximm_sum
    def get_wp_inf_array(self,z): # wp w/ pimax = inf
        # condition : alpha-beta = 2, alpha+q = 1 , beta+q = -1
        self.ask_to_compute(z)
        self.wp_terms = list()
        self.wp_terms.extend([ self.fftlog.pk2wp(self.Power['Pgg'][i]) for i in range(0,7) ])
        self.wp_sum = np.sum(self.wp_terms, axis=0)
        return self.wp_sum
    def get_ds_array(self,z): # DeltaSigma
        # condition : alpha-beta = 2, alpha+q = 1, beta+q = -1
        self.ask_to_compute(z)
        self.ds_terms = list()
        self.ds_terms.extend([ self.fftlog.pk2dwp(self.Power['Pgm'][i])*self.rho_m/1e12 for i in range(0,4)])
        self.point_mass_term = self.point_mass/self.fftlog.r**2
        self.ds_sum = np.sum(self.ds_terms, axis=0) + self.point_mass_term
        return self.ds_sum

    # get function of observables
    def get_xi(self, z): # xi_gg
        return ius(self.fftlog.r, self.get_xi_array(z) )
    def get_xigm(self, z):
        return ius(self.fftlog.r, self.get_xigm_array(z) )
    def get_ximm(self, z):
        return ius(self.fftlog.r, self.get_ximm_array(z) )
    def get_wp_inf(self,z):
        return ius(self.fftlog.r, self.get_wp_inf_array(z) )
    def get_ds(self,z):
        return ius(self.fftlog.r, self.get_ds_array(z) )

    # get_observables on given radial array and redshift
    def xi(self,r,z):
        return self.get_xi(z)(r)
    def xigm(self,r,z):
        return self.get_xigm(z)(r)
    def ximm(self,r,z):
        return self.get_ximm(z)(r)
    def wp_inf(self,r,z):
        return self.get_wp_inf(z)(r)
    def ds(self,r,z):
        return self.get_ds(z)(r)

    # wp : <rsd> or <without rsd>
    def wp(self,r,z, xigg="linear"):
        res, fac = self.wp_without_rsd(r,z), 1.0
        if self.config["rsd"]:
            fac = self.get_Kaiser_factor(r,z, xigg="linear")
        return res*fac

    # without rsd
    def wp_without_rsd(self, r, z):
        if type(self.config["pi_max"]) == float or type(self.config["pi_max"]) == int:
            xi = self.get_xi(z)
            return self._xi2wp_pimax(r, xi, self.config["pi_max"])
        else:
            return self.wp_inf(r,z)
    
    def _xi2wp_pimax(self, r, xi, pimax):
        t = np.linspace(0, pimax, 1024)
        dt = t[1]-t[0]
        wp = list()
        for rnow in r:
            wp.append( 2*integrate.trapz(xi( np.sqrt(t**2+rnow**2) ), dx=dt) )
        return np.array(wp)

    # rsd
    def get_Kaiser_factor(self,r,z, xigg="linear"):
        _1stterm_bak_ = self.config["1stterm"]
        print('computing Kaiser correction factor')
        self.set_1stterm(xigg)
        pi_max = self.config["pi_max"]
        if pi_max is None:
            pi_max = 100.0
        # 1. 
        # calculate xi
        r_ref =  np.logspace(-3, 3, 512)
        xi = self.xi(r_ref, z)
        xi_spl = ius(r_ref, xi)
        f = self.f_from_z(z)
        b = self.get_bias()["b1"]
        beta = f/b
        n = 3
        J_n = list()
        for _r in r_ref:
            t = np.linspace(1e-10, _r, 1024)
            dt = t[1]-t[0]
            J_n.append(1./_r**n*integrate.trapz(t**(n-1.)*xi_spl(t), dx = dt))
        J_3 = np.array(J_n)
        n = 5
        J_n = list()
        for _r in r_ref:
            t = np.linspace(1e-10, _r, 1024)
            dt = t[1]-t[0]
            J_n.append(1./_r**n*integrate.trapz(t**(n-1.)*xi_spl(t), dx = dt))
        J_5 = np.array(J_n)
        xi_0 = (1.+2./3.*beta+1./5.*beta**2)*xi
        xi_2 = (4./3.*beta+4./7.*beta**2)*(xi-3.*J_3)
        xi_4 = 8./35.*beta**2*(xi+15./2.*J_3-35./2.*J_5)
        r_pi =  np.logspace(-3, np.log10(pi_max), 512)
        rp, r_pi = np.meshgrid(r, r_pi, indexing = 'ij')
        s = np.sqrt(rp**2+r_pi**2)
        mu = r_pi/s
        l0 = special.eval_legendre(0, mu)
        l2 = special.eval_legendre(2, mu)
        l4 = special.eval_legendre(4, mu)
        xi_s = ius(r_ref, xi_0)(s)*l0 + ius(r_ref, xi_2)(s)*l2 + ius(r_ref, xi_4)(s)*l4
        xi_s_spl = rbs(rp[:,0], r_pi[0], xi_s)
        wp_rsd = list()
        for _r in rp:
            wp_rsd.append(2*integrate.quad(lambda t: xi_s_spl(_r, t)[0][0], 0, pi_max, epsabs = 1e-4)[0])
        wp_rsd = np.array(wp_rsd)
        # 2.
        wp_wo_rsd = self.wp_without_rsd(r, z)
        self.set_1stterm(_1stterm_bak_)
        return wp_rsd/wp_wo_rsd

    # Get each galaxy bias term of signal with finite pimax
    def get_xi_terms(self, z):
        _ = self.get_xi_array(z)
        return [ius(self.fftlog.r, t) for t in self.xi_terms]
    
    def get_wp_terms(self, z):
        if self.config["pi_max"] is None:
            _ = self.get_wp_inf_array(z)
            return [ius(self.fftlog.r, t) for t in self.wp_terms]
        else:
            r = np.logspace(-0.5, 2.5, 300)
            _ = self.get_xi_array(z)
            wp_terms_pimax = []
            for xi in self.xi_terms:
                _wp = self._xi2wp_pimax(r, ius(self.fftlog.r, xi), self.config["pi_max"])
                wp_terms_pimax.append(_wp)
            return [ius(r, t) for t in wp_terms_pimax]
    
    def get_ds_terms(self, z):
        _ = self.get_ds_array(z)
        return [ius(self.fftlog.r, t) for t in self.ds_terms]

    # some utils
    def get_sigma8_at_z(self, z, logkmin=-4, logkmax=1, nint=100):
        R = 8.
        ks = np.logspace(logkmin, logkmax, nint)
        logks = np.log(ks)
        kR = ks * R
        integrant = ks**3*self.get_pklin_from_z(ks, z)*window_tophat(kR)**2
        return np.sqrt(integrate.trapz(integrant, logks)/(2.*np.pi**2))

    def get_sigma8_at_current_redshift(self, logkmin=-4, logkmax=1, nint=100):
        return self.get_sigma8_at_z(self.current_redshift)

    # for magnification bias effect: see https://arxiv.org/pdf/1910.06400.pdf
    def dsLSS(self, r, zl, zs=1.0, N=20):
        """
        N=20 gives dsLSS as precise as 1%
        """
        z = np.linspace(1e-3, zl, N)
        h = self.halofit.cosmo.h

        xl = self.halofit.cosmo.comoving_distance(zl).value * h
        if hasattr(self, 'pz') and hasattr(self, 'source_z'):
            """
            print('computing <1/chi_s> with pz.')
            x_sources = self.halofit.cosmo.comoving_distance(self.source_z).value
            xs_inverse = simps(self.pz/x_sources, self.source_z)/simps(self.pz, self.source_z)
            xs = 1.0/xs_inverse
            print('<1/chi_s> = %f, 1/<1/chi_s> = %f'%(xs_inverse, xs))
            """
            x_sources = self.halofit.cosmo.comoving_distance(self.source_z).value * h
            norm = simps(self.pz, self.source_z)
            xs_over_xs_xl  = simps(self.pz*x_sources/(x_sources-xl), self.source_z)/norm
            one_over_xs_xl = simps(self.pz/(x_sources-xl), self.source_z)/norm
        else:
            xs = self.halofit.cosmo.comoving_distance(zs).value * h
            xs_over_xs_xl  = xs/(xs-xl)
            one_over_xs_xl = 1.0/(xs-xl)
        xz = self.halofit.cosmo.comoving_distance(z).value * h
        H0 = self.halofit.cosmo.H0.value*1e3/constants.c.value / h # /Mpc, h=1
        Hz = self.halofit.cosmo.H(z).value*1e3/constants.c.value / h # /Mpc, h=1

        dwpz = []
        for _z, _xz in zip(z, xz):
            k = self.fftlog.k
            _pklin = self.get_pklin_from_z(k, _z)
            self.halofit.set_pklin(k, _pklin, _z)
            _pkhalo = self.halofit.get_pkhalo()
            _dwpz = self.fftlog.pk2dwp(_pkhalo)
            _dwpz = ius(self.fftlog.r, _dwpz)(r * (_xz/xl))
            dwpz.append(_dwpz)
        dwpz = np.array(dwpz)

        Om = (1.0-self.cosmo.cparam[0][2])
        prefactor = 3.0/2.0 * H0* Om**2 * rho_cr / (1+zl) / 1e12

        #eff = (1+z)**2 * H0/Hz * xz**2/xl**2 * (xl-xz) *(xs-xz) / (xs-xl)
        eff = (1+z)**2 * H0/Hz * xz**2/xl**2 * (xl-xz) * (xs_over_xs_xl - xz*one_over_xs_xl)

        dsLSS = []
        for i,_r in enumerate(r):
            ans = simps(eff*dwpz[:,i], z)
            dsLSS.append(ans)
        dsLSS = np.array(dsLSS) * prefactor

        self.xs_over_xs_xl = xs_over_xs_xl
        self.one_over_xs_xl = one_over_xs_xl
        
        return dsLSS

    def wpLSS(self, r, zl, N=20):
        z = np.linspace(1e-3, zl, N)
        h = self.halofit.cosmo.h
        xl = self.halofit.cosmo.comoving_distance(zl).value * h
        xz = self.halofit.cosmo.comoving_distance(z).value * h
        H0 = self.halofit.cosmo.H0.value*1e3/constants.c.value / h # /Mpc, h=1
        Hz = self.halofit.cosmo.H(z).value*1e3/constants.c.value / h # /Mpc, h=1
        Om = (1.0-self.cosmo.cparam[0][2])

        
        wpz = []
        k = self.fftlog.k
        for _z, _xz in zip(z, xz):
            _pklin = self.get_pklin_from_z(k, _z)
            self.halofit.set_pklin(k, _pklin, _z)
            _pkhalo = self.halofit.get_pkhalo()
            _wpz = self.fftlog.pk2wp(_pkhalo)
            _wpz = ius(self.fftlog.r, _wpz)(r * (_xz/xl))
            wpz.append(_wpz)
        wpz = np.array(wpz)

        eff = 1.0/Hz * (xz*(xl-xz)/xl * (1.0+z))**2
        if type(self.config["pi_max"]) == float or type(self.config["pi_max"]) == int:
            prefactor = 2.0*self.config["pi_max"]
        else:
            print('pi max is needed. set 100Mpc/h')
            prefactor = 2.0*100.0
        prefactor *= (3.0/2.0*H0**2*Om)**2

        wpLSS = []
        for i,_r in enumerate(r):
            ans = simps(eff*wpz[:,i], z)
            wpLSS.append(ans)
        wpLSS = np.array(wpLSS) * prefactor

        return wpLSS
