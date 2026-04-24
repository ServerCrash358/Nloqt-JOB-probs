"""
Fluorescence Spectroscopy Simulation: NV Center Charge States
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

@dataclass
class NVCenterSpectralParams:
    lambda_zpl_minus = 638e-9  
    sigma_zpl_minus = 2e-9     
    lambda_zpl_zero = 575e-9   
    sigma_zpl_zero = 2.5e-9    
    phonon_energy_minus = 0.064  
    phonon_energy_zero = 0.070   
    qe_zpl_minus = 0.03   
    qe_psb_minus = 0.97   
    qe_zpl_zero = 0.025   
    qe_psb_zero = 0.975   

class FluorescenceSpectrumSimulator:
    def __init__(self):
        self.params = NVCenterSpectralParams()
        self.wavelength_array = np.linspace(500e-9, 850e-9, 2000)
        
    def lorentzian(self, w, lc, fwhm, amp):
        gamma = fwhm / 2
        return amp * (gamma**2) / ((w - lc)**2 + gamma**2)
    
    def gaussian(self, w, lc, sigma, amp):
        return amp * np.exp(-((w - lc)**2) / (2 * sigma**2))
    
    def psb(self, w, lc, E_p, T=293):
        e_zpl = (6.626e-34 * 3e8) / lc
        thermal = np.exp(-E_p * 1.602e-19 / (1.381e-23 * T))
        lc_psb = lc * (1 + E_p / (e_zpl / 1.602e-19))
        return self.gaussian(w, lc_psb, 10e-9, thermal)
    
    def simulate_nv_minus(self, conc=1.0, T=293, bg=0.02):
        zpl = self.lorentzian(self.wavelength_array, self.params.lambda_zpl_minus, self.params.sigma_zpl_minus, conc*self.params.qe_zpl_minus)
        psb = self.psb(self.wavelength_array, self.params.lambda_zpl_minus, self.params.phonon_energy_minus, T) * conc * self.params.qe_psb_minus
        return zpl + psb + bg
    
    def simulate_nv_zero(self, conc=1.0, T=293, bg=0.02):
        zpl = self.lorentzian(self.wavelength_array, self.params.lambda_zpl_zero, self.params.sigma_zpl_zero*1.2, conc*self.params.qe_zpl_zero)
        psb = self.psb(self.wavelength_array, self.params.lambda_zpl_zero, self.params.phonon_energy_zero, T) * conc * self.params.qe_psb_zero
        return zpl + psb + bg
    
    def simulate_mixed(self, ratio_minus=0.7):
        sm = self.simulate_nv_minus(conc=ratio_minus)
        sz = self.simulate_nv_zero(conc=1.0 - ratio_minus)
        return sm + sz, sm, sz
        
    def calc_fingerprint(self, spec):
        blue_mask = (self.wavelength_array >= 500e-9) & (self.wavelength_array <= 620e-9)
        blue_int = np.trapezoid(spec[blue_mask], self.wavelength_array[blue_mask])
        
        red_mask = (self.wavelength_array >= 655e-9) & (self.wavelength_array <= 750e-9)
        red_int = np.trapezoid(spec[red_mask], self.wavelength_array[red_mask])
        
        return {'red_blue_ratio': red_int / (blue_int + 1e-10)}

class pHSensingSimulator:
    def __init__(self):
        self.sim = FluorescenceSpectrumSimulator()
        
    def simulate_pH(self):
        pH_vals = np.linspace(2, 12, 21)
        ratios, spectra = [], []
        for pH in pH_vals:
            nv_minus_ratio = np.exp(-((pH - 7.0) / 2.0)**2)
            spec, _, _ = self.sim.simulate_mixed(nv_minus_ratio)
            ratios.append(self.sim.calc_fingerprint(spec)['red_blue_ratio'])
            spectra.append(spec)
        return pH_vals, ratios, spectra

def run_spectroscopy_plots():
    sim = FluorescenceSpectrumSimulator()
    ph_sim = pHSensingSimulator()
    
    # Plot 1: Spectra Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    w_nm = sim.wavelength_array * 1e9
    sm, sz = sim.simulate_nv_minus(), sim.simulate_nv_zero()
    axes[0].plot(w_nm, sm, 'r-', label='NV-')
    axes[0].plot(w_nm, sz, 'b-', label='NV0')
    axes[0].set_title('Pure Spectra', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    mixed, m_comp, z_comp = sim.simulate_mixed(0.7)
    axes[1].plot(w_nm, mixed, 'k-', label='Mixed (70% NV-)')
    axes[1].plot(w_nm, m_comp, 'r--', label='NV- component')
    axes[1].plot(w_nm, z_comp, 'b--', label='NV0 component')
    axes[1].set_title('Mixed In-Vivo Spectrum', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.savefig('./02_Fluorescence_Spectra_Comparison.png', bbox_inches='tight')
    
    # Plot 2: pH Sensing
    pH_vals, ratios, spectra = ph_sim.simulate_pH()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for i, p in enumerate(pH_vals[::3]):
        axes[0].plot(w_nm, spectra[i*3], label=f'pH {p:.1f}')
    axes[0].set_title('Spectral Evolution with pH', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(pH_vals, ratios, 'g-o', linewidth=2.5)
    axes[1].axvline(x=7.0, color='gray', linestyle='--')
    axes[1].set_title('Red/Blue Ratio vs pH', fontweight='bold')
    axes[1].grid(alpha=0.3)
    plt.savefig('./03_pH_Sensing_Response.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_spectroscopy_plots()