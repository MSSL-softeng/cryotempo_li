#!/usr/bin/python
# -*- coding: utf-8 -*-
# sampy.py

"""
Anno Domini 02/02/2021 S. DINARDO fecit :   Creation, version 1.00

                            #####################################
                            #                                   #
                            #    Classes, functions             #
                            #    and data constants which       #
                            #    allow to retrack               #
                            #    SAR altimetry waveforms        #
                            #    with SAMOSA and SAMOSA+        #
                            #    retracker                      #
                            #####################################

Classes :

  - SAMOSA  -> class to retrack SAR altimetry waveform by SAMOSA and SAMOSA+ retracker

Functions :

  - initialize_epoch -> function to provide the guess epoch
  - compute_ThNEcho ->  function to provide the Thermal Noise from Echo

### TO DO LIST  ### ->->->

References:

REF1: Ray C., Martin-Puig C., Clarizia M.P., Ruffini G., Dinardo S., Gommenginger G., Benveniste J., (2014):
      SAR Altimeter Backscattered Waveform Model, IEEE Transactions on Geoscience and Remote Sensing,
      vol. 53, no. 2, pp. 911-919, Feb. 2015, https://doi.org/10.1109/TGRS.2014.2330423.
REF2: Dinardo S.,  Fenoglio l., Buchhaupt C., Becker, M., Scharroo R., Fernandes M.J., Benveniste J., (2017):
      Coastal SAR and PLRM Altimetry in German Bight and West Baltic Sea. Advances in Space Research. 62. https://doi.org/10.1016/j.asr.2017.12.018.
REF3: Dinardo, Salvatore, (2020). Techniques and Applications for Satellite SAR Altimetry over water, land and ice, 56.
      Technische Universitat, Darmstadt, https://doi.org/10.25534/tuprints-00011343, Ph.D. Thesis, ISBN 978-3-935631-45-7

"""

## Loading Libraries
## -----------------------

import numpy as np
import os
import scipy.optimize
import bottleneck as bn
import time
import logging

logger = logging.getLogger(__name__)


## -----------------------

# -------------------------------------------------------------------------
# =========================================================================================
# ============   D E F I N I T I O N  OF  F U N C T I O N S   ========================
# -------------------------------------------------------------------------
# Add here functions which are common and used by all the classes in the library

# ----------------------------------------------------------------------
def initialize_epoch(data, tau, Raw_Elevation, CST, size_half_block=10):
    """

       Function -> initialize_epoch(data,tau,Raw_Elevation,CST,size_half_block)
                   Function providing the first-guess epoch

          Input :

                data -> waveform data matrix (dimensions are rangeXrecords)
                tau ->  time for each sample of the waveforms in data
                Raw_Elevation -> Orbit Height minus one-way tracker range
                CST -> structure of constant (speedlight in CST.c0)
                size_half_block -> half size of the moving window (10 records generally)
    """

    dr = CST.c0 / 2 * np.mean(np.diff(tau))
    DX = Raw_Elevation / dr
    DX[np.where(np.isnan(DX))] = 0

    n, m = np.shape(data)

    if m > 30e3:

        threshold_pos = 5000 / dr
        threshold_neg = -400 / dr

    else:

        threshold_pos = 9000 / dr
        threshold_neg = -400 / dr

    DX[np.where(DX > threshold_pos)] = threshold_pos
    DX[np.where(DX < threshold_neg)] = threshold_neg

    DX = bn.nanmax(DX) - DX
    DN = np.around(bn.nanmax(DX)).astype(int).item()

    REGG = np.zeros((n + DN, m), dtype=np.float32)

    data_REGG = data / data.max()

    for i in np.arange(m):
        REGG[np.around(DX[i]).astype(int).item(): np.around(DX[i]).astype(int).item() + n, i] = data_REGG[:, i]

    COR_O = np.zeros((n, m))

    for i in np.arange(m):
        fi = max(0, i - size_half_block)
        la = min(m, i + size_half_block + 1)

        block = REGG[:, fi: la].astype(np.float64)

        block = np.delete(block, np.where(np.isnan(np.sum(block, axis=0))), axis=1)

        tmp = np.prod(block, axis=1)

        # handle tmp = 0, so we do not divide with zero when computing COR
        tmp = np.where(tmp < 1e-40, 1e-40, tmp)

        COR = tmp / bn.nanmax(tmp)
        COR_O[:, i] = COR[np.around(DX[i]).astype(int).item(): np.around(DX[i]).astype(int).item() + n]

    #np.save('samosa_cor0', COR_O)
    #np.save('samosa_regg', REGG)
    del REGG

    epoch0 = tau[np.argmax(COR_O, axis=0)]
    #np.save('samosa_argmax', np.argmax(COR_O,axis=0))
    #quit()
    del COR_O
    return epoch0


# ----------------------------------------------------------------------

def compute_ThNEcho(data, NstartNoise, NendNoise):
    """

       Function -> compute_ThNEcho(data,NstartNoise,NendNoise)
                   Function providing the Thermal Noise computed from the waveform

          Input :
                data -> waveform data matrix (dimensions are rangeXrecords)
                NstartNoise -> value of the range gate from which to start the noise window (counting from 1)
                NendNoise -> value of the range gate at which to stop the noise window (counting from 1)
    """

    NstartNoise = int(NstartNoise)
    NendNoise = int(NendNoise)

    data = np.sort(data[0:np.shape(data)[0] // 2, :], axis=0)
    data[np.where(data <= 0)] = np.nan

    if NstartNoise - 1 < 0:
        NstartNoise = 0

    if NendNoise > np.shape(data)[0] // 2:
        NendNoise = np.shape(data)[0] // 2

    ThNEcho = np.nanmedian(data[NstartNoise - 1:NendNoise, :], axis=0)
    ThNEcho[np.where(np.isnan(ThNEcho))] = bn.nanmedian(ThNEcho)
    return ThNEcho


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# ===================   END  OF  F UN C T I O N S   ===============================
# =========================================================================================

# -------------------------------------------------------------------------
# ===================   START OF THE CLASSES   ===============================
# =========================================================================================

# -------------------------------------------------------------------------------------------
#                           SAMOSA CLASS
# -------------------------------------------------------------------------------------------

# ===============================================================================================

class SAMOSA:
    """

    Class -> SAMOSA(self)

        Input :

    Class Public Methods :

            - retrack_SAMOSA() -> method to retrack a SAR waveform by SAMOSA or SAMOSA+ retracker

    """

    # ----------------------------------------------------------------------
    # ------------- Initialization------------------------------------------

    def __init__(self, CST, RDB, OPT, LUT, disable_ocean_like_check: bool = False):

        """

         Private Method -> SAMOSA Class's Init

             Input :

                    - CST : structure with fields: c0 (lightspeed in m/sec), R_2 ( Reference Ellipsoid  Earth Radius in m),
                            f_e (Reference Ellipsoid Earth Flatness), gamma_3_4 (gamma function value at 3/4)
                    - RDB : structure with fields: Np_burst (number of pulses in a burst), Npulse (number of range gates per pulse),
                            PRF_SAR (Pulse Repetition Frequency in SAR mode, in Hz), BRI (Burst Repetition Interval in sec),
                            f_0 (carrier frequency in Hz), Bs (Sampled Bandwidth) , theta_3x (antenna 3dB aperture along track in radiant),
                            theta_3y (antenna 3dB aperture across track in radiant),
                    - OPT : structure with fields: method (optimization solver method), ftol (exit tolerance on f), xtol (exit tolerance on x)
                            diff_step (relative step size for the finite difference approximation of the Jacobian) max_nfev (maximum number of function evaluations)
                            gtol (exit tolerance on gradient norm of f), loss (loss function) => see scipy.optimize.least_squares for details
                    - LUT : structure with fields: F0 (filename of the F0 LUT), F1 (filename of the F1 LUT),
                            alphap_noweight (filename of the alphap LUT in case no weighting), alphap_weight (filename of the alphap LUT in case of weighting),
                            alphapower_noweight (filename of the alpha power LUT in case no weighting).
                            All the LUT files must be in a folder named auxi and located in the same folder as sampy.py

         """

        self.disable_ocean_like_check = disable_ocean_like_check

        # print('\n  Initialiating the Class ...')
        folder = os.path.dirname(__file__) + os.sep + "auxi" + os.sep

        if not hasattr(LUT, 'F0'):
            logger.error('  Error: LUT Attribute F0 not given in F0')
            self.sucess = False

        if not hasattr(LUT, 'F1'):
            logger.error('  Error: LUT Attribute F1 not given in F1')
            self.sucess = False

        if not hasattr(LUT, 'alphap_noweight'):
            logger.error('  Error: LUT Attribute alphap_noweight not given in alphap_noweight')
            self.sucess = False

        if not hasattr(LUT, 'alphap_weight'):
            logger.error('  Error: LUT Attribute alphap_weight not given in alphap_weight')
            self.sucess = False

        if not hasattr(LUT, 'alphapower_noweight'):
            logger.error('  Error: LUT Attribute alphapower_noweight not given in alphapower_noweight')
            self.sucess = False

        if not hasattr(LUT, 'alphapower_weight'):
            logger.error('  Error: LUT Attribute alphapower_weight not given in alphapower_weight')
            self.sucess = False

        if os.path.isfile(folder + os.path.sep + LUT.F0):

            self.F0_LUT = np.genfromtxt(folder + os.path.sep + LUT.F0, dtype='float', comments='#', delimiter=None)

        else:
            logger.error(
                "  Fatal Error: file " + folder + os.path.sep + LUT.F0 + ' does not exist... exiting the class ')
            self.sucess = False

        if os.path.isfile(folder + os.path.sep + LUT.F1):

            self.F1_LUT = np.genfromtxt(folder + os.path.sep + LUT.F1, dtype='float', comments='#', delimiter=None)

        else:
            logger.error(
                "  Fatal Error: file " + folder + os.path.sep + LUT.F1 + ' does not exist... exiting the class ')
            self.sucess = False

        if os.path.isfile(folder + os.path.sep + LUT.alphap_noweight):

            self.alphap_LUT_NoWght = np.genfromtxt(folder + os.path.sep + LUT.alphap_noweight, dtype='float',
                                                   comments='#', delimiter=',')

        else:
            logger.error(
                "  Fatal Error: file " + folder + os.path.sep + LUT.alphap_noweight + ' does not exist... exiting the class ')
            self.sucess = False

        if os.path.isfile(folder + os.path.sep + LUT.alphap_weight):

            self.alphap_LUT_Wght = np.genfromtxt(folder + os.path.sep + LUT.alphap_weight, dtype='float', comments='#',
                                                 delimiter=',')

        else:
            logger.error(
                "  Fatal Error: file " + folder + os.path.sep + LUT.alphap_weight + ' does not exist... exiting the class ')
            self.sucess = False

        if os.path.isfile(folder + os.path.sep + LUT.alphapower_noweight):

            self.alphapower_LUT_NoWght = np.genfromtxt(folder + os.path.sep + LUT.alphapower_noweight, dtype='float',
                                                       comments='#', delimiter=',')

        else:
            logger.error(
                "  Fatal Error: file " + folder + os.path.sep + LUT.alphapower_noweight + ' does not exist... exiting the class ')
            self.success = False

        if os.path.isfile(folder + os.path.sep + LUT.alphapower_weight):

            self.alphapower_LUT_Wght = np.genfromtxt(folder + os.path.sep + LUT.alphapower_weight, dtype='float',
                                                     comments='#', delimiter=',')

        else:
            logger.error(
                "  Fatal Error: file " + folder + os.path.sep + LUT.alphapower_weight + ' does not exist... exiting the class ')
            self.success = False

        if not hasattr(CST, 'c0'):
            logger.error('  Error: CST Attribute lightspeed not given in c0')
            self.sucess = False

        if not hasattr(CST, 'R_e'):
            logger.error('  Fatal Error: CST Attribute Earth Radius not given in R_e')
            self.sucess = False

        if not hasattr(CST, 'f_e'):
            logger.error('  Fatal Error: CST Attribute Earth Flatness not given in f_e')
            self.sucess = False

        if not hasattr(CST, 'gamma_3_4'):
            logger.error('  Fatal Error: CST Attribute Gamma Function Value at 3/4 not given in gamma_3_4')
            self.sucess = False

        if not hasattr(RDB, 'Np_burst'):
            logger.error('  Fatal Error: RDB Attribute Number of pulses per burst not given in Np_burst')
            self.sucess = False

        if not hasattr(RDB, 'PRF_SAR'):
            logger.error('  Fatal Error: RDB Attribute SAR PRF not given in PRF_SAR')
            self.sucess = False

        if not hasattr(RDB, 'BRI'):
            logger.error('  Fatal Error: RDB Attribute Burst Repetition Interval not given in BRI')
            self.sucess = False

        if not hasattr(RDB, 'f_0'):
            logger.error('  Fatal Error: RDB Attribute Carrier Frequency not given in f_0')
            self.sucess = False

        if not hasattr(RDB, 'Bs'):
            logger.error('  Fatal Error: RDB Attribute Sampled Bandwidth not given in Bs')
            self.sucess = False

        if not hasattr(RDB, 'theta_3x'):
            logger.error('  Fatal Error: RDB Attribute 3dB antenna aperture along-track not given in theta_3x')
            self.sucess = False

        if not hasattr(RDB, 'theta_3y'):
            logger.error('  Fatal Error: RDB Attribute 3dB antenna aperture across-track not given in theta_3y')
            self.sucess = False

        if not hasattr(OPT, 'method'):
            logger.error('  Fatal Error: OPT Attribute method not given in method')
            self.sucess = False

        if not hasattr(OPT, 'ftol'):
            logger.error('  Fatal Error: OPT Attribute function exit tolerance not given in ftol')
            self.sucess = False

        if not hasattr(OPT, 'xtol'):
            logger.error('  Fatal Error: OPT Attribute x exit tolerance not given in xtol')
            self.sucess = False

        if not hasattr(OPT, 'gtol'):
            logger.error('  Fatal Error: OPT Attribute gradient exit tolerance not given in gtol')
            self.sucess = False

        if not hasattr(OPT, 'diff_step'):
            logger.error('  Fatal Error: OPT Attribute difference step not given in diff_step')
            self.sucess = False

        if not hasattr(OPT, 'max_nfev'):
            logger.error('  Fatal Error: OPT Attribute max evaluated function number not given in max_nfev')
            self.sucess = False

        if not hasattr(OPT, 'loss'):
            logger.error('  Fatal Error: OPT Attribute loss function not given in loss')
            self.sucess = False

        self.CST = CST
        self.RDB = RDB
        self.OPT = OPT
        self.CONF = None

        self.RDB.PRI_SAR = 1. / self.RDB.PRF_SAR
        self.RDB.lambda_0 = self.CST.c0 / self.RDB.f_0
        self.RDB.dfa = self.RDB.PRF_SAR / self.RDB.Np_burst
        self.CST.ecc_e = np.sqrt((2. - self.CST.f_e) * self.CST.f_e)  # Earth Eccentricty
        self.CST.b_e = self.CST.R_e * np.sqrt(1. - self.CST.ecc_e ** 2)

        self.max_model = 1
        self.sucess = True
        # print('  Class initialized with success \n')
        self.earth_radius = 0
        self.kappa = 0
        self.alpha_x = 0
        self.alpha_y = 0
        self.Lg = 0
        self.Ly = 0
        self.orbit_slope = 0
        self.wf_zp = 0
        self.dr = 0
        logger.info('Initialised SAMOSA library')

        self.last_wfm_model = None
        self.last_wfm_ref = None
        self.last_nu = None
        self.check_nu_fit = None

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    def __Generate_SamosaDDM(self, epoch_ns, SWH, tau, l, GEO):

        """
         Private Method -> __Generate_SamosaDDM

             Input :
                    - self : class self
                    - epoch_ns : input epoch given in nanoseconds
                    - SWH : input SWH in meter
                    - tau : time array  (giving the time of each range gate of the waveform, tau=0 is given at the reference gate)
                    - l : input Doppler Beam index
                    - GEO : structure with fields ... (see method Retrack_Samosa)

             Output:

                    - DDM (Delay-Doppler Map)

        """

        epoch_sec = epoch_ns * 1e-9  ### epoch (convert back in second)

        tau = tau - epoch_sec  ### tau and epoch are both given in seconds
        Dk = (tau * self.RDB.Bs)

        if self.CONF.wf_weighted:

            if self.CONF.step == 1:

                ind = bn.nanargmin(abs(self.alphap_LUT_Wght[:, 0] - SWH))
                alpha_p = self.alphap_LUT_Wght[:, 1][ind]
                Lx = self.CST.c0 * GEO.Height / (2. * GEO.Vs * self.RDB.f_0 * self.RDB.Np_burst * self.RDB.PRI_SAR)

                alpha_power = 0.47356

            elif self.CONF.step == 2:

                alpha_p = 0.42349
                Lx = self.CST.c0 * GEO.Height / (
                            2. * GEO.Vs * self.RDB.f_0 * self.RDB.Np_burst * self.RDB.PRI_SAR) * self.CONF.wght_factor

                alpha_power = 0.47356

        elif ~self.CONF.wf_weighted:

            ind = bn.nanargmin(abs(self.alphap_LUT_NoWght[:, 0] - SWH))
            alpha_p = self.alphap_LUT_NoWght[:, 1][ind]
            Lx = self.CST.c0 * GEO.Height / (2. * GEO.Vs * self.RDB.f_0 * self.RDB.Np_burst * self.RDB.PRI_SAR)

            ind = bn.nanargmin(abs(self.alphapower_LUT_NoWght[:, 0] - SWH))
            alpha_power = self.alphapower_LUT_NoWght[:, 1][ind]

        else:

            logger.error('  Waveform Weighting Flag given in input ' + self.CONF.wf_weighted + ' not recognized')
            return np.nan * np.ones((len(Dk), len(l)))

        sigma_s = (SWH / (4. * self.CONF.Lz))
        sigma_z = (SWH / 4.)

        yk = 0 * Dk
        v_Dk = np.where(Dk > 0)
        yk[v_Dk] = self.Ly * np.sqrt(Dk[v_Dk])

        xl = Lx * l

        ls = self.CONF.flag_slope * self.orbit_slope * GEO.Height / (self.kappa * Lx)

        gl = 1. / np.sqrt(alpha_p ** 2 + 4. * (alpha_p ** 2) * (Lx / self.Ly) ** 4 * (l - ls) ** 2 + np.sign(SWH) * (
                    SWH / (4. * self.CONF.Lz)) ** 2)

        csi = gl[None, :] * Dk[:, None]

        z = 1. / 4. * csi ** 2

        xp = +GEO.Height * GEO.Pitch
        yp = -GEO.Height * GEO.Roll

        Gamma_0 = np.exp(-self.alpha_y * yp ** 2 - self.alpha_x * (xl[None, :] - xp) ** 2 - xl[None,
                                                                                            :] ** 2 * GEO.nu / GEO.Height ** 2 -
                         (self.alpha_y + GEO.nu / GEO.Height ** 2) * yk[:, None] ** 2) * np.cosh(
            2. * self.alpha_y * yp * yk[:, None])

        T_kappa = np.zeros(np.shape(z))
        T_kappa[v_Dk, :] = (
                    (1 + GEO.nu / ((GEO.Height ** 2) * self.alpha_y)) - yp / (self.Ly * np.sqrt(Dk[v_Dk])) * np.tanh(
                2 * self.alpha_y * yp * self.Ly * np.sqrt(Dk[v_Dk]))[None, :]).T
        T_kappa[np.where(Dk <= 0), :] = (1 + GEO.nu / ((GEO.Height ** 2) * self.alpha_y)) - 2 * self.alpha_y * yp ** 2

        csi_max_F0 = bn.nanmax(self.F0_LUT[:, 0])
        csi_min_F0 = bn.nanmin(self.F0_LUT[:, 0])
        clip_F0 = np.bitwise_and(csi >= csi_min_F0, csi <= csi_max_F0)

        v_F0 = np.where(csi > csi_max_F0)
        f0 = np.zeros(np.shape(z))
        Index = np.floor(
            (len(self.F0_LUT[:, 0]) - 1) * ((csi[clip_F0] - csi_min_F0) / (csi_max_F0 - csi_min_F0))).astype(int)
        f0[clip_F0] = (csi[clip_F0] - self.F0_LUT[Index, 0]) * ((self.F0_LUT[Index + 1, 1] - self.F0_LUT[Index, 1]) / (
                    self.F0_LUT[Index + 1, 0] - self.F0_LUT[Index, 0])) + self.F0_LUT[Index, 1]
        f0[v_F0] = 1. / 2. * np.sqrt(np.pi) / (z[v_F0]) ** (1. / 4) * (
                    1. + 3. / (32. * z[v_F0]) + 105. / (2048. * (z[(csi > csi_max_F0)]) ** 2) + 10395. / (
                        196608. * (z[v_F0]) ** 3))
        f0[np.where(csi == 0)] = (1. / 2.) * (np.pi * 2 ** (3. / 4.)) / (2. * self.CST.gamma_3_4)
        f0[np.where(csi < csi_min_F0)] = 0

        csi_max_F1 = bn.nanmax(self.F1_LUT[:, 0])
        csi_min_F1 = bn.nanmin(self.F1_LUT[:, 0])
        clip_F1 = np.bitwise_and(csi >= csi_min_F1, csi <= csi_max_F1)

        v_F1 = np.where(csi > csi_max_F1)
        f1 = np.zeros(np.shape(z))
        Index = np.floor(
            (len(self.F1_LUT[:, 0]) - 1) * ((csi[clip_F1] - csi_min_F1) / (csi_max_F1 - csi_min_F1))).astype(int)
        f1[clip_F1] = (csi[clip_F1] - self.F1_LUT[Index, 0]) * ((self.F1_LUT[Index + 1, 1] - self.F1_LUT[Index, 1]) / (
                    self.F1_LUT[Index + 1, 0] - self.F1_LUT[Index, 0])) + self.F1_LUT[Index, 1]
        f1[v_F1] = (1. / 2.) * 1. / 4. * np.sqrt(np.pi) / (z[v_F1]) ** (3. / 4.)
        f1[np.where(csi == 0)] = -(1. / 2.) * (2. ** (3. / 4.)) * self.CST.gamma_3_4 / 2.
        f1[np.where(csi < csi_min_F1)] = 0

        f = (f0 + sigma_z / self.Lg * T_kappa * gl * sigma_s * f1)

        const = np.sqrt(2. * np.pi * alpha_power ** 4)

        return const * np.sqrt(gl) * Gamma_0 * f

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    def __Compute_Residuals(self, guess_triplet, tau, wf_norm, LookAngles, MaskRanges, GEO):

        """
         Private Method -> __Compute_Residuals

             Input :

                    - self : class self
                    - guess_triplet : triplet of guess epoch (in ns), SWH (or nu for second step of SAMOSA+), and Pu
                    - tau : time array  (giving the time of each range gate of the waveform, tau=0 is given at the reference gate)
                    - wf_norm : input normalized waveform
                    - LookAngles : input Look Angle Array of each Doppler Beam
                    - MaskRanges : input Mask Range Array
                    - GEO : structure with fields ... (see method Retrack_Samosa)

             Output :

                    - residuals : residuals between model waveform and data waveform
        """

        if LookAngles is None:
            dtheta = GEO.Vs * self.RDB.BRI / (GEO.Height * self.kappa)
            Theta1 = np.pi / 2 + dtheta * self.CONF.N_Look_min
            Theta2 = np.pi / 2 + dtheta * self.CONF.N_Look_max
            LookAngles = np.rad2deg(np.arange(Theta1, Theta2, dtheta))

        DopFreqs = (2 * GEO.Vs / self.RDB.lambda_0) * np.cos(np.deg2rad(LookAngles))
        BeamIndex = np.around(self.CONF.beamsamp_factor * DopFreqs / self.RDB.dfa) / self.CONF.beamsamp_factor
        span = np.where(np.diff(BeamIndex, axis=0) == 0)
        BeamIndex = np.delete(BeamIndex, span)

        if self.CONF.rtk_type == 'samosa' or self.CONF.step == 1:

            epoch_ns = guess_triplet[0]
            SWH = guess_triplet[1]
            Pu = guess_triplet[2]

            DDM = self.__Generate_SamosaDDM(epoch_ns, SWH, tau, BeamIndex, GEO)

        elif self.CONF.rtk_type == 'samosa+' and self.CONF.step == 2:

            epoch_ns = guess_triplet[0]
            GEO.nu = guess_triplet[1]
            Pu = guess_triplet[2]

            DDM = self.__Generate_SamosaDDM(epoch_ns, 0, tau, BeamIndex, GEO)

        else:

            logger.error('  SAMOSA Retracker Generation given in input ' + self.CONF.rtk_type + ' not recognized')
            return np.nan * np.ones(np.shape(wf_norm))

        if MaskRanges is None:

            Lx = self.CST.c0 * GEO.Height / (2. * GEO.Vs * self.RDB.f_0 * self.RDB.Np_burst * self.RDB.PRI_SAR)
            MaskRanges_demin = GEO.Height * (np.sqrt(1 + (self.kappa * ((Lx * BeamIndex) / GEO.Height) ** 2)) - 1)

        else:

            MaskRanges = np.delete(MaskRanges, span)
            MaskRanges_demin = MaskRanges - min(MaskRanges)

        R = np.tile(MaskRanges_demin, (len(wf_norm), 1))
        Dr = np.tile(self.dr * np.arange(len(wf_norm) - 1, -1, -1), (len(BeamIndex), 1)).T

        DDM[np.where(R >= Dr)] = 0

        Pr = bn.nansum(DDM, 1) / len(BeamIndex)

        Pr_max = bn.nanmax(Pr)

        self.max_model = Pr_max

        Pr = Pu * (Pr / Pr_max) + GEO.ThN_norm

        self.last_wfm_model = Pr
        self.last_wfm_ref = wf_norm
        self.last_nu = GEO.nu

        residuals = Pr - np.squeeze(wf_norm)

        return residuals

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    def Retrack_Samosa(self, tau, wf, LookAngles, MaskRanges, GEO, CONF):

        """
         Public Method -> Retrack_Samosa -> method to retrack a SAR (Unfocused) Altimetry waveform by SAMOSA or SAMOSA+ retracker

             Input :
                    - self : class self
                    - tau : time array  (giving the time of each range gate of the waveform, tau=0 is given at the reference gate)
                    - wf : input waveform
                    - LookAngles : input Look Angle Array of each Doppler Beams in degrees, set it to None if you dont have this input ( generated looks will be
                                   counted between N_Look_min and N_Look_max, both given in CONF)
                    - MaskRanges : input Mask Range Array in meter, set it to None if you dont have this input (in this case they will be
                                    autonomously computed by the library)
                    - GEO  : structure with fields: LAT (latitude in deg), LON (longitude in deg), Height (Orbit Height in m),
                             Vs (Satellite Velocity in m/sec), Hrate (Orbit Height Rate in m/sec), Pitch (Altimeter Pitch in radiant),
                             Roll (Altimeter Roll in radiant), nu (inverse of mean square slope), ThN (Thermal Noise)
                             track_sign (if Track Ascending => -1, if Track Descending => +1, set it to zero if flag_slope=0 in CONF )
                    - CONF : structure with fields: flag_slope (flag to include in the model the slope of orbit and surface), wf_weighted (set it to True if waveform is weighted)
                             beamsamp_factor (1 means only one beam per resolution cell is generated in the DDM), N_Look_min (number of the first Look to generate in the DDM),
                             N_Look_max (number of the last Look to generate in the DDM), guess_epoch (first-guess epoch in second), guess_swh (first-guess swh in m),
                             guess_pu (first-guess Pu), guess_nu (first-guess nu), lb_epoch (lower bound on epoch in sec), lb_swh (lower bound in swh in m),
                             lb_pu (lower bound on Pu), lb_nu (lower bound on nu), ub_epoch (upper bound on epoch in sec), ub_swh (upper bound in swh in m),
                             ub_pu (upper bound on Pu), ub_nu (upper bound on nu), rtk_type (it can be 'samosa' to retrack the waveform with SAMOSA retracker
                             or it can be 'samosa+' to retrack the waveform with SAMOSA+ retracker)

             Output :

                    - epoch in seconds
                    - SWH in meter
                    - Amplitude Pu
                    - misfit
                    - ocean-like flag (1 means openocean, 0 means non-openocean)

        """
        if not hasattr(GEO, 'LAT'):
            logger.error('  Fatal Error: GEO Attribute Latitude not given in LAT -> output padded to nan')
            out = type('', (), {})();
            out.x = np.full([5], np.nan);
            return out.x[0], out.x[1], out.x[2], out.x[3], out.x[4]

        if not hasattr(GEO, 'LON'):
            logger.error('  Fatal Error: GEO Attribute Longitude not given in LON -> output padded to nan')
            out = type('', (), {})();
            out.x = np.full([5], np.nan);
            return out.x[0], out.x[1], out.x[2], out.x[3], out.x[4]

        if not hasattr(GEO, 'Height'):
            logger.error('  Fatal Error: GEO Attribute Orbit Height not given in Height -> output padded to nan')
            out = type('', (), {})();
            out.x = np.full([5], np.nan);
            return out.x[0], out.x[1], out.x[2], out.x[3], out.x[4]

        if not hasattr(GEO, 'Vs'):
            logger.error('  Fatal Error: GEO Attribute Satellite Velocity not given in Vs -> output padded to nan')
            out = type('', (), {})();
            out.x = np.full([5], np.nan);
            return out.x[0], out.x[1], out.x[2], out.x[3], out.x[4]

        if not hasattr(GEO, 'Hrate'):
            logger.error('  Fatal Error: GEO Attribute Orbit Height Rate not given in Hrate -> output padded to nan')
            out = type('', (), {})();
            out.x = np.full([5], np.nan);
            return out.x[0], out.x[1], out.x[2], out.x[3], out.x[4]

        if not hasattr(GEO, 'Pitch'):
            logger.error('  Fatal Error: GEO Attribute Pitch not given in Pitch -> output padded to nan')
            out = type('', (), {})();
            out.x = np.full([5], np.nan);
            return out.x[0], out.x[1], out.x[2], out.x[3], out.x[4]

        if not hasattr(GEO, 'Roll'):
            logger.error('  Fatal Error: GEO Attribute Roll not given in Roll -> output padded to nan')
            out = type('', (), {})();
            out.x = np.full([5], np.nan);
            return out.x[0], out.x[1], out.x[2], out.x[3], out.x[4]

        if not hasattr(GEO, 'nu'):
            logger.error('  Fatal Error: GEO Attribute nu not given in nu -> output padded to nan')
            out = type('', (), {})();
            out.x = np.full([5], np.nan);
            return out.x[0], out.x[1], out.x[2], out.x[3], out.x[4]

        if not hasattr(GEO, 'track_sign'):
            logger.error('  Fatal Error: GEO Attribute track sign not given in track_sign -> output padded to nan')
            out = type('', (), {})();
            out.x = np.full([5], np.nan);
            return out.x[0], out.x[1], out.x[2], out.x[3], out.x[4]

        if not hasattr(GEO, 'ThN'):
            logger.error('  Fatal Error: GEO Attribute Thermal Noise not given in ThN -> output padded to nan')
            out = type('', (), {})();
            out.x = np.full([5], np.nan);
            return out.x[0], out.x[1], out.x[2], out.x[3], out.x[4]

        if not self.sucess:
            logger.error(
                '  Fatal Error: SAMOSA Class not initialized with Success: please initialize first the class -> output padded to nan')
            out = type('', (), {})();
            out.x = np.full([5], np.nan);
            return out.x[0], out.x[1], out.x[2], out.x[3], out.x[4]

        if not np.ndim(wf) == 1:
            logger.error('  Fatal Error: waveform in wf must have only one dimension -> output padded to nan')
            out = type('', (), {})();
            out.x = np.full([5], np.nan);
            return out.x[0], out.x[1], out.x[2], out.x[3], out.x[4]

        if not np.shape(tau) == np.shape(wf):
            logger.error(
                '  Fatal Error: time in tau and waveform in wf must have the same size -> output padded to nan')
            out = type('', (), {})();
            out.x = np.full([5], np.nan);
            return out.x[0], out.x[1], out.x[2], out.x[3], out.x[4]

        if CONF.lb_epoch is None:
            CONF.lb_epoch = tau[0] * 1e9

        if CONF.ub_epoch is None:
            CONF.ub_epoch = tau[-1] * 1e9

        if CONF.flag_slope:
            CONF.flag_slope = 1
        else:
            CONF.flag_slope = 0

        CONF.lb = [CONF.lb_epoch, CONF.lb_swh, CONF.lb_pu]
        CONF.ub = [CONF.ub_epoch, CONF.ub_swh, CONF.ub_pu]

        guess_triplet = [CONF.guess_epoch * 1e9, CONF.guess_swh, CONF.guess_pu]
        self.CONF = CONF
        self.CONF.step = 1
        ## TODO Consider using first peak instead of max_peak
        wf_max = bn.nanmax(wf)
        if wf_max == 0.0:
            logger.warning(
                '  Bad Waveform: all zero -> output padded to nan')
            out = type('', (), {})();
            out.x = np.full([5], np.nan);
            return out.x[0], out.x[1], out.x[2], out.x[3], out.x[4]
        wf_norm = wf / wf_max

        with np.errstate(invalid='ignore'):
            with np.errstate(divide='ignore'):
                E = - bn.nansum(wf_norm ** 2 * np.log2(wf_norm ** 2), axis=0)
        PP = 1. / bn.nansum(wf_norm, axis=0)

        GEO.ThN_norm = GEO.ThN / wf_max

        self.earth_radius = np.sqrt(self.CST.R_e ** 2.0 * (np.cos(np.deg2rad(GEO.LAT))) ** 2 + self.CST.b_e ** 2.0 * (
            np.sin(np.deg2rad(GEO.LAT))) ** 2)  # MARIA
        self.kappa = (1. + GEO.Height / self.earth_radius)
        self.alpha_x = 8. * np.log(2.) / (GEO.Height ** 2 * self.RDB.theta_3x ** 2)
        self.alpha_y = 8. * np.log(2.) / (GEO.Height ** 2 * self.RDB.theta_3y ** 2)
        self.Lg = self.kappa / (2. * GEO.Height * self.alpha_y)
        self.Ly = np.sqrt(self.CST.c0 * GEO.Height / (self.kappa * self.RDB.Bs))
        self.orbit_slope = GEO.track_sign * (
                    (self.CST.R_e ** 2 - self.CST.b_e ** 2) / (2. * self.earth_radius ** 2)) * np.sin(
            np.deg2rad(2. * GEO.LAT)) - \
                           (-GEO.Hrate / GEO.Vs)
        self.wf_zp = len(tau) / self.RDB.Npulse
        self.dr = self.CST.c0 / (2 * self.RDB.Bs * self.wf_zp)

        try:
            out = scipy.optimize.least_squares(self.__Compute_Residuals, guess_triplet,
                                               bounds=(self.CONF.lb, self.CONF.ub), loss=self.OPT.loss,
                                               method=self.OPT.method, ftol=self.OPT.ftol, xtol=self.OPT.xtol,
                                               gtol=self.OPT.gtol,
                                               max_nfev=self.OPT.max_nfev,
                                               args=(tau, wf_norm, LookAngles, MaskRanges, GEO))

            swh = out.x[1]
            misfit = np.sqrt(1. / (len(tau)) * bn.nansum(out.fun ** 2)) * 100

            # TODO: Where does this come from? Check is this valid
            # NOTE: The ocean like check can be disabled by setting `disable_ocean_like_check = True` during
            #       the initialization of this class. This has been introduced for the retracking of sea ice
            #       waveforms.
            if self.disable_ocean_like_check:
                check = True
            else:
                check = E * PP < 0.68 or E * PP > 0.78 or (100 * PP) * self.wf_zp > 8 or (E / misfit) / self.wf_zp < 4

            if self.CONF.rtk_type == 'samosa+' and check:
                self.CONF.lb = [CONF.lb_epoch, CONF.lb_nu, CONF.lb_pu]
                self.CONF.ub = [CONF.ub_epoch, CONF.ub_nu, CONF.ub_pu]
                self.CONF.step = 2

                guess_triplet = [CONF.guess_epoch * 1e9, CONF.guess_nu, CONF.guess_pu]

                out = scipy.optimize.least_squares(self.__Compute_Residuals, guess_triplet,
                                                   bounds=(self.CONF.lb, self.CONF.ub), loss=self.OPT.loss,
                                                   method=self.OPT.method, ftol=self.OPT.ftol, xtol=self.OPT.xtol,
                                                   gtol=self.OPT.gtol,
                                                   max_nfev=self.OPT.max_nfev,
                                                   args=(tau, wf_norm, LookAngles, MaskRanges, GEO))

        except Exception as inst:

            logger.error(
                '  Fatal Error: Catched Exception in retracking: <<' + inst.__str__() + '>> ->output padded to nan')

            out = type('', (), {})();
            out.x = np.full([5], np.nan);
            return out.x[0], out.x[1], out.x[2], out.x[3], out.x[4], None, None

        Pu = (out.x[2] * bn.nanmax(wf) / self.max_model).item()
        out.model = out.fun + wf_norm
        oceanlike_flag = ~check

        return out.x[0] * 1e-9, swh, Pu, misfit, oceanlike_flag, out.model, wf_norm

# ==========================================================================================
# ===================   END OF THE C L A S S E S   ===================================
# =========================================================================================
