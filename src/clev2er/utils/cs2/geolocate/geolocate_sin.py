"""geolocate_sin.py

"""

import logging

import numpy as np
from scipy.optimize import OptimizeWarning

from clev2er.utils.cs2.geolocate import sarin_phase
from clev2er.utils.cs2.geolocate.lrm_slope import ecef_to_llh_pyproj, llh_to_ecef_pyproj

# too-many-branches, pylint: disable=R0912
# too-many-arguments, pylint: disable=R0913
# too-many-locals, pylint: disable=R0914
# too-many-statements, pylint: disable=R0915
# pylint: disable=R0801

log = logging.getLogger(__name__)


def solve_eqn(aaa, bbb, base_vec, crf_centre):
    """djb to document

    Args:
        aaa (_type_): _description_
        bbb (_type_): _description_
        base_vec (_type_): _description_
        crf_centre (_type_): _description_

    Returns:
        _type_: _description_
    """
    crf_point = np.zeros(3)

    d_xplus = 2.0 * bbb * base_vec[0] / np.power(base_vec[2], 2.0)

    d_befosqrt = 4.0 * np.power(bbb, 2.0) * np.power(base_vec[0], 2.0) / np.power(base_vec[2], 4.0)
    d_befosqrt -= (
        4.0
        * (np.power(base_vec[0], 2.0) / np.power(base_vec[2], 2.0) + 1.0)
        * (np.power(bbb, 2.0) / np.power(base_vec[2], 2.0) - aaa)
    )

    d_xplus += np.sqrt(d_befosqrt)
    d_xplus /= 2.0 * (np.power(base_vec[0], 2.0) / np.power(base_vec[2], 2.0) + 1.0)

    d_xxplus = d_xplus + crf_centre[0]

    d_zplus = (bbb - base_vec[0] * d_xplus) / base_vec[2]

    d_zzplus = d_zplus + crf_centre[2]

    d_xminus = 2.0 * bbb * base_vec[0] / np.power(base_vec[2], 2.0)
    d_xminus -= np.sqrt(d_befosqrt)
    d_xminus /= 2.0 * (np.power(base_vec[0], 2.0) / np.power(base_vec[2], 2.0) + 1.0)

    d_xxminus = d_xminus + crf_centre[0]

    d_zminus = (bbb - base_vec[0] * d_xminus) / base_vec[2]

    d_zzminus = d_zminus + crf_centre[2]

    if d_xxplus > d_xxminus:
        crf_point[0] = d_xxplus
        crf_point[2] = d_zzplus

    if d_xxminus > d_xxplus:
        crf_point[0] = d_xxminus
        crf_point[2] = d_zzminus

    return crf_point


def get_crf_in_efc(lon, lat, alt, vel_vec):
    """djb to document

    Args:
        lon (_type_): _description_
        lat (_type_): _description_
        alt (_type_): _description_
        vel_vec (_type_): _description_

    Returns:
        _type_: _description_
    """
    crf_axis = np.zeros((3, 3))
    efc_cog = np.zeros(3)
    nad = np.zeros(3)
    nad[0], nad[1], nad[2] = llh_to_ecef_pyproj(lat, lon, 0.0)
    log.debug("NADIR LLH %f %f %f", lat, lon, 0.0)
    log.debug("NADIR EFC %f %f %f len %f", nad[0], nad[1], nad[2], np.linalg.norm(nad))

    efc_cog[0], efc_cog[1], efc_cog[2] = llh_to_ecef_pyproj(lat, lon, alt)

    sat_nad_vec = nad - efc_cog
    log.debug(
        "SAT NAD VEC EFC %f %f %f len %f",
        sat_nad_vec[0],
        sat_nad_vec[1],
        sat_nad_vec[2],
        np.linalg.norm(sat_nad_vec),
    )

    ad_crf_axis1 = sat_nad_vec / np.linalg.norm(sat_nad_vec)
    ad_efc_nv = vel_vec / np.linalg.norm(vel_vec)

    scal_prod = np.dot(ad_crf_axis1, ad_efc_nv)
    ad_temp_vect = np.zeros(3)
    ad_temp_vect[0] = ad_efc_nv[0] - ad_crf_axis1[0] * scal_prod
    ad_temp_vect[1] = ad_efc_nv[1] - ad_crf_axis1[1] * scal_prod
    ad_temp_vect[2] = ad_efc_nv[2] - ad_crf_axis1[2] * scal_prod

    ad_crf_axis2 = ad_temp_vect / np.linalg.norm(ad_temp_vect)
    ad_crf_axis3 = np.cross(ad_crf_axis2, ad_crf_axis1)

    # crf_axis[0, :] = ad_crf_axis1
    # crf_axis[1, :] = ad_crf_axis2
    # crf_axis[2, :] = ad_crf_axis3
    # This fixed a problem, but I don't know if I'm abusing vector orientation here
    crf_axis[:, 0] = ad_crf_axis1
    crf_axis[:, 1] = ad_crf_axis2
    crf_axis[:, 2] = ad_crf_axis3

    return crf_axis, efc_cog


def rotation_matrix(crf_axis):
    """djb to document

    Args:
        crf_axis (_type_): _description_

    Returns:
        _type_: _description_
    """
    rot = np.zeros((3, 3))
    rot[0, 0] = np.dot([1, 0, 0], crf_axis[:, 0])
    rot[0, 1] = np.dot([1, 0, 0], crf_axis[:, 1])
    rot[0, 2] = np.dot([1, 0, 0], crf_axis[:, 2])
    rot[1, 0] = np.dot([0, 1, 0], crf_axis[:, 0])
    rot[1, 1] = np.dot([0, 1, 0], crf_axis[:, 1])
    rot[1, 2] = np.dot([0, 1, 0], crf_axis[:, 2])
    rot[2, 0] = np.dot([0, 0, 1], crf_axis[:, 0])
    rot[2, 1] = np.dot([0, 0, 1], crf_axis[:, 1])
    rot[2, 2] = np.dot([0, 0, 1], crf_axis[:, 2])
    return rot


def angle_to_poca(angle, lat, lon, alt, cor_range, vel_vec, base_vec):
    """djb to document

    Args:
        angle (_type_): _description_
        lat (_type_): _description_
        lon (_type_): _description_
        alt (_type_): _description_
        cor_range (_type_): _description_
        vel_vec (_type_): _description_
        base_vec (_type_): _description_

    Returns:
        _type_: _description_
    """
    #    base_vec[0] = 0.0
    #    base_vec[1] = 0.0
    #    base_vec[2] = -1.0
    log.debug("cor_range %f", cor_range)
    log.debug("alt-range %f", alt - cor_range)
    log.debug("vel_vec %s", str(vel_vec))
    log.debug("base_vec %s", str(base_vec))

    crf_centre = base_vec * (cor_range * np.sin(angle))
    log.debug("crf_centre %s", str(crf_centre))

    radius = cor_range * np.cos(angle)
    log.debug("radius %f", radius)

    aaa = np.power(radius, 2) - np.power(crf_centre[1], 2)
    bbb = base_vec[1] * crf_centre[1]

    crf_point = solve_eqn(aaa, bbb, base_vec, crf_centre)
    log.debug("crf_point %s", str(crf_point))

    try:
        crf_axis, efc_cog = get_crf_in_efc(lon, lat, alt, vel_vec)

    except ValueError as exc:
        log.error("Geolocation failed, Floating point exception: %s", exc)
        return np.nan, np.nan, np.nan
    rot = rotation_matrix(crf_axis)

    efc_vec = np.inner(rot, crf_point)
    log.debug("efc_vec %s len %f", str(efc_vec), np.linalg.norm(efc_vec))
    log.debug("efc_cog %s len %f", str(efc_cog), np.linalg.norm(efc_cog))
    efc_point = efc_vec + efc_cog
    log.debug("efc_point %s len %f", str(efc_point), np.linalg.norm(efc_point))
    lat_poca, lon_poca, elev_poca = ecef_to_llh_pyproj(efc_point[0], efc_point[1], efc_point[2])
    log.debug("SAT lat=%f lon=%f h=%f", lat, lon, alt)
    log.debug("POCA lat=%f lon=%f h=%f", lat_poca, lon_poca, elev_poca)
    if lon_poca > 180.0:
        lon_poca -= 360.0

    return lat_poca, lon_poca, elev_poca


def phase_to_angle(
    phase,
    wavelength=0.022084,
    baseline=1.1676,
    inferred_angle_cal_mult=1.02775,
    inferred_angle_cal_add=0.0,
):
    """djb to document

    Args:
        phase (_type_): _description_
        wavelength (float, optional): _description_. Defaults to 0.022084.
        baseline (float, optional): _description_. Defaults to 1.1676.
        inferred_angle_cal_mult (float, optional): _description_. Defaults to 1.02775.
        inferred_angle_cal_add (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    angle = phase * wavelength
    angle = -1.0 * angle / (2 * np.pi * baseline)

    angle = (inferred_angle_cal_mult * angle) + inferred_angle_cal_add
    return angle


def ambiguity_phase_offsets(phase, num_ambiguities):
    """phase offsets of the alternative solutions to test for this record

    The SARIn phase difference is only known modulo 2pi, so the POCA could lie
    in any of a series of across-track bands. With CryoSat's baseline one 2pi
    is 1.114 degrees of look angle, about 14.5km across track and 120-700m in
    height, so the candidates are well separated and a 1km DEM discriminates
    them easily.

    num_ambiguities = 0 reproduces the baselines B to F010 behaviour: a single
    alternative at -2pi or +2pi chosen by the sign of the phase. Because
    angle = -phase * wavelength / (2 pi * baseline), that alternative always
    has the opposite angle sign, ie it is the band on the far side of nadir.
    The band further out on the SAME side was never tested.

    num_ambiguities = N tests every band from -N to +N, so both neighbours are
    considered rather than only the one that crosses nadir.

    Args:
        phase (float): fitted phase difference in radians, |phase| <= pi
        num_ambiguities (int): 0 for the legacy single alternative, else the
            number of 2pi bands to search either side

    Returns:
        list : phase offsets in radians, excluding the zero offset which is
        the primary solution
    """
    if num_ambiguities <= 0:
        return [-2.0 * np.pi if phase > 0 else 2.0 * np.pi]
    return [k * 2.0 * np.pi for k in range(-num_ambiguities, num_ambiguities + 1) if k != 0]


def geolocate_sin(l1b, config, dem_ant, dem_grn, range_cor_20_ku, ind_wfm_retrack_20_ku):
    """djb to document

    Args:
        l1b (_type_): _description_
        config (_type_): _description_
        dem_ant (Dem): Antarctic DEM
        dem_grn (Dem): Greenland DEM
        range_cor_20_ku (_type_): _description_
        ind_wfm_retrack_20_ku (_type_): _description_

    Raises:
        sarin_phase.SINLocateError: _description_
        sarin_phase.SINLocateError: _description_
        sarin_phase.SINLocateError: _description_
        e: _description_

    Returns:
        _type_: _description_
    """
    # Do phase estimation
    # Use phase, vectors to get lat/lon/height

    # Extract parameter arrays from L1 file netcdf object
    lat_20_ku = l1b["lat_20_ku"][:].data
    lon_20_ku = l1b["lon_20_ku"][:].data
    ph_diff_waveform_20_ku = l1b["ph_diff_waveform_20_ku"][:].data
    coherence_waveform_20_ku = l1b["coherence_waveform_20_ku"][:].data
    sat_vel_vec_20_ku = l1b["sat_vel_vec_20_ku"][:].data
    inter_base_vec_20_ku = l1b["inter_base_vec_20_ku"][:].data
    alt_20_ku = l1b["alt_20_ku"][:].data

    # Find number of records
    nrec = len(lat_20_ku)

    # Allocate arrays for intermediate aand output parameters
    height_20_ku = np.zeros(nrec)
    final_lat_20_ku = np.zeros(nrec)
    final_lon_20_ku = np.zeros(nrec)

    angle_20_ku = np.zeros(nrec)

    lat_initial_20_ku = np.zeros(nrec)
    lon_initial_20_ku = np.zeros(nrec)

    # Alternative 2pi ambiguity solutions. num_ambiguities = 0 keeps the single
    # sign-selected alternative used by baselines B to F010; N tests bands -N
    # to +N. See ambiguity_phase_offsets.
    num_ambiguities = int(config["sin_geolocation"].get("unwrap_ambiguities", 0))
    num_alternatives = 1 if num_ambiguities <= 0 else 2 * num_ambiguities
    # reject a record if even the best candidate is this far from the DEM.
    # 0 disables, which is the baselines B to F010 behaviour.
    max_dem_diff = float(config["sin_geolocation"].get("unwrap_max_dem_diff_m", 0.0))

    lat_alt_20_ku = np.full((num_alternatives, nrec), np.nan)
    lon_alt_20_ku = np.full((num_alternatives, nrec), np.nan)
    height_alt_20_ku = np.full((num_alternatives, nrec), np.nan)

    config_fitter = config["sin_geolocation"]["phase_method"]
    if config_fitter == 1:
        fitter = sarin_phase.phase_fit_lsq
    elif config_fitter == 2:
        fitter = sarin_phase.phase_fit_cuf
    else:
        fitter = (
            sarin_phase.phase_fit_sample
        )  # (sample window) 3: used in config/baseline_b_stage1.yml

    log_completed = 10
    log.info("Processing %d records", nrec)
    bad_1 = 0
    bad_2 = 0
    bad_3 = 0

    log.info(
        "Computing SARIN geolocation with method %s",
        str(config["sin_geolocation"]["phase_method"]),
    )
    log.info("Phase unwrapping is %s", str(config["sin_geolocation"]["unwrap"]))

    # ------------------------------------------------------------------------------
    # Process each record
    # ------------------------------------------------------------------------------

    for i in range(nrec):
        log.debug("processing record %d", i)
        complete = (i + 1) * 100.0 / nrec
        if complete >= log_completed:
            log_completed = log_completed + 10
            log.info("Completed %d%%", int(complete))
        try:
            # Check if inputs are OK
            if ind_wfm_retrack_20_ku[i] == -32768:  # This is the fill value used in stage1
                height_20_ku[i] = np.nan
                final_lat_20_ku[i] = np.nan
                final_lon_20_ku[i] = np.nan
                continue

            # Get the phase

            phase, bad_1, bad_2, bad_3 = fitter(
                ph_diff_waveform_20_ku[i],
                coherence_waveform_20_ku[i],
                ind_wfm_retrack_20_ku[i],
                bad_1,
                bad_2,
                bad_3,
                config,
            )
            if np.isnan(phase):
                raise sarin_phase.SINLocateError("Phase retrieval failed")
            if np.abs(phase) > np.pi:
                raise sarin_phase.SINLocateError("Phase out of bounds")
            # Get angle
            angle = phase_to_angle(phase)
            angle_20_ku[i] = angle

            # Calculate the POCA location and height
            log.debug("-- GEOLOCATING  --")
            lat_poca, lon_poca, elev_poca = angle_to_poca(
                angle,
                lat_20_ku[i],
                lon_20_ku[i],
                alt_20_ku[i],
                range_cor_20_ku[i],
                sat_vel_vec_20_ku[i],
                inter_base_vec_20_ku[i],
            )
            final_lat_20_ku[i] = lat_poca
            final_lon_20_ku[i] = lon_poca
            height_20_ku[i] = elev_poca

            if config["sin_geolocation"]["unwrap"]:
                offsets = ambiguity_phase_offsets(phase, num_ambiguities)
                for alt, offset in enumerate(offsets):
                    log.debug("-- GEOLOCATING  ambiguity offset %+.3f rad --", offset)
                    lat_poca, lon_poca, elev_poca = angle_to_poca(
                        phase_to_angle(phase + offset),
                        lat_20_ku[i],
                        lon_20_ku[i],
                        alt_20_ku[i],
                        range_cor_20_ku[i],
                        sat_vel_vec_20_ku[i],
                        inter_base_vec_20_ku[i],
                    )
                    lat_alt_20_ku[alt, i] = lat_poca
                    lon_alt_20_ku[alt, i] = lon_poca
                    height_alt_20_ku[alt, i] = elev_poca

            if (
                height_20_ku[i] > config["sin_geolocation"]["height_max"]
                or height_20_ku[i] < config["sin_geolocation"]["height_min"]
            ):
                raise sarin_phase.SINLocateError("Height out of bounds")
        except OptimizeWarning as exc:
            # Here to catch them for debugging but currently handled before here
            # by defaulting results
            # Doesn't indicate an error, indicates the model not fitting the data
            raise exc
        except sarin_phase.SINLocateError as exc:
            log.debug("Defaulting results. Reason is %s", exc.msg)
            final_lat_20_ku[i] = np.nan
            final_lon_20_ku[i] = np.nan
            height_20_ku[i] = np.nan
            # raise e
    #
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    if config["sin_geolocation"]["unwrap"]:
        dem = dem_ant if lat_20_ku[0] < 0 else dem_grn

        def dem_at(lats, lons):
            """reference DEM elevation at the candidate locations"""
            return dem.interp_dem(lats, lons, method="linear", xy_is_latlon=True)

        # Deviation from the DEM for the primary solution and each alternative.
        # The trigger is evaluated against the PRIMARY deviation, as it was
        # when there was only one alternative.
        orig_dem = dem_at(final_lat_20_ku, final_lon_20_ku)
        orig_diff = np.abs(height_20_ku - orig_dem)
        best_diff = orig_diff.copy()

        lat_initial_20_ku[:] = final_lat_20_ku[:]
        lon_initial_20_ku[:] = final_lon_20_ku[:]

        triggered = orig_diff >= config["sin_geolocation"]["unwrap_trigger_m"]
        num_replaced = 0
        for alt in range(num_alternatives):
            alt_dem = dem_at(lat_alt_20_ku[alt], lon_alt_20_ku[alt])
            alt_diff = np.abs(height_alt_20_ku[alt] - alt_dem)
            # np.less treats NaN as False, so candidates that failed to
            # geolocate can never win
            idx = np.where(np.bitwise_and(np.less(alt_diff, best_diff), triggered))[0]
            if idx.size > 0:
                height_20_ku[idx] = height_alt_20_ku[alt][idx]
                final_lat_20_ku[idx] = lat_alt_20_ku[alt][idx]
                final_lon_20_ku[idx] = lon_alt_20_ku[alt][idx]
                best_diff[idx] = alt_diff[idx]
                num_replaced += idx.size
        log.info(
            "Phase unwrapping: %d replacements from %d alternative solution(s) per record",
            num_replaced,
            num_alternatives,
        )

        # Reject records whose best candidate is still implausibly far from the
        # DEM. Widening the search gives more chances for a spurious solution
        # to sit closer to the DEM than the true one, so this bounds the damage.
        if max_dem_diff > 0:
            reject = np.where(best_diff > max_dem_diff)[0]
            if reject.size > 0:
                height_20_ku[reject] = np.nan
                final_lat_20_ku[reject] = np.nan
                final_lon_20_ku[reject] = np.nan
                log.info(
                    "Rejected %d of %d measurements more than %.1fm from the DEM",
                    reject.size,
                    nrec,
                    max_dem_diff,
                )

    log.debug("bad counts 1=%d 2=%d 3=%d", bad_1, bad_2, bad_3)
    log.info("Processed %d records", i + 1)

    return height_20_ku, final_lat_20_ku, final_lon_20_ku
