import numpy as np

parameters_real = {
    "overexposed": {"min_gamma": 0.35, "max_gamma": 0.6},
    "dark": {"min_gamma": 1.5, "max_gamma": 2.5},
    "heavy_noise": {"mean_noise": 30, "variance_noise": np.random.randint(20, 55)},
    "motion_blur": {"min_proportion": 0.018, "max_proportion": 0.028},
    "small_motion_blur": {"min_proportion": 0.012, "max_proportion": .02},
    "blur": {"min_proportion": 0.012, "max_proportion": 0.022},
    "rdmrot": {"min_proportion": -10, "max_proportion": 10},
}

parameters_synth = {
    "overexposed": {"min_gamma": 0.3, "max_gamma": 0.6},
    "dark": {"min_gamma": 2, "max_gamma": 4},
    "heavy_noise": {"mean_noise": 0.40, "variance_noise": np.random.randint(20, 60)},
    "motion_blur": {"min_proportion": 0.018, "max_proportion": 0.028},
    "small_motion_blur": {"min_proportion": 0.012, "max_proportion": 0.02},
    "blur": {"min_proportion": 0.008, "max_proportion": 0.014},
    "initial": {"mean_noise": 6, "variance_noise": 5},
}


probs = {
    'dark': .5,
    'overexposed' : .5,
    'occluded': .5,
    'dark_occluded': .33,
    'rtp': 1,
    'dark_rtp': .5,
    'ccw_rpt': 1,
    'ccw': 1,
    'dark_ccw': .5,
    'ocluded_rtp': .25,
    'blur': .5,
    'rpt_blur': .66,
    'ccw_blur' : .66,
    'upside_down_blur': .4,
    'upside_down_dark': .2,
    'upside_down_ccw': .4,
    'upside_down_ocluded': .2,
    'heavy_noise_rdmrot': .4,
    'overexposed_occluded_rpt_ccw': .3,
    'dark_occluded_rpt_ccw': .2,
    'occluded_rpt_ccw': .2,
    'overexposed_occluded': .2,
    'heavy_noise': .5,
    'motion_blur': 1,
    'blur_ccw_rdmrot': .6,
    'dark_rdmrot': .3,
    'blur_rdmrot': .3,
    'heavy_noise_rdmrot': .2,
    'mblur_rdmrot': .6,
    'ccw_rpt_mblur': 1,
    'ccw_overexposed_mblur': .3,
    'ccw_dark_mblur': .8,
    'mblur_dark': .4,
    'mblur_overexposed': .4,
    'dark_rdmrot_mblur': .2
}