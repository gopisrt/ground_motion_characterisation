# ground_motion_characterisation

1. read_PEER_NGA_file.py reads the PEER NGA ground motion file and outputs time series (ground acceleration, velocity or displacement)
2. ground_motion_processing.py performs the processing through i) bandpass butterworth filter, ii) de-mean, and iii) cosine tapering.
3. ground_motion_fft.py computes Fourier transform of the ground motion
4. spectral_densities.py computes the cross-spectral density between two ground motions of same duration
5. smoothing_Hamming.py performs Haaming window sommthing of spectral densities
6. coherencies.py estimates complex coherency, lagged and unlagged coherencies between two ground motions of same duration


