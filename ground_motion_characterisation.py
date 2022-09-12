import math
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.fft import fft
	
## Read PEER NGA file

def read_PEER_NGA_file( filepath, scale_to_SI_units = True ):
	
	try:
		
		file_object = open( filepath, 'r' )
		raw_data  = file_object.readlines( )

		n_rows = len( raw_data )
		n_headers = 4
		
		last_header = raw_data[ n_headers - 1 ]
		npts = int( last_header[ last_header.index( 'NPTS= ' ) + 6 : last_header.index( ', ' ) ] )
		dt = float( last_header[ last_header.index( 'DT= ' ) + 4 : last_header.index( ' SEC' ) ] )
		
		raw_series = [ ]

		for i1 in range( n_headers, n_rows ):
			
			raw_series += str( raw_data[ i1 ] ).split( )
		   
		time_vector = np.arange( 0,  npts * dt, dt )
		
		if scale_to_SI_units:
			
			units_header = raw_data[ n_headers - 2 ]
			units = units_header.index( 'OF ' )
			
			match units_header[ units + 3 ]:
				
				case 'G':
					
					scale_fctr = 9.81
					
				case _:
					
					scale_fctr = 0.01
						
			raw_series = np.asarray( raw_series, dtype = float) * scale_fctr
		
		file_object.close( )
		
		return raw_series, npts, dt, time_vector
	
	except IOError:
		
		print( "File not found!" )
		
## Ground motion processing

def ground_motion_processing( raw_series, fs, low_cut_freq = 0.2, high_cut_freq = 20, order = 4, cosine_taper_frctn = 0.1 ):
	
	npts = len( raw_series )

	# bandpass filter
	nyq = 0.5 * fs	
	low = low_cut_freq / nyq
	high = high_cut_freq / nyq
	
	sos = butter( order, [ low, high ], analog = False, btype = 'band', output = 'sos' )	
	processed_series = sosfilt( sos, raw_series )
	 
	# De-mean
	processed_series = processed_series - sum( processed_series ) / npts
	
	# cosine tapering
	cosine_taper_npts = int( cosine_taper_frctn * npts )
	cosine_taper_vctr = 0.5 * ( 1 - np.cos( math.pi * np.linspace(0, cosine_taper_npts, cosine_taper_npts +1, endpoint=True) / npts ) )
	
	processed_series[ 0 : cosine_taper_npts + 1] = np.multiply( processed_series[ 0 : cosine_taper_npts + 1], cosine_taper_vctr )
	processed_series[ npts - cosine_taper_npts - 1 :  ] = np.multiply( processed_series[ npts - cosine_taper_npts - 1 :  ], cosine_taper_vctr )
	
	return processed_series

def ground_motion_fft( ground_motion, fs, f_upper_limit ):
    
    # Frequency vector genration
    npts = len( ground_motion )
    nfft = 2 ** math.ceil( math.log2( npts ) )
    fh = np.linspace( 0, fs / 2, int( nfft / 2+1 ) )
    fh_upper_indx = np.argmin( abs( np.array(fh) - f_upper_limit ) )
    fh = fh[ : fh_upper_indx ]
    
    # Computing FFT
    fft_ground_motion = fft( ground_motion ) / fs
    fft_ground_motion = fft_ground_motion[ : fh_upper_indx ]    
    abs_fft_ground_motion = abs( fft_ground_motion )
        
    return fh, fft_ground_motion, abs_fft_ground_motion

def spectral_densities( ground_motion_1, ground_motion_2, fs, f_upper_limit ):
    
    # This routine assumes ground motion 1 and 2 have idential duration    
    T_duration = ( len( ground_motion_1 ) - 1 ) / fs
    # FFT of ground motion 1
    fh_upper, fft_ground_motion_1, abs_fft_ground_motion_1 = ground_motion_fft( ground_motion_1, fs, f_upper_limit )
    
    # FFT of ground motion 1
    fh_upper, fft_ground_motion_2, abs_fft_ground_motion_2 = ground_motion_fft( ground_motion_2, fs, f_upper_limit )
    
    # Cross spectral density between ground motions 1 and 2
    cross_spectral_density = ( np.multiply( fft_ground_motion_1, np.conj( fft_ground_motion_2 ) ) ) / T_duration

    return fh_upper, cross_spectral_density

def smoothing_Hamming( M, unsmoothed_series ):
    
    # Smoothing vector
    m = np.linspace( -M, M, 2 * M + 1)
    hamm_fctr = 0.538
    smoothing_vctr= ( hamm_fctr - ( 1 - hamm_fctr ) * np.cos( math.pi * ( m + M ) / M ) ) / ( 1.08 * M )
    
    # Moving average through convolution
    smoothed_series = np.convolve( smoothing_vctr, unsmoothed_series )
    smoothed_series = smoothed_series[ M : -M ]

    return smoothed_series

def coherencies( ground_motion_1, ground_motion_2, fs, f_upper_limit, M ):
    
    # Unsmoothed cross spectral density between ground motions 1 and 2
    fh_upper, cross_spectral_density = spectral_densities( ground_motion_1, ground_motion_2, fs, f_upper_limit )
    
    # Unsmoothed auto (power) spectral densities
    fh_upper, auto_spectral_density_1 = spectral_densities( ground_motion_1, ground_motion_1, fs, f_upper_limit )
    fh_upper, auto_spectral_density_2 = spectral_densities( ground_motion_2, ground_motion_2, fs, f_upper_limit )

    # Smoothed spectral densities
    smoothed_cross_spectral_density = smoothing_Hamming( M, cross_spectral_density )
    smoothed_auto_spectral_density_1 = smoothing_Hamming( M, auto_spectral_density_1 )
    smoothed_auto_spectral_density_2 = smoothing_Hamming( M, auto_spectral_density_2 )
    
    # Coherency estimates
    coherency = np.divide( smoothed_cross_spectral_density, np.power ( np.multiply( smoothed_auto_spectral_density_1, 
                                                                              smoothed_auto_spectral_density_2 ), 0.5 ))
    lagged_coherency = abs(coherency)
    unlagged_coherency = np.real(coherency)

    return fh_upper, coherency, lagged_coherency, unlagged_coherency