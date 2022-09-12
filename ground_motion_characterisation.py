from math import pi
import numpy as np
from scipy.signal import butter, sosfilt
	
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
	cosine_taper_vctr = 0.5 * ( 1 -np.cos( pi * np.linspace(0, cosine_taper_npts, cosine_taper_npts +1, endpoint=True) / npts ) )
	
	processed_series[ 0 : cosine_taper_npts + 1] = np.dot( processed_series[ 0 : cosine_taper_npts + 1], cosine_taper_vctr )
	processed_series[ npts - cosine_taper_npts - 1 :  ] = np.dot( processed_series[ npts - cosine_taper_npts - 1 :  ], cosine_taper_vctr )
	
	return processed_series