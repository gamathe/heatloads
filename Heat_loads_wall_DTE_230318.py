import pandas as pd                 # Data tables
import os                           # Returns current directory, create files
import numpy as np                  # Arrays
import matplotlib.pyplot as plt	    # Graphs

from math import sqrt, atan, log, exp, sin, cos, tan

from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import *

pi = np.pi

def wall_matrix(n_layers,thickness,lambdaval,rho,c_layer):
    
    #! n_div finite elements of order two by layer"
    n_div=2 # mandatory"
    n_elem=n_div*n_layers

    #! n_div finite elements of order two by layer#
    Res=thickness/lambdaval
    R_nobl=np.sum(Res)
    
    # Matrix lines and rows numbered from 1 to 2*n_elem+1 > at the end of the procedure suppress first row and first column

    #!Define coefficients of C (thermal mass) and L (conductivity) symetric matrixes #
    C_el = np.zeros((4,4))
    L_el = np.zeros((4,4))
    C_el[1,1]=2/15 ; C_el[2,2]=8/15 ; C_el[3,3]=2/15 ; C_el[2,1]=1/15 ;C_el[3,1]=-1/30 ; C_el[3,2]=1/15 
    L_el[1,1]=7/3 ; L_el[2,2]=16/3 ; L_el[3,3]=7/3 ; L_el[2,1]=-8/3 ;L_el[3,1]=1/3 ; L_el[3,2]=-8/3

    #! Define H matrix#
    #! Define vectors : RLE=lambda/epaisseur d'élément, RCE=rho*c_p*epaisseur d'élément#
    RLE0=lambdaval/(thickness/n_div)
    RCE0=rho*c_layer*thickness/n_div
    
    RLE= [0]
    RLE.extend(RLE0)
    RCE= [0]
    RCE.extend(RCE0)

    #!Define symetric matrixes M_1=L*deltaT/2+C and M_0=-L*deltaT/2+C   :#
    shape = (2*n_elem+2,2*n_elem+2)
    L = np.zeros(shape)
    C = np.zeros(shape)
    
    #Termes diagonaux extrêmes#
    L[1,1]= RLE[1]*L_el[1,1]
    C[1,1]= RCE[1]*C_el[1,1]

    L[2*n_elem+1,2*n_elem+1]=RLE[n_layers]*L_el[3,3]
    C[2*n_elem+1,2*n_elem+1]=RCE[n_layers]*C_el[3,3]

    #Termes diagonaux aux endroits de superpositions entre couches différentes#
    for k in range(1,n_layers):
        L[2*n_div*k+1,2*n_div*k+1]=RLE[k]*L_el[3,3]+RLE[k+1]*L_el[1,1]
        C[2*n_div*k+1,2*n_div*k+1]=RCE[k]*C_el[3,3]+RCE[k+1]*C_el[1,1]
   
    #Termes diagonaux aux endroits de superpositions entre mêmes matériaux
    #cette boucle ne fonctionne que pour n_div=2, sinon ajouter des lignes}
    for k in range(1,n_layers+1):
        L[2*n_div*k-1,2*n_div*k-1]=2*RLE[k]*L_el[3,3]
        C[2*n_div*k-1,2*n_div*k-1]=2*RCE[k]*C_el[3,3]

    #Sous-matrices des termes [2;1] [3;1] [2;2] [3;2]#
    for k in range(1,n_layers+1):
        #Première sous-couche i=1, seconde i=2#
        for i in range(1,n_div+1):
            #Terme diagonal#
            L[4*k+2*i-6+2,4*k+2*i-6+2]=RLE[k]*L_el[2,2]
            C[4*k+2*i-6+2,4*k+2*i-6+2]=RCE[k]*C_el[2,2]
            #Trois termes sous la diagonale#
            L[4*k+2*i-6+2,4*k+2*i-6+1]=RLE[k]*L_el[2,1]
            C[4*k+2*i-6+2,4*k+2*i-6+1]=RCE[k]*C_el[2,1]
            L[4*k+2*i-6+3,4*k+2*i-6+1]=RLE[k]*L_el[3,1]
            C[4*k+2*i-6+3,4*k+2*i-6+1]=RCE[k]*C_el[3,1]
            L[4*k+2*i-6+3,4*k+2*i-6+2]=RLE[k]*L_el[3,2]
            C[4*k+2*i-6+3,4*k+2*i-6+2]=RCE[k]*C_el[3,2]

    #Symetry#
    for i in range(1,n_elem*2+2):
        for j in range(1,i):
            L[j,i]=L[i,j]
            C[j,i]=C[i,j]
            
    L=L[1:,1:]
    C=C[1:,1:]

    return n_elem,R_nobl,L,C

def CLF(alpha_wall, M_A, U, azimuth_w_deg, slope_w_deg, iflag_shading, month, t_in, imposed_t_out, t_out_max, DELTAt_out, method="LSODA"):

    M_per_A_wall=max(10,M_A)
    U_value_wall = U if U > 0 else 0.01
    R_value_wall = 1/U_value_wall

    t_init=26 #[°C]

    # "Air properties:"
    v_a=0.8401 # [m^3/kg] "specific volume of humid air per kg of dry air"
    c_p_a=1020 # [J/kg-K] "specific heat capacity of humid air per kg of dry air"
    sigma_boltzman=5.67E-8

    # "!Boundary layers"
    h_r=5 # [W/m^2-K]
    h_c=3 # [W/m^2-K]
    h_in=h_r + h_c
    h_out=17.5 # [W/m^2-K]

    # "!Days of simulation"
    n_day_sim=3

    # Wall azimuth angle gamma is comprised between -180° and 180°#
    if azimuth_w_deg > 180 : 
        gamma_w_deg = azimuth_w_deg-360
    else:
        gamma_w_deg = azimuth_w_deg

    # concrete bloc
    rho_concrete_bloc=1200 #[kg/m^3]
    c_p_concrete_bloc=840 #[J/kg.K]
#     lambda_concrete_bloc=1.273 #[W/m.K]

    # "!Total number of finite element layers, with two degree two elements by layer"
    n_layers = 2
    nl=n_layers

    thickness_w    = M_per_A_wall / rho_concrete_bloc
    
    # Imposed lambda value to meet the imposed thermal resistance
    lambda_concrete_bloc    = thickness_w / max(0.001, R_value_wall - 1/h_in - 1/h_out)
  
    # "! internal vertical wall layers"    
    thickness_wall = thickness_w/n_layers * np.ones(n_layers) 
    lambda_wall    = lambda_concrete_bloc * np.ones(n_layers)
    rho_wall       = rho_concrete_bloc * np.ones(n_layers)
    c_layer_wall   = c_p_concrete_bloc * np.ones(n_layers)

    # Matrixes of vertical wall layers"
    n_elem,R_nobl_wall,L_wall,C_wall = wall_matrix(n_layers,thickness_wall,lambda_wall,rho_wall,c_layer_wall)

    n_nodes=2*n_elem+1

    # Initial conditions"
    t_a_in_set = t_in + 0.01 #[°C]
    t_a_in_init=t_a_in_set

    # Simulation period
    n_day_sim=3
    hour_start=0
    n_hours_sim=24*n_day_sim
    hour_stop=hour_start + n_hours_sim

    tau_initial=hour_start*3600
    tau_final=hour_stop*3600

    DELTAtau=600 #[s]

    # Time in s : Create an array of evenly-spaced values
    tau = np.arange(tau_initial,tau_final+1,DELTAtau)

    # Hour and Day from the start of the simulation
    hour       = tau/3600
    hour_per_0 = hour-24*np.trunc(hour/24)
    # np.choose(condition,[action if condition = 0 or false, action if condition = 1 or true])
    # np.choose(array, [action if condition = 0, action if condition = 1 , action if condition = 2 ...)])
    hour_per=np.choose(hour_per_0 > 0.000001,[24,hour_per_0])

    day = hour/24
    day_int_0 = np.trunc(hour/24)+1
    day_int   = day_int_0-1

    # Sarting hour in sun data table according to month of simulation
    month_start=max(1,min(12,month))
    hour_start = np.zeros(13)
    hour_start[1]=1+15*24; hour_start[2]=1+(31+15)*24; hour_start[3]=1+(31+28+15)*24; hour_start[4]=1+(2*31+28+15)*24; hour_start[5]=1+(2*31+28+30+15)*24; hour_start[6]=1+(3*31+28+30+15)*24
    hour_start[7]=1+(3*31+28+2*30+15)*24; hour_start[8]=1+(4*31+28+2*30+15)*24; hour_start[9]=1+(5*31+28+2*30+15)*24; hour_start[10]=1+(5*31+28+3*30+15)*24
    hour_start[11]=1+(6*31+28+3*30+15)*24; hour_start[12]=1+(6*31+28+4*30+15)*24

    # Hour and Day from the start of the year (simulation starts at 15th of the considered month)
    hour_yr = hour + float(hour_start[month])
    day_yr = hour_yr/24

    # External dry and wet temperatures for July: hour by hour from 0 to 24h (local solar hour)
    h_sol = np.arange(25).astype(np.float32)

#     t_dry_july = np.array([21. , 18.5, 16. , 15.5, 15. , 15.5, 16. , 18.5, 21. , 24. , 27. , \
#                            29. , 31. , 31.5, 32. , 31.5, 31. , 30. , 29. , 27.5, 26. , 24.5, 23. , 22. , 21. ])
#     t_wet_july = np.array([16.15,15.24,14.3,14.11,13.92,14.11,14.3,15.24,16.15,17.21,18.22, \
#                            18.88,19.52,19.67,19.83,19.67,19.52,19.2,18.88,18.39,17.89,17.38,16.86,16.51,16.15])

    t_dry_july = np.array([20.6 , 20.3, 20.2 , 20.0, 20.1 , 20.3, 20.7, 21.3, 21.9 , 22.5 , 23.2 , \
                           23.8 , 26.6 , 29.2, 31.5 , 32.0, 31.5 , 30.3 , 28. , 25.6, 22.7 , 22.1, 21.6 , 21.1 , 20.6 ])
    
    # # Correction month by month - Max daily External dry and wet temperatures 
    dt_dry_m = np.array([-11. , -10. ,  -7.8,  -5.5,  -2.5,  -0.5,   0. ,   0. ,  -2.5,  -4.1,  -8.2, -10.2])
    dt_wet_m = np.array([-5.5, -5. , -3.9, -2.7, -2.3,  0. ,  0. ,  0. , -0.5, -2.3, -3.9, -5. ])

    dt_dry = dt_dry_m[month-1]
    dt_wet = dt_wet_m[month-1]

    # External dry and wet temperatures for the current month: hour by hour from 0 to 24h (local solar hour)
    t_dry_std = t_dry_july + dt_dry
    
    if imposed_t_out == 0 :
        t_dry = t_dry_std
    else :
        # Profile adapted to the data given for t_out_max and DELTAt_out
        t_dry_max_std  = np.max(t_dry_std)
        DELTAt_dry_std = np.max(t_dry_std) - np.min(t_dry_std)
        t_dry = t_out_max - (t_dry_max_std-t_dry_std) * (DELTAt_out/DELTAt_dry_std)

#     t_wet_std = t_wet_july - dt_wet
#     t_wet_max_std = np.max(t_wet_std)
#     DELTAt_wet_std = np.max(t_wet_std) - np.min(t_wet_std)
#     t_wet = t_out_max - (np.average(t_dry_std) - np.average(t_wet_std)) - (t_wet_max_std-t_wet_std) * (DELTAt_out/DELTAt_dry_std)

    df = pd.DataFrame(tau, columns=['tau'])

    df['hour']     = hour
    df['day']      = day
    df['day_int']  = day_int
    df['hour_yr']  = hour_yr
    df['day_yr']   = day_yr
    df['hour_per'] = hour_per

    df1 = pd.DataFrame(h_sol, columns=['hour_per'])
    df1['t_dry'] = t_dry
#     df1['t_wet'] = t_wet

    df = df.merge(df1, how='left')

    # replace NaN values with interpolated values
    df['t_dry'] = df.interpolate(method = 'values')['t_dry']
#     df['t_wet'] = df.interpolate(method = 'values')['t_wet']

    # df

    #Atmospheric pressure at sea level [Pa]
    p_0 = 101325 
    #Estimation of atmospheric pressure at local height
    #Scale height of the Rayleigh atmosphere near the earth surface [m]"
    z_h = 8434.5
    #Local height above the sea level [m]"
    z_local = 62 
    p_atm = exp(-z_local/z_h)*p_0

    # Relative humidity from t_dry and t_wet
    t_dry = df['t_dry'].values
    # t_wet = df['t_wet'].values
    # p_atm_hPa = p_atm/100
    # rh_out_0 = (np.exp(1.8096+(17.2694*t_wet/(237.3+t_wet))) - 7.866*10**-4 * p_atm_hPa * (t_dry-t_wet) * (1+t_wet/610)) \
    #         /(np.exp(1.8096+(17.2694*t_dry/(237.3+t_dry))))

    # # Where True, yield x, otherwise yield y.
    # rh_out = np.where(rh_out_0 >1, 1, rh_out_0)

    # T_dry = t_dry + 273
    # p_v_sat_out = np.exp(77.3450 + 0.0057 * T_dry- 7235 / T_dry)/ T_dry**8.2
    # p_v_out = rh_out * p_v_sat_out

    np.set_printoptions(edgeitems=25)

    phi_deg = 50.90        # Latitude
    lambda_deg = -4.53     # Longitude

    n_days_year =  365

    pi = np.pi

    # Longitude expressed in hours
    lambda_h = lambda_deg/15

    sin_phi = sin(phi_deg * pi/180)
    cos_phi = cos(phi_deg * pi/180)
    tan_phi = tan(phi_deg * pi/180)

    hour_sol_local = hour_yr

    # Equation of time ET in hours
    # beta = 2*pi/365 in rad/day, J = hour_sol_local/24, betaJ in rad
    betaJ = (2*pi/n_days_year)*(hour_sol_local/24)
    ET    = (1/60) * (-0.00037+0.43177*np.cos(betaJ) - 3.165*np.cos(2*betaJ) - 0.07272*np.cos(3*betaJ) \
                - 7.3764*np.sin(betaJ) - 9.3893*np.sin(2*betaJ) - 0.24498*np.sin(3*betaJ))

    hour_sol_local_per = hour_sol_local-24*np.trunc(hour_sol_local/24)
    # Assign 24 h  to the zero h elements 
    hour_sol_local_per[hour_sol_local_per == 0] = 24
    # day=1+np.trunc(hour_sol_local/24)

    time_rad=2*pi*hour_sol_local/(24*365)
    cos_time=np.cos(time_rad)

    # hour_south_per = heure périodique égale à 0h quand le soleil est au Sud (azimut gamma = 0)
    hour_south_per = hour_sol_local_per - 12

    # Angle horaire omega en degres : omega = 0 quand le soleil est au Sud (azimut gamma = 0)
    omega_deg = hour_south_per*15   
    sin_omega = np.sin(omega_deg * pi/180)
    cos_omega = np.cos(omega_deg * pi/180)

    # Sun declination delta en degres
    time_rad=2*pi*hour_sol_local/(24*n_days_year)
    time_lag_rad = 2*pi*(284/n_days_year)
    sin_time_decl = np.sin(time_rad+time_lag_rad)
    delta_rad=0.40928*sin_time_decl
    delta_deg=(180/pi)*delta_rad

    sin_delta = np.sin(delta_rad)
    cos_delta = np.cos(delta_rad)
    tan_delta = np.tan(delta_rad)

    # Angle theta_z between sun beams and vertical"
    theta_z_rad = np.abs(np.arccos(sin_delta*sin_phi+cos_delta*cos_phi*cos_omega))
    cos_theta_z= np.cos(theta_z_rad)
    sin_theta_z= np.sin(theta_z_rad)
    theta_z_deg= (180/pi)*theta_z_rad

    # Compute gamma_s : Sun azimuth "
    # Azimut value comprised between -pi and +pi
    gamma_s_rad = np.arctan2(sin_omega, cos_omega * sin_phi - tan_delta * cos_phi)

    sin_gamma_s = np.sin(gamma_s_rad)
    cos_gamma_s = np.cos(gamma_s_rad)
    # Azimut value comprised between -180 and +180
    gamma_s_deg = (180/pi)*gamma_s_rad 

    # Components of the unit vector parallel to sun  beams in axes: South, East, Vertical
    n_sun_beam_South = cos_gamma_s*sin_theta_z
    n_sun_beam_East = sin_gamma_s*sin_theta_z
    n_sun_beam_Vert = cos_theta_z

    # Direct horizontal irradiance calculation
    # Solar altitude angle: only when the sun is over the horizontal plane
    h_s_deg = np.choose(theta_z_deg < 90,[0, 90 - theta_z_deg])
    h_s_rad = (pi/180) * h_s_deg

    # Compute I_th_cs from the solar radiation I_dot_n_0 external to the atmosphere

    # Direct normal irradiance calculation
    # Solar constant [W/m^2]
    I_dot_0 = 1367  
    # Correction for the variation of sun-earth distance [W/m^2]
    delta_I_dot_0 = 45.326 
    I_dot_n_0 = I_dot_0 + delta_I_dot_0*cos_time

    I_th_0= np.where(cos_theta_z > 0,I_dot_n_0*cos_theta_z,0)

    # Direct horizontal irradiance calculation
    # Solar altitude angle: only when the sun is over the horizontal plane
    h_s_deg = np.where(theta_z_deg < 90,90 - theta_z_deg , 0)
    h_s_rad = (pi/180) * h_s_deg

    # Correction of solar altitude angle for refraction
    DELTAh_s_deg = 0.061359*(180/pi)*(0.1594+1.1230*(pi/180)*h_s_deg+0.065656*(pi/180)**2 * h_s_deg**2)/(1+28.9344*(pi/180)*h_s_deg+277.3971*(pi/180)**2 *h_s_deg**2)
    h_s_true_deg = h_s_deg + DELTAh_s_deg
    h_s_true_rad = (pi/180)* h_s_true_deg

    # m_r: relative optical air mass
    # m_r: ratio of the optical path lenght through atmosphere "
    # and the optical path lenght through a standard atmosphere at sea level with the sun at the zenith"
    # Kasten and Young (1989)"
    m_r = p_atm / p_0 /(np.sin(h_s_true_rad) + 0.50572 *((h_s_true_deg + 6.07995)**(-1.6364)))

    # delta_R: Integral Rayleigh optical thickness"
    delta_R = np.where(m_r > 20, 1/(10.4+0.718*m_r), 1/ (6.62960+1.75130*m_r-0.12020*m_r**2+0.00650*m_r**3-0.00013*m_r**4))

    # T_L_2: Linke turbidity factor for relative air mass = 2"
    # Site turbidity beta_site: Country: 0.05  Urban:0.20"
    beta_site = 0.05

    T_L_summer = 3.302
    T_L_winter = 2.455
    T_L_avg=(T_L_summer+T_L_winter)/2
    DELTAT_L=(T_L_summer-T_L_winter)/2
    time_lag_w_deg=360*(-30.5/360-1/4)
    time_lag_w_rad=2*pi*(-30.5/360-1/4)
    sin_time_w = np.sin(time_rad+time_lag_w_rad)
    T_L_2=T_L_avg+DELTAT_L*sin_time_w

    # Direct horizontal irradiance"
    I_bh_cs = I_dot_n_0 * np.sin(h_s_rad) * np.exp(-0.8662*T_L_2*m_r*delta_R)

    # Direct normal irradiance
    # Not considered if sun flicking the wall with an angle < 2°
    I_beam_cs = np.where(cos_theta_z > 0.035, I_bh_cs / cos_theta_z, 0)

    # Diffuse horizontal irradiance"
    # T_rd: diffuse transmission for sun in zenith"
    T_rd = -1.5843E-2+3.0543E-2*T_L_2+3.797E-4*T_L_2**2

    # F_d: diffuse angular function"
    A_0 = 2.6463E-1-6.1581E-2*T_L_2+3.1408E-3*T_L_2**2
    A_1 = 2.0402+1.8945E-2*T_L_2-1.1161E-2*T_L_2**2
    A_2 = -1.3025+3.9231E-2*T_L_2+8.5079E-3*T_L_2**2
    F_d = A_0+A_1*np.sin(h_s_rad)+A_2*(np.sin(h_s_rad)**2)

    # Diffuse horizontal irradiance"
    I_dh_cs = np.where(h_s_deg > 2, I_dot_n_0*T_rd*F_d, 0)
    I_test = I_dot_n_0*T_rd*F_d

    # Total horizontal irradiance"
    I_th_cs= I_bh_cs+I_dh_cs

    I_bh = I_bh_cs
    I_dh = I_dh_cs
    I_th = I_th_cs

    theta_z_rad = theta_z_deg * (pi/180)
    cos_theta_z= np.cos(theta_z_rad)
    sin_theta_z= np.sin(theta_z_rad)
    tan_theta_z= np.tan(theta_z_rad)

    # Define the gamma_w : wall azimuth
    gamma_w_rad = gamma_w_deg*pi/180
    sin_gamma_w = sin(gamma_w_rad)
    cos_gamma_w = cos(gamma_w_rad)

    # Define the p : slope
    p_deg = slope_w_deg
    p_rad = p_deg *pi/180
    sin_p = sin(p_rad)
    cos_p = cos(p_rad)

    # Components of the wall unit normal vector in axes: South, East, Vertical
    n_wall_South = cos_gamma_w*sin_p
    n_wall_East = sin_gamma_w*sin_p
    n_wall_Vert = cos_p

    # Mask effect
    prod_scal_v_n = n_sun_beam_South * n_wall_South + n_sun_beam_East * n_wall_East + n_sun_beam_Vert * n_wall_Vert
    # Sun beams hitting the wall with an angle >2°
    iflag_no_mask = np.where(prod_scal_v_n > 0.035, 1, 0)

    # Difference of azimuth between sun and wall
    # deltagamma_deg comprised between 0° and 180°
    # Where True, yield x, otherwise yield y.
    deltagamma_deg = np.where(abs(gamma_s_deg - gamma_w_deg) > 180, abs(abs(gamma_s_deg - gamma_w_deg) - 360),
                              abs(gamma_s_deg - gamma_w_deg))
    # deltagamma_rad comprised between 0 and pi
    deltagamma_rad = deltagamma_deg *pi/180
    cos_dgamma = np.cos(deltagamma_rad)
    sin_dgamma = np.sin(deltagamma_rad)
    tan_dgamma = np.tan(deltagamma_rad)

    # Compute ratio= cos(theta)/cos(theta_z)
    # Cos of angle theta between sun beams and normal direction to the wall"
    # Mask effect if sun flicking the wall with an angle < 2°
    cos_theta = cos_dgamma * sin_theta_z
    # Where True, yield x, otherwise yield y.
    cos_theta_cos_theta_z = np.where(sin_theta_z > 0.035, cos_theta / cos_theta_z, 0)

    I_beam = np.where(cos_theta_z > 0.035, I_bh / cos_theta_z, 0)
    I_b = np.where(cos_theta > 0.035, cos_theta * I_beam, 0)

    # Ground reflexion: grass 0.2, snow 0.9"
    rho_ground=0.2

    # Diffuse and reflected solar gains"
    I_dr = I_dh * (1+cos_p) / 2 + I_th * rho_ground * (1-cos_p)/2

    # Solar irradiance on the facade"
    I_tv = I_b + I_dr

    # Ground reflexion
    rho_ground=0

    # Diffuse and reflected solar gains
    I_dr_cs=I_dh_cs*(1+cos_p)/2+I_th_cs*rho_ground*(1-cos_p)/2
    #direct solar gains#
    I_b_cs=iflag_no_mask*I_bh_cs*np.where((cos_p+sin_p*cos_theta_cos_theta_z) > 0, (cos_p+sin_p*cos_theta_cos_theta_z), 0)
    # Solar gains on the wall
    I_t_cs=I_b_cs+I_dr_cs

    I_wall=(1- iflag_shading)*I_t_cs+iflag_shading*I_dr_cs
    t_eq_out_wall= t_dry +alpha_wall*I_wall/h_out

    tau_0  = tau[0]

    Ti0  = t_init * np.ones(n_nodes)


    def model_dTi_t(Ti, tau):
        Tw = Ti[0:]

        ind = int((tau - tau_0) /DELTAtau) 
        if ind>len(t_eq_out_wall)-1: ind=len(t_eq_out_wall)-1

        # Radiative and convective heat exchanges at wall surfaces
        t_s_wall_in = Tw[0]
        t_s_wall_out = Tw[-1]

        #! All walls
        shape = (n_nodes)
        fh = np.zeros(shape)
        dTw_t = np.zeros(shape)

        # Inside boundary conditions
        fh[0] = h_in*(t_a_in_set-t_s_wall_in)

        # Outside boundary conditions
        fh[-1] = h_out*(t_eq_out_wall[ind]-t_s_wall_out)

        dTw_t[:] = np.linalg.inv(C_wall[:,:]) @ (fh[:] - L_wall[:,:] @ Tw[:])

        dTi_t = dTw_t

        q_dot_in_to_wall_A = fh[0]
        q_dot_wall_to_in_A = - fh[0]
        DELTAt_eq = q_dot_wall_to_in_A/U_value_wall

        return (dTi_t, DELTAt_eq) # allows to return more outputs than only dTi_T

    
# SOLVER odeint    
    
#     def dTi_t_odeint(Ti, tau):
#         ret = model_dTi_t(Ti, tau) 
#         return ret[0]

#     Ti = odeint(dTi_t_odeint , Ti0, tau)
#     print(Ti.shape,Ti)
#     DELTAt_eq = np.asarray([model_dTi_t(Ti[tt],tau[tt])[1] for tt in range(len(tau))])


# SOLVER solve_ivp

    def dTi_t_solveivp(Ti, tau):
        ret = model_dTi_t(tau, Ti)
        return ret[0]
    
    T_array = solve_ivp(dTi_t_solveivp, (tau[0],tau[-1]), Ti0, t_eval=tau, method=method)
    Ti = T_array.y.T
    DELTAt_eq = np.asarray([model_dTi_t(Ti[tt],tau[tt])[1] for tt in range(len(tau))])

    
    DELTAt_eq_max = np.max(DELTAt_eq)
    
    DELTAt_ext_int = t_dry-t_a_in_set
    DELTAt_eq_out=t_eq_out_wall-t_a_in_set
    
    dfl = pd.DataFrame( {'DTE':DELTAt_eq, 'DTE_out':DELTAt_eq_out, 'TOUT':t_dry, \
                         'T_0':t_in * np.ones(len(t_dry))})
    dfT = pd.DataFrame(Ti[0:],columns=['T_' + str(j+1) for j in range(n_nodes)])
    dfT['T_' + str(n_nodes+1)] = t_in + DELTAt_eq_out
    dfl = pd.merge(dfl, dfT, left_index=True, right_index=True)
    
    
    dfl['tau'] = tau
    dfl['hour']= dfl['tau']/3600
    dfl = dfl.drop(['tau'], axis=1)
    dfl['hour_per']= hour_per

    dfl.index = dfl['hour']
    dfl = dfl.loc[n_hours_sim-24:n_hours_sim]

    first_index = dfl.index[0]  # Gets the label of the first row
    dfl.loc[first_index, 'hour_per'] = 0
    
    dfp = dfl.copy()
    dfp['inthour']= dfp['hour'].astype(int)
    dfp = dfp.loc[dfp['hour'] == dfp['inthour']]
    dfp['hour_per'] = dfp['hour_per'].astype(int)
    dfp.index = dfp['hour_per']
    dfp = dfp[['DTE']].copy()
    dfp['DTE'] = round(dfp['DTE'],2)

    dfc = dfp.copy()
    dfc.index = dfc['DTE']
    dfc = dfc.drop(['DTE'], axis=1)
    
    dfpa = dfl.copy()
    dfpa['inthour']= dfpa['hour'].astype(int)
    dfpa = dfpa.loc[dfpa['hour'] == dfpa['inthour']]
    dfpa['hour_per'] = dfpa['hour_per'].astype(int)
    dfpa.index = dfpa['hour_per']
    dfpa = dfpa[['TOUT']].copy()
    dfpa['TOUT'] = round(dfpa['TOUT'],2)

    dfca = dfpa.copy()
    dfca.index = dfca['TOUT']
    dfca = dfca.drop(['TOUT'], axis=1)
    
    return dfl, dfp, dfc, dfpa, dfca