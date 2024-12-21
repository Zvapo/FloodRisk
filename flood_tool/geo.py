from numpy import array, asarray, mod, sin, cos, tan, sqrt, arctan2, floor, rad2deg, deg2rad, stack
from scipy.linalg import inv

__all__ = ['get_easting_northing_from_gps_lat_long',
           'get_gps_lat_long_from_easting_northing',
           'GeoTransformer']

class Ellipsoid:
    """ Data structure for a global ellipsoid. """

    def __init__(self, a, b, F_0):
        self.a = a
        self.b = b
        self.n = (a-b)/(a+b)
        self.e2 = (a**2-b**2)/a**2
        self.F_0 = F_0
        self.H=0

class Datum(Ellipsoid):
    """ Data structure for a global datum. """

    def __init__(self, a, b, F_0, phi_0, lam_0, E_0, N_0, H):
        super().__init__(a, b, F_0)
        self.phi_0 = phi_0
        self.lam_0 = lam_0
        self.E_0 = E_0
        self.N_0 = N_0
        self.H = H

def dms2rad(deg, min=0, sec=0):
    """Convert degrees, minutes, seconds to radians.
    
    Parameters
    ----------
    deg: array_like
        Angle in degrees.
    min: array_like
        (optional) Angle component in minutes.
    sec: array_like
        (optional) Angle component in seconds.

    Returns
    -------
    numpy.ndarray
        Angle in radians.
    """
    deg = asarray(deg)
    return deg2rad(deg+min/60.+sec/3600.)

def rad2dms(rad, dms=False):
    """Convert radians to degrees, minutes, seconds.

    Parameters
    ----------

    rad: array_like
        Angle in radians.
    dms: bool
        Use degrees, minutes, seconds format. If False, use decimal degrees.

    Returns
    -------
    numpy.ndarray
        Angle in degrees, minutes, seconds or decimal degrees.
    """

    rad = asarray(rad)
    deg = rad2deg(rad)
    if dms:
        min = 60.0*mod(deg, 1.0)
        sec = 60.0*mod(min, 1.0)
        return stack((floor(deg), floor(min), sec.round(4)))
    else:
        return deg

osgb36 = Datum(a=6377563.396,
               b=6356256.910,
               F_0=0.9996012717,
               phi_0=deg2rad(49.0),
               lam_0=deg2rad(-2.),
               E_0=400000,
               N_0=-100000,
               H=24.7)

wgs84 = Ellipsoid(a=6378137, 
                  b=6356752.3142,
                  F_0=0.9996)

def lat_long_to_xyz(phi, lam, rads=False, datum=osgb36):
    """Convert input latitude/longitude in a given datum into
    Cartesian (x, y, z) coordinates.

    Parameters
    ----------

    phi: array_like
        Latitude in degrees (if radians=False) or radians (if radians=True).
    lam: array_like
        Longitude in degrees (if radians=False) or radians (if radians=True).
    rads: bool (optional)
        If True, input latitudes and longitudes are in radians.
    datum: Datum (optional)
        Datum to use for conversion.
    """
    if not rads:
        phi = deg2rad(phi)
        lam = deg2rad(lam)

    nu = datum.a*datum.F_0/sqrt(1-datum.e2*sin(phi)**2)
  
    return array(((nu+datum.H)*cos(phi)*cos(lam),
                  (nu+datum.H)*cos(phi)*sin(lam),
                  ((1-datum.e2)*nu+datum.H)*sin(phi)))

def xyz_to_lat_long(x,y,z, rads=False, datum=osgb36):

    p = sqrt(x**2+y**2)

    lam = arctan2(y, x)
    phi = arctan2(z,p*(1-datum.e2))

    for _ in range(10):

        nu = datum.a*datum.F_0/sqrt(1-datum.e2*sin(phi)**2)
        dnu = -datum.a*datum.F_0*cos(phi)*sin(phi)/(1-datum.e2*sin(phi)**2)**1.5

        f0 = (z + datum.e2*nu*sin(phi))/p - tan(phi)
        f1 = datum.e2*(nu**cos(phi)+dnu*sin(phi))/p - 1.0/cos(phi)**2
        phi -= f0/f1

    if not rads:
        phi = rad2dms(phi)
        lam = rad2dms(lam)

    return phi, lam

class Transformer():
    
    def __init__(self, datum):
        
        self.N_0 = datum.N_0
        self.E_0 = datum.E_0
        self.phi_0 = datum.phi_0
        self.lam_0 = datum.lam_0
        self.a = datum.a
        self.b = datum.b
        self.e2 = datum.e2
        self.n = datum.n
        self.F_0 = datum.F_0
        
        self.M = 0.0
        self.nu = 0.0
        self.rho = 0.0
        self.eta2 = 0.0
        
    def _compute_M(self, phi):
        self.M = self.b * self.F_0 * (
                (1 + self.n + 5/4*self.n**2 + 5/4*self.n**3) * (phi - self.phi_0) - 
                (3*self.n + 3*self.n**2 + 21/8*self.n**3) * sin(phi - self.phi_0) * cos(phi + self.phi_0) + 
                (15/8*self.n**2 + 15/8*self.n**3) * sin(2*(phi - self.phi_0)) * cos(2*(phi + self.phi_0)) - 
                35/24*self.n**3 * sin(3*(phi - self.phi_0)) * cos(3*(phi + self.phi_0))
                )
        
    def _compute_nu(self, phi):
        self.nu = self.a * self.F_0 * (1 - self.e2*sin(phi)**2)**(-0.5)
        
    def _compute_rho(self, phi):
        self.rho = self.a * self.F_0 * (1 - self.e2) * (1 - self.e2*sin(phi)**2)**(-1.5)
        
    def _compute_eta2(self):
        self.eta2 = self.nu / self.rho - 1
        
    def get_easting_northing_from_gps_lat_long(self, phi, lam, rads=False):
        """ Get OSGB36 easting/northing from GPS latitude and longitude pairs.

        Parameters
        ----------
        phi: float/arraylike
            GPS (i.e. WGS84 datum) latitude value(s)
        lam: float/arrayling
            GPS (i.e. WGS84 datum) longitude value(s).
        rads: bool (optional)
            If true, specifies input is is radians.
        Returns
        -------
        numpy.ndarray
            Easting values (in m)
        numpy.ndarray
            Northing values (in m)
        
        Examples
        --------
        >>> get_easting_northing_from_gps_lat_long([55.5], [-1.54])
        (array([429157.0]), array([623009]))
        References
        ----------
        Based on the formulas in "A guide to coordinate systems in Great Britain".
        See also https://webapps.bgs.ac.uk/data/webservices/convertForm.cfm
        """
        
        phi = array(phi, ndmin=1)
        lam = array(lam, ndmin=1)
            
        assert phi.size == lam.size, 'Length of easting data and northing data does not match!'
        assert (phi.size>0) and (lam.size>0), 'Please make sure data is entered!'

        east_list = []
        north_list = []
        for p, l in zip(phi, lam):
            if not rads:
                p = dms2rad(p)
                l = dms2rad(l)
                
            self._compute_nu(p)
            self._compute_rho(p)
            self._compute_eta2()
            self._compute_M(p)

            one = self.M + self.N_0
            two = self.nu/2 * sin(p) * cos(p)
            three = self.nu/24 * sin(p) * cos(p)**3 * (5 - tan(p)**2 + 9*self.eta2)
            threeA = self.nu/720 * sin(p) * cos(p)**5 * (61 - 58*tan(p)**2 + tan(p)**4)

            four = self.nu * cos(p)
            five = self.nu/6 * cos(p)**3 * (self.nu/self.rho - tan(p)**2)
            six = self.nu/120 * cos(p)**5 * (5 - 18*tan(p)**2 + tan(p)**4 + 14*self.eta2 - 58*tan(p)**2*self.eta2)

            north = one + two*(l-self.lam_0)**2 + three*(l-self.lam_0)**4 + threeA*(l-self.lam_0)**6
            east = self.E_0 + four*(l-self.lam_0) + five*(l-self.lam_0)**3 + six*(l-self.lam_0)**5
            
            north_list.append(north)
            east_list.append(east)

        return array(east_list), array(north_list)
        
    def get_gps_lat_long_from_easting_northing(self, east, north, rads=False, dms=True, ms=False):
        """ Get OSGB36 easting/northing from GPS latitude and
        longitude pairs.
        Parameters
        ----------
        east: float/arraylike
            OSGB36 easting value(s) (in m).
        north: float/arrayling
            OSGB36 easting value(s) (in m).
        rads: bool (optional)
            If true, specifies ouput is is radians.
        dms: bool (optional)
            If true, output is in degrees/minutes/seconds. Incompatible
            with rads option.
        ms: bool (optional)
            If ture, output is in degrees/minutes/seconds and contains minutes/seconds.
        Returns
        -------
        numpy.ndarray
            GPS (i.e. WGS84 datum) latitude value(s).
        numpy.ndarray
            GPS (i.e. WGS84 datum) longitude value(s).
        Examples
        --------
        >>> get_gps_lat_long_from_easting_northing([429157], [623009])
        (array([55.5]), array([-1.540008]))
        References
        ----------
        Based on the formulas in "A guide to coordinate systems in Great Britain".
        See also https://webapps.bgs.ac.uk/data/webservices/convertForm.cfm
        """
        
        east = array(east, ndmin=1)
        north = array(north, ndmin=1)
            
        assert east.size == north.size, 'Length of easting data and northing data does not match!'
        assert (east.size>0) and (north.size>0), 'Please make sure data is entered!'
        assert rads != dms, 'Output can either be in radians or degrees/minutes/seconds.'

        phi_list = []
        lam_list = []
        for e, n in zip(east, north):
            phi_prime = (n-self.N_0) / (self.a*self.F_0) + self.phi_0
            self._compute_M(phi_prime)

            while abs(n - self.N_0 - self.M) >= 0.01:
                phi_prime += (n-self.N_0 - self.M) / (self.a*self.F_0)
                self._compute_M(phi_prime)
            self._compute_nu(phi_prime)
            self._compute_rho(phi_prime)
            self._compute_eta2()

            tan_phi = tan(phi_prime)
            seven = tan_phi / (2*self.rho*self.nu)
            eight = tan_phi / (24*self.rho*self.nu**3) * (5 + 3*tan_phi**2 + self.eta2 - 
                                                                 9*tan_phi**2*self.eta2**2)
            nine = tan_phi / (720*self.rho*self.nu**5) * (61 + 90*tan_phi**2 * 45*tan_phi**4)

            sec_phi = 1 / cos(phi_prime)
            ten = sec_phi / self.nu
            eleven = sec_phi / (6*self.nu**3) * (self.nu/self.rho + 2*tan_phi**2)
            twelve = sec_phi / (120*self.nu**5) * (5 + 28*tan_phi**2 + 24*tan_phi**4)
            thirteen = sec_phi / (5040*self.nu**7) * (61 + 662*tan_phi**2 + 1320*tan_phi**4 + 720*tan_phi**6)

            phi = phi_prime - seven*(e-self.E_0)**2 + eight*(e-self.E_0)**4 - nine*(e-self.E_0)**6
            lam = self.lam_0 + ten*(e-self.E_0) - eleven*(e-self.E_0)**3 + \
                  twelve*(e-self.E_0)**5 - thirteen*(e-self.E_0)**7
            
            if dms:
                phi_list.append(rad2dms(phi, dms=ms))
                lam_list.append(rad2dms(lam, dms=ms))
            else:
                phi_list.append(phi)
                lam_list.append(lam)

        return array(phi_list), array(lam_list)

class HelmertTransform(object):
    """Callable class to perform a Helmert transform."""
    
    def __init__(self, s, rx, ry, rz, T):

        self.T = T.reshape((3, 1))
        
        self.M = array([[1+s, -rz, ry],
                        [rz, 1+s, -rx],
                        [-ry, rx, 1+s]])

    def __call__(self, X):
        X = X.reshape((3,-1))
        return self.T + self.M@X

class HelmertInverseTransform(object):
    """Callable class to perform the inverse of a Helmert transform."""
    
    def __init__(self, s, rx, ry, rz, T):

        self.T = T.reshape((3, 1))
        
        self.M = inv(array([[1+s, -rz, ry],
                        [rz, 1+s, -rx],
                        [-ry, rx, 1+s]]))

    def __call__(self, X):
        X = X.reshape((3,-1))
        return self.M@(X-self.T)

OSGB36transform = HelmertTransform(20.4894e-6,
                             -dms2rad(0,0,0.1502),
                             -dms2rad(0,0,0.2470),
                             -dms2rad(0,0,0.8421),
                             array([-446.448, 125.157, -542.060]))

WGS84transform = HelmertInverseTransform(20.4894e-6,
                             -dms2rad(0,0,0.1502),
                             -dms2rad(0,0,0.2470),
                             -dms2rad(0,0,0.8421),
                             array([-446.448, 125.157, -542.060]))


def WGS84toOSGB36(phi, lam, rads=False):
    """Convert WGS84 latitude/longitude to OSGB36 latitude/longitude.
    
    Parameters
    ----------
    phi : array_like or float
        Latitude in degrees or radians on WGS84 datum.
    lam : array_like or float
        Longitude in degrees or radians on WGS84 datum.
    rads : bool, optional
        If True, phi and lam are in radians. If False, phi and lam are in degrees.

    Returns
    -------
    tuple of numpy.ndarrays
        Latitude and longitude on OSGB36 datum in degrees or radians.
    """
    xyz = OSGB36transform(lat_long_to_xyz(asarray(phi), asarray(lam),
                                  rads=rads, datum=wgs84))
    return xyz_to_lat_long(*xyz, rads=rads, datum=osgb36)

def OSGB36toWGS84(phi, lam, rads=False):
    """Convert OSGB36 latitude/longitude to WGS84 latitude/longitude.
    
    Parameters
    ----------
    phi : array_like or float
        Latitude in degrees or radians on OSGB36 datum.
    lam : array_like or float
        Longitude in degrees or radians on OSGB36 datum.
    rads : bool, optional
        If True, phi and lam are in radians. If False, phi and lam are in degrees.

    Returns
    -------
    tuple of numpy.ndarrays
        Latitude and longitude on WGS84 datum in degrees or radians.
    """
    xyz = WGS84transform(lat_long_to_xyz(asarray(phi), asarray(lam),
                                  rads=rads, datum=osgb36))
    return xyz_to_lat_long(*xyz, rads=rads, datum=wgs84)


GeoTransformer = Transformer(osgb36)

'''
Usage (in tool.py): 

from .geo import GeoTransformer

GeoTransformer.get_gps_lat_long_from_easting_northing([429157], [623009])
output: (array([55.49941394]), array([-1.53844546]))

GeoTransformer.get_easting_northing_from_gps_lat_long([55.5], [-1.54])
output: (array([429058.36712719]), array([623073.56982745]))
'''

