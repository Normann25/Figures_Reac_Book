import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.transforms import IdentityTransform, TransformedBbox, Bbox
from typing import Sequence
from sympy import solve, Poly, Eq, Function, exp, symbols
import seaborn as sns
from scipy.stats import norm

def morse(D_e, b, r, r_e):
    """
    Parameters
    ----------
    D_e: int or float
        Dissociation energy

    b: int or float
        Controls width of potential

    r: array of floats
        Internuclear separation

    r_e: int or float
        Equilibrium bond distance
    """
    V = D_e*(1-np.exp(-b*(r-r_e)))**2
    return V

def energy_levels_chap2(levels, wavenumber, anharmonicity, D_e, b, r_e):
    """
    Calculates energy levels with the same length as the width of the Morse potential it is used for.
    The spacing between the energy levels decreases with increasing energy.
   
    Parameters
    ----------
    levels: int
        Number of energy levels to calculate

    wavenumber: int or float

    anharmonicity: int or float

    D_e: int or float
        Dissociation energy

    b: int or float
        Controls width of potential

    r_e: int or float
        Equilibrium bond distance
    """
    energy_levels = []
    distance = []
    for v in range(levels+1):
        E_v = (v+0.5)*wavenumber-((v+0.5)**2)*wavenumber*anharmonicity
        energy_levels.append(E_v)
        r = symbols('r')
        dist = solve(D_e*(1-exp(-b*(r-r_e)))**2-E_v)
        distance.append(dist)
    distance_array = np.array(distance)
    return energy_levels, distance_array

def gauss(sigma, x, mu):
    """
    Parameters
    ----------
    sigma: int or float
        Standard deviation of the destribution

    x: array of floats

    mu: int or float
        Mean or expectation of the distribution
    """
    y = (1/sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)
    return y

def morse_antimorse(D_e, b, r, r_e):
    """
    Parameters
    ----------
    D_e: int or float
        Dissociation energy

    b: int or float
        Controls width of potential

    r: array of floats
        Internuclear separation

    r_e: int or float
        Equilibrium bond distance
    """
    if r_e == None:
        V_morse = D_e*(1-np.exp(-b*(r)))**2
        V_anti = (D_e/2)*(np.exp(-2*b*r)+2*np.exp(-b*r))
    else:
        q = r-r_e
        V_morse = morse(D_e, b, r, r_e)
        V_anti = (D_e/2)*(np.exp(-2*b*q)+2*np.exp(-b*q))
    return V_morse, V_anti

def make_subplots_chap2a(axes, xdataname, ydataname, xcon, ycon, n, m):
    """
    Parameters
    ----------
    axes : `matplotlib.axes.Axes`
        The Axes to add the graphs to.
    
    axdataname, ydataname, xcon, ycon : list, array or dataframe column

    n, m : int or float
        Defines frame in plot
    """
    ax = axes
    # pathway along reaction path
    ax.plot(xdataname, ydataname, color = 'k')
    sns.lineplot(x = xcon, y = ycon, color = 'k', ax = ax)
    # Frame
    ax.hlines(ycon[0]-5, xdataname.min()-n, m*xdataname.max()+n, color = 'k', lw = 1.2)
    ax.hlines(ycon[0]+5, xdataname.max()+n, m*xdataname.max()+n, color = 'k', lw = 1.2)
    ax.vlines(xdataname.min()-n, ycon[0]-5, ydataname[-1], color = 'k', lw = 1.2)
    ax.vlines(xdataname.max()+n, ycon[0]+5, ydataname[-1], color = 'k', lw = 1.2)
    # Remove ticks
    ax.set_ylabel(None, fontsize = 0)
    ax.set_xlabel(None, fontsize = 0)
    plt.xticks([], [])
    plt.yticks([], [])
    ax.set_xticks([]) # for major ticks
    ax.set_yticks([])
    ax.set_xticks([], minor=True) # for minor ticks
    ax.set_yticks([], minor=True)
    sns.despine(top=True, right=True, left=True, bottom=True)

def make_subplots_chap2bd(axes, xdataname, ydataname, xmin1, xmax, xmin2, ymin, ymax):
    """
    Parameters
    ----------
    axes : `matplotlib.axes.Axes`
        The Axes to add the graphs to.
    
    axdataname, ydataname : list, array or dataframe column

    xmin1, xmax, xmin2, ymin, ymax : int or float
        Defines frame in plot
    """
    ax = axes
    # pathway along reaction path
    ax.plot(xdataname, ydataname, color = 'k')
    # Frame
    ax.hlines(-ymin, xmin1, xmax, color = 'k', lw = 1.2)
    ax.hlines(ymin, xmin2, xmax, color = 'k', lw = 1.2)
    ax.vlines(xmin1, -ymin, ymax, color = 'k', lw = 1.2)
    ax.vlines(xmin2, ymin, ymax, color = 'k', lw = 1.2)
    # Remove ticks
    ax.set_ylabel(None, fontsize = 0)
    ax.set_xlabel(None, fontsize = 0)
    plt.xticks([], [])
    plt.yticks([], [])
    ax.set_xticks([]) # for major ticks
    ax.set_yticks([])
    ax.set_xticks([], minor=True) # for minor ticks
    ax.set_yticks([], minor=True)
    sns.despine(top=True, right=True, left=True, bottom=True)

def make_subplots_chap2c(axes, xdataname, ydataname, xcon, ycon, n, xmin1, xmax, xmin2, m):
    """
    Parameters
    ----------
    axes : `matplotlib.axes.Axes`
        The Axes to add the graphs to.
    
    axdataname, ydataname, xcon, ycon : list, array or dataframe column

    n, m, xmin1, xmax, xmin2 : int or float
        Defines frame in plot
    """
    ax = axes
    # pathway along reaction path
    ax.plot(xdataname, ydataname, color = 'k')
    sns.lineplot(x = xcon, y = ycon, color = 'k', ax = ax)
    # Frame
    ax.hlines(ydataname.min()-n, xmin1, xmax, color = 'k', lw = 1.2)
    ax.hlines(ydataname.max()+n, xmin2, xmax, color = 'k', lw = 1.2)
    ax.vlines(xmin1, ydataname.min()-n, m*ydataname.max()+n, color = 'k', lw = 1.2)
    ax.vlines(xmin2, ydataname.max()+n, m*ydataname.max()+n, color = 'k', lw = 1.2)
    # Remove ticks
    ax.set_ylabel(None, fontsize = 0)
    ax.set_xlabel(None, fontsize = 0)
    plt.xticks([], [])
    plt.yticks([], [])
    ax.set_xticks([]) # for major ticks
    ax.set_yticks([])
    ax.set_xticks([], minor=True) # for minor ticks
    ax.set_yticks([], minor=True)
    sns.despine(top=True, right=True, left=True, bottom=True)

def rot_xy(x,y,phi):
    """
    Rotates x and y coordinates a certain angle phi

    Parameters
    ----------
    x, y : array of floats
        Coordinates to be rotated

    phi : int or float
        Angle the coordinates should be rotated in radians
    """
    x_rot, y_rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])@np.array([x,y])
    return x_rot, y_rot

def find_intercept(c1, c2, m1, m2):
    """
    Calculates the x and y coordinates of the intercept between two straight lines defined as:
    y = m1*x + c1 and y = m2*x + c2

    Parameters
    ----------
    c1, c2, m1, m2 : int or float
    """
    x_intercept = (c2-c1)/(m1-m2)
    y_intercept = m1*x_intercept+c1
    return x_intercept, y_intercept

def orb_motion(D_e, b, R, r_eq, l, mu):
    """
    Parameters
    ----------
    D_e: int or float
        Dissociation energy

    b: int or float
        Controls width of potential

    R: array of floats
        Internuclear separation

    r_e: int or float
        Equilibrium bond distance

    l: int or float
        Rotational quantum number, takes values 0, 1, 2, ...

    mu: int or float
        Reduced mass
    """
    q = R-r_eq
    V_morse = D_e*(1-np.exp(-b*(q)))**2 -D_e
    L = np.sqrt(l*(l+1)) 
    pot = (L**(2))/(2*mu*R**(2)) - 4.5
    V_eff = V_morse + pot
    return V_morse, pot, V_eff

def ellipse(x_val, b, a):
    """
    Parameters
    ----------
    x_val: array of floats
        Range of x-values defining the width of th ellipse if it where centered at the origin
    
    a, b: int or float
        The width of the ellipse is 2*a and the height 2*b
    """
    y_val = (a*np.sqrt(b**2-x_val**2))/b
    y_val = np.append(y_val[:-1], -y_val[::-1][1:])
    x_array = np.append(x_val[:-1], x_val[::-1][1:])
    
    y_array = np.append(y_val, y_val[0])
    x_array = np.append(x_array, x_array[0])

    return x_array, y_array

def potential_surface(x, x_b, n1, n2):
    """
    Parameters
    ----------
    x: array of floats
        Reaction coordinate

    x_b: int or float
        Mid point (reaction coordinate) of potential

    n1, n2: int or float
        Scaling factors of first and second half of the potental
    """
    y = np.ones_like(x)
    y[:x_b] += n1*norm.pdf(x[:x_b], x[x_b], 1)
    y[x_b:] += n2*norm.pdf(x[x_b:], x[x_b], 1)
    y[:x_b] = y[:x_b]+y[x_b]-y[x_b-1]
    return y

def arrowed_spines(fig, ax):

    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    # removing the axis ticks
    plt.xticks([]) # labels 
    plt.yticks([])
    ax.xaxis.set_ticks_position('none') # tick markers
    ax.yaxis.set_ticks_position('none')

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./30.*(ymax-ymin) 
    hl = 1./20.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
             head_width=hw, head_length=hl, overhang = ohg, 
             length_includes_head= True, clip_on = False) 

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
             head_width=yhw, head_length=yhl, overhang = ohg, 
             length_includes_head= True, clip_on = False)

def potentials_chap15(D_e, b, r, r_e):
    """
    Parameters
    ----------
    D_e: int or float
        Dissociation energy

    b: int or float
        Controls width of potential

    r: array of floats
        Internuclear separation

    r_e: int or float
        Equilibrium bond distance
    """
    q = r-r_e
    V_morse = morse(D_e, b, r, r_e)-D_e
    V_anti = (D_e/2)*(np.exp(-0.5*b*(q+0.2))+2*np.exp(-b*(q+0.2)))
    return V_morse, V_anti

class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """
    def __init__(self, xy, p1, p2, size = 75, unit = 'points', ax = None, 
                 text = '', textposition = 'inside', text_kw = None, **kwargs):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy   # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(self._xydata, size, size, angle = 0.0, 
                         theta1 = self.theta1, theta2 = self.theta2, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(ha = 'center', va = 'center', 
                       xycoords = IdentityTransform(),
                       xytext = (0,0), textcoords = 'offset points',
                       annotation_clip = True)
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy = self._center, **self.kw)
    
    def get_size(self):
        factor = 1.
        if self.unit == 'points':
            factor = self.ax.figure.dpi / 72.
        elif self.unit[:4] == 'axes':
            b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            dic = {'max': max(b.width, b.height),
                   'min': min(b.width, b.height),
                   'width': b.width, 'height': b.height}
            factor = dic[self.unit[5:]]
        return self.size * factor
    
    def set_size(self, size):
        self.size = size
    
    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)
    
    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center 
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)
    
    def get_theta2(self):
        return self.get_theta(self.vec2)
    
    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)
    
    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == 'inside':
            r = s / np.interp(angle_span, [60, 90, 135, 180],
                                          [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == 'outside':
            def R90(a, r, w, h):
                if a < np.arctan(h/2/(r+w/2)):
                    return np.sqrt((r+w/2)**2 + (np.tan(a)*(r+w/2))**2)
                else:
                    c = np.sqrt((w/2)**2+(h/2)**2)
                    T = np.arcsin(c * np.cos(np.pi/2 - a + np.arcsin(h/2/c))/r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w/2, h/2])
                    return np.sqrt(np.sum(xy**2))
            
            def R(a, r, w, h):
                aa = (a % (np.pi/4))*((a % (np.pi/2)) <= np.pi/4) + \
                     (np.pi/4 - (a % (np.pi/4)))*((a % (np.pi/2)) >= np.pi/4)
                return R90(aa, r, *[w, h][::int(np.sign(np.cos(2*a)))])
            
            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X-s/2), 0))[0] * 72
            self.text.set_position([offs*np.cos(angle), offs*np.sin(angle)])
            