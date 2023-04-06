import numpy as np
import matplotlib.pyplot as plt

from .create_initsuperdrops import initSDsinputsdict
from ..readbinary import readbinary


def get_superdroplet_attributes(configfile, constsfile, initSDsfile):
    ''' get gridbox boundaries from binary file and 
    re-dimensionalise usign COORD0 const from constsfile '''

    inputs = initSDsinputsdict(configfile, constsfile)
    
    sd_gbxindex, eps, radius, m_sol, coord3 = read_dimless_superdrops_binary(initSDsfile)

    radius = radius * inputs["R0"]
    m_sol = m_sol * inputs["MASS0"]
    coord3 = coord3 * inputs["COORD0"]

    return sd_gbxindex, eps, radius, m_sol, coord3


def read_dimless_superdrops_binary(filename):
    ''' return dimenionsless gbx boundaries by reading binary file'''

    datatypes = [np.uintc, np.uint, np.double, np.double, np.double]
    data, ndata_pervar = readbinary(filename)

    print(ndata_pervar)
    idxs = []
    for n in range(1, len(ndata_pervar)):
        # indexs for division of data list between each variable
        idxs.append(np.sum(ndata_pervar[:n]))

    sd_gbxindex = np.asarray(data[:idxs[0]], dtype=datatypes[0])
    eps = np.asarray(data[idxs[0]:idxs[1]], dtype=datatypes[1])
    radius = np.asarray(data[idxs[1]:idxs[2]], dtype=datatypes[2])
    m_sol = np.asarray(data[idxs[2]:idxs[3]], dtype=datatypes[3])
    coord3 = np.asarray(data[idxs[3]:], dtype=datatypes[4])

    print("attribute shapes: ", sd_gbxindex.shape, eps.shape,
          radius.shape, m_sol.shape, coord3.shape)
    
    return sd_gbxindex, eps, radius, m_sol, coord3


def plot_initdistribs(configfile, constsfile, initSDsfile,
                      vol, binpath, savefig):

    plt.rcParams.update({'font.size': 14})

    eps, radius, m_sol, coord3 = get_superdroplet_attributes(configfile,
                                                               constsfile,
                                                               initSDsfile)[1:]

    print(eps, radius, m_sol, coord3)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    axs = axs.flatten()

    # create nbins evenly spaced in log10(r)
    nbins = 100
    minr, maxr = np.min(radius)/10, np.max(radius)*10
    hedgs = np.linspace(np.log10(minr), np.log10(maxr),
                        nbins+1)  # edges to lnr bins

    l0 = plot_radiusdistrib(axs[0], hedgs, radius, eps)

    l1 = plot_numconcdistrib(axs[1], hedgs, eps, radius, vol)

    l3 = plot_masssolutedistrib(axs[3], hedgs, eps, radius, m_sol, vol)

    l2 = plot_coord3distrib(axs[2], hedgs, coord3, radius)

    fig.tight_layout()
    if savefig:
        fig.savefig(binpath+"/initdistribs.png", dpi=400,
                    bbox_inches="tight", facecolor='w', format="png")
        print("Figure .png saved as: "+binpath+"/gridboxboundaries.png")
    plt.show()


def log10r_frequency_distribution(radius, hedgs, wghts):
    ''' get distribution of data with weights 'wghts' against 
    log10(r). Uses np.histogram to get frequency of a particular
    value of data that falls in each bin (with each bin defined
    by it's edges 'hedgs'). Return distirbution alongside the radius
    bin centers and widths in [m]'''

    if type(wghts) != np.ndarray:
        wghts = np.full(np.shape(radius), wghts)

    hist, hedgs = np.histogram(np.log10(radius), bins=hedgs,
                               weights=wghts, density=None)

    # convert [m] to [micron]
    hedgs = (10**(hedgs))*1e6
    # radius bin widths [micron]
    hwdths = hedgs[1:] - hedgs[:-1]
    # radius bin centres [micron]
    hcens = (hedgs[1:]+hedgs[:-1])/2

    return hist, hedgs, hwdths, hcens


def plot_radiusdistrib(ax, hedgs, radius, eps):
    ''' get and plotthe superdroplet radius in each log10(r)
    bin and as a scatter on a twinx axis with their multiplicities'''

    l1 = ax.scatter(radius*1e6, eps, zorder=1,
                    color="purple", label="multiplicities")

    ax2 = ax.twinx()
    hist, hedgs, hwdths, hcens = log10r_frequency_distribution(radius, hedgs, 1)
    l2 = ax2.step(hcens, hist, where='mid', alpha=0.8, zorder=0,
                  color="grey", label="number distribution")

    ax.set_xscale("log")
    ax.set_xlabel("radius, r, /\u03BCm")
    ax.set_yscale("log")

    ax.set_ylabel("superdroplet multiplicity")
    ax2.set_ylabel("superdroplet number distribution")

    ax.legend(loc="lower left")
    ax2.legend(loc="lower right")

    return [l1, l2]


def plot_numconcdistrib(ax, hedgs, eps, radius, vol):
    ''' get and plot frequency of real droplets in each log10(r) bin '''

    wghts = eps / vol / 1e6  # [cm^-3]
    hist, hedgs, hwdths, hcens = log10r_frequency_distribution(
        radius, hedgs, wghts)

    line = ax.bar(hcens, hist, hwdths, color="teal",
                  label="binned distribution")
    ax.set_xscale("log")
    ax.set_xlabel("radius, r, /\u03BCm")
    ax.set_ylabel("real droplet number concentration / cm$^{-3}$")
    ax.legend(loc="lower left")

    return line


def plot_masssolutedistrib(ax, hedgs, eps, radius, m_sol, vol):
    ''' get and plot frequency of real droplets in each log10(r) bin '''

    wghts = m_sol*eps/vol * 1000 / 1e6  # [g cm^-3]
    hist, hedgs, hwdths, hcens = log10r_frequency_distribution(
        radius, hedgs, wghts)

    line = ax.bar(hcens, hist, hwdths, color="teal")
    ax.set_xscale("log")
    ax.set_xlabel("radius, r, /\u03BCm")
    ax.set_ylabel("solute mass per unit volume / g cm$^{-3}$")

    return line


def plot_coord3distrib(ax, hedgs, coord3, radius):

    line = None
    if any(coord3):
        line = ax.scatter(radius*1e6, coord3, c="purple")

    ax.set_xscale("log")
    ax.set_xlabel("radius, r, /\u03BCm")
    ax.set_ylabel("superdroplet coord3 / m")

    return line
