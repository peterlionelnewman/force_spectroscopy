
"""


    This script takes a .ARDF file generated on Asylum AFM and calculates a best fit
    elastic modulus for the sample, later dumping the result to a .txt file

    Written by: Pete, 2022, (peterlionelnewman)
    Paper doi: doi.org/10.1002/advs.202204741
    Paper url: https://onlinelibrary.wiley.com/doi/10.1002/advs.202204741
    Cite us if you use any part of this script in anyway :-)

    This script is free to use and modify for non-commercial purposes, and
    please cite us if you use any part of this script in anyway :-)



"""


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import load_ardf
from scipy.optimize import differential_evolution
from multiprocessing import Pool
import datetime
import time
import traceback

pathmaster = '/Users/shinweichong/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/pycharm/afm/data/20230509/'

spring_constant = 0.126     # N/m
poisson_ratio = 0.5         # PEG
radius = 0.000_005_0        # sphere radius
model = 'Ogden'             # Hertz, Ogden
alpha = 2                   # 2 gives neo-hookean
plot = True
count_break = 20
live_plot = True
multiprocess = False

global ax


def objective_function(x, *args):

    """

    Objective function for minimization.
    Calculates the difference between
        1) a theoretical force-displacement curve, from elastic modulus and contact point
        2) the experimental force-displacement curve.


    """

    if np.isnan(x[0]):
        return 1E99

    E = x[0]            # the elastic modulus
    ind = int(x[1])     # the index at which contact starts
    offset_RHS = x[2]   # a offset in defl
    offset_LHS = x[3]   # the index at which the offset starts

    [delz, defl, force, radius, poisson_ratio, model, alpha] = args

    #boundary conditions again, scipy optimize is broken and doesn't respect bounds, not sure why
    if E < 100 or E > 1_000_000 \
            or ind > len(defl)-2 or ind < 2 \
            or offset_RHS < -0.2 * np.max(defl) or offset_RHS > 0.2 * np.max(defl) \
            or offset_LHS < -0.2 * np.max(defl) or offset_LHS > 0.2 * np.max(defl):
        return 1E99
    else:

        m = ((force[ind] - offset_RHS) - (force[0] - offset_LHS)) / delz[ind]

        # rejection criteria based on LHS gradient
        if m > ((0.025 * np.max(force)) - np.min(force)) / delz[ind]:
            return 1E99

        c = (force[0] - offset_LHS)
        force_LHS = m * delz[:ind] + c

        # calc delta from ind
        delta = delz[ind:] - defl[ind:]
        delta = delta - delta[0]

        if model == 'Hertz':
            force_RHS = (4 / 3) * (E / (1 - poisson_ratio ** 2)) * (radius ** 0.5) * (delta ** (3 / 2))

        elif model == 'Ogden':  # https: // www.ncbi.nlm.nih.gov / pmc / articles / PMC3615644 / pdf / nihms415272.pdf
            a = radius ** 0.5 * delta ** 0.5
            B = (40 * E) / (9 * np.pi * (1 - poisson_ratio ** 2))
            force_RHS = ((B * np.pi * (a ** 2)) / alpha) * (((1 - 0.2 * (a / radius)) ** (- alpha / 2 - 1)) - ((1 - 0.2 * (a / radius)) ** (alpha - 1)))

        force_RHS = force_RHS + force_LHS[-1]

        # minimize the area between curves
        # LHS = (np.sum(np.abs((force[:ind] - force_LHS))))
        # RHS = (np.sum(np.abs((force[ind:] - force_RHS))))

        # minimize the weighted area between curves
        weight_LHS = np.linspace(1, 0, int(0.2 * len(force_LHS)))
        weight_LHS = np.hstack((np.ones(len(force_LHS) - len(weight_LHS)), weight_LHS))
        weight_RHS = np.linspace(0, 1, int(0.2 * len(force_RHS)))
        weight_RHS = np.hstack((weight_RHS, np.ones(len(force_RHS) - len(weight_RHS))))
        LHS = np.sum(np.abs((force[:ind] - force_LHS) * weight_LHS))
        RHS = np.sum(np.abs((force[ind:] - force_RHS) * weight_RHS))

        # mean squared difference, norm by the average magnitude, norm by number of points
        # LHS = (1 * np.sum(np.abs((force[:ind] - force_LHS))) / np.mean(force_LHS) / len(force_LHS)) ** 1
        # RHS = (1 * np.sum(np.abs((force[ind:] - force_RHS))) / np.mean(force_RHS) / len(force_RHS)) ** 1

        # mean squared difference, norm by the average magnitude, norm by number of points
        # weight_LHS = np.linspace(1, 0, int(0.1 * len(force_LHS)))
        # weight_LHS = np.hstack((np.ones(len(force_LHS) - len(weight_LHS)), weight_LHS))
        # weight_RHS = np.linspace(0, 1, int(0.1 * len(force_RHS)))
        # weight_RHS = np.hstack((weight_RHS, np.ones(len(force_RHS) - len(weight_RHS))))
        # LHS = (1 * np.sum(np.abs((force[:ind] - force_LHS) * weight_LHS)) / np.mean(force_LHS) / len(force_LHS)) ** 1
        # RHS = (1 * np.sum(np.abs((force[ind:] - force_RHS) * weight_RHS)) / np.mean(force_RHS) / len(force_RHS)) ** 1

        return LHS + RHS

def process_ardf(filename):

    out = []

    # try:
    [[raw, deflect, delp], metalist] = load_ardf.ardf2hdf5(filename)

    if '(' in filename and ')' in filename:
        conditions_of_fabrication = filename[filename.index('(') + 1:filename.index(')')]
        out.append(f'{filename.replace(","," ")}, ')
        out.append(f'{conditions_of_fabrication}, ')
    else:
        conditions_of_fabrication = filename
        out.append(f'{filename}')

    for i in range(len(raw)):

        print(f"i: {i}")
        start = time.perf_counter()

        # load the data, ::x downsampling if desired, minimal effect on processing time
        delz = delp[i][::1]
        defl = deflect[i][::1]

        # preprocessing

        # remove nans
        defl[np.isnan(defl)] = 10E-16
        delz[np.isnan(delz)] = 10E-16

        # offset to be positive
        defl += defl.min()
        delz += delz.min()

        # remove zeros
        defl[defl == 0] = 10E-16
        delz[delz == 0] = 10E-16

        # kernel find where delz changes from extend to retract
        kernel = np.convolve((np.diff(delz) > 0), np.array([0.1, 0.2, 0.4, 0.2, 0.1]), mode='valid')

        # take extend only,
        delz = delz[0:np.where(kernel < 0.5)[0][0]]
        delz = delz - np.min(delz)  # in m
        defl = defl[0:np.where(kernel < 0.5)[0][0]]
        defl = defl - np.min(defl)  # in m

        if len(delz) < 2 or len(defl) < 2 or sum(defl[::-2]) == 0:
            continue

        # identify discontinuities
        dis = np.diff((defl[1::] - defl[0]) / (delz[1::] - delz[0]))
        print(f'defl: {defl}')
        print(f'dis: {dis}')
        dis = np.argwhere(dis < (dis.min() * 0.5))[-1] + int(len(dis) * 0.1)
        delz = delz[dis[0]::]
        defl = defl[dis[0]::]

        # clip height of the defl
        # ind = defl < 0.1 * np.max(defl)  # only calculate over 0-1 fractional height of the defl
        # defl = defl[ind]
        # delz = delz[ind]

        force = defl * spring_constant

        count = 0
        min_val = [1E100]
        X = [0, 0, 0, 0]
        offset_guess = 0

        while count < count_break:  # keep going until the top 5 results have an E std within 1% of each other, check every 5 evaluations

            x = differential_evolution(objective_function,
                                       bounds=((100, 1_000_000), (2, len(defl) - 2), (-0.1 * np.max(force), 0.1 * np.max(force)),
                                               (-0.1 * np.max(force), 0.1 * np.max(force))),
                                       maxiter=1000,
                                       disp=False,
                                       popsize=10,
                                       tol=0.01,
                                       mutation=0.5,
                                       recombination=0.7,
                                       seed=None,
                                       callback=None,
                                       polish=True,
                                       init='latinhypercube',
                                       atol=0,
                                       args=[delz, defl, force, radius, poisson_ratio, model, alpha]
                                       )

            if np.abs(x.x[0] - 100) < 5:
                count += 1
            else:
                X = np.vstack([X, x.x])
                min_val.append(x.fun)
                count += 1
                if count == count_break:
                    print(f"...{conditions_of_fabrication}, {i}", end='')
                    count += count_break
                if count % 5 == 0:
                    X_sorted = X[np.argsort(min_val), :]
                    if np.std(X_sorted[0:5, 0]) / np.mean(X_sorted[0:5, 0]) < 0.01:
                        count += count_break  # break when the top 5 results have an E std within 1% of each other

        X_sorted = X[np.argsort(min_val), :]
        E = X_sorted[0, 0]
        ind = int(X_sorted[0, 1])
        offset_RHS = X_sorted[0, 2]
        offset_LHS = X_sorted[0, 3]
        force_LHS = ((force[ind] - offset_RHS) - (force[0] - offset_LHS)) / delz[ind] * delz[:ind] + (force[0] - offset_LHS)
        delta = delz[ind:] - defl[ind:]
        delta = delta - delta[0]
        if model == 'Hertz':
            force_RHS = (4 / 3) * (E / (1 - poisson_ratio ** 2)) * (radius ** 0.5) * (delta ** (3 / 2))
        elif model == 'Ogden':  # https: // www.ncbi.nlm.nih.gov / pmc / articles / PMC3615644 / pdf / nihms415272.pdf
            a = radius ** 0.5 * delta ** 0.5
            B = (40 * E) / (9 * np.pi * (1 - poisson_ratio ** 2))
            force_RHS = ((B * np.pi * (a ** 2)) / alpha) * (
                    ((1 - 0.2 * (a / radius)) ** (- alpha / 2 - 1)) - ((1 - 0.2 * (a / radius)) ** (alpha - 1)))
        force_RHS = force_RHS + force_LHS[-1]
        delz = delz * 1E6

        if np.log10(E) < 3:
            E_print = str(np.around(E, 2)) + ' Pa'
        elif np.log10(E) < 6:
            E_print = str(np.around(E / 1E3, 2)) + ' kPa'
        elif np.log10(E) < 9:
            E_print = str(np.around(E / 1E6, 2)) + ' MPa'
        else:
            E_print = str(np.around(E / 1E9, 2)) + ' GPa'


        if plot:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
            ax.plot(delz, defl * spring_constant, color='black', linewidth=1)
            y_lim = [force_RHS.min() - 0.2 * force_RHS.max(), force_RHS.max() * 1.2]
            ax.grid(which='both', axis='both', alpha=0.5)
            ax.set_ylim([y_lim[0], y_lim[1]])
            ax.set_xlim([-0.25, 14])
            ax.xaxis.set_minor_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05E-7))
            ax.plot(delz[ind:], force_RHS, color='magenta', alpha=1, lw=5, ls='--')
            ax.plot(delz[0:ind], force_LHS, color='green', alpha=0.95, lw=4, ls='--')
            ax.set_ylabel('force (N)')
            ax.set_xlabel('del z (µm)')
            ax.text(delz[0], y_lim[1] * 0.66,
                    f'{conditions_of_fabrication} \n'
                    f'using {model} fitype \n'
                    f'alpha = {alpha} \n'
                    f'recording #{i} \n'
                    f'E = {E_print} \n'
                    f'ind = {ind} \n'
                    f'offset R,L = {np.around(offset_LHS * 10e9, 1)}, {np.around(offset_RHS * 10e9, 1)} 10e-9\n'
                    f'took {count - count_break} iterations \n'
                    f'time: {datetime.timedelta(seconds=(time.perf_counter() - start))}')
            plt.draw()
            plt.show()

        # store E for writing to 'processed' text file
        out.append(f'{E}, ')

        print(f'took {count - count_break} iter: {datetime.timedelta(seconds=(time.perf_counter() - start))} d HH:MM:SS.ms, {conditions_of_fabrication}, {model} fitype, alpha = {alpha}, #{i}, E = {E_print}, ind = {ind}, offset L, R = {np.around(offset_LHS * 10e9, 1)}, {np.around(offset_RHS * 10e9, 1)}')

    out.append('\n')
    # except Exception as e:
    #
    #     # Set color to cyan
    #     print('\033[96m', end='')
    #     print(f'Failed to process {filename}')
    #     print(f'Error: {str(e)}')
    #     print('\033[0m', end='')
    #
    #     # Set color to red
    #     print('\033[91m', end='')
    #     print(traceback.format_exc())  # Print the full traceback
    #     # Reset color to default
    #     print('\033[0m', end='')
    #     # tk.messagebox.showerror('Python Error',
    #     #                         f'{traceback.format_exc()}')
    #     return
    #
    return out


if __name__ == '__main__':

    # crawl around your computer
    absfilelist = []
    os.chdir(pathmaster)
    for dirpath, _, filenames in os.walk("."):
        for f in filenames:
            if f.endswith('.ARDF') and not 'glass' in f:
                absfilelist.append(dirpath + '/' + f)

    # save the results
    with open(f'elastic_modulus_processed_alpha-{alpha}.txt', 'w') as processed:
        processed.write(f'spring_constant: {spring_constant}\n')
        processed.write(f'radius: {radius}\n')
        processed.write(f'model: {model}\n')
        try:
            processed.write(f'alpha: {alpha}\n')
        except:
            pass
        processed.write(f'datetime: {datetime.datetime.now()}\n')
        processed.write(f'path of data: {pathmaster}\n')


        processed.write('\n')

        # open each ardf force map fit/calculate each curves Young's mod
        script_start = time.perf_counter()

        # multiprocess the data
        if multiprocess:
            with Pool() as pool:
                out = pool.map(process_ardf, absfilelist)

        # single process
        else:
            out = []
            for fi, filename in enumerate(absfilelist):
                out.append(process_ardf(filename))

        # write the results
        for lines in out:
            for line in lines:
                processed.write(line)

    print(f'finished processing {len(absfilelist)} files in {datetime.timedelta(seconds=(time.perf_counter() - script_start))} d HH:MM:SS.ms')
