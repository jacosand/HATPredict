#!/usr/bin/env python

"""
Script to compute all DFT descriptors needed as input to the ML model to predict
the free energy barriers for a list of hydrogen atom transfer (HAT) reactions
specified in reactions.txt, using pre-computed geometry optimization and single-point
Gaussian .out files for all molecules found in reactions.txt.

This script can be used in training mode, with transition states specified in reactions.txt,
and in inference mode, without transition specified in reactions.txt.  Training mode
requires the appropriate transition state Gaussian .out files, and extracts the ground-truth
free energy barrier in addition to the input descriptors.

Run from the command line:
python get_descriptors.py [arguments] /path/to/directory/containing/reactions.txt

For possible arguments, see argument parser in __main__ section or run:
python get_descriptors.py -h
"""

import os
import argparse
import csv
from functools import lru_cache
from morfeus import BuriedVolume


periodic_table = ["",
    "H","He",
    "Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
]


@lru_cache(maxsize=None)
def read_sp_file(spdir, name, suffix, reacting_atom):
    """
    Function which reads a Gaussian .out output file from a single-point computation and extracts
    the raw features needed to compute the DFT descriptors for input into the ML model, returning
    them as a dictionary.  The output of this function is cached, since multiple reactions may
    contain the same molecule, and thus require reading the same single-point output files.

    Arguments:
    
        - spdir: path to the directory where the single-point .out file is located

        - name: base name which identifies the molecule whose raw features are
                to be extracted
        
        - suffix: filename suffix, which identifies whether the single-point output
                file corresponds to the neutral molecule ('sp0'), the cation ('sp+'),
                or the anion ('sp-').
                
                The overall filename to be read is the name, followed by a hyphen,
                followed by the suffix, followed by the '.out' extension.
        
        - reacting_atom: integer identifying the number of the heavy atom in the single-point
                 .out file which undergoes the hydrogen atom transfer (HAT) reaction.
    
    Returns:

        - features: dictionary with all raw features needed to compute the DFT descriptors
                    for input into the ML model, containing the following keys:
            
            - 'elements': tuple listing the element symbols of all atoms in the molecule

            - 'coords': molecular geometry as a tuple of 3-tuples.  Each atom in the molecule corresponds to
                        one 3-tuple containing (X-coord, Y-coord, Z-coord).  Note that the order of atoms
                        is the same in 'elements' and 'coords'.
            
            - 'energy': floating point containing the single-point energy of the molecule in Hartrees

            - 'HOMO': floating point containing the HOMO energy of the molecule in Hartrees

            - 'LUMO': floating point containing the LUMO energy of the molecule in Hartrees

            - 'charge': floating point containing the NBO charge on the reacting_atom

            - 'bonds': tuple containing list of bond orders for all bonds to the reacting_atom.  A bond
                       is defined as a bond order > 0.5

            - 'bondsH': tuple containing list of bond orders for all bonds from hydrogen atoms to the
                        reacting_atom.  A bond is defined as a bond order > 0.5

            - 'bondsX': tuple containing list of bond orders for all bonds from non-hydrogen atoms to the
                        reacting_atom.  A bond is defined as a bond order > 0.5
    """

    start_coords = False
    stable_wavefunction = False
    start_orbitals = False
    start_charges = False
    start_bonds = False

    elements = []
    coords = []
    bonds = []

    features = {}

    spdir_file_out = os.path.join(spdir, name + suffix + '.out')

    with open(spdir_file_out, 'r') as f:
        for line in f:
            # Signifies start of the molecular geometry
            if line.strip() == 'Standard orientation:':
                start_coords = True
                # Skip next four lines until coordinates begin
                for _ in range(4):
                    f.readline()
            
            # Start reading coordinates
            elif start_coords:

                # Detect whether coordinates are finished
                if '--' in line.strip():
                    start_coords = False
                    continue

                # Read coordinates
                words = line.split()
                elements.append(periodic_table[int(words[1])])
                coords.append((float(words[3]), float(words[4]), float(words[5])))

            # Energy line: in the event of unstable wavefunctions, the energy kept will be the last in the file,
            # corresponding to the final stable wavefunction
            elif line.startswith(' SCF Done:'):
                words = line.split()
                energy = float(words[4])

            # A stable wavefunction has been found, and everything below in the output file corresponds
            # to this stable wavefunction
            elif line.strip() == 'The wavefunction is stable under the perturbations considered.':
                stable_wavefunction = True

            # Only collect orbital energies, NBO charges, and Wiberg bond orders after a stable wavefunction has been found
            if stable_wavefunction:

                # Beginning of orbital energies section
                if line.strip() == 'Orbital energies and kinetic energies (alpha):':
                    start_orbitals = True

                elif start_orbitals:
                    words = line.split()

                    # Store the last occupied orbital as the HOMO (by continually replacing the HOMO energy
                    # with each new occupied orbital)
                    if words[1].endswith('O'):
                        HOMO = float(words[2])

                    # Store the first virtual orbital as the LUMO
                    elif words[1].endswith('V'):
                        LUMO = float(words[2])

                        # Prevent later virtual orbitals from replacing the LUMO
                        start_orbitals = False
                
                # Beginning of NBO charges section
                elif line.strip() == 'Summary of Natural Population Analysis:':
                    start_charges = True

                    # Skip next five lines until NBO charges begin
                    for _ in range(5):
                        f.readline()

                elif start_charges:
                    words = line.split()

                    # Check for the reacting atom
                    if reacting_atom == int(words[1]):

                        # Store the NBO charge of the reacting atom
                        charge = float(words[2])
                        start_charges = False

                # Beginning of Wiberg bond index section
                elif line.strip() == 'Wiberg bond index matrix in the NAO basis:':
                    start_bonds = True
                
                # Check for the reacting atom
                elif start_bonds and line.strip().startswith(str(reacting_atom)+'.'):
                    words = line.split()

                    # Append bond orders for bonds to the reacting atom
                    bonds += [float(bond) for bond in words[2:]]

                # End of Wiberg bond index section
                elif line.strip() == 'Wiberg bond index, Totals by atom:':
                    start_bonds = False
                    break
    
    features = {
        'elements': tuple(elements),
        'coords': tuple(coords),
        'energy': energy,
        'HOMO': HOMO,
        'LUMO': LUMO,
        'charge': charge,
        'bonds': tuple(bond for bond in bonds if bond > 0.5),
        'bondsH': tuple(bond for bond, element in zip(bonds, elements) if bond > 0.5 and element == 'H'),
        'bondsX': tuple(bond for bond, element in zip(bonds, elements) if bond > 0.5 and element != 'H')
    }

    return features


@lru_cache(maxsize=None)
def get_enthalpy_and_free_energy(optdir, spdir, name):
    """
    Function which reads a series of Gaussian .out output files in order to compute the
    enthalpy and free energy of a molecule.  These ouput files include a single-point file,
    a combined geometry optimization and frequency file, and optionally two other single-point
    files to compute solvation corrections.  The output of this function is cached, since
    multiple reactions may contain the same molecule, and thus require reading the same
    single-point output files.

    Arguments:

        - optdir: path to the directory where the optimization and frequency .out file is
                  located

        - spdir: path to the directory where the single-point .out files are located

        - name: base name which identifies the molecule whose enthalpy and free energy
                are to be computed.  The files to be read will have this name, followed
                by a hyphen, followed by a suffix (specified in the function body),
                followed by the '.out' extension.

    Returns:

        - enthalpy: floating point containing the enthalpy of the molecule in Hartrees

        - free_energy: floating point containing the Gibbs free energy of the molecule
                       in Hartrees
    """    

    # Get single-point energy for the molecule
    spdir_file_out = os.path.join(spdir, name + '-sp0' + '.out')
    with open(spdir_file_out, 'r') as f:
        for line in f:
            if line.startswith(' SCF Done:'):
                words = line.split()
                sp_energy = float(words[4])

    # Get vibrational corrections to compute the enthalpy and free energy
    with open(os.path.join(optdir, name + '.out'), 'r') as f:
        for line in f:
            # Vibrational correction to enthalpy
            if line.startswith(' Thermal correction to Enthalpy='):
                words = line.split()
                enthalpy_correction = float(words[-1])

            # Vibrational correction to free energy
            elif line.startswith(' Thermal correction to Gibbs Free Energy='):
                words = line.split()
                free_energy_correction = float(words[-1])

    # The solvation correction is computed at a different (lower) level of theory,
    # so it is the difference between the solvated single-point energy and the
    # gas-phase single-point energy at this different (lower) level of theory.

    # Solvated single-point energy at solvation level of theory
    with open(os.path.join(spdir, name + '-sps' + '.out'), 'r') as f:
        for line in f:
            if line.startswith(' SCF Done:'):
                words = line.split()
                solv_energy = float(words[4])

    # Gas-phase single-point energy at solvation level of theory
    with open(os.path.join(spdir, name + '-spg' + '.out'), 'r') as f:
        for line in f:
            if line.startswith(' SCF Done:'):
                words = line.split()
                gas_energy = float(words[4])

    # Combine contributions to enthalpy
    enthalpy = sp_energy + enthalpy_correction + solv_energy - gas_energy

    # Combine contributions to free energy
    free_energy = sp_energy + free_energy_correction + solv_energy - gas_energy

    return enthalpy, free_energy


@lru_cache(maxsize=None)
def bv(elements, coords, atom):
    """
    Function which computes the buried volume around a particular atom in a molecule
    using radii of 3, 4, and 5 Angstroms.  This function is a wrapper around the
    BuriedVolume functionality in the morfeus software package.  The output of this function 
    is cached, since multiple reactions may contain the same molecule, and thus would require
    the same buried volume computations.

    Arguments:

        - elements: tuple listing the element symbols of all atoms in the molecule

        - coords: molecular geometry as a tuple of 3-tuples.  Each atom in the molecule corresponds to
                  one 3-tuple containing (X-coord, Y-coord, Z-coord).  Note that the order of atoms
                  must be the same in elements and coords.
        
        - atom: an integer containing the number of the atom around which the buried volume
                is to be computed.
    
    Returns:

        - bv3: buried volume using a radius of 3 Angstroms

        - bv4: buried volume using a radius of 4 Angstroms

        - bv5: buried volume using a radius of 5 Angstroms
    """

    bv3 = BuriedVolume(elements, coords, atom, include_hs=True, radius=3, radii_scale=1).fraction_buried_volume
    bv4 = BuriedVolume(elements, coords, atom, include_hs=True, radius=4, radii_scale=1).fraction_buried_volume
    bv5 = BuriedVolume(elements, coords, atom, include_hs=True, radius=5, radii_scale=1).fraction_buried_volume

    return bv3, bv4, bv5


def compute_descriptors(optdir, spdir, reaction, training_mode):
    """
    Function which computes all DFT descriptors needed for the input of one hydrogen atom
    transfer (HAT) reaction into the ML model, returning them as a dictionary.

    Arguments:

        - optdir: path to directory where all geometry optimization Gaussian .out output
                  files are located

        - spdir: path to directory where all single-point Gaussian .out output files
                 are located

        - reaction: a dictionary describing the hydrogen atom transfer reaction whose
                    free energy barrier is to be estimated by the ML model, containing
                    the following keys:
            
            - 'subH_file': base filename for substrate starting material with H atom attached

            - 'cat_file': base filename for catalyst starting material without H atom attached

            - 'sub_file': base filename for substrate product without H atom attached

            - 'catH_file': base filename for catalyst product with H atom attached

            - 'subH_atom', 'cat_atom', 'sub_atom', 'catH_atom': integer atom number of the heavy atom
                                                                that is undergoing hydrogen atom
                                                                transfer (HAT) in the respective molecule

            - 'ts_file': (optional, only for model training) base filename for the hydrogen
                         atom transfer (HAT) transition state

        - training_mode: boolean describing whether or not the model is to be used in training mode.
                         If True, also extracts the ground-truth enthalpy and free energy barriers
                         (this requires the appropriate transition state Gaussian .out files to be present).

    Returns:

        - desc: a dictionary containing all DFT descriptors needed for the input of one
                hydrogen atom transfer (HAT) reaction into the ML model.  May optionally
                also contain the true output ('DGdd', 'DHdd') in the case of model training.
                See the function body for all the descriptors and how they are calculated.
    """

    desc = {}

    # Compute bond dissociation enthalpy (DH) and bond dissociation free energy (DG)
    # descriptors in kcal/mol

    HARTREE_TO_KCAL = 627.509469

    subH_H, subH_G = get_enthalpy_and_free_energy(optdir, spdir, reaction['subH_file'])
    sub_H, sub_G = get_enthalpy_and_free_energy(optdir, spdir, reaction['sub_file'])
    cat_H, cat_G = get_enthalpy_and_free_energy(optdir, spdir, reaction['cat_file'])
    catH_H, catH_G = get_enthalpy_and_free_energy(optdir, spdir, reaction['catH_file'])

    desc['DH'] = (sub_H + catH_H - subH_H - cat_H) * HARTREE_TO_KCAL
    desc['DG'] = (sub_G + catH_G - subH_G - cat_G) * HARTREE_TO_KCAL

    # Optionally, in model training, compute the true outputs: enthalpy of activation (DHdd)
    # and free energy of activation (DGdd) in kcal/mol
    if training_mode:
        ts_H, ts_G = get_enthalpy_and_free_energy(optdir, spdir, reaction['ts_file'])
        desc['DHdd'] = (ts_H - subH_H - cat_H) * HARTREE_TO_KCAL
        desc['DGdd'] = (ts_G - subH_G - cat_G) * HARTREE_TO_KCAL


    # Extract raw features needed to compute descriptors
    # ... for substrate starting material with H atom attached
    subH = read_sp_file(spdir, reaction['subH_file'], '-sp0', reaction['subH_atom'])
    subH_p = read_sp_file(spdir, reaction['subH_file'], '-sp+', reaction['subH_atom'])
    subH_m = read_sp_file(spdir, reaction['subH_file'], '-sp-', reaction['subH_atom'])

    # ... for catalyst starting material without H atom attached
    cat = read_sp_file(spdir, reaction['cat_file'], '-sp0', reaction['cat_atom'])
    cat_p = read_sp_file(spdir, reaction['cat_file'], '-sp+', reaction['cat_atom'])
    cat_m = read_sp_file(spdir, reaction['cat_file'], '-sp-', reaction['cat_atom'])

    # ... for substrate product without H atom attached
    sub = read_sp_file(spdir, reaction['sub_file'], '-sp0', reaction['sub_atom'])
    sub_p = read_sp_file(spdir, reaction['sub_file'], '-sp+', reaction['sub_atom'])
    sub_m = read_sp_file(spdir, reaction['sub_file'], '-sp-', reaction['sub_atom'])

    # ... for catalyst product with H atom attached
    catH = read_sp_file(spdir, reaction['catH_file'], '-sp0', reaction['catH_atom'])
    catH_p = read_sp_file(spdir, reaction['catH_file'], '-sp+', reaction['catH_atom'])
    catH_m = read_sp_file(spdir, reaction['catH_file'], '-sp-', reaction['catH_atom'])


    # Compute all electronic descriptors
    # ... for substrate starting material with H atom attached
    desc['subH_HOMO'] = subH['HOMO']                                    # HOMO
    desc['subH_LUMO'] = subH['LUMO']                                    # LUMO
    desc['subH_CP'] = (subH['HOMO'] + subH['LUMO']) / 2                 # chemical potential
    desc['subH_Hard'] = (subH['LUMO'] - subH['HOMO']) / 2               # hardness
    desc['subH_Soft'] = 1 / desc['subH_Hard']                           # softness
    desc['subH_Ephil'] = desc['subH_CP'] ** 2 / (2 * desc['subH_Hard']) # electrophilicity

    # ... for catalyst starting material without H atom attached
    desc['cat_SOMO'] = cat['HOMO']                                      # HOMO
    desc['cat_IE'] = cat_p['energy'] - cat['energy']                    # ionization energy
    desc['cat_EA'] = cat['energy'] - cat_m['energy']                    # electron affinity
    desc['cat_Eneg'] = (desc['cat_IE'] + desc['cat_EA']) / 2            # electronegativity
    desc['cat_Soft'] = 1 / (desc['cat_IE'] - desc['cat_EA'])            # softness

    # ... for substrate product without H atom attached
    desc['sub_SOMO'] = sub['HOMO']                                      # HOMO
    desc['sub_IE'] = sub_p['energy'] - sub['energy']                    # ionization energy
    desc['sub_EA'] = sub['energy'] - sub_m['energy']                    # electron affinity
    desc['sub_Eneg'] = (desc['sub_IE'] + desc['sub_EA']) / 2            # electronegativity
    desc['sub_Soft'] = 1 / (desc['sub_IE'] - desc['sub_EA'])            # softness

    # ... for catalyst product with H atom attached
    desc['catH_HOMO'] = catH['HOMO']                                    # HOMO
    desc['catH_LUMO'] = catH['LUMO']                                    # LUMO
    desc['catH_CP'] = (catH['HOMO'] + catH['LUMO']) / 2                 # chemical potential
    desc['catH_Hard'] = (catH['LUMO'] - catH['HOMO']) / 2               # hardness
    desc['catH_Soft'] = 1 / desc['catH_Hard']                           # softness
    desc['catH_Ephil'] = desc['catH_CP'] ** 2 / (2 * desc['catH_Hard']) # electrophilicity


    # Compute all charge descriptors
    # ... for substrate starting material with H atom attached
    desc['subH_Q'] = subH['charge']                                     # NBO charge
    desc['subH_FukP'] = subH_p['charge'] - subH['charge']               # Fukui f+
    desc['subH_FukM'] = subH['charge'] - subH_m['charge']               # Fukui f-
    desc['subH_FukD'] = desc['subH_FukP'] - desc['subH_FukM']           # Fukui Df

    # ... for catalyst starting material without H atom attached
    desc['cat_Q'] = cat['charge']                                      # NBO charge
    desc['cat_Fuk0'] = (cat_p['charge'] - cat_m['charge']) / 2         # Fukui f0

    # ... for substrate product without H atom attached
    desc['sub_Q'] = sub['charge']                                      # NBO charge
    desc['sub_Fuk0'] = (sub_p['charge'] - sub_m['charge']) / 2         # Fukui f0

    # ... for catalyst product with H atom attached
    desc['catH_Q'] = catH['charge']                                     # NBO charge
    desc['catH_FukP'] = catH_p['charge'] - catH['charge']               # Fukui f+
    desc['catH_FukM'] = catH['charge'] - catH_m['charge']               # Fukui f-
    desc['catH_FukD'] = desc['catH_FukP'] - desc['catH_FukM']           # Fukui Df


    # Compute all bond order descriptors
    # ... for substrate starting material with H atom attached
    desc['subH_BO'] = sum(subH['bonds']) / len(subH['bonds'])           # Mean bond order

    if len(subH['bondsH']) == 0:  # no bonds to H
        desc['subH_BOH'] = desc['subH_BO']
    else:
        desc['subH_BOH'] = sum(subH['bondsH']) / len(subH['bondsH'])    # Mean bond order to H

    if len(subH['bondsX']) == 0:  # no bonds not to H
        desc['subH_BOX'] = desc['subH_BO']
    else:
        desc['subH_BOX'] = sum(subH['bondsX']) / len(subH['bondsX'])    # Mean bond order not to H

    # ... for catalyst starting material without H atom attached
    desc['cat_BO'] = sum(cat['bonds']) / len(cat['bonds'])              # Mean bond order

    # ... for substrate product without H atom attached
    desc['sub_BO'] = sum(sub['bonds']) / len(sub['bonds'])              # Mean bond order

    # ... for catalyst product with H atom attached
    desc['catH_BO'] = sum(catH['bonds']) / len(catH['bonds'])           # Mean bond order

    if len(catH['bondsH']) == 0:  # no bonds to H
        desc['catH_BOH'] = desc['catH_BO']
    else:
        desc['catH_BOH'] = sum(catH['bondsH']) / len(catH['bondsH'])    # Mean bond order to H

    if len(catH['bondsX']) == 0:  # no bonds not to H
        desc['catH_BOX'] = desc['catH_BO']
    else:
        desc['catH_BOX'] = sum(catH['bondsX']) / len(catH['bondsX'])    # Mean bond order not to H


    # Compute all steric parameters (buried volume)
    desc['subH_BV3'], desc['subH_BV4'], desc['subH_BV5'] = bv(subH['elements'], subH['coords'], reaction['subH_atom'])
    desc['cat_BV3'], desc['cat_BV4'], desc['cat_BV5'] = bv(cat['elements'], cat['coords'], reaction['cat_atom'])
    desc['sub_BV3'], desc['sub_BV4'], desc['sub_BV5'] = bv(sub['elements'], sub['coords'], reaction['sub_atom'])
    desc['catH_BV3'], desc['catH_BV4'], desc['catH_BV5'] = bv(catH['elements'], catH['coords'], reaction['catH_atom'])

    return desc


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extracts DFT descriptors used in the machine \
        learning model for all reactions described in reactions.txt, using the geometry optimization \
        and single-point Gaussian .out output files generated previously.  Each line in the .csv \
        output file (default name: descriptors.csv) corresponds to one line in reactions.txt, and contains \
        all the DFT descriptors needed for the machine learning model to predict the free energy barrier \
        of that reaction.")

    parser.add_argument("dir", help="path to directory which contains reactions.txt file")
    parser.add_argument("-o", "--opt", help="relative path to subdirectory where geometry optimization Gaussian .out output files are located", default='opt')
    parser.add_argument("-s", "--sp", help="relative path to subdirectory where single-point Gaussian .out output files are located", default='sp')
    parser.add_argument("-c", "--csv", help="base filename of .csv file (without extension) in which to store DFT descriptors", default='descriptors')
    parser.add_argument("-t", "--train", help="run in training mode, also extracting ground-truth free energy barriers", action='store_true')

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise FileNotFoundError('Dataset directory %s does not exist!' % args.dir)

    dir_reactions_txt = os.path.join(args.dir, 'reactions' + '.txt')
    dir_descriptors_csv = os.path.join(args.dir, args.csv + '.csv')

    optdir = os.path.join(args.dir, args.opt)
    spdir = os.path.join(args.dir, args.sp)

    field_names = [
        # Base filenames
        'subH_file', 'sub_file', 'cat_file', 'catH_file', 'ts_file',

        # Reaction enthalpy and free energy
        'DH', 'DG',

        # Descriptors for the substrate with the H attached
        'subH_HOMO', 'subH_LUMO', 'subH_CP', 'subH_Hard', 'subH_Soft', 'subH_Ephil', # electronic descriptors
        'subH_Q', 'subH_FukP', 'subH_FukM', 'subH_FukD',                             # charge descriptors
        'subH_BO', 'subH_BOH', 'subH_BOX',                                           # bond order descriptors
        'subH_BV3', 'subH_BV4', 'subH_BV5',                                          # steric descriptors

        # Descriptors for the substrate radical with the H removed
        'sub_SOMO', 'sub_IE', 'sub_EA', 'sub_Eneg', 'sub_Soft',                      # electronic descriptors
        'sub_Q', 'sub_Fuk0',                                                         # charge descriptors
        'sub_BO',                                                                    # bond order descriptors
        'sub_BV3', 'sub_BV4', 'sub_BV5',                                             # steric descriptors

        # Descriptors for the catalyst radical with the H removed
        'cat_SOMO', 'cat_IE', 'cat_EA', 'cat_Eneg', 'cat_Soft',                      # electronic descriptors
        'cat_Q', 'cat_Fuk0',                                                         # charge descriptors
        'cat_BO',                                                                    # bond order descriptors
        'cat_BV3', 'cat_BV4', 'cat_BV5',                                             # steric descriptors

        # Descriptors for the catalyst with the H attached
        'catH_HOMO', 'catH_LUMO', 'catH_CP', 'catH_Hard', 'catH_Soft', 'catH_Ephil', # electronic descriptors
        'catH_Q', 'catH_FukP', 'catH_FukM', 'catH_FukD',                             # charge descriptors
        'catH_BO', 'catH_BOH', 'catH_BOX',                                           # bond order descriptors
        'catH_BV3', 'catH_BV4', 'catH_BV5',                                          # steric descriptors

        # Transition state descriptors to be predicted
        'DHdd', 'DGdd'
    ]

    # Open and write header to .csv output file.
    csv_file = open(dir_descriptors_csv, 'w')
    writer = csv.DictWriter(csv_file, fieldnames=field_names, extrasaction='ignore')
    writer.writeheader()

    with open(dir_reactions_txt, 'r') as f:
        # Loop over each reaction in reactions.txt
        for line in f:

            # Treat lines that start with # as a comment, and allow blank lines
            if line.startswith('#') or line.strip() == '':
                continue

            # Parse reaction described on the line.
            reaction = {}
            words = line.strip().split(',')
            reaction['subH_file'] = words[0]
            reaction['subH_atom'] = int(words[1])
            reaction['sub_file'] = words[2]
            reaction['sub_atom'] = int(words[3])
            reaction['cat_file'] = words[4]
            reaction['cat_atom'] = int(words[5])
            reaction['catH_file'] = words[6]
            reaction['catH_atom'] = int(words[7])

            # Include transition state file if in training mode.
            if args.train:
                reaction['ts_file'] = words[8]

            try:
                # Compute descriptors for the reaction.
                desc = compute_descriptors(optdir, spdir, reaction, args.train)

                # Write the descriptors to the .csv file
                writer.writerow({**reaction, **desc})
            except:
                print('Could not extract descriptors for the reaction: ' + line.strip())

    csv_file.close()
