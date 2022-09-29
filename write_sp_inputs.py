#!/usr/bin/env python

"""
Script to generate all Gaussian single-point input .com files needed to
compute the DFT features used in the machine learning model for all reactions
described in reactions.txt, using pre-computed geometry optimization Gaussian
.out output files for all the molecules found in reactions.txt.

Run from the command line:
python write_sp_inputs.py [arguments] /path/to/directory/with/reactions.txt

For possible arguments, see argument parser in __main__ section or run:
python write_sp_inputs.py -h
"""

import os
import argparse
from xyz_to_com import write_com_file


periodic_table = [
    "",
    "H", "He",

    "Li", "Be", "B", "C", "N", "O", "F", "Ne",

    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",

    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
    "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",

    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
    "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",

    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu",
    "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn",

    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am",
    "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
    "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
    "Fl", "Mc", "Lv", "Ts", "Og",
]


def read_opt_file(optdir_file_out):
    """
    Function which reads a Gaussian .out output file from a combined geometry
    optimization and frequency job.  Extracts the charge, spin, and optimized
    geometry suitable for submitting further Gaussian jobs that require the
    optimized geometry.

    Arguments:

        - dir_file_out: string containing the full path to a .out file,
                        including the .out extension.

    Returns:

        - charge: integer charge of the molecule

        - spin: integer spin of the molecule

        - geometry: a list of 4-tuples, where each atom in the molecule
                    corresponds to one 4-tuple.  Each 4-tuple contains
                    (Element Abbreviation, X-coord, Y-coord, Z-coord)
    """
    freq_job = False
    start_coords = False

    geometry = []

    with open(optdir_file_out, 'r') as f:
        for line in f:
            # Signifies start of frequency job, using optimized geometry
            if line.startswith(' Link1:'):
                freq_job = True

            # Charge and spin multiplicity line
            elif freq_job and line.startswith(' Charge ='):
                words = line.split()
                charge = int(words[2])
                spin = int(words[5])

            # Signifies start of the optimized molecular geometry used in the
            # frequency job
            elif freq_job and line.strip() == 'Standard orientation:':
                start_coords = True
                # Skip next four lines until coordinates begin
                f.readline()
                f.readline()
                f.readline()
                f.readline()

            # Start reading coordinates
            elif freq_job and start_coords:

                # Detect whether coordinates are finished
                if '--' in line.strip():
                    start_coords = False
                    break

                # Read coordinates
                words = line.split()
                geometry.append(
                    (periodic_table[int(words[1])],
                     float(words[3]), float(words[4]), float(words[5]))
                )

    return (charge, spin, geometry)


def write_feature_inputs(spdir, name, geometry, charge, spin):
    """
    Function which generates all the Gaussian single-point input .com files
    needed to compute all the DFT features used in the ML model.

    Arguments:

        - spdir: path where all of these single-point input .com
                 files are to be written

        - name: base filename for the molecule whose features are to be
                computed.  Note that suffixes will be added by this function to
                this name for each of the single-point input .com files needed.

        - geometry: a list of 4-tuples, where each atom in the molecule
                    corresponds to one 4-tuple.  Each 4-tuple contains
                    (Element Abbreviation, X-coord, Y-coord, Z-coord)

        - charge: integer charge of the molecule

        - spin: integer spin of the molecule

    Returns:

        Nothing
    """

    # This input file is used to compute the free energy of a transition state,
    # which is used to check the performance of the ML model, as in a training
    # or validation set.
    if name.startswith('TS'):

        route_ts = '#p def2tzvpp um062x stable=opt'
        spdir_tsfile_com = os.path.join(spdir, name + '-sp0' + '.com')
        write_com_file(spdir_tsfile_com, geometry, route_ts,
                       charge=charge, spin=spin)

    else:

        route_nbo = '#p def2tzvpp um062x pop=(full,nboread) stable=opt'
        add_input = '$nbo bndidx $end'

        # Compute DFT features for ML model that come from neutral molecule.
        spdir_neutral_com = os.path.join(spdir, name + '-sp0' + '.com')
        write_com_file(spdir_neutral_com, geometry, route_nbo,
                       charge=charge, spin=spin,
                       add_input=add_input)

        # Compute DFT features for ML model that come from molecular cation.
        # Note modified charge and spin.
        spdir_cation_com = os.path.join(spdir, name + '-sp+' + '.com')
        write_com_file(spdir_cation_com, geometry, route_nbo,
                       charge=charge + 1, spin=(spin % 2) + 1,
                       add_input=add_input)

        # Compute DFT features for ML model that come from molecular anion.
        # Note modified charge and spin.
        spdir_anion_com = os.path.join(spdir, name + '-sp-' + '.com')
        write_com_file(spdir_anion_com, geometry, route_nbo,
                       charge=charge - 1, spin=(spin % 2) + 1,
                       add_input=add_input)

    # Compute solvation corrections for the neutral molecule.
    route_sol = '#p 6-31g(d) um062x scrf=(smd,solvent=acetonitrile) stable=opt'
    spdir_solfile_com = os.path.join(spdir, name + '-sps' + '.com')
    write_com_file(spdir_solfile_com, geometry, route_sol,
                   charge=charge, spin=spin)

    # Applying solvation correction requires gas phase computation at same
    # level of theory.
    route_gas = '#p 6-31g(d) um062x stable=opt'
    spdir_gasfile_com = os.path.join(spdir, name + '-spg' + '.com')
    write_com_file(spdir_gasfile_com, geometry, route_gas,
                   charge=charge, spin=spin)


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generates Gaussian \
        single-point input .com files needed to compute the DFT features used \
        in the machine learning model for all reactions described in \
        reactions.txt, using the geometry optimization Gaussian .out output \
        files for all the molecules found in reactions.txt")

    parser.add_argument(
        "dir",
        help="path to directory which contains reactions.txt file",
    )

    parser.add_argument(
        "-o", "--opt",
        help="relative path to subdirectory where geometry optimization \
              Gaussian .out output files are located",
        default='opt',
    )

    parser.add_argument(
        "-s", "--sp",
        help="relative path to subdirectory where single-point Gaussian .com \
              input files will be written",
        default='sp',
    )

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise FileNotFoundError('Main directory %s does not exist!' % args.dir)

    dir_reactions_txt = os.path.join(args.dir, 'reactions' + '.txt')

    optdir = os.path.join(args.dir, args.opt)
    spdir = os.path.join(args.dir, args.sp)

    # Build up a set containing the filenames of all the molecules present
    # anywhere in reactions.txt
    files = set()

    with open(dir_reactions_txt, 'r') as f:
        for line in f:

            # Treat lines that start with # as a comment, and allow blank lines
            if line.startswith('#') or line.strip() == '':
                continue

            words = line.strip().split(',')

            # Add filenames of all molecules in reaction to the growing set.
            files |= {words[0], words[2], words[4], words[6]}

            # An optional final element corresponds to a transition state,
            # for use in training or validation sets.
            if len(words) == 9:
                files |= {words[8]}

    try:
        os.mkdir(spdir)
    except Exception:
        pass

    for name in files:

        try:
            # Read final geometry, charge, and spin from the geometry
            # optimization Gaussian .out output file
            optdir_file_out = os.path.join(optdir, name + '.out')
            charge, spin, geometry = read_opt_file(optdir_file_out)

            # Write all necessary single-point Gaussian .com input files to
            # compute the input features for the ML model.
            write_feature_inputs(spdir, name, geometry, charge, spin)

        except Exception:

            print('Did not find geometry optimization output file for %s'
                  % name)
