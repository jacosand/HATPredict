#!/usr/bin/env python

"""
Script to convert molecular geometries in .xyz files into Gaussian
input .com files.

Run from the command line:
python xyz_to_com.py [arguments] /path/to/.xyz/files

For possible arguments, see argument parser in __main__ section or run:
python xyz_to_com.py -h

Script outputs one .com file for each .xyz file that matches the path,
in the same directory and with the same base filename.
"""

import os
import argparse


def read_xyz_file(dir_file_xyz):
    """
    Function which extracts a molecular geometry from a .xyz file.

    Arguments:

        - xyz_file: string containing the full path to a .xyz file,
                    including the .xyz extension.

    Returns:

        - geometry: a list of 4-tuples, where each atom in the molecule
                    corresponds to one 4-tuple.  Each 4-tuple contains
                    (Element Abbreviation, X-coord, Y-coord, Z-coord)
    """

    geometry = []

    # TODO: Incorporate this into a verbose option.
    print(dir_file_xyz)

    with open(dir_file_xyz, 'r') as f:

        # Skip the first two lines
        f.readline()
        f.readline()

        # The remaining lines correspond to individual atoms
        for line in f:
            words = line.split()
            if len(words) == 4:
                atom, x, y, z = line.split()
                geometry.append((atom, float(x), float(y), float(z)))

    return geometry


def write_com_file(dir_file_com, geometry, route, link='', charge=0, spin=1,
                   add_input='', nproc=8, mem='16GB', chk=False):
    """
    Function which writes a Gaussian input .com file given a molecular geometry
    and other arguments.

    Arguments:

        - dir_file_com: string containing the full path to the desired output
                        .com file, including the .com extension

        - geometry: a list of 4-tuples, where each atom in the molecule
                    corresponds to one 4-tuple.  Each 4-tuple contains
                    (Element Abbreviation, X-coord, Y-coord, Z-coord)

        - route: string containing the routeline for the Gaussian job

        - link: optional string containing a routeline for a secondary Gaussian
                job to be run after the first job is complete

        - charge: integer charge of the molecule

        - spin: integer spin of the molecule

        - add_input: optional string containing additional lines to append at
                     the end of the main Gaussian job, used for scans and other
                     constraints

        - nproc: integer number of processors to use for the Gaussian job

        - mem: string describing amount of memory to use for the Gaussian job

        - chk: boolean describing whether to write a Gaussian .chk checkpoint
               file

    Returns:

        Nothing
    """

    # Strip off .com extension
    dir_file = os.path.splitext(dir_file_com)[0]

    # Extract the base filename without the path
    file = os.path.basename(dir_file)

    with open(dir_file_com, 'w') as f:

        f.write('%nprocshared=' + str(nproc) + '\n')
        f.write('%mem=' + mem + '\n')
        if chk:
            f.write('%chk=' + file + '.chk\n')

        f.write(route + '\n')
        f.write('\n')

        f.write(file + '\n')
        f.write('\n')

        f.write(str(charge) + ' ' + str(spin) + '\n')

        for line in geometry:
            f.write(' %-15s %13.8f %13.8f %13.8f\n' % line)
        f.write('\n')

        if add_input != '':
            f.write(add_input + '\n')

        if link != '':
            f.write('--Link1--' + '\n')
            f.write('%nprocshared=' + str(nproc) + '\n')
            f.write('%mem=' + mem + '\n')
            if chk:
                f.write('%chk=' + file + '.chk\n')
            f.write(link + '\n')

        f.write('\n')


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert .xyz output files \
        into .com Gaussian input files")

    parser.add_argument(
        "files",
        help="pattern that matches .xyz output files",
        nargs='+',
    )

    parser.add_argument(
        "-r", "--route",
        help="routeline (default '#p opt freq=noraman ub3lyp/6-31+g(d,p)')",
        default='#p opt freq=noraman ub3lyp/6-31+g(d,p)',
    )

    parser.add_argument(
        "-l", "--link",
        help="link1 routeline for second job (default None)",
        default='',
    )

    parser.add_argument(
        "-c", "--charge",
        help="charge (default 0)",
        default=0,
        type=int,
    )

    parser.add_argument(
        "-s", "--spin",
        help="spin (default 1)",
        default=1,
        type=int,
    )

    parser.add_argument(
        "-n", "--nproc",
        help="number of processors (default 8)",
        default=8,
        type=int,
    )

    parser.add_argument(
        "-m", "--mem",
        help="memory as a string (default '16GB')",
        default='16GB',
    )

    parser.add_argument(
        "-a", "--add",
        help="lines to add to file for constraints and scans (default None)",
        default='',
    )

    parser.add_argument(
        "--chk",
        help="write checkpoint files",
        action='store_true',
    )

    args = parser.parse_args()

    # Loop over files that match the path
    for dir_file_xyz in args.files:

        # Extract molecular geometry from .xyz file
        geometry = read_xyz_file(dir_file_xyz)

        # Strip off .xyz extension
        dir_file = os.path.splitext(dir_file_xyz)[0]

        # Add .com extension
        dir_file_com = dir_file + '.com'

        # Write Guassian .com file
        write_com_file(
            dir_file_com, geometry, args.route, link=args.link,
            charge=args.charge, spin=args.spin, add_input=args.add,
            nproc=args.nproc, mem=args.mem, chk=args.chk,
        )
