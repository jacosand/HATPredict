#!/usr/bin/env python

"""
Script to apply the trained machine learning model to predict the free energy
barriers for a list of hydrogen atom transfer (HAT) reactions whose DFT
descriptors are in a .csv descriptors file (default: descriptors.csv).
Writes predictions to another .csv file (default: barriers.csv).

This script can be used in training mode, if DFT-computed free energy barriers
are included in the .csv descriptors file, in which case ML-predicted and
DFT-computed free energy barriers are compared.  In the default inference mode,
only ML-predicted free energy barriers are saved (the ground-truth DFT-computed
free energy barriers are assumed to be unknown).

Run from the command line:
python get_barriers.py [arguments] /path/to/directory/with/descriptors.csv

For possible arguments, see argument parser in __main__ section or run:
python get_barriers.py -h
"""

import os
import argparse
import pandas as pd
import xgboost


def reverse_reactions(df_):
    """
    Function which reads a Pandas dataframe whose rows contain the descriptors,
    and optionally the ground-truth DFT enthalpy and/or free energy barriers,
    for hydrogen atom transfer reactions, and returns a Pandas dataframe whose
    rows correspond to the reverse reactions.  The reverse reactions are in the
    same order as the corresponding forward reactions, and the input Pandas
    dataframe is unmodified.

    Arguments:

        - df_: Pandas dataframe whose rows contain the descriptors, and
               optionally the ground-truth DFT enthalpy and/or free energy
               barriers, for hydrogen atom transfer reactions

    Returns:

        - df_reverse: Pandas dataframe whose rows contain the descriptors,
                      and optionally the ground-truth DFT enthalpy and/or
                      free energy barriers, for the reverse reactions
    """

    df_reverse = df_.copy(deep=True)

    # Descriptors pertaining to the "substrate" in the forward reaction
    # pertain to the "catalyst" in the reverse reaction, and vice versa.
    # So swap the corresponding "substrate" and "catalyst" descriptors.
    for col in list(df_.columns):
        if col.startswith('sub_'):
            df_reverse[[col, 'cat_' + col[4:]]
                       ] = df_[['cat_' + col[4:], col]]
        elif col.startswith('subH_'):
            df_reverse[[col, 'catH_' + col[5:]]
                       ] = df_[['catH_' + col[5:], col]]

    # Compute the enthalpy and free energy barriers of the reverse reaction.
    df_reverse['DHdd'] = df_['DHdd'] - df_['DH']
    df_reverse['DGdd'] = df_['DGdd'] - df_['DG']

    # The enthalpy and free energy of reaction is negated for the reverse
    # reaction.
    df_reverse['DH'] = -df_['DH']
    df_reverse['DG'] = -df_['DG']

    return df_reverse


def add_reverse_reactions(df):
    """
    Function which reads a Pandas dataframe whose rows correspond to hydrogen
    atom transfer reactions, and returns a Pandas dataframe where the reverse
    reactions have been appended.

    In addition, a new "group" column is created for use in group k-fold
    cross-validation; this column always contains the name of the substrate
    reactant.  In particular, a forward reaction and its reverse reaction are
    always in the same group.

    Arguments:

        - df: Pandas dataframe whose rows correspond to hydrogen atom transfer
              reactions

    Returns:

        - Pandas dataframe where the reverse reactions have been appended (and
          a "group" column has been added)
    """

    return (df
            # Add "group" column
            .assign(group=lambda df_: df_.subH_file)
            # Append reverse reactions
            .pipe(lambda df_: pd.concat([df_, reverse_reactions(df_)],
                                        ignore_index=True))
            )


def split_and_tweak(df):
    """
    Function which reads a Pandas dataframe whose rows correspond to hydrogen
    atom transfer reactions, and splits the columns into input descriptors (x),
    ground-truth outputs (y), filenames of the reaction species (rxns), and
    groups (group) used for group k-fold cross-validation.

    If this function is called in testing mode, rxns['ts_file'] and y will be
    NaN.

    The outputs of this function are suitable for use in scikit-learn machine
    learning models.

    Arguments:

        - df: Pandas dataframe whose rows correspond to hydrogen atom transfer
              reactions

    Returns:

        - x: Pandas dataframe whose rows correspond to input descriptors for
             hydrogen atom transfer reactions

        - y: Pandas series whose entries correspond to the ground-truth output
             for hydrogen atom transfer reactions, in this case the
             ground-truth free energy barrier computed via DFT.
             NaN if function is called in testing mode.

        - rxns: Pandas dataframe whose rows correspond to the filenames of the
                reaction species for hydrogen atom transfer reactions.
                rxns['ts_file'] is NaN if function is called in testing mode.

        - group: Pandas series whose entries correspond to the groups used in
                 group k-fold cross-validation (see the function
                 add_reverse_reactions above for how the groups are chosen)
    """

    rxn_files = ['subH_file', 'cat_file', 'sub_file', 'catH_file', 'ts_file']
    rxns = df[rxn_files]

    y = df['DGdd']

    group = df['group']

    x = (df
         .drop(columns=rxn_files)
         .drop(columns=['DHdd', 'DGdd'])
         .drop(columns=['group'])
         )

    return x, y, rxns, group


def test_with_reverse_reactions(x, model):
    """
    Function which applies a fitted machine learning model to input descriptors
    and returns the predictions.

    Arguments:

        - x: Pandas dataframe whose rows correspond to input descriptors for
             hydrogen atom transfer reactions.  The reactions found in the
             second half of the rows must be the reverse of the reactions found
             in the first half of the rows, in the same order.  An appropriate
             dataframe can be created via sequential application of
             add_reverse_reactions followed by split_and_tweak.

        - model: A fitted machine learning model, as with scikit-learn

    Returns:

        - n_rxns: Number of forward reactions present in the input (half the
                  number of rows in the input), returned for convenience in
                  plotting only the forward reactions

        - y_pred: Pandas series whose entries correspond to the outputs
                  predicted with the machine learning model.  Outputs for the
                  forward reactions are returned followed immediately by
                  outputs for the reverse reactions.
    """

    # Obtain raw model predictions for both forward and reverse reactions
    y_pred_raw = model.predict(xgboost.DMatrix(x))

    # Compute the number of forward reactions
    n_rxns = y_pred_raw.shape[0] // 2

    # Combine the raw predictions of the forward and reverse reactions to
    # generate better predictions...

    # ... of the forward reactions.
    y_pred_fwd = pd.Series(
        y_pred_raw[:n_rxns] + y_pred_raw[n_rxns:] + x['DG'][:n_rxns]) / 2

    # ... and of the reverse reactions.
    y_pred_rev = pd.Series(
        y_pred_raw[:n_rxns] + y_pred_raw[n_rxns:] + x['DG'][n_rxns:]) / 2

    y_pred = pd.concat([y_pred_fwd, y_pred_rev], ignore_index=True)

    return n_rxns, y_pred


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Applies the trained machine \
        learning model to predict the free energy barrier for each reaction \
        whose DFT descriptors are in the .csv descriptors file (default: \
        descriptors.csv), and writes the predictions to another .csv file \
        (default: barriers.csv).  Can optionally be used in training mode to \
        compare ML-predicted free energy barriers to ground-truth \
        DFT-computed barriers.")

    parser.add_argument(
        "dir",
        help="path to directory which contains .csv file containing DFT input \
              descriptors",
    )

    parser.add_argument(
        "-c", "--csv",
        help="base filename of .csv file (without extension) in which DFT \
              input descriptors are stored",
        default='descriptors',
    )

    parser.add_argument(
        "-o", "--out",
        help="base filename of .csv file (without extension) in which ML \
              predicted barriers are to be written",
        default='barriers',
    )

    parser.add_argument(
        "-t", "--train",
        help="run in training mode, comparing ML-predicted to ground-truth \
              DFT-computed free energy barriers",
        action='store_true',
    )

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise FileNotFoundError(
            'Dataset directory %s does not exist!' % args.dir)

    dir_descriptors_csv = os.path.join(args.dir, args.csv + '.csv')
    dir_barriers_csv = os.path.join(args.dir, args.out + '.csv')

    # Read and preprocess test data set
    data_test = pd.read_csv(dir_descriptors_csv)
    x_test, y_test, rxns_test, group_test = split_and_tweak(
        add_reverse_reactions(data_test))

    # Import trained XGBoost model and run it on test data set
    xgb = xgboost.Booster(model_file='xgb_model.json')
    n_test, y_test_pred = test_with_reverse_reactions(x_test, xgb)

    # Create Pandas dataframe storing all free energy barriers
    # If not in training mode, the 'TS', 'DFT Barrier (kcal/mol)', and
    # 'Difference (kcal/mol)' columns should be NaN.
    barriers = (rxns_test
                .rename(columns={'subH_file': 'A–H',
                                 'sub_file': 'A·',
                                 'cat_file': 'B·',
                                 'catH_file': 'B–H',
                                 'ts_file': 'TS'})
                .assign(**{'ML Barrier (kcal/mol)': y_test_pred,
                           'DFT Barrier (kcal/mol)': y_test,
                           'Difference (kcal/mol)': y_test_pred - y_test})
                )

    # Drop NaN columns if not in training mode.
    if not args.train:
        barriers.drop(['TS', 'DFT Barrier (kcal/mol)',
                      'Difference (kcal/mol)'], axis=1, inplace=True)

    # Print ML-predicted free energy barriers and save them to .csv file.
    print()
    print('ML-Predicted Free Energy Barriers:')

    print()
    pd.options.display.float_format = '{:.2f}'.format
    pd.options.display.max_rows = 200
    print(barriers[:n_test])
    barriers[:n_test].to_csv(
        dir_barriers_csv, index=False, float_format='{:.2f}'.format)

    print()
    print('The ML-predicted free energy barriers were saved to: '
          + dir_barriers_csv)
    print()
