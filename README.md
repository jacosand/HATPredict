# HATPredict

HATPredict is a Python program that uses a machine learning model to predict the **free energy reaction barriers (∆G‡)** for **hydrogen atom transfer (HAT)** reactions of the form:

**A–H  +  B·  ⇌  A·  +  B–H**

The machine learning model uses input features derived from density functional theory (DFT) computations performed on the four ground-state compounds (**A–H**,  **A·**, **B–H**, and **B·**) with the Gaussian quantum chemistry software package.  At test time, **no transition states [A--H--B]‡ are needed**, thus eliminating the most expensive part of computing free energy reaction barriers and creating both human and computational time savings.

Often we imagine that **A–H** corresponds to a substrate undergoing a hydrogen abstraction by catalyst **B·**, so we will often view the reaction above as:

**Sub–H  +  Cat·  ⇌  Sub·  +  Cat–H**

However, HATPredict can be used to predict the free energy barrier (∆G‡) for *any* HAT reaction.

The program uses a machine learning approach adapted from a [paper](https://doi.org/10.1039/D1QO01325D) by Professor Xin Hong at Zhejiang University, but with several original improvements:

- The original AdaBoost model is replaced by an [XGBoost](https://xgboost.readthedocs.io/en/stable/index.html) model, which outperforms the original model particularly when applied to new catalysts (**B·** or **Cat·**) not present in the original training data.

- The original manual hyperparameter search is replaced by a much more extensive hyperparameter search based on Bayesian optimization ([Hyperopt](http://hyperopt.github.io/hyperopt/)).

- The new model is trained not only on forward reactions, as illustrated above, but also on reverse reactions (B–H  +  A·  ⇌  B·  +  A–H).  This doubles the size of the training data without requiring any new data collection, and improves model performance on new and unseen catalysts.

- At test time, the new model automatically combines predictions of forward and reverse reactions to generate better overall predictions by making use of the known free energy of reaction (∆G_rxn) present in the input features.

# Software Dependencies

- Python, NumPy, Pandas (as with [Anaconda](https://www.anaconda.com/))
- xgboost ([installation instructions](https://xgboost.readthedocs.io/en/stable/install.html)): currently tested with version 1.6.2 installed via `pip`
- morfeus ([installation instructions](https://kjelljorner.github.io/morfeus/)): used to compute buried volumes of molecules

In addition, you need access to the [Gaussian 16](https://gaussian.com/gaussian16/) quantum chemistry software package to perform the quantum chemistry computations used to generate the input descriptors for the machine learning model.  Older versions of Gaussian may be compatible but have not been tested.

# Model Training

The training of HATPredict is explained and performed in the Jupyter notebook `Training.ipynb`.  All details related to training the machine learning model are contained in that notebook.  If you simply want to use the already-trained model, you do not need to look at the training notebook.

Model training also requires the following additional software dependencies (not required to use the already-trained model):

- Matplotlib, seaborn, scikit-learn (as with [Anaconda](https://www.anaconda.com/))
- Hyperopt ([installation instructions](http://hyperopt.github.io/hyperopt/))
- SHAP ([installation instructions](https://shap.readthedocs.io/en/latest/index.html))

# How to Run HATPredict on Your Own Reactions

The following section walks through a complete example of using HATPredict to predict the free energy reaction barriers (∆G‡) for a small set of three hydrogen atom transfer (HAT) reactions.  All files needed for this example are included in the `test_data/example` subdirectory.  To run HATPredict on your own reactions, create your own subdirectory of `test_data`, mimic the directory structure as described below, and replace `test_data/example` with `test_data/[my subdirectory]` in any calls to a script.  You can easily run HATPredict on thousands of reactions at a time.

## Step 1: Optimize the Geometries of All Starting Materials and Products

Use Gaussian 16 to optimize all starting materials and products present in all of your HAT reactions.  The optimizations must be performed at the **B3LYP/6-31+G(d,p)** level of theory using the following route line:

```
#p opt freq=noraman b3lyp/6-31+g(d,p)
```

Be sure to optimize **A–H**,  **A·**, **B–H**, and **B·** for each HAT reaction, and be sure the charge and spin for each species is correct.  The Gaussian files may be named according to convenience for your particular project.  If a species appears in more than one reaction (such as when considering many reactions of a single catalyst), the species only needs to be optimized once.

Place all Gaussian optimization output files in the `test_data/[my subdirectory]/opt` subdirectory.  You can find example files in `test_data/example/opt`.

If you have input geometries present as `.xyz` files, you can optionally use the `xyz_to_com.py` script to generate appropriate Gaussian input files.  For usage information, run:

```
python xyz_to_com.py -h
```

## Step 2: Prepare Text File Describing All HAT Reactions to Compute with HATPredict

HATPredict requires a text file, named `reactions.txt`, where each line describes one HAT reaction whose free energy barrier (∆G‡) is to be estimated with the machine learning model.  Here is the contents of `test_data/example/reactions.txt` needed for our small set of three HAT reactions:

```
SUB-PRE-S04,10,SUB-POST-S04-P3,10,CAT-PRE-O8,4,CAT-POST-O8,4
SUB-PRE-S10,12,SUB-POST-S10-P1,12,CAT-PRE-O8,4,CAT-POST-O8,4
SUB-PRE-S04,1,SUB-POST-S04-P5,1,CAT-PRE-O7,5,CAT-POST-O7,5
```

As you can see, each line corresponds to one HAT reaction and contains the following eight items separated by commas:

1. Output filename for **A–H** (without directory or .out extension)

2. Atom number of the heavy atom in the above file undergoing HAT

3. Output filename for **A·** (without directory or .out extension)

4. Atom number of the heavy atom in the above file undergoing HAT

5. Output filename for **B·** (without directory or .out extension)

6. Atom number of the heavy atom in the above file undergoing HAT

7. Output filename for **B–H** (without directory or .out extension)

8. Atom number of the heavy atom in the above file undergoing HAT

As the example above shows, it is fine to use the same filename in more than one reaction (such as when considering many reactions of a single catalyst or substrate).

Prepare your `reactions.txt` file according to the format described above, and save it to `test_data/[my subdirectory]/reactions.txt`.

## Step 3: Perform Single-Point Computations on All Starting Materials and Products

HATPredict requires a set of single-point density functional theory (DFT) computations for all starting materials and products (**A–H**,  **A·**, **B–H**, and **B·**) that are included anywhere in the `reactions.txt` file.  These single-point computations, performed with Gaussian 16, provide the input features for the machine learning model.

To generate all necessary Gaussian input files for all species included in the `reactions.txt`, call the `write_sp_inputs.py` script and pass your subdirectory as an argument:

```
python write_sp_inputs.py test_data/[my subdirectory]
```

The script will create a new directory, `test_data/[my subdirectory]/sp`, and generate all needed Gaussian input files there.  (For a full list of options, call the script with the `-h` argument.)

Run all of these computations with Gaussian 16, and then transfer the output files back to `test_data/[my subdirectory]/sp`.

As an example, for our small set of three HAT reactions, the script would be called as:

```
python write_sp_inputs.py test_data/example
```

You can check out the Gaussian input files generated by this script, as well as the associated output files, in `test_data/examples/sp`.

## Step 4: Extract the Input Features for the Machine Learning Model

After all Gaussian computations are complete, extract the input features for the machine learning model by calling the `get_descriptors.py` script and passing your subdirectory as an argument:

```
python get_descriptors.py test_data/[my subdirectory]
```

The script will create a new `.csv` file, `test_data/[my subdirectory]/descriptors.csv`, which contains all of the input features for the machine learning model.  (For a full list of options, call the script with the `-h` argument.)

As an example, for our small set of three HAT reactions, the script would be called as:

```
python get_descriptors.py test_data/example
```

You can check out the `descriptors.csv` file generated by this script at `test_data/examples/descriptors.csv`.

## Step 5: Run the Machine Learning Model to Obtain Predicted Free Energy Barriers

After the DFT input descriptors have been computed, predict the free energy barriers (∆G‡) with the machine learning model by calling the `get_barriers.py` script and passing your subdirectory as an argument:

```
python get_barriers.py test_data/[my subdirectory]
```

The script will create a new `.csv` file, `test_data/[my subdirectory]/barriers.csv`, which contains the ML-predicted free energy barriers for each reaction.  (For a full list of options, call the script with the `-h` argument.)

As an example, for our small set of three HAT reactions, the script would be called as:

```
python get_barriers.py test_data/example
```

You can check out the `barriers.csv` file generated by this script at `test_data/examples/barriers.csv`.

# Future Work

I am currently working to expand the training data to a larger number of catalysts in order to improve the performance of HATPredict on new and unseen catalysts.  Several other model and data improvements are planned as well.
