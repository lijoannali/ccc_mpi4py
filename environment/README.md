# Manual conda environment installation and data download

If you want to run the scripts/notebooks, you need to follow these steps to create a conda environment and download the necessary data.

Keep in mind that although unit tests are automatically run on Linux, macOS and MS Windows, the software is manually tested only on Linux/Ubuntu.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.

2. Open a terminal, clone this repository, and `cd` into the repository root folder.

3. Run `cd environment`.

4. (optional) Adjust your environment variables:

    ```bash
    # (optional, will default to subfolder 'cm_gene_expr' under the system's temporary directory)
    # Root directory where all data will be downloaded to
    export CM_ROOT_DIR=/tmp/cm_gene_expr

    # (optional, will default to half the number of cores)
    # Adjust the number of cores available for general tasks
    export CM_N_JOBS=2

    # (optional)
    # Export this variable if you downloaded the manuscript sources and want to
    # generate the figures for it
    export CM_MANUSCRIPT_DIR=/tmp/manuscript
    ```

5. (optional) Adjust other settings (i.e. root directory, available computational
   resources, etc.) by modifying the file `../libs/clustermatch/settings.py`

6. Adjust your `PYTHONPATH` variable to include the `libs` directory:

    ```bash
    export PYTHONPATH=`readlink -f ../libs/`:$PYTHONPATH
    ```

    `readlink` might not work on macOS. In that case, simply replace it with
    the absolute path to the `../libs/` folder.

7. Run `bash scripts/setup_environment.sh`.
This will create a conda environment and download the data needed to run the analyses.
It will download `XXX` GB, so it will take a while to finish.
Keep in mind two things here:
1) the script requires conda >= 4.9.0 to work;
if you cannot upgrade, just remove the `--no-capture-output` parameter in the script.
2) we use conda packages from the `conda-forge` channel;
if you find any issues with the installation, it is advisible to configure the channel priority of your conda to `strict` if possible (`conda config --set channel_priority strict`).

# Developer usage

These steps are only for developers.

1. Modify `scripts/environment_base.yml` accordingly (if needed).
1. Run:
 
    ```bash
    conda env create -n clustermatch_gene_expr -f scripts/environment_base.yml
    conda activate clustermatch_gene_expr
    bash scripts/install_other_packages.sh
    ```

<!-- 
1. (CHECK!) Install JupyterLab extensions (MIGHT NOT BE NECESSARY IN VERSION 3.0+):
 
    ```bash
    jupyter labextension install @jupyterlab/toc
    ``` -->

1. Export conda environment:

    ```
    conda env export --name clustermatch_gene_expr --file environment.yml
    ```

1. Modify `environment.yml` and leave only manually installed packages (not their dependencies).