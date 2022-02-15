# Installing


## How to install

Honestly, installing and versions are a bit of a nightmare for Python, Jupyter, and some combinations of data science packages.

Our instructions are oriented towards a MacOS system. Similar should work on Linux or Windows, however Linux does also have its own package managers.

The basic install idea is:

  * `git clone git@github.com:WinVector/ATasteOfDataScience.git`
  * Read XKCD 1987: https://xkcd.com/1987/
  * Install Anaconda from https://www.anaconda.com
  * Start the `Anaconda Navigator` App (either installed where applications are installed or in HOME/opt)
  * Select the `Environments` panel
  * Press `Import` and use the pop-up file browser to import `data_science_examples.yaml`
  * Return to the `Home` panel

To run we then, in a running `Anaconda Navigator`:

  * Make sure the `Applications` pull-down is on data_science_examples
  * Click Launch on the `JupyterLab` pane (if that fails one can fall back to `JupyterNotebook` or `PyCharm`)

The installation YAML is [data_science_examples.yaml](data_science_examples.yaml), and the exact versions used (listed via `conda list`) is [data_science_examples_versions.txt](data_science_examples_versions.txt).

We suggest re-running some of the example `.ipynb` files to see if the install is working.

  
## Why to install

Installing software is pain. However, we feel it is worth the effort when possible.

However, *always* using remote services and pre-built containers has its own risks and promotes a learned helplessness. By working through a single install once we are trying to isolate many issues into one session. Also, installing must be possible- else how are remote services and containers provisioned in the first place?

## Known Issues / Fixes


### Can't use "conda activate data_science_examples" from the shell

Find out which of your home startup dot-files Anaconda wrote "added by Anaconda" into, and copy this code into which startup file is actually executed on shell startup. Candidates include: `.bash_profile`, `.zsh_profile`, `.profile`, `.bashrc` (depending on your system).

### JupyterLab won't launch

Note: to get JupyterLab from [Anaconda](https://www.anaconda.com) to run on a Mac we have found one must run once:

```
conda activate data_science_examples.yaml 
jupyter server extension disable nbclassic
```

on the command line in the conda environment ([source](https://lifesaver.codes/answer/extensionmanager-object-has-no-attribute-extensions-when-starting-jupyterlab-10228)).

Baring that, one can run JupyterNotebook, [VSCode](https://code.visualstudio.com), or [DataSpell](https://www.jetbrains.com/dataspell/). [PyCharm](https://www.jetbrains.com/help/pycharm/jupyter-notebook-support.html) JupyterNotebook support may also have issues.

### Packages don't seem to be installed

Make sure one has selected the data_science_examples environment.


