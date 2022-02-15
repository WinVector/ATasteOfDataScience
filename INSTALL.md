# Installing

Honestly, installing and versions are a bit of a nightmare for Python, Jupyter, and some combinations of data science packages.

Our instructions are oriented towards a MacOS system. Similar should work on Linux or Windows, however Linux does also have its own package managers.

The basic install idea is:

  * Read XKCD 1987: https://xkcd.com/1987/
  * Install Anaconda from https://www.anaconda.com
  * Start the `Anaconda Navigator` App (either installed where applications are installed or in HOME/opt)
  * Select the `Environments` panel
  * Press `Import` and use the pop-up file browser to import `data_science_examples.yaml`
  * Return to the `Home` panel

To run we then, in a running `Anaconda Navigator`:

  * Make sure the `Applications` pull-down is on data_science_examples
  * Click Launch on the `JupyterLab` pane (if that fails one can fall back to `JupyterNotebook` or `PyCharm`)
  

-------


Note: to get JupyterLab from [Anaconda](https://www.anaconda.com) to run on a Mac we have found one must run once:

```
conda activate data_science_examples.yaml 
jupyter server extension disable nbclassic
```

on the command line in the conda environment ([source](https://lifesaver.codes/answer/extensionmanager-object-has-no-attribute-extensions-when-starting-jupyterlab-10228)).

Running conda commands requires sourcing the code that the anaconda install writes into one of `.bash_profile`, `.zsh_profile`, `.profile`, `.bashrc` (depending on your system).

Baring that one can run JupyterNotebook, [VSCode](https://code.visualstudio.com), or [DataSpell](https://www.jetbrains.com/dataspell/) ([PyCharm](https://www.jetbrains.com/help/pycharm/jupyter-notebook-support.html)'s JupyterNotebook support may also have issues).

--------

Copyright Win Vector LLC 2022 https://www.win-vector.com/

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.

