# Training Quantum Embedding Kernels on Near-Term Quantum Computers

Here we share code to run the numeric experiments and generate the figures for our paper on [trainable quantum embedding kernels (QEKs)](https://arxiv.org/).

### How to use this repository
Install the `requirements.txt` using pip in order to execute all notebooks.

The .py files in the home directory are jupyter notebooks, stored with [Jupytext](https://jupytext.readthedocs.io/en/latest/) for convenient synchronization.
The requirements contain this package, so that after installation you can open any of the python files in jupyter-notebook, hit save and will obtain a `.ipynb`
copy automatically, in which we then recommend to run the computations.

Some of the notebooks take considerable time to run the full simulations or even require a AWS bucket and credits to run (for the QPU experiment).
However, we fixed the pseudo-randomness seeds where possible and stored the corresponding output so you can reuse those results for visualization and tinkering.
For the hardware experiment we included a surrogate simulation of the experiment, but excluding our noise model.

### Structure
The first numeric experiment is about noiseless simulations of the QEKs and training them via kernel target alignment. These experiments can be found in the files
* `noiseless_checkerboard.py`
* `noiseless_symmetricdonuts.py`
* `noiseless_MNIST.py`
and the decision boundary plots are generated in
* `noiseless_plot.py`

The second experiment is about device noise mitigation and regularization of the kernel matrix. The data is generated and processed in
* `noisy_generating.py`
* `noisy_processing.py`
respectively. A small demo on how to use the mitigation and regularization methods introduced in our `qml.kernels` patch to pennylane and on the combinations
of these methods we looked at can be found in
* `post_processing_demo.py`

The third experiment is about evaluation and mitigation of a QEK on a real QPU. The data generation and processing again is split up into
* `qpu_generating.py`
* `qpu_processing.py`
