# Training Quantum Embedding Kernels on Near-Term Quantum Computers

Here we share code to run the numeric experiments and generate the figures for our paper on [trainable quantum embedding kernels (QEKs)](https://arxiv.org/).

_All following descriptions refer to the _`paper/`_ subdirectory._ In the subdirectory `qhack2021/` you can find our hackathon submission to [qhack 2021](https://pennylane.ai/blog/2021/03/qhack-the-quantum-machine-learning-hackathon/).

### How to use this repository
Install the `requirements.txt` using pip in order to be able to execute all notebooks (assuming you are in the `paper/` subdirectory):
```
pip install -r src/requirements.txt --upgrade
```
This might take a moment if you start from a fresh virtual environment as packages like sklearn and tensorflow are required.

The .py files in the home directory are jupyter notebooks, stored with [Jupytext](https://jupytext.readthedocs.io/en/latest/) for convenient synchronization.
The requirements contain this package, so that after installation you can open any of the python files in jupyter-notebook, hit save and will obtain a `.ipynb`
copy automatically, in which we then recommend to run the computations.

Some of the notebooks take considerable time to run the full simulations or even require a AWS bucket and credits to run (for the QPU experiment).
In particular for the kernel training notebooks (`noiseless_*`), the recomputation will require hours of runtime.
However, we fixed the pseudo-randomness seeds where possible and stored the corresponding output for the mitigation numerics so you can reuse those results for visualization and tinkering.
For the hardware experiment we included a surrogate simulation of the experiment, though without our noise model. The mitigation of the original hardware kernel matrix is stored and seed-fixed as well but can be recomputed within about 15 minutes.

### Structure
The first numeric experiment is about noiseless simulations of the QEKs and training them via kernel target alignment. These experiments can be found in the files
* `noiseless_checkerboard_and_donuts.py`
* `noiseless_MNIST.py`
and the decision boundary plots are generated in
* `noiseless_plot.py`

The second experiment is about device noise mitigation and regularization of the kernel matrix. The data is generated and processed in
* `noisy_generating.py`
* `noisy_processing.py`
respectively. A small demo on how to use the mitigation and regularization methods introduced in our `qml.kernels` patch to [PennyLane](https://github.com/PennyLaneAI/pennylane) and on the combinations
of these methods we looked at can be found in
* `post_processing_demo.py`

The third experiment is about evaluation and mitigation of a QEK on a real QPU. The data generation and processing again is split up into
* `qpu_generating.py`
* `qpu_processing.py`

Finally there are directories
* `data/` for data (no, really!)
* `images/` for the images that are produced by the notebooks in the home directory.
* `src/` for some source code like helper functions, dataset generation etc.
* `style/` for reference files to be used in calls to the python package [RSMF](https://github.com/johannesjmeyer/rsmf)
