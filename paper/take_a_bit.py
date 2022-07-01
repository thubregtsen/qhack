from dill import dump, load

with open("data/noisy_sim/kernel_matrices_Checkerboard_untrained.dill", "rb+") as f:
    matrices = load(f)

new_matrices = {}
for key, mat in matrices.items():
    if len(key)>2 and key[2]<10:
        new_matrices[key] = mat

with open("data/noisy_sim/kernel_matrices_Checkerboard_untrained.selection", "wb+") as f:
    dump(new_matrices, f)
