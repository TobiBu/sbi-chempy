rule compute_APE:
    input:
        "src/data/pytorch_state_dict.pt"
        "src/data/chempy_data/chempy_TNG_val_data.npz"
    output:
        "src/tex/output/ape_NN.txt"
    script:
        "src/scripts/train_torch_chempy.py"