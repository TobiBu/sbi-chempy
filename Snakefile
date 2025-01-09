rule compute_APE:
    input:
        "src/data/pytorch_state_dict.pt", "src/data/chempy_data/chempy_TNG_val_data.npz"
    output:
        "src/tex/output/ape_NN.txt"
    script:
        "src/scripts/train_torch_chempy.py"

rule compute_posterior_APE:
    input:
        "src/data/pytorch_state_dict.pt", "src/data/chempy_data/chempy_TNG_val_data.npz"
    output:
        "src/tex/output/global_posterior_APE_NPE_C.txt", "src/tex/output/posterior_APE.txt", "src/data/posterior_NPE_C.pickle"
    script:
        "src/scripts/train_sbi.py"