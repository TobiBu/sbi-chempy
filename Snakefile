rule train_emulator:
    input: 
        "src/data/chempy_data/chempy_TNG_train_data.npz",
        "src/data/chempy_data/chempy_TNG_val_data.npz"
    output:
        "src/data/pytorch_state_dict.pt"
    script:
        "src/scripts/train_torch_chempy.py"
rule compute_APE:
    input:
        "src/data/pytorch_state_dict.pt",
        "src/data/chempy_data/chempy_TNG_val_data.npz"
    output:
        "src/tex/output/ape_NN.txt",
        "src/tex/output/ape_NN_log.txt",
        "src/tex/figures/ape_NN.pdf"
    script:
        "src/scripts/evaluate_emulator.py"
rule train_posterior:
    input:
        "src/data/pytorch_state_dict.pt"
    output:
        "src/data/posterior_NPE_C.pickle"
    script:
        "src/scripts/train_sbi.py"
rule compute_posterior_APE:
    input:
        "src/data/pytorch_state_dict.pt",
        "src/data/chempy_data/chempy_TNG_val_data.npz",
        "src/data/posterior_NPE_C.pickle"
    output:
        "src/tex/output/global_posterior_APE.txt",
        "src/tex/output/posterior_APE.txt",
        "src/data/ape_posterior_NPE_C.pt"
    script:
        "src/scripts/evaluate_posterior.py"
rule sample_posterior:
    input:
        "src/data/pytorch_state_dict.pt",
        "src/data/posterior_NPE_C.pickle"
    output:
        "src/data/alpha_IMF_obs_NPE_C.npy",
        "src/data/log10_N_Ia_obs_NPE_C.npy"
    script:
        "src/scripts/sample_posterior.py"

rule tng_inference:
    input:
        "src/data/alpha_IMF_obs_NPE_C.npy",
        "src/data/log10_N_Ia_obs_NPE_C.npy"
    output:
        "src/tex/output/CHEMPY_TNG_sbi.txt"
    script:
        "src/scripts/chempy_tng_inference.py"