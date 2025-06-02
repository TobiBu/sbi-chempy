import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

import paths


##########################################################################################################
# --- Calculate mean and error ---
def mean_std(data, mu_prior, sigma_prior):

    prior_mean = mu_prior
    prior_cov = np.array([[sigma_prior[0] ** 2, 0], [0, sigma_prior[1] ** 2]])
    mean_n = np.zeros_like(prior_mean)
    inv_cov = np.zeros_like(prior_cov)
    n = len(data)

    for x in data:
        mean = x.mean(axis=0)
        cov_n = np.cov(x, rowvar=False)
        inv_cov_n = np.linalg.inv(cov_n)

        mean_n += inv_cov_n @ mean
        inv_cov += inv_cov_n

    post_cov = inv_cov - ((1 - n) * np.linalg.inv(prior_cov))
    post_cov = np.linalg.inv(post_cov)

    post_mean = post_cov @ (mean_n - (1 - n) * (np.linalg.inv(prior_cov) @ prior_mean))

    return post_mean, post_cov


##########################################################################################################
# --- N-Star parameter plot ---
def n_stars_plot(
    x1,
    x2,
    x_true,
    save_name,
    no_stars=np.array([1, 10, 100, 500, 1000]),
    simulations=1000,
    prior=np.array([[-2.3, -2.89], [0.3, 0.3]]),
):
    fit = []
    err = []

    # --- Fit a 2D Gaussian to the data ---
    for n in no_stars:
        data = np.stack([x1[:n], x2[:n]], 2)
        mu, cov = mean_std(data, prior[0], prior[1])
        mu_alpha, mu_log10N_Ia = mu[0], mu[1]
        sigma_alpha, sigma_log10N_Ia = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])

        fit.append([mu_alpha, mu_log10N_Ia])
        err.append([sigma_alpha, sigma_log10N_Ia])

    fit = np.array(fit)
    err = np.array(err)

    # --- Plot the data ---
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(26, 6))

    def plot(fit, err, true, ax, name):
        ax.plot(no_stars, fit, color="b", label="Fit")
        ax.fill_between(
            no_stars,
            fit - err,
            fit + err,
            alpha=0.3,
            color="b",
            label=r"1 & 2 $\sigma$",
        )
        ax.fill_between(no_stars, fit - 2 * err, fit + 2 * err, alpha=0.2, color="b")

        ax.axhline(true, color="k", linestyle=":", linewidth=2, label="Ground Truth")

        ax.set_xlabel(r"$N_{\rm stars}$", fontsize=40)
        ax.set_ylabel(name, fontsize=40)
        ax.set_ylim([true - 0.2 * abs(true), true + 0.2 * abs(true)])
        ax.set_xscale("log")
        ax.set_xlim([1, no_stars[-1]])
        ax.tick_params(labelsize=30, size=10, width=3)
        ax.tick_params(which="minor", size=5, width=2)

    for i, name in enumerate([r"$\alpha_{\rm IMF}$", r"$\log_{10} N_{\rm Ia}$"]):
        plot(fit[:, i], err[:, i], x_true[0, i], ax[i], name)

    ax[0].legend(fontsize=15, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(paths.figures / f"{save_name}.pdf")


##########################################################################################################
# --- N-Star comparison plot ---


def n_stars_plot_comp(
    x1,
    x2,
    x_true,
    dat,
    save_name,
    no_stars=np.array([1, 10, 100, 500, 1000]),
    simulations=1000,
    prior=np.array([[-2.3, -2.89], [0.3, 0.3]]),
):
    fit = []
    err = []

    # Extract the lambda values and n-stars
    all_Lambdas = dat.f.Lambdas
    n_stars = dat.f.n_stars

    # Here we compute the statistical variances, averaged across realizations with the same value of n-stars
    med, lo, up, lo2, up2, sample_lo, sample_hi = [
        np.zeros((len(n_stars), 2)) for _ in range(7)
    ]
    for i in range(len(n_stars)):
        # Select only the Lambda estimates for this value of n-stars
        theseL = all_Lambdas[i]
        # Now compute the median, 1- and 2-sigma parameter ranges from the output chains for each realization using this n-stars.
        lowL2, lowL, medianL, upL, upL2 = [
            [np.percentile(L, p, axis=0) for L in theseL]
            for p in [2.275, 15.865, 50.0, 84.135, 97.725]
        ]
        # Take the average over all realizations
        up[i] = np.median(upL, axis=0)
        lo[i] = np.median(lowL, axis=0)
        up2[i] = np.median(upL2, axis=0)
        lo2[i] = np.median(lowL2, axis=0)
        med[i] = np.median(medianL, axis=0)

    # --- Fit a 2D Gaussian to the data ---
    for n in no_stars:
        data = np.stack([x1[:n], x2[:n]], 2)
        mu, cov = mean_std(data, prior[0], prior[1])
        mu_alpha, mu_log10N_Ia = mu[0], mu[1]
        sigma_alpha, sigma_log10N_Ia = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])

        fit.append([mu_alpha, mu_log10N_Ia])
        err.append([sigma_alpha, sigma_log10N_Ia])

    fit = np.array(fit)
    err = np.array(err)

    # --- Plot the data ---
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(26, 6))

    def plot(fit, err, true, ax, name):
        ax.plot(no_stars, fit, color="b", label="Fit")
        ax.fill_between(
            no_stars,
            fit - err,
            fit + err,
            alpha=0.1,
            color="b",
            label=r"1 & 2 $\sigma$",
        )
        ax.fill_between(no_stars, fit - 2 * err, fit + 2 * err, alpha=0.1, color="b")

        ax.axhline(true, color="k", linestyle=":", linewidth=2, label="Ground Truth")

        ax.set_xlabel(r"$N_{\rm stars}$", fontsize=40)
        ax.set_ylabel(name, fontsize=40)
        ax.set_ylim([true - 0.2 * abs(true), true + 0.2 * abs(true)])
        ax.set_xscale("log")
        ax.set_xlim([1, no_stars[-1]])
        ax.tick_params(labelsize=30, size=10, width=3)
        ax.tick_params(which="minor", size=5, width=2)
        # Add Philcox
        ax.plot(n_stars, med[:, i], c="r", label="HMC")
        ax.fill_between(n_stars, lo[:, i], up[:, i], alpha=0.2, color="r")
        ax.fill_between(n_stars, lo2[:, i], up2[:, i], alpha=0.1, color="r")

    for i, name in enumerate([r"$\alpha_{\rm IMF}$", r"$\log_{10} N_{\rm Ia}$"]):
        plot(fit[:, i], err[:, i], x_true[0, i], ax[i], name)

    ax[0].legend(fontsize=20, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(paths.figures / f"{save_name}.pdf")


##########################################################################################################
# --- Absolute percentage error plot ---


def ape_plot(ape, labels_in, save_path):
    fig, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.20, 0.80)}
    )
    colors = ["tomato", "skyblue", "olive", "gold", "teal", "orchid"]

    print("\nAPE of the Posterior:")
    print("Median + upper quantile - lower quantile")
    l_quantile, median, u_quantile = np.percentile(ape, [25, 50, 75])
    print(f"Total : {median:.1f}% + {u_quantile-median:.1f} - {median-l_quantile:.1f}")
    print("")

    for i in range(ape.shape[1]):
        l_quantile, median, u_quantile = np.percentile(ape[:, i], [25, 50, 75])
        ax_hist.hist(
            ape[:, i],
            bins=25,
            density=True,
            range=(0, 100),
            label=labels_in[i],
            color=colors[i],
            alpha=0.5,
        )
        median = np.percentile(ape[:, i], 50)
        ax_hist.axvline(median, color=colors[i], linestyle="--")
        print(
            labels_in[i]
            + f" : {median:.1f}% + {u_quantile-median:.1f} - {median-l_quantile:.1f}"
        )

    print()

    ax_hist.set_xlabel("Error (%)", fontsize=15)
    ax_hist.set_ylabel("Density", fontsize=15)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.legend()

    bplot = ax_box.boxplot(
        ape,
        vert=False,
        autorange=False,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="tomato"),
        medianprops=dict(color="black"),
    )
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
    ax_box.set(yticks=[])
    ax_box.spines["left"].set_visible(False)
    ax_box.spines["right"].set_visible(False)
    ax_box.spines["top"].set_visible(False)

    fig.suptitle("APE of the Posterior", fontsize=20)
    plt.xlim(0, 100)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.clf()


##########################################################################################################
# --- Gaussian Posterior plot ---


def gaussian_posterior_plot(
    alpha_IMF,
    log10_N_Ia,
    global_params,
    title,
    prior=np.array([[-2.3, -2.89], [0.3, 0.3]]),
):

    data = np.stack([alpha_IMF, log10_N_Ia], 2)
    mu, cov = mean_std(data, prior[0], prior[1])
    mu_alpha, mu_log10N_Ia = mu[0], mu[1]
    sigma_alpha, sigma_log10N_Ia = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])

    xlim = [-0.03, 0.03]
    ylim = [-0.03, 0.03]

    if np.abs(global_params[0, 0] - mu_alpha) > 0.03:
        if mu_alpha - global_params[0, 0] < 0:
            xlim[0] = mu_alpha - global_params[0, 0] - 10 * sigma_alpha
        elif mu_alpha - global_params[0, 0] > 0:
            xlim[1] = mu_alpha - global_params[0, 0] + 10 * sigma_alpha

    if np.abs(global_params[0, 1] - mu_log10N_Ia) > 0.03:
        if mu_log10N_Ia - global_params[0, 1] < 0:
            ylim[0] = mu_log10N_Ia - global_params[0, 1] - 10 * sigma_log10N_Ia
        elif mu_log10N_Ia - global_params[0, 1] > 0:
            ylim[1] = mu_log10N_Ia - global_params[0, 1] + 10 * sigma_log10N_Ia

    grid_x = [global_params[0, 0] + xlim[0], global_params[0, 0] + xlim[1]]
    grid_y = [global_params[0, 1] + ylim[0], global_params[0, 1] + ylim[1]]

    x, y = np.mgrid[grid_x[0] : grid_x[1] : 0.001, grid_y[0] : grid_y[1] : 0.001]
    pos = np.dstack((x, y))

    # create a multivariate normal
    posterior = multivariate_normal(mean=mu, cov=cov)

    # create a figure
    plt.figure(figsize=(15, 15))

    # labels
    label_gt = (
        r"Ground Truth"
        + f"\n"
        + r"$\alpha_{\rm IMF} = $"
        + f"${global_params[0,0]:.2f}$"
        + f"\n"
        + r"$\log_{10} N_{\rm Ia} = $"
        + f"${global_params[0,1]:.2f}$"
    )
    label_fit = (
        r"Fit"
        + f"\n"
        + r"$\alpha_{\rm IMF} = $"
        + f"${mu_alpha:.3f} \\pm {sigma_alpha:.3f}$"
        + f"\n"
        + r"$\log_{10} N_{\rm Ia} = $"
        + f"${mu_log10N_Ia:.3f} \\pm {sigma_log10N_Ia:.3f}$"
    )

    # Sigma levels
    levels = []
    sigma = np.array([3, 2, 1, 0], dtype=float)
    for n in sigma:
        levels.append(
            posterior.pdf(
                [mu_alpha + n * sigma_alpha, mu_log10N_Ia + n * sigma_log10N_Ia]
            )
        )
    cmap = plt.get_cmap("Blues")
    colors = [cmap(0.3), cmap(0.5), cmap(0.8)]
    cf = plt.contourf(x, y, posterior.pdf(pos), levels=levels, colors=colors, alpha=0.7)
    CS = plt.contour(
        x, y, posterior.pdf(pos), levels=levels, colors="k", linestyles="dashed"
    )
    text = plt.clabel(CS, inline=True, fontsize=15)
    for t in text:
        i = np.abs(np.array(levels) - float(t._text)).argmin()
        s = int(sigma[i])
        t.set(text=f"{s} $\\sigma$")

    legend_true = plt.scatter(
        global_params[0, 0],
        global_params[0, 1],
        color="k",
        label=label_gt,
        s=400,
        marker="x",
        linewidths=5,
    )
    legend_fit = plt.scatter(
        mu_alpha, mu_log10N_Ia, color="blue", label=label_fit, s=100
    )

    legend_fit = plt.legend(
        handles=[legend_fit],
        fontsize=15,
        shadow=True,
        fancybox=True,
        loc=2,
        bbox_to_anchor=(0, 0.9),
    )
    legend_true = plt.legend(
        handles=[legend_true],
        fontsize=15,
        shadow=True,
        fancybox=True,
        loc=2,
        bbox_to_anchor=(0, 0.99),
    )

    plt.xlabel(r"$\alpha_{\rm IMF}$", fontsize=40)
    plt.ylabel(r"$\log_{10} N_{\rm Ia}$", fontsize=40)
    plt.tick_params(labelsize=30)

    plt.gca().add_artist(legend_fit)
    plt.gca().add_artist(legend_true)

    # plt.title(title, fontsize=60)

    plt.tight_layout()
    plt.savefig(paths.figures / f"{title}.png")
    plt.clf()


##########################################################################################################
# --- Gaussian Posterior plot Comparison ---


def gaussian_posterior_plot_n_stars(
    alpha_IMF,
    log10_N_Ia,
    global_params,
    title,
    philcox,
    no_stars=100,
    prior=np.array([[-2.3, -2.89], [0.3, 0.3]]),
):

    data = np.stack([alpha_IMF, log10_N_Ia], 2)
    mu, cov = mean_std(data, prior[0], prior[1])
    mu_alpha, mu_log10N_Ia = mu[0], mu[1]
    sigma_alpha, sigma_log10N_Ia = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])

    xlim = [-0.03, 0.03]
    ylim = [-0.03, 0.03]

    if np.abs(global_params[0, 0] - mu_alpha) > 0.03:
        if mu_alpha - global_params[0, 0] < 0:
            xlim[0] = mu_alpha - global_params[0, 0] - 10 * sigma_alpha
        elif mu_alpha - global_params[0, 0] > 0:
            xlim[1] = mu_alpha - global_params[0, 0] + 10 * sigma_alpha

    if np.abs(global_params[0, 1] - mu_log10N_Ia) > 0.03:
        if mu_log10N_Ia - global_params[0, 1] < 0:
            ylim[0] = mu_log10N_Ia - global_params[0, 1] - 10 * sigma_log10N_Ia
        elif mu_log10N_Ia - global_params[0, 1] > 0:
            ylim[1] = mu_log10N_Ia - global_params[0, 1] + 10 * sigma_log10N_Ia

    if philcox is not None:

        # add Philcox ellipses
        idx = np.where(philcox["n_stars"] == no_stars)[0][0]

        # Calculate the sigmas for specific index
        sigma_h_1 = philcox["up"][idx, 0] - philcox["med"][idx, 0]
        sigma_l_1 = philcox["med"][idx, 0] - philcox["lo"][idx, 0]
        sigma_philcox_1 = (sigma_h_1 + sigma_l_1) / 2

        sigma_h_2 = philcox["up"][idx, 1] - philcox["med"][idx, 1]
        sigma_l_2 = philcox["med"][idx, 1] - philcox["lo"][idx, 1]
        sigma_philcox_2 = (sigma_h_2 + sigma_l_2) / 2

        if np.abs(global_params[0, 0] - philcox["med"][idx, 0]) > 0.03:
            if philcox["med"][idx, 0] - global_params[0, 0] < 0:
                xlim[0] = (
                    philcox["med"][idx, 0] - global_params[0, 0] - 2 * sigma_philcox_1
                )
            elif philcox["med"][idx, 0] - global_params[0, 0] > 0:
                xlim[1] = (
                    philcox["med"][idx, 0] - global_params[0, 0] + 2 * sigma_philcox_1
                )

        if np.abs(global_params[0, 1] - philcox["med"][idx, 1]) > 0.03:
            if philcox["med"][idx, 1] - global_params[0, 1] < 0:
                ylim[0] = (
                    philcox["med"][idx, 1] - global_params[0, 1] - 2 * sigma_philcox_2
                )
            elif philcox["med"][idx, 1] - global_params[0, 1] > 0:
                ylim[1] = (
                    philcox["med"][idx, 1] - global_params[0, 1] + 2 * sigma_philcox_2
                )

    grid_x = [global_params[0, 0] + xlim[0], global_params[0, 0] + xlim[1]]
    grid_y = [global_params[0, 1] + ylim[0], global_params[0, 1] + ylim[1]]

    x, y = np.mgrid[grid_x[0] : grid_x[1] : 0.001, grid_y[0] : grid_y[1] : 0.001]
    pos = np.dstack((x, y))

    # create a multivariate normal
    posterior = multivariate_normal(mean=mu, cov=cov)

    # create a figure
    plt.figure(figsize=(15, 15))

    # Add Philcox values to plot
    if philcox is not None:

        # Create label with scalar values
        label_hmc = (
            r"HMC"
            + f"\n"
            + r"$\alpha_{\rm IMF} = $"
            + f'${philcox["med"][idx, 0]:.3f} \\pm {sigma_philcox_1:.3f}$'
            + f"\n"
            + r"$\log_{10} N_{\rm Ia} = $"
            + f'${philcox["med"][idx, 1]:.3f} \\pm {sigma_philcox_2:.3f}$'
        )

        legend_philcox = plt.scatter(
            philcox["med"][idx, 0],
            philcox["med"][idx, 1],
            color="red",
            label=label_hmc,
            s=100,
        )

        # create a multivariate normal
        posterior_philcox = multivariate_normal(
            mean=[philcox["med"][:, 0][-1], philcox["med"][:, 1][-1]],
            cov=[[sigma_philcox_1**2, 0], [0, sigma_philcox_2**2]],
        )

        # Sigma levels
        levels = []
        sigma = np.array([3, 2, 1, 0], dtype=float)
        for n in sigma:
            levels.append(
                posterior_philcox.pdf(
                    [
                        philcox["med"][:, 0][-1] + n * sigma_philcox_1,
                        philcox["med"][:, 1][-1] + n * sigma_philcox_2**2,
                    ]
                )
            )

        cmap = plt.get_cmap("Reds")
        colors = [cmap(0.3), cmap(0.5), cmap(0.8)]
        cf = plt.contourf(
            x, y, posterior_philcox.pdf(pos), levels=levels, colors=colors, alpha=0.7
        )
        CS = plt.contour(
            x,
            y,
            posterior_philcox.pdf(pos),
            levels=levels,
            colors="gray",
            linewidths=3,
            linestyles="dotted",
        )
        text = plt.clabel(CS, inline=True, fontsize=15)
        for t in text:
            i = np.abs(np.array(levels) - float(t._text)).argmin()
            s = int(sigma[i])
            t.set(text=f"{s} $\\sigma$", color="gray")
        legend_philcox = plt.legend(
            handles=[legend_philcox],
            fontsize=15,
            shadow=True,
            fancybox=True,
            loc=2,
            bbox_to_anchor=(0, 0.81),
        )
        plt.gca().add_artist(legend_philcox)

    # labels
    label_gt = (
        r"Ground Truth"
        + f"\n"
        + r"$\alpha_{\rm IMF} = $"
        + f"${global_params[0,0]:.2f}$"
        + f"\n"
        + r"$\log_{10} N_{\rm Ia} = $"
        + f"${global_params[0,1]:.2f}$"
    )
    label_fit = (
        r"SBI"
        + f"\n"
        + r"$\alpha_{\rm IMF} = $"
        + f"${mu_alpha:.3f} \\pm {sigma_alpha:.3f}$"
        + f"\n"
        + r"$\log_{10} N_{\rm Ia} = $"
        + f"${mu_log10N_Ia:.3f} \\pm {sigma_log10N_Ia:.3f}$"
    )

    # Sigma levels
    levels = []
    sigma = np.array([3, 2, 1, 0], dtype=float)
    for n in sigma:
        levels.append(
            posterior.pdf(
                [mu_alpha + n * sigma_alpha, mu_log10N_Ia + n * sigma_log10N_Ia]
            )
        )
    cmap = plt.get_cmap("Blues")
    colors = [cmap(0.3), cmap(0.5), cmap(0.8)]
    cf = plt.contourf(x, y, posterior.pdf(pos), levels=levels, colors=colors, alpha=0.7)
    CS = plt.contour(
        x, y, posterior.pdf(pos), levels=levels, colors="gray", linestyles="dashed"
    )
    text = plt.clabel(CS, inline=True, fontsize=15)
    for t in text:
        i = np.abs(np.array(levels) - float(t._text)).argmin()
        s = int(sigma[i])
        t.set(text=f"{s} $\\sigma$", color="gray")

    legend_true = plt.scatter(
        global_params[0, 0],
        global_params[0, 1],
        color="k",
        label=label_gt,
        s=400,
        marker="x",
        linewidths=5,
    )
    legend_fit = plt.scatter(
        mu_alpha, mu_log10N_Ia, color="blue", label=label_fit, s=100
    )

    legend_fit = plt.legend(
        handles=[legend_fit],
        fontsize=15,
        shadow=True,
        fancybox=True,
        loc=2,
        bbox_to_anchor=(0, 0.9),
    )
    legend_true = plt.legend(
        handles=[legend_true],
        fontsize=15,
        shadow=True,
        fancybox=True,
        loc=2,
        bbox_to_anchor=(0, 0.99),
    )

    plt.xlabel(r"$\alpha_{\rm IMF}$", fontsize=40)
    plt.ylabel(r"$\log_{10} N_{\rm Ia}$", fontsize=40)
    plt.tick_params(labelsize=30)

    plt.gca().add_artist(legend_fit)
    plt.gca().add_artist(legend_true)

    # plt.title(title, fontsize=60)

    plt.tight_layout()
    plt.savefig(paths.figures / f"{title}.png")
    plt.clf()


##########################################################################################################
# --- N-Star comparison plot 2 ---


def n_stars_plot_comp2(
    x1,
    x2,
    x_true,
    philcox,
    save_name,
    no_stars=np.array([1, 10, 100, 500, 1000]),
    prior=np.array([[-2.3, -2.89], [0.3, 0.3]]),
):
    fit = []
    err = []

    # --- Fit a 2D Gaussian to the data ---
    for n in no_stars:
        data = np.stack([x1[:n], x2[:n]], 2)
        mu, cov = mean_std(data, prior[0], prior[1])
        mu_alpha, mu_log10N_Ia = mu[0], mu[1]
        sigma_alpha, sigma_log10N_Ia = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])

        fit.append([mu_alpha, mu_log10N_Ia])
        err.append([sigma_alpha, sigma_log10N_Ia])

    fit = np.array(fit)
    err = np.array(err)

    # --- Plot the data ---
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(26, 6))

    def plot(fit, err, x_true, ax, name):
        two_sigma_h = philcox["med"][:, i] + 2 * (
            philcox["up"][:, i] - philcox["med"][:, i]
        )
        two_sigma_l = philcox["med"][:, i] - 2 * (
            philcox["med"][:, i] - philcox["lo"][:, i]
        )
        # Add Philcox
        ax.plot(philcox["n_stars"], philcox["med"][:, i], c="r", label="HMC")
        ax.fill_between(
            philcox["n_stars"],
            philcox["lo"][:, i],
            philcox["up"][:, i],
            alpha=0.1,
            color="r",
        )
        ax.fill_between(
            philcox["n_stars"], two_sigma_l, two_sigma_h, alpha=0.1, color="r"
        )

        ax.plot(no_stars, fit, color="b", label="Fit")
        ax.fill_between(
            no_stars,
            fit - err,
            fit + err,
            alpha=0.1,
            color="b",
            label=r"1 & 2 $\sigma$",
        )
        ax.fill_between(no_stars, fit - 2 * err, fit + 2 * err, alpha=0.1, color="b")

        ax.axhline(x_true, color="k", linestyle=":", linewidth=2, label="Ground Truth")

        ax.set_xlabel(r"$N_{\rm stars}$", fontsize=40)
        ax.set_ylabel(name, fontsize=40)
        ax.set_ylim([x_true - 0.2 * abs(x_true), x_true + 0.2 * abs(x_true)])
        ax.set_xscale("log")
        ax.set_xlim([1, no_stars[-1]])
        ax.tick_params(labelsize=30, size=10, width=3)
        ax.tick_params(which="minor", size=5, width=2)

    for i, name in enumerate([r"$\alpha_{\rm IMF}$", r"$\log_{10} N_{\rm Ia}$"]):
        plot(fit[:, i], err[:, i], x_true[0, i], ax[i], name)

    ax[0].legend(fontsize=20, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(paths.figures / f"{save_name}.pdf")
    plt.clf()
