import json
import os

# Backend settings (keep as in your original)
os.environ["MPLBACKEND"] = "TkAgg"

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def load_losses(loss_log_path):
    with open(loss_log_path, "r", encoding="utf-8") as f:
        losses = json.load(f)
    return losses


def get_loss_data(losses, split, loss_type):
    return losses[split][loss_type]


def plot_discriminator_losses(
    ax, disc_loss, best_epoch, label_prefix="", show_legend=False
):
    ax.plot(
        disc_loss["first_loss"],
        label=f"{label_prefix}First Domain",
        alpha=0.7,
        linewidth=2,
    )
    ax.plot(
        disc_loss["second_loss"],
        label=f"{label_prefix}Second Domain",
        alpha=0.7,
        linewidth=2,
    )
    ax.plot(
        disc_loss["latent_loss"],
        label=f"{label_prefix}Latent Space",
        alpha=0.7,
        linewidth=2,
    )
    ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=2, alpha=0.7)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Discriminator Losses", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)


def plot_enc_gen_losses(
    ax, enc_gen_loss, best_epoch, label_prefix="", show_legend=False
):
    ax.plot(
        enc_gen_loss["first_loss"],
        label=f"{label_prefix}First Domain",
        alpha=0.7,
        linewidth=2,
    )
    ax.plot(
        enc_gen_loss["second_loss"],
        label=f"{label_prefix}Second Domain",
        alpha=0.7,
        linewidth=2,
    )
    ax.plot(
        enc_gen_loss["latent_loss"],
        label=f"{label_prefix}Latent Space",
        alpha=0.7,
        linewidth=2,
    )
    ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=2, alpha=0.7)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Encoder-Generator Losses", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)


def plot_cycle_consistency_losses(
    ax, cc_loss, best_epoch, label_prefix="", show_legend=False
):
    ax.plot(
        cc_loss["first_cycle_loss"],
        label=f"{label_prefix}First Cycle",
        alpha=0.7,
        linewidth=2,
    )
    ax.plot(
        cc_loss["second_cycle_loss"],
        label=f"{label_prefix}Second Cycle",
        alpha=0.7,
        linewidth=2,
    )
    ax.plot(
        cc_loss["first_full_cycle_loss"],
        label=f"{label_prefix}First Full Cycle",
        alpha=0.7,
        linewidth=2,
    )
    ax.plot(
        cc_loss["second_full_cycle_loss"],
        label=f"{label_prefix}Second Full Cycle",
        alpha=0.7,
        linewidth=2,
    )
    ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=2, alpha=0.7)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Cycle Consistency Losses", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)


def plot_component_losses(losses, split="train", figsize=(12, 16), save_path=None):
    if split not in losses:
        print(f"No data found for split: {split}")
        return

    fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)
    best_epoch = losses["trainer_info"]["best_epoch"]

    disc_loss = get_loss_data(losses, split, "disc_loss")
    plot_discriminator_losses(axes[0], disc_loss, best_epoch, show_legend=False)

    enc_gen_loss = get_loss_data(losses, split, "enc_gen_loss")
    plot_enc_gen_losses(axes[1], enc_gen_loss, best_epoch, show_legend=False)

    cc_loss = get_loss_data(losses, split, "cc_loss")
    plot_cycle_consistency_losses(axes[2], cc_loss, best_epoch, show_legend=False)

    # Legends outside on the right per axes
    for ax in axes:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=True,
            framealpha=1.0,
        )

    # Make space for right-side legends
    fig.set_size_inches(figsize[0] + 2.5, figsize[1])
    fig.suptitle(
        f"{split.capitalize()} Component Losses", fontsize=14, fontweight="bold"
    )

    if save_path:
        fig.savefig(f"{save_path}/component_losses.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_total_losses(losses, figsize=(10, 6), save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    # Train
    train_loss = losses["trainer_info"]["train_loss"]
    ax.plot(train_loss, label="Train Total Loss", linewidth=2, alpha=0.8)

    # Val if present
    val_loss = losses.get("trainer_info", {}).get("val_loss")
    if val_loss:
        ax.plot(val_loss, label="Validation Total Loss", linewidth=2, alpha=0.8)

    best_epoch = losses["trainer_info"]["best_epoch"]
    print(f"best_epoch: {best_epoch}")
    ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=2, alpha=0.7)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(
        "Total Training and Validation Loss", fontsize=14, fontweight="bold", pad=12
    )
    ax.set_ylim(bottom=0)

    # Legend outside right
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, framealpha=1.0
    )
    fig.set_size_inches(figsize[0] + 2.0, figsize[1])

    if save_path:
        fig.savefig(f"{save_path}/total_losses.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def compare_total_losses(
    losses1,
    losses2,
    label1="Model 1",
    label2="Model 2",
    figsize=(14, 12),
    save_path=None,
):
    has_val_data = bool(losses1.get("trainer_info", {}).get("val_loss")) and bool(
        losses2.get("trainer_info", {}).get("val_loss")
    )

    if has_val_data:
        fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
    else:
        fig, axes = plt.subplots(
            1, 1, figsize=(figsize[0], max(6, figsize[1] // 2)), constrained_layout=True
        )
        axes = [axes]

    # Train
    ax = axes[0]
    ax.plot(
        losses1["trainer_info"]["train_loss"], label=f"{label1}", linewidth=2, alpha=0.8
    )
    ax.plot(
        losses2["trainer_info"]["train_loss"], label=f"{label2}", linewidth=2, alpha=0.8
    )
    best_epoch1 = losses1["trainer_info"]["best_epoch"]
    best_epoch2 = losses2["trainer_info"]["best_epoch"]

    ax.axvline(
        best_epoch1,
        color="grey",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"{label1} Best Epoch",
    )
    ax.axvline(
        best_epoch2,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"{label2} Best Epoch",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Comparison", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(bottom=0)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    # Val (only if both present)
    if has_val_data:
        ax = axes[1]
        ax.plot(
            losses1["trainer_info"]["val_loss"],
            label=f"{label1}",
            linewidth=2,
            alpha=0.8,
        )
        ax.plot(
            losses2["trainer_info"]["val_loss"],
            label=f"{label2}",
            linewidth=2,
            alpha=0.8,
        )
        best_epoch1 = losses1["trainer_info"]["best_epoch"]
        best_epoch2 = losses2["trainer_info"]["best_epoch"]
        if best_epoch1 is not None:
            ax.axvline(
                best_epoch1,
                color="grey",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f"{label1} Best Epoch",
            )
        if best_epoch2 is not None:
            ax.axvline(
                best_epoch2,
                color="black",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f"{label2} Best Epoch",
            )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(
            "Validation Loss Comparison", fontsize=13, fontweight="bold", pad=10
        )
        ax.set_ylim(bottom=0)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    # Widen to accommodate outside legends
    fig.set_size_inches(fig.get_size_inches()[0] + 2.5, fig.get_size_inches()[1])
    fig.suptitle(
        f"Total Loss Comparison: {label1} vs {label2}", fontsize=14, fontweight="bold"
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def compare_component_losses(
    losses1,
    losses2,
    label1="Model 1",
    label2="Model 2",
    split="train",
    figsize=(10, 20),
    save_path=None,
):
    """
    Compare component losses (discriminator, encoder-generator, cycle consistency)
    between two models.
    """
    if split not in losses1 or split not in losses2:
        print(f"No data found for split: {split} in one or both models")
        return

    fig, axes = plt.subplots(4, 3, figsize=figsize, constrained_layout=True)

    disc_loss1 = get_loss_data(losses1, split, "disc_loss")
    disc_loss2 = get_loss_data(losses2, split, "disc_loss")
    enc_gen_loss1 = get_loss_data(losses1, split, "enc_gen_loss")
    enc_gen_loss2 = get_loss_data(losses2, split, "enc_gen_loss")
    cc_loss1 = get_loss_data(losses1, split, "cc_loss")
    cc_loss2 = get_loss_data(losses2, split, "cc_loss")
    best_epoch1 = losses1["trainer_info"]["best_epoch"]
    best_epoch2 = losses2["trainer_info"]["best_epoch"]

    # Row 1
    ax = axes[0, 0]
    if "first_loss" in disc_loss1:
        ax.plot(disc_loss1["first_loss"], label=label1, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch1, color="grey", linestyle="--", linewidth=2, alpha=0.7, label=f"{label1} Best Epoch")
    if "first_loss" in disc_loss2:
        ax.plot(disc_loss2["first_loss"], label=label2, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch2, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"{label2} Best Epoch")
    
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("First Domain Disc Loss", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if "second_loss" in disc_loss1:
        ax.plot(disc_loss1["second_loss"], label=label1, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch1, color="grey", linestyle="--", linewidth=2, alpha=0.7, label=f"{label1} Best Epoch")
    if "second_loss" in disc_loss2:
        ax.plot(disc_loss2["second_loss"], label=label2, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch2, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"{label2} Best Epoch")
        
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("Second Domain Disc Loss", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    if "latent_loss" in disc_loss1:
        ax.plot(disc_loss1["latent_loss"], label=label1, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch1, color="grey", linestyle="--", linewidth=2, alpha=0.7, label=f"{label1} Best Epoch")
    if "latent_loss" in disc_loss2:
        ax.plot(disc_loss2["latent_loss"], label=label2, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch2, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"{label2} Best Epoch")

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("Latent Space Disc Loss", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Row 2
    ax = axes[1, 0]
    if "first_loss" in enc_gen_loss1:
        ax.plot(enc_gen_loss1["first_loss"], label=label1, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch1, color="grey", linestyle="--", linewidth=2, alpha=0.7, label=f"{label1} Best Epoch")
    if "first_loss" in enc_gen_loss2:
        ax.plot(enc_gen_loss2["first_loss"], label=label2, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch2, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"{label2} Best Epoch")
    
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("First Domain Enc-Gen Loss", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "second_loss" in enc_gen_loss1:
        ax.plot(enc_gen_loss1["second_loss"], label=label1, alpha=0.7, linewidth=2)
                ax.axvline(best_epoch1, color="grey", linestyle="--", linewidth=2, alpha=0.7, label=f"{label1} Best Epoch")
    if "second_loss" in enc_gen_loss2:
        ax.plot(enc_gen_loss2["second_loss"], label=label2, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch2, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"{label2} Best Epoch")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("Second Domain Enc-Gen Loss", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    if "latent_loss" in enc_gen_loss1:
        ax.plot(enc_gen_loss1["latent_loss"], label=label1, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch1, color="grey", linestyle="--", linewidth=2, alpha=0.7, label=f"{label1} Best Epoch")
    if "latent_loss" in enc_gen_loss2:
        ax.plot(enc_gen_loss2["latent_loss"], label=label2, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch2, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"{label2} Best Epoch")

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("Latent Space Enc-Gen Loss", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Row 3
    ax = axes[2, 0]
    if "first_cycle_loss" in cc_loss1:
        ax.plot(cc_loss1["first_cycle_loss"], label=label1, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch1, color="grey", linestyle="--", linewidth=2, alpha=0.7, label=f"{label1} Best Epoch")
    if "first_cycle_loss" in cc_loss2:
        ax.plot(cc_loss2["first_cycle_loss"], label=label2, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch2, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"{label2} Best Epoch")
        
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("First Cycle Loss", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    if "second_cycle_loss" in cc_loss1:
        ax.plot(cc_loss1["second_cycle_loss"], label=label1, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch1, color="grey", linestyle="--", linewidth=2, alpha=0.7, label=f"{label1} Best Epoch")
    if "second_cycle_loss" in cc_loss2:
        ax.plot(cc_loss2["second_cycle_loss"], label=label2, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch2, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"{label2} Best Epoch")

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("Second Cycle Loss", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    if "first_cycle_loss" in cc_loss1:
        ax.plot(
            cc_loss1["first_cycle_loss"],
            label=f"{label1} - First",
            alpha=0.7,
            linewidth=2,
            linestyle="--",
        )
    if "second_cycle_loss" in cc_loss1:
        ax.plot(
            cc_loss1["second_cycle_loss"],
            label=f"{label1} - Second",
            alpha=0.7,
            linewidth=2,
        )
    if "first_cycle_loss" in cc_loss2:
        ax.plot(
            cc_loss2["first_cycle_loss"],
            label=f"{label2} - First",
            alpha=0.7,
            linewidth=2,
            linestyle="--",
        )
    if "second_cycle_loss" in cc_loss2:
        ax.plot(
            cc_loss2["second_cycle_loss"],
            label=f"{label2} - Second",
            alpha=0.7,
            linewidth=2,
        )

    ax.axvline(best_epoch1, color="grey", linestyle="--", linewidth=2, alpha=0.7, label=f"{label1} Best Epoch")
    ax.axvline(best_epoch2, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"{label2} Best Epoch")

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("Cycle Losses Combined", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Row 4
    ax = axes[3, 0]
    if "first_full_cycle_loss" in cc_loss1:
        ax.plot(cc_loss1["first_full_cycle_loss"], label=label1, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch1, color="grey", linestyle="--", linewidth=2, alpha=0.7, label=f"{label1} Best Epoch")
    if "first_full_cycle_loss" in cc_loss2:
        ax.plot(cc_loss2["first_full_cycle_loss"], label=label2, alpha=0.7, linewidth=2)
        ax.axvline(best_epoch2, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"{label2} Best Epoch")

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("First Full Cycle Loss", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    if "second_full_cycle_loss" in cc_loss1:
        ax.plot(
            cc_loss1["second_full_cycle_loss"], label=label1, alpha=0.7, linewidth=2
        )
        ax.axvline(best_epoch1, color="grey", linestyle="--", linewidth=2, alpha=0.7, label=f"{label1} Best Epoch")
    if "second_full_cycle_loss" in cc_loss2:
        ax.plot(
            cc_loss2["second_full_cycle_loss"], label=label2, alpha=0.7, linewidth=2
        )
        ax.axvline(best_epoch2, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"{label2} Best Epoch")

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("Second Full Cycle Loss", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[3, 2]
    if "first_full_cycle_loss" in cc_loss1:
        ax.plot(
            cc_loss1["first_full_cycle_loss"],
            label=f"{label1} - First Full",
            alpha=0.7,
            linewidth=2,
            linestyle="--",
        )
    if "second_full_cycle_loss" in cc_loss1:
        ax.plot(
            cc_loss1["second_full_cycle_loss"],
            label=f"{label1} - Second Full",
            alpha=0.7,
            linewidth=2,
        )
    if "first_full_cycle_loss" in cc_loss2:
        ax.plot(
            cc_loss2["first_full_cycle_loss"],
            label=f"{label2} - First Full",
            alpha=0.7,
            linewidth=2,
            linestyle="--",
        )
        ax.axvline(best_epoch1, color="grey", linestyle="--", linewidth=2, alpha=0.7, label=f"{label1} Best Epoch")
    if "second_full_cycle_loss" in cc_loss2:
        ax.plot(
            cc_loss2["second_full_cycle_loss"],
            label=f"{label2} - Second Full",
            alpha=0.7,
            linewidth=2,
        )
        ax.axvline(best_epoch2, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"{label2} Best Epoch")

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("Full Cycle Losses Combined", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Put legends OUTSIDE for each axes
    for ax in axes.flat:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            framealpha=1.0,
        )

    # Adjust to accommodate legends, but keep image tall
    fig.set_size_inches(figsize[0], figsize[1])
    fig.suptitle(
        f"{split.capitalize()} Component Losses: {label1} vs {label2}",
        fontsize=15,
        fontweight="bold",
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def compare_component_losses_split(
    losses1,
    losses2,
    label1="Model 1",
    label2="Model 2",
    split="train",
    save_path=None,
):
    """
    Compare component losses between two models, with one image per loss type.
    Each image has one row: domains/components as columns.
    """
    if split not in losses1 or split not in losses2:
        print(f"No data found for split: {split} in one or both models")
        return

    disc_loss1 = get_loss_data(losses1, split, "disc_loss")
    disc_loss2 = get_loss_data(losses2, split, "disc_loss")
    enc_gen_loss1 = get_loss_data(losses1, split, "enc_gen_loss")
    enc_gen_loss2 = get_loss_data(losses2, split, "enc_gen_loss")
    cc_loss1 = get_loss_data(losses1, split, "cc_loss")
    cc_loss2 = get_loss_data(losses2, split, "cc_loss")

    fig, axes = plt.subplots(3, 1, figsize=(8, 18), constrained_layout=True)

    ax = axes[0]
    ax.plot(disc_loss1["first_loss"], label=label1, alpha=0.7, linewidth=2)
    ax.plot(disc_loss2["first_loss"], label=label2, alpha=0.7, linewidth=2)
    ax.set_title("First Domain Disc Loss", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(disc_loss1["second_loss"], label=label1, alpha=0.7, linewidth=2)
    ax.plot(disc_loss2["second_loss"], label=label2, alpha=0.7, linewidth=2)
    ax.set_title("Second Domain Disc Loss", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(disc_loss1["latent_loss"], label=label1, alpha=0.7, linewidth=2)
    ax.plot(disc_loss2["latent_loss"], label=label2, alpha=0.7, linewidth=2)
    ax.set_title("Latent Space Disc Loss", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, framealpha=1.0
        )
    fig.set_size_inches(8, 18)
    fig.suptitle(
        f"{split.capitalize()} Discriminator Losses Comparison",
        fontsize=15,
        fontweight="bold",
    )

    if save_path:
        fig.savefig(
            f"{save_path}/comparison_disc_losses.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()

    fig, axes = plt.subplots(3, 1, figsize=(8, 18), constrained_layout=True)

    ax = axes[0]
    ax.plot(enc_gen_loss1["first_loss"], label=label1, alpha=0.7, linewidth=2)
    ax.plot(enc_gen_loss2["first_loss"], label=label2, alpha=0.7, linewidth=2)
    ax.set_title("First Domain Enc-Gen Loss", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(enc_gen_loss1["second_loss"], label=label1, alpha=0.7, linewidth=2)
    ax.plot(enc_gen_loss2["second_loss"], label=label2, alpha=0.7, linewidth=2)
    ax.set_title("Second Domain Enc-Gen Loss", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(enc_gen_loss1["latent_loss"], label=label1, alpha=0.7, linewidth=2)
    ax.plot(enc_gen_loss2["latent_loss"], label=label2, alpha=0.7, linewidth=2)
    ax.set_title("Latent Space Enc-Gen Loss", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, framealpha=1.0
        )
    fig.set_size_inches(8, 18)
    fig.suptitle(
        f"{split.capitalize()} Encoder-Generator Losses Comparison",
        fontsize=15,
        fontweight="bold",
    )

    if save_path:
        fig.savefig(
            f"{save_path}/comparison_encgen_losses.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()

    fig, axes = plt.subplots(4, 1, figsize=(8, 20), constrained_layout=True)

    ax = axes[0]
    ax.plot(cc_loss1["first_cycle_loss"], label=label1, alpha=0.7, linewidth=2)
    ax.plot(cc_loss2["first_cycle_loss"], label=label2, alpha=0.7, linewidth=2)
    ax.set_title("First Cycle Loss", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(cc_loss1["second_cycle_loss"], label=label1, alpha=0.7, linewidth=2)
    ax.plot(cc_loss2["second_cycle_loss"], label=label2, alpha=0.7, linewidth=2)
    ax.set_title("Second Cycle Loss", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(cc_loss1["first_full_cycle_loss"], label=label1, alpha=0.7, linewidth=2)
    ax.plot(cc_loss2["first_full_cycle_loss"], label=label2, alpha=0.7, linewidth=2)
    ax.set_title("First Full Cycle Loss", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(cc_loss1["second_full_cycle_loss"], label=label1, alpha=0.7, linewidth=2)
    ax.plot(cc_loss2["second_full_cycle_loss"], label=label2, alpha=0.7, linewidth=2)
    ax.set_title("Second Full Cycle Loss", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Legends outside for each axes
    for ax in axes:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            framealpha=1.0,
        )

    fig.set_size_inches(8, 20)
    fig.suptitle(
        f"{split.capitalize()} Cycle Consistency Losses Comparison",
        fontsize=15,
        fontweight="bold",
    )

    if save_path:
        fig.savefig(
            f"{save_path}/comparison_cycle_losses.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()


def check_if_folder_exists(folder_name):
    if not os.path.exists(folder_name):
        print(f"Error: Folder not found: {folder_name}")
        exit(1)


def single_loss_analysis(cmd_args):
    losses = load_losses(f"{cmd_args.folder1}/loss_logs.json")

    if cmd_args.output_folder:
        os.makedirs(f"{cmd_args.folder1}/{cmd_args.output_folder}", exist_ok=True)

    save_path = ""
    if cmd_args.output_folder:
        save_path = f"{cmd_args.folder1}/{cmd_args.output_folder}"

    if cmd_args.plot_type == "component":
        plot_component_losses(losses, split=cmd_args.split, save_path=f"{save_path}")
    elif cmd_args.plot_type == "total":
        plot_total_losses(losses, save_path=f"{save_path}")


def comparison_loss_analysis(cmd_args):
    losses1 = load_losses(f"{cmd_args.folder1}/loss_logs.json")
    losses2 = load_losses(f"{cmd_args.folder2}/loss_logs.json")
    save_path1 = None
    save_path2 = None
    if cmd_args.output_folder:
        save_path1 = f"{cmd_args.folder1}/{cmd_args.output_folder}"
        save_path2 = f"{cmd_args.folder2}/{cmd_args.output_folder}"
        os.makedirs(save_path1, exist_ok=True)
        os.makedirs(save_path2, exist_ok=True)

    labels = dict(
        label1=os.path.basename(cmd_args.folder1),
        label2=os.path.basename(cmd_args.folder2),
    )

    if cmd_args.plot_type == "component":
        if save_path1 and save_path2:
            # Save to both folders
            compare_component_losses_split(
                losses1,
                losses2,
                split=cmd_args.split,
                save_path=save_path1,
                **labels,
            )
            compare_component_losses_split(
                losses1,
                losses2,
                split=cmd_args.split,
                save_path=save_path2,
                **labels,
            )
        else:
            # Display if no output path provided
            compare_component_losses_split(
                losses1, losses2, split=cmd_args.split, save_path=None, **labels
            )
    elif cmd_args.plot_type == "total":
        if save_path1 and save_path2:
            compare_total_losses(
                losses1,
                losses2,
                save_path=f"{save_path1}/comparison_total_losses.png",
                **labels,
            )
            compare_total_losses(
                losses1,
                losses2,
                save_path=f"{save_path2}/comparison_total_losses.png",
                **labels,
            )
        else:
            compare_total_losses(losses1, losses2, save_path=None, **labels)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze and plot losses from a specified folder."
    )
    parser.add_argument(
        "--folder1",
        type=str,
        required=True,
        help="Folder containing the loss_logs.json file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Which split to plot",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        default="component",
        choices=["component", "total"],
        help="Type of plot to generate",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison between folder1 and folder2",
    )
    parser.add_argument(
        "--folder2",
        type=str,
        default=None,
        help="Second folder containing the loss_logs.json file for comparison (required with --compare)",
    )
    parser.add_argument(
        "--output_folder", type=str, default=None, help="Folder to save output plots"
    )
    args = parser.parse_args()

    if args.compare:
        if not args.folder2:
            print("Error: For comparison, please provide --folder2 as well.")
            exit(1)
        comparison_loss_analysis(args)
    else:
        single_loss_analysis(args)
