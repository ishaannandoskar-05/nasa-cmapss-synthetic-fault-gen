"""
CGAN training utilities.

BUG-008: cross-sensor correlation penalty for generator loss
BUG-013: early stopping for CGAN training loops

Usage in notebooks (05b, 11):
    from src.utils.cgan_training import correlation_penalty, EarlyStopping
"""

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# BUG-008: Cross-sensor correlation penalty
# ---------------------------------------------------------------------------

def correlation_penalty(x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:
    """
    Compute a cross-sensor correlation penalty for the CGAN generator.

    Penalises the difference between the pairwise feature correlation
    matrices of real and generated windows.  Adding this term to the
    generator loss encourages the synthetic data to reproduce joint
    sensor relationships, not just marginal distributions — directly
    addressing the KS pass-rate gap documented in BUG-008.

    Args:
        x_real: real batch, shape (B, T, D)
        x_fake: generated batch, shape (B, T, D)

    Returns:
        Scalar tensor (mean squared difference of correlation matrices).
    """
    B = x_real.size(0)

    # Flatten time and feature dims: (B, T*D)
    real_flat = x_real.reshape(B, -1)
    fake_flat = x_fake.reshape(B, -1)

    # Mean-centre across the batch
    real_c = real_flat - real_flat.mean(dim=0, keepdim=True)
    fake_c = fake_flat - fake_flat.mean(dim=0, keepdim=True)

    # Unscaled covariance matrix (T*D x T*D)
    corr_real = (real_c.T @ real_c) / max(B - 1, 1)
    corr_fake = (fake_c.T @ fake_c) / max(B - 1, 1)

    # Detach real-side so gradient only flows through fake
    return torch.mean((corr_real.detach() - corr_fake) ** 2)


# ---------------------------------------------------------------------------
# BUG-013: Discriminative-score-based early stopping for CGAN
# ---------------------------------------------------------------------------

def compute_discriminative_score(
    G,
    X_real: np.ndarray,
    y_real: np.ndarray,
    latent_dim: int,
    device,
    n_samples: int = 500,
    seed: int = 42,
) -> float:
    """
    Estimate how distinguishable synthetic samples are from real ones.

    Draws n_samples real and n_samples synthetic windows, trains a
    RandomForest classifier in 5-fold CV to tell them apart, and
    returns the mean accuracy.

    Ideal (perfect generator): ~0.50
    Bad (trivially detectable): close to 1.0

    Args:
        G           : CGAN generator (nn.Module), called in eval mode.
        X_real      : real training array, shape (N, T, D).
        y_real      : class labels, shape (N,).
        latent_dim  : size of the noise vector fed to G.
        device      : torch device.
        n_samples   : number of real/fake samples to evaluate.
        seed        : random seed for reproducibility.

    Returns:
        Mean 5-fold CV accuracy (float in [0, 1]).
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X_real), size=n_samples, replace=False)
    X_real_samp = X_real[indices].reshape(n_samples, -1)
    y_real_cls = y_real[indices]

    G.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        c = torch.tensor(y_real_cls, dtype=torch.long).to(device)
        X_fake = G(z, c).cpu().numpy()
    G.train()

    X_fake_flat = X_fake.reshape(n_samples, -1)

    X_all = np.vstack([X_real_samp, X_fake_flat])
    y_disc = np.array([0] * n_samples + [1] * n_samples)

    sc = StandardScaler()
    X_all = sc.fit_transform(X_all)

    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    scores = cross_val_score(clf, X_all, y_disc, cv=5)
    return float(scores.mean())


class EarlyStopping:
    """
    BUG-013: Patience-based early stopping for CGAN training.

    Evaluates the discriminative score every `eval_interval` epochs and
    stops training when the distance from the ideal score (0.5) has not
    improved for `patience` consecutive evaluations.

    Best generator/discriminator checkpoints are saved automatically so
    the caller can restore the best weights after training ends.

    Usage:
        stopper = EarlyStopping(
            G=G, D=D, X_real=X, y_real=labels,
            latent_dim=LATENT_DIM, device=DEVICE,
            ckpt_dir=MODELS_DIR,
            eval_interval=50, patience=3,
        )

        for epoch in range(MAX_EPOCHS):
            # ... training step ...
            if stopper.step(epoch):
                print("Early stopping triggered.")
                break

        # Restore best weights
        stopper.restore_best()
    """

    def __init__(
        self,
        G,
        D,
        X_real: np.ndarray,
        y_real: np.ndarray,
        latent_dim: int,
        device,
        ckpt_dir,
        eval_interval: int = 50,
        patience: int = 3,
        seed: int = 42,
    ):
        self.G = G
        self.D = D
        self.X_real = X_real
        self.y_real = y_real
        self.latent_dim = latent_dim
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.eval_interval = eval_interval
        self.patience = patience
        self.seed = seed

        self._best_dist = float("inf")
        self._no_improve = 0

    def step(self, epoch: int) -> bool:
        """
        Call at the end of each epoch.

        Returns True when training should stop.
        """
        if (epoch + 1) % self.eval_interval != 0:
            return False

        score = compute_discriminative_score(
            self.G, self.X_real, self.y_real,
            self.latent_dim, self.device, seed=self.seed,
        )
        dist = abs(score - 0.5)
        print(
            f"  [EarlyStopping] epoch {epoch + 1}: "
            f"disc_score={score:.4f}  dist={dist:.4f}"
        )

        if dist < self._best_dist:
            self._best_dist = dist
            self._no_improve = 0
            torch.save(
                self.G.state_dict(),
                self.ckpt_dir / "generator_best.pt",
            )
            torch.save(
                self.D.state_dict(),
                self.ckpt_dir / "discriminator_best.pt",
            )
        else:
            self._no_improve += 1
            if self._no_improve >= self.patience:
                return True  # stop

        return False

    def restore_best(self):
        """Load the best-checkpoint weights back into G and D."""
        import torch
        best_g = self.ckpt_dir / "generator_best.pt"
        best_d = self.ckpt_dir / "discriminator_best.pt"
        if best_g.exists():
            self.G.load_state_dict(torch.load(best_g, map_location=self.device))
        if best_d.exists():
            self.D.load_state_dict(torch.load(best_d, map_location=self.device))
        print(f"Restored best weights from {self.ckpt_dir}")
