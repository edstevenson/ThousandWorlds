from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PPCAFit:
    mu: torch.Tensor  # (F, A)
    loadings: torch.Tensor  # (F, A, q)
    sigma2: torch.Tensor  # ()
    Z: torch.Tensor  # (n, q)


def fit_ppca(
    Y: torch.Tensor,  # (n, A, F) normalized spectral coeffs
    *,
    field_mask: torch.Tensor,  # (n, F) bool
    sh_mask: torch.Tensor,  # (A, F) bool
    latent_dim: int,
    num_iters: int = 50,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> PPCAFit:
    torch.manual_seed(seed)
    ref = Y
    Y = Y.to(dtype)
    field_mask_f = field_mask.to(dtype=dtype)
    sh_idx = [torch.where(sh_mask[:, f])[0] for f in range(int(Y.shape[2]))]

    n, A, F = map(int, Y.shape)
    q = int(latent_dim)
    Iq = torch.eye(q, device=ref.device, dtype=dtype)

    mu = ref.new_zeros((F, A), dtype=dtype)
    loadings = (ref.new_ones((F, A, q), dtype=dtype) * 0.01) * torch.randn((F, A, q), device=ref.device, dtype=dtype)
    sigma2 = ref.new_tensor(1.0, dtype=dtype)

    log_every = max(int(num_iters) // 10, 1)
    for it in range(int(num_iters)):
        sigma2_old = sigma2
        # E-step: latent posterior moments with whole-field missingness.
        WtW_f = torch.stack([(loadings[f, sh_idx[f]].T @ loadings[f, sh_idx[f]]) for f in range(F)], dim=0)
        S = torch.einsum("nf,fij->nij", field_mask_f, WtW_f)

        b = ref.new_zeros((n, q), dtype=dtype)
        for f in range(F):
            idx = sh_idx[f]
            if idx.numel() == 0:
                continue
            resid = (Y[:, idx, f] - mu[f, idx][None, :]) * field_mask_f[:, f : f + 1]
            b = b + resid @ loadings[f, idx]

        M = S + sigma2 * Iq[None]
        L = torch.linalg.cholesky(M)
        Z = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
        Cov = sigma2 * torch.cholesky_solve(Iq[None].expand(n, -1, -1), L)
        Ezz = Cov + Z[:, :, None] * Z[:, None, :]

        # M-step: fieldwise expected regression for mean/loadings, then sigma2.
        total_sse = sigma2.new_zeros(())
        total_count = 0
        for f in range(F):
            idx = sh_idx[f]
            obs = field_mask[:, f]
            if idx.numel() == 0 or not bool(obs.any()):
                continue

            Yf = Y[obs][:, idx, f]
            Zf = Z[obs]
            Ezzf = Ezz[obs]
            n_f = int(Yf.shape[0])

            Uf = torch.cat([Zf.new_ones((n_f, 1)), Zf], dim=1)
            sumz = Zf.sum(dim=0)
            S_uu = Uf.new_zeros((q + 1, q + 1))
            S_uu[0, 0] = n_f
            S_uu[0, 1:] = sumz
            S_uu[1:, 0] = sumz
            S_uu[1:, 1:] = Ezzf.sum(dim=0)
            S_uu = S_uu + 1.0e-6 * torch.eye(q + 1, device=Uf.device, dtype=Uf.dtype)

            B = Yf.T @ Uf
            Aparams = torch.linalg.solve(S_uu.T, B.T).T
            mu[f, idx] = Aparams[:, 0]
            loadings[f, idx] = Aparams[:, 1:]

            pred = Uf @ Aparams.T
            resid = Yf - pred
            total_sse = total_sse + (resid**2).sum()
            total_sse = total_sse + (Cov[obs] * (loadings[f, idx].T @ loadings[f, idx])[None]).sum()
            total_count += n_f * int(idx.numel())

        sigma2 = (total_sse / max(total_count, 1)).clamp_min(1.0e-8)
        step = it + 1
        if verbose and (step == 1 or step == int(num_iters) or step % log_every == 0):
            delta = float((sigma2 - sigma2_old).abs().item())
            print(f"[ppca] iter {step:>4}/{int(num_iters)} | sigma2={float(sigma2.item()):.3e} | delta={delta:.3e}")

    mu = mu * sh_mask.T.to(dtype=dtype)
    loadings = loadings * sh_mask.T.to(dtype=dtype)[:, :, None]
    return PPCAFit(mu=mu, loadings=loadings, sigma2=sigma2, Z=Z)
