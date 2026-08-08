import torch

torch.set_default_dtype(torch.float64)
cdtype = torch.complex128

I2 = torch.eye(2, dtype=cdtype)
X = torch.tensor([[0, 1], [1, 0]], dtype=cdtype)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=cdtype)
Z = torch.diag(torch.tensor([1, -1], dtype=cdtype))


def rot_Z(theta):
    return torch.diag(torch.stack([torch.exp(1j * theta), torch.exp(-1j * theta)]))


def rot_X(phi):
    c = torch.cos(phi).to(cdtype)
    s = 1j * torch.sin(phi).to(cdtype)
    return torch.stack([torch.stack([c, s]), torch.stack([s, c])])


def depolarizing_channel(rho, p):
    p = p.to(cdtype)
    K0 = torch.sqrt(1 - p) * I2
    K1 = torch.sqrt(p / 3) * X
    K2 = torch.sqrt(p / 3) * Y
    K3 = torch.sqrt(p / 3) * Z
    out = K0 @ rho @ K0.conj().t().contiguous()
    out = out + K1 @ rho @ K1.conj().t().contiguous()
    out = out + K2 @ rho @ K2.conj().t().contiguous()
    out = out + K3 @ rho @ K3.conj().t().contiguous()
    return out


def build_rho(theta, Phi, p, rho):
    R0 = rot_X(Phi[0])
    rho = R0 @ rho @ R0.conj().t().contiguous()
    for phi_j in Phi[1:]:
        Rz = rot_Z(theta)
        rho = Rz @ rho @ Rz.conj().t().contiguous()
        rho = depolarizing_channel(rho, p)
        Rx = rot_X(phi_j)
        rho = Rx @ rho @ Rx.conj().t().contiguous()
    return rho


def initial_state():
    rho0 = torch.zeros((2, 2), dtype=cdtype)
    rho0[0, 0] = 1.0
    return rho0


def f_E(theta, Phi, p, rho0=None):
    if rho0 is None:
        rho0 = initial_state()
    rho = build_rho(theta, Phi, p, rho0)
    return rho[0, 0].real


def fisher_sqrt(theta, Phi, p_val):
    p = torch.tensor(p_val, dtype=torch.float64, requires_grad=True)
    f = f_E(theta, Phi, p)
    (df_dp,) = torch.autograd.grad(f, p, create_graph=True)
    F_p = df_dp**2 / (f * (1 - f) + 1e-12)
    return torch.sqrt(F_p.clamp(min=0.0))


def worst_case_sensitivity(theta, Phi, p_lo, p_hi, n_grid=5):
    p_grid = torch.linspace(p_lo, p_hi, n_grid).tolist()
    vals = torch.stack([fisher_sqrt(theta, Phi, pv) for pv in p_grid])
    return vals.min()


def optimize_noise_sensitivity(
    d, p_lo, p_hi, theta_init=None, Phi_init=None,
    n_grid=5, steps=200, lr=0.05, verbose=True, seed=0,
):
    torch.manual_seed(seed)

    theta = torch.tensor(
        0.3 if theta_init is None else float(theta_init), requires_grad=True
    )
    if Phi_init is None:
        Phi = (0.1 * torch.randn(d + 1)).requires_grad_(True)
    else:
        Phi = torch.tensor(Phi_init, dtype=torch.float64, requires_grad=True)

    opt = torch.optim.Adam([theta, Phi], lr=lr)

    L = None
    for step in range(steps):
        opt.zero_grad()
        L = worst_case_sensitivity(theta, Phi, p_lo, p_hi, n_grid=n_grid)
        (-L).backward()
        opt.step()
        if verbose and (step % 20 == 0 or step == steps - 1):
            print(f"step {step:4d}  L_E = {L.item():.4f}  theta = {theta.item():.4f}")

    L_E = L.item()
    kappa = L_E / d
    return theta.item(), Phi.detach().numpy(), L_E, kappa
