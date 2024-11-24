import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from plasma_model import (
    TwoFluidModel, device,
    mass_electron, mass_ion,
    epsilon_0, mu_0, c
)

def setup_shock_conditions(nx, dx):
    """
    Set up initial conditions for electromagnetic two-fluid shock
    Using normalized units where:
    - Ion mass = 1.0
    - Ion charge = 1.0
    - Speed of light = 1.0
    - μ₀ = 1.0
    - ε₀ = 1.0
    
    Initial conditions:
    Left state (x < 0):
        n_e = 1.0
        n_i = 1.0
        v_e = v_i = 0
        T_e = T_i = 0.5e-4/(γ-1)
        Bx = 0.0075
        By = 0.01
        
    Right state (x > 0):
        n_e = 0.125
        n_i = 0.125
        v_e = v_i = 0
        T_e = T_i = 0.05e-4/(γ-1)
        Bx = 0.0075
        By = -0.01
    """
    # Spatial grid centered at x=0
    x = torch.linspace(-0.5, 0.5, nx, device=device)
    mid_point = nx // 2
    
    # Initialize state vectors
    n_e = torch.empty(nx, device=device)
    n_i = torch.empty(nx, device=device)
    v_e = torch.zeros((3, nx), device=device)
    v_i = torch.zeros((3, nx), device=device)
    
    # Left state (x < 0)
    left_mask = x < 0
    n_e[left_mask] = 1.0      # Number density
    n_i[left_mask] = 1.0      # Number density
    
    # Right state (x > 0)
    right_mask = x >= 0
    n_e[right_mask] = 0.125   # Number density
    n_i[right_mask] = 0.125   # Number density
    
    # Energy density (E = p/(γ-1))
    gamma = 5/3  # Adiabatic index
    
    # Electron energy
    E_e = torch.empty(nx, device=device)
    E_e[left_mask] = 0.5e-4/(gamma-1)
    E_e[right_mask] = 0.05e-4/(gamma-1)
    
    # Ion energy
    E_i = torch.empty(nx, device=device)
    E_i[left_mask] = 0.5e-4/(gamma-1)
    E_i[right_mask] = 0.05e-4/(gamma-1)
    
    # Initialize electromagnetic fields
    B = torch.tensor([[0.0075], [0.01], [0.0]], device=device).repeat(1, nx)
    B[1, right_mask] = -0.01
    
    E = torch.zeros((3, nx), device=device)
    
    # Convert to conserved variables
    U_e = torch.zeros((5, nx), device=device)
    U_i = torch.zeros((5, nx), device=device)
    
    # Density
    U_e[0] = n_e
    U_i[0] = n_i
    
    # Momentum (ρv)
    U_e[1:4] = n_e * v_e
    U_i[1:4] = n_i * v_i
    
    # Total energy density (ρe)
    U_e[4] = E_e
    U_i[4] = E_i
    
    return U_e, v_e, E_e/n_e, U_i, v_i, E_i/n_i, E, B, x

def compute_plasma_frequency(n_e):
    """
    Compute electron plasma frequency
    """
    # In normalized units:
    # ω_pe = sqrt(n_e * e^2 / (ε_0 * m_e))
    # With e = 1, ε_0 = 1, this becomes:
    return torch.sqrt(torch.abs(n_e) / mass_electron)

def compute_acoustic_speed(n, T, mass, gamma=5/3):
    """
    Compute acoustic speed cs = √(γp/ρ)
    In normalized units: p = nT, ρ = nm
    Thus cs = √(γT/m)
    """
    return torch.sqrt(gamma * T / mass)

def plot_state(model, x, t, save_path=None):
    """Plot the current state of the simulation"""
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(12, 16))
    
    # Get primitive variables
    n_e = model.U_e[0].cpu().numpy()
    n_i = model.U_i[0].cpu().numpy()
    v_e = (model.U_e[1:4] / model.U_e[0].unsqueeze(0)).cpu().numpy()
    v_i = (model.U_i[1:4] / model.U_i[0].unsqueeze(0)).cpu().numpy()
    x = x.cpu().numpy()
    
    # Plot densities
    ax1.plot(x, n_e, label='Electron')
    ax1.plot(x, n_i, label='Ion')
    ax1.set_title('Number Density')
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('Density (m⁻³)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot pressures
    p_e = n_e * model.get_electron_temperature().cpu().numpy()
    p_i = n_i * model.get_ion_temperature().cpu().numpy()
    ax2.plot(x, p_e, label='Electron')
    ax2.plot(x, p_i, label='Ion')
    ax2.set_title('Pressure')
    ax2.set_xlabel('Position (m)')
    ax2.set_ylabel('Pressure')
    ax2.legend()
    ax2.grid(True)
    
    # Plot velocities
    ax3.plot(x, v_e[0], label='Electron')
    ax3.plot(x, v_i[0], label='Ion')
    ax3.set_title('X-Velocity')
    ax3.set_xlabel('Position (m)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.legend()
    ax3.grid(True)
    
    ax4.plot(x, v_e[1], label='Electron')
    ax4.plot(x, v_i[1], label='Ion')
    ax4.set_title('Y-Velocity')
    ax4.set_xlabel('Position (m)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.legend()
    ax4.grid(True)
    
    # Plot electric field
    E = model.E.cpu().numpy()
    ax5.plot(x, E[0], label='Ex')
    ax5.plot(x, E[1], label='Ey')
    ax5.plot(x, E[2], label='Ez')
    ax5.set_title('Electric Field')
    ax5.set_xlabel('Position (m)')
    ax5.set_ylabel('E (V/m)')
    ax5.legend()
    ax5.grid(True)
    
    # Plot magnetic field
    B = model.B.cpu().numpy()
    ax6.plot(x, B[0], label='Bx')
    ax6.plot(x, B[1], label='By')
    ax6.plot(x, B[2], label='Bz')
    ax6.set_title('Magnetic Field')
    ax6.set_xlabel('Position (m)')
    ax6.set_ylabel('B (T)')
    ax6.legend()
    ax6.grid(True)
    
    # Plot temperatures
    T_e = model.get_electron_temperature().cpu().numpy()
    T_i = model.get_ion_temperature().cpu().numpy()
    ax7.plot(x, T_e, label='Electron')
    ax7.plot(x, T_i, label='Ion')
    ax7.set_title('Temperature')
    ax7.set_xlabel('Position (m)')
    ax7.set_ylabel('Temperature')
    ax7.legend()
    ax7.grid(True)
    
    # Plot current density
    j, _ = model.compute_maxwell_source_terms()
    ax8.plot(x, j[0].cpu().numpy(), label='jx')
    ax8.plot(x, j[1].cpu().numpy(), label='jy')
    ax8.plot(x, j[2].cpu().numpy(), label='jz')
    ax8.set_title('Current Density')
    ax8.set_xlabel('Position (m)')
    ax8.set_ylabel('Current Density')
    ax8.legend()
    ax8.grid(True)
    
    plt.suptitle(f'Time: {t:.2e} s')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def run_simulation(nx=200, tmax=1e-9, plot_interval=1e-10, charge=10.0):
    """
    Run the two-fluid shock simulation
    
    Args:
        nx: Number of spatial points
        tmax: Maximum simulation time
        plot_interval: Time between plots
        charge: Magnitude of species charge (ion charge = +charge, electron charge = -charge)
    """
    # Set up grid
    L = 1.0  # Domain length in normalized units
    dx = L/nx
    
    # Initialize model and set initial conditions
    model = TwoFluidModel(nx, dx, charge=charge)
    U_e, v_e, T_e, U_i, v_i, T_i, E, B, x = setup_shock_conditions(nx, dx)
    model.set_initial_conditions(U_e, v_e, T_e, U_i, v_i, T_i, E, B)
    
    # Time stepping parameters
    cfl = 0.4  # CFL number
    
    # Create output directory
    os.makedirs('shock_results', exist_ok=True)
    
    # Time evolution
    t = 0
    step = 0
    next_plot = 0
    
    print("Starting shock simulation...")
    
    while t < tmax:
        # Compute maximum wave speeds
        max_speed_e = torch.max(model.compute_max_wavespeed(model.U_e, mass_electron)).item()
        max_speed_i = torch.max(model.compute_max_wavespeed(model.U_i, mass_ion)).item()
        max_speed = max(max_speed_e, max_speed_i, c)
        
        # Compute timestep
        dt = cfl * dx / max_speed
        
        # Take a step
        model.step(dt)
        t += dt
        step += 1
        
        # Print diagnostic information
        if step == 1 or t >= next_plot:
            print(f"Time: {t:.2e} s (Step {step})")
            print(f"dt: {dt:.2e}")
            print(f"Max electron density: {torch.max(model.U_e[0]):.3e}")
            print(f"Max ion density: {torch.max(model.U_i[0]):.3e}")
            print(f"Max E field: {torch.max(torch.abs(model.E)):.3e}")
            print(f"Max B field: {torch.max(torch.abs(model.B)):.3e}")
            print("---")
            
            # Plot current state
            plot_state(model, x, t, f'shock_results/state_t{t:.2e}.png')
            next_plot = t + plot_interval
    
    return model, x

def main():
    # Simulation parameters
    nx = 10000  # Number of spatial points
    L = 1.0   # Domain length
    dx = L / nx
    
    # Species charge magnitude
    charge = 10.0  # qi = +10, qe = -10
    
    # Time parameters
    tmax = 20.0     # Run until t = 20.0
    plot_interval = 2.0  # Plot every 2.0 time units (10 snapshots)
    
    # Run simulation
    model, x = run_simulation(nx, tmax, plot_interval, charge=charge)

if __name__ == "__main__":
    main()
