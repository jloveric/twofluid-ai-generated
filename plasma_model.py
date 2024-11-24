import torch
import torch.nn.functional as F
import numpy as np

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Physical constants (normalized units)
c = 1.0  # Speed of light
epsilon_0 = 1.0  # Vacuum permittivity
mu_0 = 1.0  # Vacuum permeability

# Species parameters
mass_ion = 1.0
mass_electron = 5.447e-4

class TwoFluidModel:
    def __init__(self, nx, dx, charge=10.0):
        """
        Initialize two-fluid plasma model
        
        Args:
            nx: Number of spatial points
            dx: Grid spacing
            charge: Magnitude of species charge (ion charge = +charge, electron charge = -charge)
        """
        self.nx = nx
        self.dx = dx
        self.charge_ion = charge
        self.charge_electron = -charge
        
        # Initialize state vectors
        self.U_e = torch.zeros((5, nx), device=device)  # Electron fluid state vector
        self.U_i = torch.zeros((5, nx), device=device)  # Ion fluid state vector
        self.E = torch.zeros((3, nx), device=device)    # Electric field
        self.B = torch.zeros((3, nx), device=device)    # Magnetic field
        
    def compute_pressure(self, U, mass, gamma):
        """
        Compute pressure from conservative variables
        
        Args:
            U (torch.Tensor): State vector [5, nx]
            mass (torch.Tensor): Particle mass
            gamma (float): Adiabatic index
            
        Returns:
            torch.Tensor: Pressure [nx]
        """
        n = U[0]
        momentum = U[1:4]
        energy = U[4]
        
        # Compute kinetic energy density
        v = momentum / (n.unsqueeze(0))  # velocity
        kinetic_energy = 0.5 * mass * torch.sum(v**2, dim=0) * n
        
        # Pressure from ideal gas law
        pressure = (gamma - 1) * (energy - kinetic_energy)
        return pressure
    
    def compute_flux(self, U, mass, charge):
        """
        Compute flux terms F(U) for fluid equations
        
        Args:
            U (torch.Tensor): State vector [5, nx]
            mass (torch.Tensor): Particle mass
            charge (torch.Tensor): Particle charge
            
        Returns:
            torch.Tensor: Flux vector [5, nx]
        """
        n = U[0]
        momentum = U[1:4]
        energy = U[4]
        
        # Compute derived quantities
        v = momentum / (n.unsqueeze(0))  # velocity
        p = self.compute_pressure(U, mass, 5/3)
        
        # Initialize flux tensor
        F = torch.zeros_like(U)
        
        # Continuity equation flux
        F[0] = n * v[0]  # x-component only
        
        # Momentum equation flux
        for i in range(3):
            F[1+i] = momentum[i] * v[0]
        F[1] += p  # Add pressure term to x-momentum flux
        
        # Energy equation flux
        F[4] = (energy + p) * v[0]
        
        return F
    
    def compute_source_terms(self, U, mass, charge):
        """
        Compute source terms for fluid equations (Lorentz force)
        """
        n = U[0]
        momentum = U[1:4]
        energy = U[4]
        
        v = momentum / (n.unsqueeze(0))
        
        # Lorentz force density
        j = charge * n * v
        F_lorentz = charge * n * (self.E + torch.cross(v, self.B, dim=0))
        
        # Work done by electromagnetic field
        W_em = torch.sum(j * self.E, dim=0)
        
        # Initialize source terms
        source = torch.zeros_like(U)
        
        # Momentum source terms (Lorentz force)
        source[1:4] = F_lorentz
        
        # Energy source term (electromagnetic work)
        source[4] = W_em
        
        return source

    def compute_maxwell_source_terms(self):
        """
        Compute source terms for Maxwell's equations
        """
        # Compute current density
        j_e = self.charge_electron * self.U_e[0] * self.U_e[1:4] / self.U_e[0].unsqueeze(0)
        j_i = self.charge_ion * self.U_i[0] * self.U_i[1:4] / self.U_i[0].unsqueeze(0)
        j = j_e + j_i
        
        # Compute charge density
        rho_e = self.charge_electron * self.U_e[0]
        rho_i = self.charge_ion * self.U_i[0]
        rho = rho_e + rho_i
        
        return j, rho

    def compute_em_flux(self, E, B):
        """
        Compute electromagnetic flux tensor for Maxwell's equations
        Written in conservative form:
        ∂B/∂t + ∇×E = 0
        ∂E/∂t - c²∇×B = -j/ε₀
        
        Returns flux tensor for [E, B] state vector
        """
        # Initialize flux tensor for [E, B] (6 components)
        F = torch.zeros((6, E.shape[1]), device=device)
        
        # Faraday's law fluxes: ∂B/∂t + ∇×E = 0
        # Only include x-derivatives since we're in 1D
        F[3] = 0.0           # ∂Bx/∂t = 0 (no y or z derivatives)
        F[4] = -E[2]         # ∂By/∂t = -∂Ez/∂x
        F[5] = E[1]          # ∂Bz/∂t = ∂Ey/∂x
        
        # Ampere's law fluxes: ∂E/∂t - c²∇×B = -j/ε₀
        # Only include x-derivatives since we're in 1D
        F[0] = 0.0           # ∂Ex/∂t = 0 (no y or z derivatives)
        F[1] = c**2 * B[2]   # ∂Ey/∂t = c²∂Bz/∂x
        F[2] = -c**2 * B[1]  # ∂Ez/∂t = -c²∂By/∂x
        
        return F
    
    def lax_friedrichs_em_flux(self, EL, BL, ER, BR):
        """
        Compute local Lax-Friedrichs flux for electromagnetic fields
        """
        # Compute fluxes at left and right states
        FL = self.compute_em_flux(EL, BL)
        FR = self.compute_em_flux(ER, BR)
        
        # Maximum wave speed for EM waves is c
        wave_speed = c
        
        # Combine E and B fields into single state vector
        UL = torch.cat([EL, BL], dim=0)
        UR = torch.cat([ER, BR], dim=0)
        
        # Local Lax-Friedrichs flux
        flux = 0.5 * (FL + FR - wave_speed * (UR - UL))
        
        return flux

    def compute_em_derivatives(self):
        """
        Compute derivatives for electromagnetic fields using local Lax-Friedrichs flux
        with second-order reconstruction using minmod limiter
        """
        # Compute limited slopes for E and B fields
        E_slopes = self.compute_limited_slopes(self.E)
        B_slopes = self.compute_limited_slopes(self.B)
        
        # Reconstruct states at interfaces
        EL, ER = self.reconstruct_states(self.E, E_slopes, self.dx)
        BL, BR = self.reconstruct_states(self.B, B_slopes, self.dx)
        
        # Compute fluxes at cell interfaces
        flux = torch.zeros((6, self.nx-1), device=device)
        
        # Interior interfaces
        flux = self.lax_friedrichs_em_flux(EL[:, :-1], BL[:, :-1], 
                                          ER[:, 1:], BR[:, 1:])
        
        # Compute flux differences
        dF = torch.zeros((6, self.nx), device=device)
        dF[:, 1:-1] = (flux[:, 1:] - flux[:, :-1]) / self.dx
        
        # Get current density for source term in Ampere's law
        j, _ = self.compute_maxwell_source_terms()
        
        # Initialize derivatives
        dE_dt = -dF[:3]  # Electric field derivatives
        dB_dt = -dF[3:]  # Magnetic field derivatives
        
        # Add source term for Ampere's law
        dE_dt -= j / epsilon_0
        
        return dE_dt, dB_dt

    def compute_E_field_derivatives(self):
        """
        Compute time derivatives for electric field
        """
        j, rho = self.compute_maxwell_source_terms()
        
        # Ampere's law
        curl_B = self.curl(self.B)
        dE_dt = c**2 * (curl_B - mu_0 * j)
        
        return dE_dt

    def compute_B_field_derivatives(self):
        """
        Compute time derivatives for magnetic field
        """
        # Faraday's law
        dB_dt = -self.curl(self.E)
        
        return dB_dt

    def curl(self, F):
        """
        Compute curl of a vector field
        """
        curl_F = torch.zeros_like(F)
        curl_F[0, 1:-1] = (F[2, 2:] - F[2, :-2]) / (2 * self.dx)
        curl_F[1, 1:-1] = -(F[2, 2:] - F[2, :-2]) / (2 * self.dx)
        curl_F[2, 1:-1] = (F[1, 2:] - F[1, :-2] - F[0, 2:] + F[0, :-2]) / (2 * self.dx)
        
        return curl_F

    def lax_friedrichs_flux(self, UL, UR, mass, charge):
        """
        Compute Lax-Friedrichs flux at cell interfaces
        """
        # Compute fluxes at cell interfaces
        FL = self.compute_flux(UL, mass, charge)
        FR = self.compute_flux(UR, mass, charge)
        
        # Average flux
        flux = 0.5 * (FL + FR)
        
        # Add numerical viscosity
        alpha = 0.5  # Numerical viscosity coefficient
        wave_speed = alpha * (torch.abs(self.compute_max_wavespeed(UL, mass)) + 
                             torch.abs(self.compute_max_wavespeed(UR, mass)))
        flux = flux - 0.5 * wave_speed.unsqueeze(0) * (UR - UL)
        
        return flux

    def tvd_rk3_step(self, dt):
        """
        Third-order TVD Runge-Kutta time integration
        """
        # Store initial state
        U_e_0 = self.U_e.clone()
        U_i_0 = self.U_i.clone()
        E_0 = self.E.clone()
        B_0 = self.B.clone()
        
        # First stage
        k1_e = self.compute_fluid_derivatives(self.U_e, mass_electron, self.charge_electron)
        k1_i = self.compute_fluid_derivatives(self.U_i, mass_ion, self.charge_ion)
        k1_E, k1_B = self.compute_em_derivatives()
        
        self.U_e = U_e_0 + dt * k1_e
        self.U_i = U_i_0 + dt * k1_i
        self.E = E_0 + dt * k1_E
        self.B = B_0 + dt * k1_B
        
        # Second stage
        k2_e = self.compute_fluid_derivatives(self.U_e, mass_electron, self.charge_electron)
        k2_i = self.compute_fluid_derivatives(self.U_i, mass_ion, self.charge_ion)
        k2_E, k2_B = self.compute_em_derivatives()
        
        self.U_e = U_e_0 * (3/4) + (self.U_e + dt * k2_e) * (1/4)
        self.U_i = U_i_0 * (3/4) + (self.U_i + dt * k2_i) * (1/4)
        self.E = E_0 * (3/4) + (self.E + dt * k2_E) * (1/4)
        self.B = B_0 * (3/4) + (self.B + dt * k2_B) * (1/4)
        
        # Third stage
        k3_e = self.compute_fluid_derivatives(self.U_e, mass_electron, self.charge_electron)
        k3_i = self.compute_fluid_derivatives(self.U_i, mass_ion, self.charge_ion)
        k3_E, k3_B = self.compute_em_derivatives()
        
        self.U_e = U_e_0 * (1/3) + (self.U_e + dt * k3_e) * (2/3)
        self.U_i = U_i_0 * (1/3) + (self.U_i + dt * k3_i) * (2/3)
        self.E = E_0 * (1/3) + (self.E + dt * k3_E) * (2/3)
        self.B = B_0 * (1/3) + (self.B + dt * k3_B) * (2/3)

    def compute_fluid_derivatives(self, U, mass, charge):
        """
        Compute spatial derivatives for fluid equations using local Lax-Friedrichs flux
        with second-order reconstruction using minmod limiter
        """
        # Compute limited slopes for reconstruction
        slopes = self.compute_limited_slopes(U)
        
        # Reconstruct states at interfaces
        UL, UR = self.reconstruct_states(U, slopes, self.dx)
        
        # Compute fluxes at cell interfaces
        flux = torch.zeros((5, self.nx-1), device=device)
        
        # Interior interfaces
        flux = self.lax_friedrichs_flux(UL[:, :-1], UR[:, 1:], mass, charge)
        
        # Compute flux differences
        dF = torch.zeros((5, self.nx), device=device)
        dF[:, 1:-1] = (flux[:, 1:] - flux[:, :-1]) / self.dx
        
        # Add source terms (Lorentz force and electromagnetic work)
        source = self.compute_source_terms(U, mass, charge)
        
        return -dF + source

    def step(self, dt):
        """
        Advance solution by one time step using TVD-RK3
        """
        self.tvd_rk3_step(dt)
        
        # Apply boundary conditions
        self.apply_boundary_conditions()
    
    def set_initial_conditions(self, U_e, v_e, T_e, U_i, v_i, T_i, E, B):
        """Set initial conditions for all variables"""
        self.U_e = U_e
        self.U_i = U_i
        self.E = E
        self.B = B

    def apply_boundary_conditions(self):
        """
        Apply copy boundary conditions to all fields
        Copy the last interior cell value to the boundary cells
        """
        # Copy boundary conditions for electron fluid
        self.U_e[:, 0] = self.U_e[:, 1]    # Left boundary
        self.U_e[:, -1] = self.U_e[:, -2]  # Right boundary
        
        # Copy boundary conditions for ion fluid
        self.U_i[:, 0] = self.U_i[:, 1]    # Left boundary
        self.U_i[:, -1] = self.U_i[:, -2]  # Right boundary
        
        # Copy boundary conditions for electromagnetic fields
        self.E[:, 0] = self.E[:, 1]    # Left boundary
        self.E[:, -1] = self.E[:, -2]  # Right boundary
        
        self.B[:, 0] = self.B[:, 1]    # Left boundary
        self.B[:, -1] = self.B[:, -2]  # Right boundary

    def compute_max_wavespeed(self, U, mass):
        """
        Compute maximum wave speed for Lax-Friedrichs flux
        """
        n = U[0]
        momentum = U[1:4]
        energy = U[4]
        
        # Compute velocity
        v = momentum / (n.unsqueeze(0))
        
        # Compute sound speed
        p = self.compute_pressure(U, mass, 5/3)
        cs = torch.sqrt(5/3 * p / (mass * n))
        
        # Compute Alfven speed
        B_mag = torch.sqrt(torch.sum(self.B[:, :n.shape[0]]**2, dim=0))
        va = B_mag / torch.sqrt(mu_0 * mass * n)
        
        # Maximum wave speed is max of sound speed, Alfven speed, and fluid velocity
        max_speed = torch.max(torch.abs(v[0]))
        max_speed = torch.max(max_speed, cs)
        max_speed = torch.max(max_speed, va)
        
        return max_speed

    def get_electron_temperature(self):
        """Get electron temperature from conserved variables"""
        n = self.U_e[0]
        v = self.U_e[1:4] / n.unsqueeze(0)
        E = 0.5 * (v * v).sum(dim=0) * n  # Kinetic energy density
        return (self.U_e[4] - E) / n  # T = (E - 0.5*ρv²)/n

    def get_ion_temperature(self):
        """Get ion temperature from conserved variables"""
        n = self.U_i[0]
        v = self.U_i[1:4] / n.unsqueeze(0)
        E = 0.5 * (v * v).sum(dim=0) * n  # Kinetic energy density
        return (self.U_i[4] - E) / n  # T = (E - 0.5*ρv²)/n

    def minmod(self, a, b):
        """
        Minmod limiter function
        """
        return 0.5 * (torch.sign(a) + torch.sign(b)) * torch.minimum(torch.abs(a), torch.abs(b))

    def compute_limited_slopes(self, U):
        """
        Compute limited slopes for each variable using minmod limiter
        Returns slopes that give second-order accuracy in smooth regions
        while maintaining monotonicity near discontinuities
        """
        # Forward and backward differences
        dU_forward = torch.zeros_like(U)
        dU_backward = torch.zeros_like(U)
        
        # Interior points
        dU_forward[:, :-1] = (U[:, 1:] - U[:, :-1]) / self.dx
        dU_backward[:, 1:] = (U[:, 1:] - U[:, :-1]) / self.dx
        
        # Compute limited slopes using minmod
        slopes = self.minmod(dU_forward, dU_backward)
        
        # Set boundary slopes to zero
        slopes[:, 0] = 0
        slopes[:, -1] = 0
        
        return slopes

    def reconstruct_states(self, U, slopes, dx):
        """
        Reconstruct left and right states at cell interfaces
        using limited slopes
        """
        # Left state at right interface of each cell
        UL = U + 0.5 * dx * slopes
        
        # Right state at left interface of each cell
        UR = U - 0.5 * dx * slopes
        
        return UL, UR

# Example usage
if __name__ == "__main__":
    # Set up grid
    nx = 100  # Number of spatial points
    dx = 0.01  # Spatial step size (meters)
    dt = 1e-12  # Time step (seconds)
    
    # Initialize model
    model = TwoFluidModel(nx, dx)
    
    # Set up initial conditions
    # Gaussian perturbation in electron density
    x = torch.linspace(0, (nx-1)*dx, nx, device=device)
    n_e = torch.ones(nx, device=device) + 0.1 * torch.exp(-(x - nx*dx/2)**2 / (0.1*nx*dx)**2)
    v_e = torch.zeros((3, nx), device=device)
    T_e = torch.ones(nx, device=device) * 1e4  # 10,000 K
    
    # Ion initial conditions
    n_i = torch.ones(nx, device=device)
    v_i = torch.zeros((3, nx), device=device)
    T_i = torch.ones(nx, device=device) * 1e4  # 10,000 K
    
    # Electromagnetic field initial conditions
    E = torch.zeros((3, nx), device=device)
    B = torch.zeros((3, nx), device=device)
    B[2, :] = 1e-3  # Small uniform magnetic field in z-direction
    
    # Set initial conditions
    U_e = torch.zeros((5, nx), device=device)
    U_e[0] = n_e
    U_e[1:4] = n_e.unsqueeze(0) * v_e
    U_e[4] = 1.5 * n_e * T_e + 0.5 * mass_electron * n_e * torch.sum(v_e**2, dim=0)
    
    U_i = torch.zeros((5, nx), device=device)
    U_i[0] = n_i
    U_i[1:4] = n_i.unsqueeze(0) * v_i
    U_i[4] = 1.5 * n_i * T_i + 0.5 * mass_ion * n_i * torch.sum(v_i**2, dim=0)
    
    model.set_initial_conditions(U_e, v_e, T_e, U_i, v_i, T_i, E, B)
    
    # Run simulation for 1000 steps
    print("Starting simulation...")
    for i in range(1000):
        if i % 100 == 0:
            print(f"Step {i}")
            print(f"Max electron density: {torch.max(model.U_e[0]):.3e}")
            print(f"Max ion density: {torch.max(model.U_i[0]):.3e}")
            print(f"Max E field: {torch.max(torch.abs(model.E)):.3e}")
            print(f"Max B field: {torch.max(torch.abs(model.B)):.3e}")
            print("---")
        
        model.step(dt)
