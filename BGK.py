import numpy as np



class BGK_D2Q9_Diffusion_Advection:    
    # Setup methods
    def __init__(self, lattice_dimensions: np.ndarray, omega: float, omega_d: float, alpha: float = 0.0):
        # D2Q9 weights and velocities
        self.W = np.array([4/9] + [1/9]*4 + [1/36]*4)
        self.c_int = np.array([[0, 0],[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=int)
        self.c = self.c_int.astype(float)
        self.lattice_dimensions = lattice_dimensions
        self.N = np.zeros(shape=(9, lattice_dimensions[0], lattice_dimensions[1]))
        self.N_eq = np.zeros_like(self.N)   
        self.C = np.zeros(shape=(9, lattice_dimensions[0], lattice_dimensions[1]))
        self.C_eq = np.zeros_like(self.C)   
        self.Q_iab = np.zeros(shape=(9, 2, 2))
        
        for i in range(9):
            self.Q_iab[i] = np.outer(self.c[i], self.c[i]) - (1/3)*np.eye(2)
        self.u = np.zeros(shape=(lattice_dimensions[0], lattice_dimensions[1], 2))
        self.rho = np.zeros(shape=(lattice_dimensions[0], lattice_dimensions[1]))
        
        # Forcing term
        self.F = np.zeros(shape=(lattice_dimensions[0], lattice_dimensions[1], 2))
        self.omega = omega
        self.omega_d = omega_d
        self.g = np.array([0.0, 9.81])
        self.alpha = alpha
    
    def initialize_from_initial_conditions(self, u: np.ndarray, C_init: np.ndarray):
        # Need to initialize C[i] and N[i] from the lattice
        # averages of u and C_init
        
        #C[i] must satisfy:
        # C_init(x,y) = sum_i C_i(x,y)
        self.C = np.random.rand(9, self.lattice_dimensions[0], self.lattice_dimensions[1])   
        C_avg = np.sum(self.C, axis=0)
        self.C *= C_init[np.newaxis, :, :] / C_avg[np.newaxis, :, :]
        
        # N[i] must satisfy: 
        # u = sum_i c_i * N_i / sum_i N_i
        
        
        
        
        pass
    # Equilibrium distribution functions
    # ---------------------------------
    def compute_equilibrium(self):  
        self.rho = np.sum(self.N, axis=0)
        self.u = np.einsum("ia,ixy->xya", self.c, self.N) / self.rho[:, :, np.newaxis]

        # N_i(x,y) = W_i*rho(x,y)*[1 + 3(c_i*u(x,y)) + 3Q_iab*u_a(x,y)*u_b(x,y)]

        cu = np.einsum("ia,xya->ixy", self.c, self.u)  # c_i * u(x,y)
        uu = np.einsum("xya, xyb->xyab", self.u, self.u)  # u_a(x,y) * u_b(x,y)
        Q_uu = np.einsum("iab, xyab->ixy", self.Q_iab, uu)  # Q_iab * u_a(x,y) * u_b(x,y)
        self.N_eq = self.W[:, np.newaxis, np.newaxis] * self.rho[np.newaxis, :, :] * (
            1 + 3 * cu + 4.5 * cu**2 - 1.5 * np.einsum("xya,xya->xy", self.u, self.u)[np.newaxis, :, :] + 4.5 * Q_uu
        )
        
    def compute_C_eq(self):
        # C_i(x,y) = W_i*[1 + 3(c_i*U(x,y))]
        self.C_eq = self.W[:, np.newaxis, np.newaxis] * (
            1 + 3 * np.einsum("ia,xya->ixy", self.c, self.u)
        )
    
    # Collision steps
    # ---------------
    def collision_step(self):
        # F = alpha*rho*g*C
        F = self.alpha * self.rho[:, :, np.newaxis] * self.g[np.newaxis, np.newaxis, :] * self.C
        
        self.N += -self.omega*(self.N - self.N_eq) - self.W[:, np.newaxis, np.newaxis] * np.einsum("ia,xya->ixy", self.c, F)
        pass
        
    def collision_step_C(self):
        self.C += -self.omega_d*(self.C - self.C_eq)
        pass
    
    # Propagation steps
    # -----------------
    def propagation_step_periodic(self):
        # With periodic boundary conditions the 
        # Propagation step for N_i can be efficiently implemented using np.roll
        for i in range(9):
            self.N[i] = np.roll(self.N[i], shift=self.c_int[i], axis=(0, 1))
        pass
    
    def propagation_step_C_periodic(self):
        for i in range(9):
            self.C[i] = np.roll(self.C[i], shift=self.c_int[i], axis=(0, 1))
        pass
    
    def propagation_step_noslip(self):
        pass
    
    def propagation_step_C_noslip(self):
        pass
    
    # Iteration step for BGK model
    def iterate_BGK(self, bc_type = 'periodic'):
        bc_types = ['periodic', 'noslip']
        bc_methods = [[self.propagation_step_periodic, self.propagation_step_C_periodic],
                      [self.propagation_step_noslip, self.propagation_step_C_noslip]]
        assert bc_type in bc_types, f"bc_type must be one of {bc_types}"
        idx = bc_types.index(bc_type)
        propagation_step_method, propagation_step_C_method = bc_methods[idx]
        
        # Steps for BGK iteration:
        # --------------------------
        # Compute N_eq, rho and u
        self.compute_equilibrium()
        # Colision steps
        self.collision_step()
        self.collision_step_C()
        # propagation steps
        propagation_step_method()
        propagation_step_C_method()
        pass
    
    def Simulate_BGK(self, num_iterations: int):
        N_t = np.zeros(shape=(1+num_iterations, *self.N.shape))
        C_t = np.zeros(shape=(1+num_iterations, *self.N.shape))
        
        N_t[0] = self.N.copy()
        C_t[0] = self.N.copy()
        
        for i in range(num_iterations):
            self.iterate_BGK()  
            N_t[i+1] = self.N.copy()
            C_t[i+1] = self.N.copy()
        return N_t, C_t

def nu2w(nu: float) -> float:
    """Convert kinematic viscosity to relaxation parameter omega."""
    return 1 /(3*nu) + 1/6

def D2omega_d(D: float) -> float:
    """Convert diffusion coefficient to relaxation parameter omega."""
    return 1/(3*D) + 1/6

def problem_1():
    Lattice_dimensions = np.array([200, 200])
    delta = 5
    U = 0.1
    u_0y = 10e-5
    k = 2*np.pi /( Lattice_dimensions[0]/10)
    c_0 = 1.0
    X,Y = np.meshgrid(np.arange(Lattice_dimensions[0]), np.arange(Lattice_dimensions[1]))
    
    u_x = U*(np.tanh((Y - 0.25*Lattice_dimensions[0])/delta) - np.tanh((Y - 0.75*Lattice_dimensions[0])/delta) - 1)
    u_y = u_0y * (np.sin(2*np.pi*X/Lattice_dimensions[0]))
    u_0 = np.zeros(shape=(Lattice_dimensions[0], Lattice_dimensions[1], 2))
    u_0[:, :, 0] = u_x
    u_0[:, :, 1] = u_y
    
    
    C_0 = c_0*(np.tanh((Y - 0.25*Lattice_dimensions[0])/delta) - np.tanh((Y - 0.75*Lattice_dimensions[0])/delta))
    
    w = nu2w(nu=0.01)
    w_d = D2omega_d(D=0.01)
    # Not included in the problem description
    alpha = 0.1
    
    System = BGK_D2Q9_Diffusion_Advection(lattice_dimensions=Lattice_dimensions, omega=w, omega_d=w_d, alpha=alpha)
    System.initialize_from_initial_conditions(u=u_0, C_init=C_0)
    

if __name__ == "__main__":
    pass
