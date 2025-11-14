import numpy as np



class BGK_D2Q9_Diffusion_Advection:    
    def __init__(self, lattice_dimensions: np.ndarray, omega: float, alpha: float = 0.0):
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
        self.N += -self.omega*(self.N - self.N_eq) - self.W[:, np.newaxis, np.newaxis] * np.einsum("ia,xya->ixy", self.c, self.F)
        pass
        
    def collision_step_C(self):
        self.C += -self.omega*(self.C - self.C_eq)
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

if __name__ == "__main__":
    pass
