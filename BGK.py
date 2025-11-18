import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

class BGK_D2Q9_Diffusion_Advection:    
    # Setup methods
    def __init__(self, lattice_dimensions: np.ndarray, omega: float, omega_d: float, alpha: float = 0.0, bc: str = 'periodic'):
        # D2Q9 weights and velocities
        self.W = np.array([4/9] + [1/9]*4 + [1/36]*4)
        self.c_int = np.array([[0, 0],[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=int)
        self.c = self.c_int.astype(float)

         
        self.lattice_dimensions = lattice_dimensions
        self.N = np.zeros(shape=(9, lattice_dimensions[0], lattice_dimensions[1]))
        self.N_eq = np.zeros_like(self.N)   
        self.C = np.zeros(shape=(9, lattice_dimensions[0], lattice_dimensions[1]))
        self.C_avg = np.zeros(shape=(lattice_dimensions[0], lattice_dimensions[1]))
        self.C_eq = np.zeros_like(self.C)   
        self.Q_iab = np.zeros(shape=(9, 2, 2))
        
        for i in range(9):
            self.Q_iab[i] = (3/2)*np.outer(self.c[i], self.c[i]) - (1/2)*np.eye(2)
        self.u = np.zeros(shape=(lattice_dimensions[0], lattice_dimensions[1], 2))
        self.rho = np.zeros(shape=(lattice_dimensions[0], lattice_dimensions[1]))
        
        # Forcing term
        self.F = np.zeros(shape=(lattice_dimensions[0], lattice_dimensions[1], 2))
        self.omega = omega
        self.omega_d = omega_d
        self.g = np.array([0.0, -9.81])
        self.alpha = alpha
        self.set_bcs(bc)
    
    def initialize_from_initial_conditions(self, u: np.ndarray, C_init: np.ndarray, rho : float = 1.0, rho_array: np.ndarray = np.zeros((0,0))):
        """
        Initialize distributions from macroscopic fields.

        Standard LBM practice for underdetermined u→N_i is to set the
        populations to their equilibrium with a chosen density field (often
        constant). Likewise for the passive scalar, set C_i to scalar-equilibrium.

        Parameters
        - u: array (Nx, Ny, 2), initial velocity field
        - C_init: array (Nx, Ny), initial scalar (concentration/temperature)
        - rho: float or array (Nx, Ny), initial density (default 1.0)
        """
        # Shape checks (lightweight)
        Nx, Ny = self.lattice_dimensions
        assert u.shape == (Nx, Ny, 2), f"u must have shape {(Nx, Ny, 2)}, got {u.shape}"
        assert C_init.shape == (Nx, Ny), f"C_init must have shape {(Nx, Ny)}, got {C_init.shape}"

        # Set macroscopic fields
        if rho_array.shape != (self.lattice_dimensions[0], self.lattice_dimensions[1]):
            self.rho = np.full((Nx, Ny), rho)
        else:
            self.rho = rho_array.astype(float)
        self.u = u.astype(float)
        self.C_avg = C_init.astype(float)

        # Calculates equilibrium distribution N_eq
        self.compute_equilibrium()
        self.compute_C_eq()
        
        # Set distributions to equilibrium
        self.N = self.N_eq.copy()
        self.C = self.C_eq.copy()

    # Macroscopic field computations
    # ----------------------------
    def compute_Macroscopic_fields(self):
        self.rho = np.sum(self.N, axis=0)
        self.u = np.einsum("ia,ixy->xya", self.c, self.N) / self.rho[:, :, np.newaxis]
        self.C_avg = np.sum(self.C, axis=0)



    # Equilibrium distribution functions
    # ---------------------------------
    
    # Verified without for loops, produce same result as with for loops
    def compute_equilibrium(self):  

        # N_i(x,y) = W_i*rho(x,y)*[1 + 3(c_i*u(x,y)) + 3Q_iab*u_a(x,y)*u_b(x,y)]

        cu = np.einsum("ia,xya->ixy", self.c, self.u)  # c_i * u(x,y)
        uu = np.einsum("xya, xyb->xyab", self.u, self.u)  # u_a(x,y) * u_b(x,y)
        Q_uu = np.einsum("iab, xyab->ixy", self.Q_iab, uu)  # Q_iab * u_a(x,y) * u_b(x,y)
        self.N_eq = self.W[:, np.newaxis, np.newaxis] * self.rho[np.newaxis, :, :] * (
            1 + 3 * cu + 3 * Q_uu
        )

        


    def compute_C_eq(self):
        #self.C_avg = np.sum(self.C, axis=0)
        # C_eq_i = W_i * C_avg * (1 + 3*c_i·u)
        
        self.C_eq = self.C_avg[np.newaxis, :, :]*self.W[:, np.newaxis, np.newaxis] * (
            1 + 3 * np.einsum("ia,xya->ixy", self.c, self.u)
        ) 
    
    # Collision steps
    # ---------------
    # Verified without for loops, produce same result as with for loops
    def collision_step(self):
        # F = alpha*rho*g*C
        F = self.alpha * self.rho[:, :, np.newaxis] * self.g[np.newaxis, np.newaxis, :] * self.C_avg[:, :, np.newaxis]
        
        # N_i(x,y) += -omega*(N_i(x,y) - N_eq_i(x,y)) + 3*W_i*c_i*F(x,y)
        self.N += -self.omega*(self.N - self.N_eq) + 3*self.W[:, np.newaxis, np.newaxis] * np.einsum("ia,xya->ixy", self.c, F)
        pass
        
    # Collision step for N_i without with for loops



    def collision_step_C(self):
        self.C += -self.omega_d*(self.C - self.C_eq)
        pass
    
    # Propagation steps
    # -----------------
    def propagation_step_periodic(self):
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
        propagation_step, propagation_step_C = self.propagation_step_methods
        
        # Steps for BGK iteration:
        # --------------------------

        self.compute_equilibrium() # Verified
        self.compute_C_eq()

        # Colision steps
        self.collision_step() # Verified
        self.collision_step_C()
        # propagation steps
        propagation_step, propagation_step_C = self.propagation_step_methods
        propagation_step() # Verified
        propagation_step_C()
        
        # Compute the new rho, u and C_avg
        self.compute_Macroscopic_fields() # Verified
        pass
    
    def Simulate_BGK(self, num_iterations: int, save_interval: int = 1):
        """
        Simulate the BGK model for a given number of iterations.
        Parameters
        ----------
        num_iterations : int
            Number of iterations to simulate.
        
        Returns
        -------
        u_t : array (num_iterations+1, Nx, Ny, 2)
            Velocity field at each time step.
        C_t : array (num_iterations+1, Nx, Ny)
            Scalar field at each time step.
        """
        
        saved_frames = num_iterations // save_interval + 1
        
        u_t = np.zeros(shape=(saved_frames, self.lattice_dimensions[0], self.lattice_dimensions[1], 2))
        C_t = np.zeros(shape=(saved_frames, self.lattice_dimensions[0], self.lattice_dimensions[1]))

        
        u_t[0] = self.u.copy()
        C_t[0] = self.C_avg.copy()
        
    
        
        for i in tqdm(range(num_iterations)):
            if (i+1) % save_interval == 0:
                frame_index = (i+1) // save_interval
                u_t[frame_index] = self.u.copy()
                C_t[frame_index] = np.sum(self.C, axis=0)
            self.iterate_BGK()  
        return u_t, C_t

    
    def set_bcs(self, bc: str):
        bc_types = ['periodic', 'noslip']
        bc_methods = [[self.propagation_step_periodic, self.propagation_step_C_periodic],
                        [self.propagation_step_noslip, self.propagation_step_C_noslip]]
        assert bc in bc_types, f"bc must be one of {bc_types}"  
        self.propagation_step_methods = bc_methods[bc_types.index(bc)]
    
    
    def plot_u(self):
                # plot the initial velocity field with quiver
        X, Y = np.meshgrid(np.arange(self.lattice_dimensions[0]), np.arange(self.lattice_dimensions[1]), indexing='ij')
        
        plt.figure(figsize=(8, 6))
        plt.quiver(X[::5, ::5], Y[::5, ::5], self.u[::5, ::5, 0], self.u[::5, ::5, 1], color='r')
        plt.title('Initial Velocity Field')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.show()

    def plot_C(self):
        # C[x,y], so need to transpose and flip vertically
        C_plot = self.C_avg.T
        C_plot = C_plot[::-1, :]
        plt.imshow(C_plot, origin='lower', cmap='viridis')
        plt.colorbar(label='Concentration')
        plt.title('Scalar Field Concentration')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()



def nu2w(nu: float) -> float:
    """Convert kinematic viscosity to relaxation parameter omega."""
    return 1/(3*nu + 0.5)

def D2omega_d(D: float) -> float:
    """Convert diffusion coefficient to relaxation parameter omega."""
    return 1/(3*D + 0.5)


def animate_u_t(u: np.ndarray, fps = 10, quiver_interval=10, frame_jump=1, time_step_per_frame=1):
    # u(t,x,y) = (u_x, u_y) 
    U = np.linalg.norm(u, axis=3)

    
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[2]), indexing='ij')
    quiver = ax.quiver(X[::quiver_interval, ::quiver_interval], Y[::quiver_interval, ::quiver_interval], u[0, ::quiver_interval, ::quiver_interval, 0], u[0, ::quiver_interval, ::quiver_interval, 1], color='k')
    Magnitude = ax.imshow(U[0].T[::-1, :], origin='lower', cmap='bwr', alpha=0.5)
    ax.set_title('Velocity Field Over Time')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')
    cb = fig.colorbar(Magnitude, ax=ax, label='Velocity Magnitude')
    def update(frame):
        ax.clear()
        quiver = ax.quiver(X[::quiver_interval, ::quiver_interval], Y[::quiver_interval, ::quiver_interval], u[frame, ::quiver_interval, ::quiver_interval, 0], u[frame, ::quiver_interval, ::quiver_interval, 1], color='k')
        Magnitude = ax.imshow(U[frame].T[::-1, :], origin='lower', cmap='bwr', alpha=0.5)
        cb.update_normal(Magnitude)
        ax.set_title(f'Velocity Field at time step {frame*time_step_per_frame}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
        return quiver, Magnitude
    
    ani = animation.FuncAnimation(fig, update, frames=range(0, u.shape[0], frame_jump), blit=False, interval=1000/fps)
    plt.show()
    
def animate_C_t(C: np.ndarray, fps = 10, frame_jump=1, time_step_per_frame=1):
    # C(t,x,y)

    fig, ax = plt.subplots()
    im = ax.imshow(C[0].T[::-1, :], origin='lower', cmap='bwr')
    ax.set_title('Scalar Field Concentration Over Time')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    cb = fig.colorbar(im, ax=ax, label='Concentration')
    
    def update(frame):
        ax.clear()
        im = ax.imshow(C[frame].T[::-1, :], origin='lower', cmap='bwr')
        ax.set_title(f'Scalar Field Concentration at time step {frame*time_step_per_frame}')
        cb.update_normal(im)
        
        return im 
    
    ani = animation.FuncAnimation(fig, update, frames=range(0, C.shape[0], frame_jump), blit=False, interval=1000/fps) # type: ignore
    plt.show()
    
    
    
    
# Problems
# ----------------------------

def problem_1():
    Lattice_dimensions = np.array([200, 200])
    delta = 5
    U = 0.1
    u_0y = 10e-5
    k = 2*np.pi /( Lattice_dimensions[0]/10)
    c_0 = 1.0
    X,Y = np.meshgrid(np.arange(Lattice_dimensions[0]), np.arange(Lattice_dimensions[1]), indexing='ij')
    
    u_x = U*(np.tanh((Y - 0.25*Lattice_dimensions[0])/delta) - np.tanh((Y - 0.75*Lattice_dimensions[0])/delta) - 1)
    u_y = u_0y * (np.sin(2*np.pi*X/Lattice_dimensions[0]))
    u_0 = np.zeros(shape=(Lattice_dimensions[0], Lattice_dimensions[1], 2))
    u_0[:, :, 0] = u_x
    u_0[:, :, 1] = u_y
    
    
    C_0 = c_0*(np.tanh((Y - 0.25*Lattice_dimensions[1])/delta) - np.tanh((Y - 0.75*Lattice_dimensions[1])/delta))
    
    
    w = nu2w(nu=0.01)
    print(f"Relaxation parameter omega: {w}")
    w_d = D2omega_d(D=0.01)
    # Not included in the problem description

    
    System = BGK_D2Q9_Diffusion_Advection(lattice_dimensions=Lattice_dimensions, omega=w, omega_d=w_d)
    System.initialize_from_initial_conditions(u=u_0, C_init=C_0)

    # Too many frames to store all, so save every 1000th frame
    u_t, C_t = System.Simulate_BGK(num_iterations=50000, save_interval=1000)
    
    animate_u_t(u_t, fps=5, time_step_per_frame=1000)
    animate_C_t(C_t, fps=5, time_step_per_frame=1000)




# Main script
if __name__ == "__main__":
    problem_1()

    
    pass
