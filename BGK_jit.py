import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from typing import NamedTuple
import jax
import functools

class BGKState(NamedTuple):
    # Microscopic distribution functions
    N: jnp.ndarray
    N_eq: jnp.ndarray
    C: jnp.ndarray
    C_eq: jnp.ndarray
    # Macroscopic fields
    rho: jnp.ndarray
    u: jnp.ndarray
    C_avg: jnp.ndarray


class BGK_D2Q9_Diffusion_Advection:    
    # Setup methods
    def __init__(self, lattice_dimensions: jnp.ndarray, omega: float, omega_d: float, alpha: float = 0.0, bc: str = 'periodic'):
        # D2Q9 weights and velocities
        self.W = jnp.array([4/9] + [1/9]*4 + [1/36]*4)
        self.c_int = jnp.array([[0, 0],[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=int)
        self.c = self.c_int.astype(float)
        self.lattice_dimensions = lattice_dimensions
        
        self.state = BGKState(
            N = jnp.zeros(shape=(9, lattice_dimensions[0], lattice_dimensions[1])),
            N_eq =  jnp.zeros(shape=(9, lattice_dimensions[0], lattice_dimensions[1])),   
            C = jnp.zeros(shape=(9, lattice_dimensions[0], lattice_dimensions[1])),
            C_avg = jnp.zeros(shape=(lattice_dimensions[0], lattice_dimensions[1])),
            C_eq = jnp.zeros(shape=(9, lattice_dimensions[0], lattice_dimensions[1])),   
            u = jnp.zeros(shape=(lattice_dimensions[0], lattice_dimensions[1], 2)),
            rho = jnp.zeros(shape=(lattice_dimensions[0], lattice_dimensions[1]))
        )

        self.Q_iab = (3/2)*jnp.einsum("ia, ib->iab", self.c, self.c) - (1/2)*jnp.eye(2)[jnp.newaxis, :, :]
        


        self.omega = omega
        self.omega_d = omega_d
        self.g = jnp.array([0.0, -9.81])
        self.alpha = alpha
        self.set_bcs(bc)
    
    def initialize_from_initial_conditions(self, u: jnp.ndarray, C_init: jnp.ndarray, rho : float = 1.0, rho_array: jnp.ndarray = jnp.zeros((0,0))):
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
            rho_helper = jnp.full((Nx, Ny), rho)
        else:
            rho_helper = rho_array.astype(float)

        self.state = self.state._replace(rho=rho_helper, u=u, C_avg=C_init)

        # Calculates equilibrium distribution N_eq
        self.state = self.compute_equilibrium(self.state)
        self.state = self.compute_C_eq(self.state)
        
        # Set distributions to equilibrium
        self.state = self.state._replace(N=self.state.N_eq.copy(), C=self.state.C_eq.copy())

    

    # Macroscopic field computations
    # ----------------------------
    def compute_Macroscopic_fields(self, state: BGKState) -> BGKState:
        rho = jnp.sum(state.N, axis=0)
        u = jnp.einsum("ia,ixy->xya", self.c, state.N) / rho[:, :, jnp.newaxis]
        C_avg = jnp.sum(state.C, axis=0)
        return state._replace(rho=rho, u=u, C_avg=C_avg)



    # Equilibrium distribution functions
    # ---------------------------------
    
    # Verified without for loops, produce same result as with for loops
    def compute_equilibrium(self, state: BGKState) -> BGKState:  

        # N_i(x,y) = W_i*rho(x,y)*[1 + 3(c_i*u(x,y)) + 3Q_iab*u_a(x,y)*u_b(x,y)]

        cu = jnp.einsum("ia,xya->ixy", self.c, state.u)  # c_i * u(x,y)
        uu = jnp.einsum("xya, xyb->xyab", state.u, state.u)  # u_a(x,y) * u_b(x,y)
        Q_uu = jnp.einsum("iab, xyab->ixy", self.Q_iab, uu)  # Q_iab * u_a(x,y) * u_b(x,y)
        N_eq = self.W[:, jnp.newaxis, jnp.newaxis] * state.rho[jnp.newaxis, :, :] * (
            1 + 3 * cu + 3 * Q_uu
        ) 
        
        return state._replace(N_eq=N_eq)

    def compute_C_eq(self, state: BGKState) -> BGKState:
        #self.C_avg = jnp.sum(self.C, axis=0)
        # C_eq_i = W_i * C_avg * (1 + 3*c_i·u)
        
        C_eq = self.state.C_avg[jnp.newaxis, :, :]*self.W[:, jnp.newaxis, jnp.newaxis] * (
            1 + 3 * jnp.einsum("ia,xya->ixy", self.c, self.state.u)
        ) 
        return state._replace(C_eq=C_eq)
    
    # Collision steps
    # ---------------
    # Verified without for loops, produce same result as with for loops
    def collision_step(self, state: BGKState) -> BGKState:
        # F = alpha*rho*g*C
        F = self.alpha * state.rho[:, :, jnp.newaxis] * self.g[jnp.newaxis, jnp.newaxis, :] * state.C_avg[:, :, jnp.newaxis]
        
        # N_i(x,y) += -omega*(N_i(x,y) - N_eq_i(x,y)) + 3*W_i*c_i*F(x,y)
        N = state.N + -self.omega*(state.N - state.N_eq) + 3*self.W[:, jnp.newaxis, jnp.newaxis] * jnp.einsum("ia,xya->ixy", self.c, F)
        return state._replace(N=N)
        




    def collision_step_C(self, state: BGKState) -> BGKState:
        C = state.C -self.omega_d*(state.C - state.C_eq)
        return state._replace(C=C)
    
    # Propagation steps
    # -----------------
    def propagation_step_periodic(self, state: BGKState) -> BGKState:
        for i in range(9):    
            state = state._replace(N=state.N.at[i].set(jnp.roll(state.N[i], shift=self.c_int[i], axis=(0, 1))))
        return state
    
    def propagation_step_C_periodic(self, state: BGKState) -> BGKState:
        for i in range(9):
            state = state._replace(C=state.C.at[i].set(jnp.roll(state.C[i], shift=self.c_int[i], axis=(0, 1))))
        return state
    
    def propagation_step_noslip(self, state: BGKState) -> BGKState:
        return state
    
    def propagation_step_C_noslip(self, state: BGKState)-> BGKState:
        return state
    
    
    @functools.partial(jax.jit, static_argnums=0)
    def step_periodic(self, state: BGKState) -> BGKState:
        state = self.compute_equilibrium(state)
        state = self.compute_C_eq(state)
        state = self.collision_step(state)
        state = self.collision_step_C(state)
        state = self.propagation_step_periodic(state)
        state = self.propagation_step_C_periodic(state)
        return self.compute_Macroscopic_fields(state)
    
    # Iteration step for BGK model
    def iterate_BGK(self, bc_type = 'periodic'):
        self.state = self.step_periodic(self.state)
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
        
        u_t = np.zeros(shape=(0, self.lattice_dimensions[0], self.lattice_dimensions[1], 2))
        C_t = np.zeros(shape=(0, self.lattice_dimensions[0], self.lattice_dimensions[1]))

        u_t = jnp.append(u_t, self.state.u.copy()[jnp.newaxis, ...], axis=0)
        C_t = jnp.append(C_t, self.state.C_avg.copy()[jnp.newaxis, ...], axis=0)
        
    
        
        for i in tqdm(range(num_iterations)):
            if (i+1) % save_interval == 0:
                frame_index = (i+1) // save_interval
                u_t = jnp.append(u_t, self.state.u.copy()[jnp.newaxis, ...], axis=0)
                C_t = jnp.append(C_t, self.state.C_avg.copy()[jnp.newaxis, ...], axis=0)
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
        X, Y = jnp.meshgrid(jnp.arange(self.lattice_dimensions[0]), jnp.arange(self.lattice_dimensions[1]), indexing='ij')
        
        plt.figure(figsize=(8, 6))
        plt.quiver(X[::5, ::5], Y[::5, ::5], self.state.u[::5, ::5, 0], self.state.u[::5, ::5, 1], color='r')
        plt.title('Initial Velocity Field')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.show()

    def plot_C(self):
        # C[x,y], so need to transpose and flip vertically
        C_plot = self.state.C_avg.T
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


def animate_u_t(u: jnp.ndarray, fps = 10, quiver_interval=10, frame_jump=1, time_step_per_frame=1):
    # u(t,x,y) = (u_x, u_y) 
    U = jnp.linalg.norm(u, axis=3)

    
    fig, ax = plt.subplots()
    X, Y = jnp.meshgrid(jnp.arange(u.shape[1]), jnp.arange(u.shape[2]), indexing='ij')
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
    
def animate_C_t(C: jnp.ndarray, fps = 10, frame_jump=1, time_step_per_frame=1):
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
    Lattice_dimensions = jnp.array([200, 200])
    delta = 5
    U = 0.1
    u_0y = 10e-5
    k = 2*jnp.pi /( Lattice_dimensions[0]/10)
    c_0 = 1.0
    X,Y = jnp.meshgrid(jnp.arange(Lattice_dimensions[0]), jnp.arange(Lattice_dimensions[1]), indexing='ij')
    
    u_x = U*(jnp.tanh((Y - 0.25*Lattice_dimensions[0])/delta) - jnp.tanh((Y - 0.75*Lattice_dimensions[0])/delta) - 1)
    u_y = u_0y * (jnp.sin(2*jnp.pi*X/Lattice_dimensions[0]))
    x_hat = jnp.array([1.0, 0.0])
    y_hat = jnp.array([0.0, 1.0])
    u_0 = x_hat[jnp.newaxis, jnp.newaxis, :]*u_x[:, :, jnp.newaxis] + y_hat[jnp.newaxis, jnp.newaxis, :]*u_y[:, :, jnp.newaxis]
    
    
    C_0 = c_0*(jnp.tanh((Y - 0.25*Lattice_dimensions[1])/delta) - jnp.tanh((Y - 0.75*Lattice_dimensions[1])/delta))
    
    
    w = nu2w(nu=0.01)
    print(f"Relaxation parameter omega: {w}")
    w_d = D2omega_d(D=0.01)
    # Not included in the problem description

    
    System = BGK_D2Q9_Diffusion_Advection(lattice_dimensions=Lattice_dimensions, omega=w, omega_d=w_d)
    System.initialize_from_initial_conditions(u=u_0, C_init=C_0)

    # Too many frames to store all, so save every 100th frame
    u_t, C_t = System.Simulate_BGK(num_iterations=5000, save_interval=100)
    
    animate_u_t(u_t, fps=5, time_step_per_frame=100)
    animate_C_t(C_t, fps=5, time_step_per_frame=100)



# Main script
if __name__ == "__main__":
    problem_1()

    
    pass
