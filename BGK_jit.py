import jax.numpy as jnp
from matplotlib.colors import CenteredNorm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import NamedTuple
import jax
import functools
import scipy.interpolate
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)


# Data structure to hold the state of the BGK model. Necessary for JAX JIT compilation, 
# as JAX requires immutable data structures. More complex than a simple class, but it 
# makes the jax an order of magnitude faster. Apart from the strict non-mutation requirement,
# Jax.numpy are essentially the same as numpy arrays.

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

@functools.partial(jax.jit)
def roll_dir(distribution: jnp.ndarray, shift_x: int, shift_y: int):
    shifted = jnp.roll(distribution, shift=shift_x, axis=0)
    return jnp.roll(shifted, shift=shift_y, axis=1)

class BGK_D2Q9_Diffusion_Advection:    
    # Setup methods
    def __init__(self, lattice_dimensions: jnp.ndarray, omega: float, omega_d: float, alpha: float = 0.0, bc: str = 'periodic'):
        # D2Q9 weights and velocities. These are constant, and are therefore immutable.
        self.W = jnp.array([4/9] + [1/9]*4 + [1/36]*4)
        self.c_int = jnp.array([[0, 0],[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=int)
        self.c = self.c_int.astype(float)
        self.lattice_dimensions = lattice_dimensions
        self.omega = omega
        self.omega_d = omega_d
        self.g = jnp.array([0.0, -9.81])
        self.alpha = alpha
        self.set_bcs(bc)

        # Mutable variables, need to be stored in state for JAX JIT     
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
        



    
    def initialize_from_initial_conditions(self, u: jnp.ndarray, C_init: jnp.ndarray, rho : float = 1.0, rho_array: jnp.ndarray = jnp.zeros((0,0))):
        """
        Initialize distributions from macroscopic fields.

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
        self.state = self._compute_equilibrium(self.state)
        self.state = self._compute_C_eq(self.state)
        
        # Set distributions to equilibrium
        self.state = self.state._replace(N=self.state.N_eq.copy(), C=self.state.C_eq.copy())

    

    # Macroscopic field computations
    # ----------------------------
    @functools.partial(jax.jit, static_argnums=0)
    def _compute_Macroscopic_fields(self, state: BGKState) -> BGKState:
        rho = jnp.sum(state.N, axis=0)
        u = jnp.einsum("ia,ixy->xya", self.c, state.N) / rho[:, :, jnp.newaxis]
        C_avg = jnp.sum(state.C, axis=0)
        return state._replace(rho=rho, u=u, C_avg=C_avg)



    # Equilibrium distribution functions
    # ---------------------------------
    
    @functools.partial(jax.jit, static_argnums=0)
    def _compute_equilibrium(self, state: BGKState) -> BGKState:  

        # N_i(x,y) = W_i*rho(x,y)*[1 + 3(c_i*u(x,y)) + 3Q_iab*u_a(x,y)*u_b(x,y)]

        cu = jnp.einsum("ia,xya->ixy", self.c, state.u)  # c_i * u(x,y)
        uu = jnp.einsum("xya, xyb->xyab", state.u, state.u)  # u_a(x,y) * u_b(x,y)
        Q_uu = jnp.einsum("iab, xyab->ixy", self.Q_iab, uu)  # Q_iab * u_a(x,y) * u_b(x,y)
        N_eq = self.W[:, jnp.newaxis, jnp.newaxis] * state.rho[jnp.newaxis, :, :] * (
            1 + 3 * cu + 3 * Q_uu
        ) 
        
        return state._replace(N_eq=N_eq)

    @functools.partial(jax.jit, static_argnums=0)
    def _compute_C_eq(self, state: BGKState) -> BGKState:
        #self.C_avg = jnp.sum(self.C, axis=0)
        # C_eq_i = W_i * C_avg * (1 + 3*c_i·u)
        
        C_eq = state.C_avg[jnp.newaxis, :, :]*self.W[:, jnp.newaxis, jnp.newaxis] * (
            1 + 3 * jnp.einsum("ia,xya->ixy", self.c, state.u)
        ) 
        return state._replace(C_eq=C_eq)
    
    # Collision steps
    # ---------------
    @functools.partial(jax.jit, static_argnums=0)
    def _collision_step(self, state: BGKState) -> BGKState:
        # F = alpha*rho*g*C
        F = self.alpha * state.rho[:, :, jnp.newaxis] * self.g[jnp.newaxis, jnp.newaxis, :] * state.C_avg[:, :, jnp.newaxis]
        
        # N_i(x,y) += -omega*(N_i(x,y) - N_eq_i(x,y)) + 3*W_i*c_i*F(x,y)
        N = state.N + -self.omega*(state.N - state.N_eq) + 3*self.W[:, jnp.newaxis, jnp.newaxis] * jnp.einsum("ia,xya->ixy", self.c, F)
        return state._replace(N=N)
        

    @functools.partial(jax.jit, static_argnums=0)
    def _collision_step_C(self, state: BGKState) -> BGKState:
        C = state.C -self.omega_d*(state.C - state.C_eq)
        return state._replace(C=C)
    
    # Propagation steps
    # -----------------
    @functools.partial(jax.jit, static_argnums=0)
    def _propagation_step_periodic(self, state: BGKState) -> BGKState:
        shift_x = self.c_int[:, 0]
        shift_y = self.c_int[:, 1]
        N_streamed = jax.vmap(roll_dir, in_axes=(0, 0, 0))(state.N, shift_x, shift_y)
        return state._replace(N=N_streamed)
    
    @functools.partial(jax.jit, static_argnums=0)
    def _propagation_step_C_periodic(self, state: BGKState) -> BGKState:
        shift_x = self.c_int[:, 0]
        shift_y = self.c_int[:, 1]
        
        C_streamed = jax.vmap(roll_dir, in_axes=(0, 0, 0))(state.C, shift_x, shift_y)
        return state._replace(C=C_streamed)
    
    # Not implemented (yet)
    def _propagation_step_noslip(self, state: BGKState) -> BGKState:
        return state
    
    def _propagation_step_C_noslip(self, state: BGKState)-> BGKState:
        return state
    
    # Method to choose propagation step with boundary conditions
    # ------------------------------------------------------------
    @functools.partial(jax.jit, static_argnums=0)
    def _propagation_step_with_bc(self, state: BGKState) -> BGKState:
        return jax.lax.cond(
            self.bc == 0,
            lambda s: self._propagation_step_periodic(s),
            lambda s: self._propagation_step_noslip(s),
            state,
        )
    
    @functools.partial(jax.jit, static_argnums=0)
    def _propagation_step_C_with_bc(self, state: BGKState) -> BGKState:
        return jax.lax.cond(
            self.bc == 0,
            lambda s: self._propagation_step_C_periodic(s),
            lambda s: self._propagation_step_C_noslip(s),
            state,
        )


    # Internal step and simulation methods, JIT compiled
    @functools.partial(jax.jit, static_argnums=0)
    def _itterate_BGK(self, state: BGKState) -> BGKState:
        state = self._compute_equilibrium(state)
        state = self._compute_C_eq(state)
        state = self._collision_step(state)
        state = self._collision_step_C(state)
        state = self._propagation_step_with_bc(state)
        state = self._propagation_step_C_with_bc(state)
        return self._compute_Macroscopic_fields(state)
    




    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def _simulate_bgk_compiled(self, state: BGKState, num_iterations: int, save_interval: int):
        if save_interval <= 0:
            raise ValueError("save_interval must be positive")

        num_full_blocks, remainder = divmod(num_iterations, save_interval)
        num_saves = num_full_blocks + 1  # initial snapshot + one per completed block

        u_hist = jnp.zeros((num_saves,) + state.u.shape, dtype=state.u.dtype)
        c_hist = jnp.zeros((num_saves,) + state.C_avg.shape, dtype=state.C_avg.dtype)
        u_hist = u_hist.at[0].set(state.u)
        c_hist = c_hist.at[0].set(state.C_avg)

        # Wrap single step for use in fori_loop, which requires a specific signature
        def single_step(_, current_state):
            return self._itterate_BGK(current_state)

        # Block body for fori_loop, which performs save_interval steps
        def block_body(i, carry):
            block_state, u_buf, c_buf = carry
            block_state = jax.lax.fori_loop(0, save_interval, single_step, block_state)
            u_buf = u_buf.at[i + 1].set(block_state.u)
            c_buf = c_buf.at[i + 1].set(block_state.C_avg)
            return (block_state, u_buf, c_buf)

        carry = (state, u_hist, c_hist)
        # Iterate over full blocks of save_interval steps
        state, u_hist, c_hist = jax.lax.fori_loop(0, num_full_blocks, block_body, carry)
        
        # Handle remaining steps if total iterations is not a multiple of save_interval
        if remainder:
            state = jax.lax.fori_loop(0, remainder, single_step, state)

        return state, u_hist, c_hist


    # User methods for iteration and simulation. These update the internal state, and
    # makes it so that the user does not have to pass the state around.
    # ----------------------------
    def itterate_BGK(self):
        self.state = self._itterate_BGK(self.state)
        pass



    def Simulate_BGK(self, num_iterations: int, save_interval: int = 1):
        """
        Run the BGK model for ``num_iterations`` steps, storing the initial
        state and the state after every ``save_interval`` iterations.

        Returns
        -------
        u_t : array (floor(num_iterations / save_interval)+1, Nx, Ny, 2)
            Velocity snapshots.
        C_t : array (floor(num_iterations / save_interval)+1, Nx, Ny)
            Scalar-field snapshots.
        """
        num_iterations = int(num_iterations)
        save_interval = int(save_interval)
        state, u_hist, c_hist = self._simulate_bgk_compiled(
            self.state, num_iterations, save_interval
        )
        self.state = state
        return u_hist, c_hist



    
    def set_bcs(self, bc: str):
        bc_types = ['periodic', 'noslip']
        assert bc in bc_types, f"bc must be one of {bc_types}"
        self.bc = bc_types.index(bc)

    
    
    def plot_u(self, quiver_interval=5):
                # plot the initial velocity field with quiver
        X, Y = jnp.meshgrid(jnp.arange(self.lattice_dimensions[0]), jnp.arange(self.lattice_dimensions[1]), indexing='ij')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.quiver(X[::quiver_interval, ::quiver_interval], Y[::quiver_interval, ::quiver_interval], self.state.u[::quiver_interval, ::quiver_interval, 0], self.state.u[::quiver_interval, ::quiver_interval, 1], color='k')
        ax.imshow(jnp.linalg.norm(self.state.u, axis=2).T[::-1, :], origin='lower', cmap='Reds', alpha=0.5)
        ax.set_title('Initial Velocity Field')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
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


def animate_u_t(u: jnp.ndarray, fps = 10, quiver_interval=10, frame_jump=1, time_step_per_frame=1, filename=""):
    # u(t,x,y) = (u_x, u_y) 
    U = jnp.linalg.norm(u, axis=3)

    
    fig, ax = plt.subplots()
    X, Y = jnp.meshgrid(jnp.arange(u.shape[1]), jnp.arange(u.shape[2]), indexing='ij')
    quiver = ax.quiver(X[::quiver_interval, ::quiver_interval], Y[::quiver_interval, ::quiver_interval], u[0, ::quiver_interval, ::quiver_interval, 0], u[0, ::quiver_interval, ::quiver_interval, 1], color='k')
    Magnitude = ax.imshow(U[0].T[::-1, :], origin='lower', cmap='Reds', alpha=0.5, interpolation='bilinear')
    ax.set_title('Velocity Field Over Time')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')
    cb = fig.colorbar(Magnitude, ax=ax, label='Velocity Magnitude')
    def update(frame):
        ax.clear()
        quiver = ax.quiver(X[::quiver_interval, ::quiver_interval], Y[::quiver_interval, ::quiver_interval], u[frame, ::quiver_interval, ::quiver_interval, 0], u[frame, ::quiver_interval, ::quiver_interval, 1], color='k')
        Magnitude = ax.imshow(U[frame].T[::-1, :], origin='lower', cmap='Reds', alpha=0.5, interpolation='bilinear')
        cb.update_normal(Magnitude)
        ax.set_title(f'Velocity Field at time step {frame*time_step_per_frame}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
        return quiver, Magnitude
    
    ani = animation.FuncAnimation(fig, update, frames=range(0, u.shape[0], frame_jump), blit=False, interval=1000/fps)
    if filename!="":
        ani.save(filename, writer='ffmpeg', fps=fps)

    plt.show()
    
def animate_C_t(C: jnp.ndarray, fps = 10, frame_jump=1, time_step_per_frame=1, filename=""):
    # C(t,x,y)
    C_0 = jnp.max(C[0])

    fig, ax = plt.subplots()
    im = ax.imshow(C[0].T[::-1, :], origin='lower', cmap='bwr', interpolation='bilinear', norm=CenteredNorm(vcenter=C_0/2))
    ax.set_title('Scalar Field Concentration Over Time')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    cb = fig.colorbar(im, ax=ax, label='Concentration')
    
    def update(frame):
        ax.clear()
        im = ax.imshow(C[frame].T[::-1, :], origin='lower', cmap='bwr', interpolation='bilinear', norm=CenteredNorm(vcenter=C_0/2))
        ax.set_title(f'Scalar Field Concentration at time step {frame*time_step_per_frame}')
        cb.update_normal(im)
        
        return im 
    
    ani = animation.FuncAnimation(fig, update, frames=range(0, C.shape[0], frame_jump), blit=False, interval=1000/fps) # type: ignore
    if filename!="":
        ani.save(filename, writer='ffmpeg', fps=fps)
    plt.show()

def animate_vorticity_t(vorticity: jnp.ndarray, fps = 10, frame_jump=1, time_step_per_frame=1, filename=""):
    # vorticity(t,x,y)

    fig, ax = plt.subplots()
    im = ax.imshow(vorticity[0].T[::-1, :], origin='lower', cmap='bwr', interpolation='bilinear', norm=CenteredNorm(vcenter=0))
    ax.set_title('Vorticity Over Time')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    cb = fig.colorbar(im, ax=ax, label='Vorticity')
    
    def update(frame):
        ax.clear()
        im = ax.imshow(vorticity[frame].T[::-1, :], norm=CenteredNorm(vcenter=0), origin='lower', cmap='bwr', interpolation='bilinear')
        ax.set_title(f'Vorticity at time step {frame*time_step_per_frame}')
        cb.update_normal(im)
        
        return im 
    
    ani = animation.FuncAnimation(fig, update, frames=range(0, vorticity.shape[0], frame_jump), blit=False, interval=1000/fps) # type: ignore
    if filename!="":
        ani.save(filename, writer='ffmpeg', fps=fps)
    plt.show()

def get_vorticity(u: jnp.ndarray) -> jnp.ndarray:    
    #u[t, x, y, 0] = u_x
    #u[t, x, y, 1] = u_y

    dudy = jnp.gradient(u[:, :, :, 0], axis=2)
    dvdx = jnp.gradient(u[:, :, :, 1], axis=1)
 
    vorticity = dvdx - dudy
    return vorticity

def get_y_mid(C: jnp.ndarray, c_0: float) -> jnp.ndarray:
    """
    Determine y position of the minimum scalar field C relative to its initial condition C_0.
    Parameters
    ----------
    C : jnp.ndarray
        The current scalar field, shape (t, x, y).
    c_0 : float
        The initial maximum concentration value.
    Returns
    -------
    jnp.ndarray
        The mid-point y position for each x-coordinate, shape (t, x).
    """
    N_y = C.shape[2]

    rellative_C = jnp.abs(C - c_0/2)
    y_mid = jnp.argmin(rellative_C[:, :, N_y//2:N_y], axis=2)

    return y_mid 


def interpolate_y_mid(C: jnp.ndarray, c_0: float, Lattice_dimensions: jnp.ndarray, sublattice_presicion = 10, tol=1e-5) -> jnp.ndarray:
    """
    Interpolate the y mid-point positions to sub-lattice precision.
    Parameters
    ----------
    C : jnp.ndarray
        The current scalar field, shape (t, x, y).
    c_0 : float
        The initial maximum concentration value.
    Lattice_dimensions : jnp.ndarray
        The dimensions of the lattice, shape (2,).
    sublattice_presicion : int
        The factor by which to increase the resolution in the y-direction.
    tol : float
        The tolerance for finding the mid-point.
    Returns
    -------
    jnp.ndarray
        The interpolated mid-point y position for each x-coordinate, shape (t, x)
        We find the roots of the function C - c_0/2 using cubic spline interpolation and
        newton's method.
    """

    C_upper_rellative = C[:, :, Lattice_dimensions[1]//2:] - c_0/2
    
    
    # We need to first interpolate the C field such that the partial derivative in y direction exists
    y_mid_interp = np.zeros((C_upper_rellative.shape[0], C_upper_rellative.shape[1]*sublattice_presicion))
    # Initial guess for y_mid_interp is the center of the lattice, store this as the last row, as
    # then we can use for every t: y_mid_interp[t-1, x] as initial guess for y_mid_interp[t, x] 
    y_mid_interp[-1, :] = np.ones_like(y_mid_interp[0, :]) * C_upper_rellative.shape[2]//2  

    x_grid = np.linspace(0, Lattice_dimensions[0]-1, Lattice_dimensions[0])
    y_grid = np.linspace(0, C_upper_rellative.shape[2]-1, C_upper_rellative.shape[2])

    x_subgrid = np.linspace(0, Lattice_dimensions[0]-1, Lattice_dimensions[0]*sublattice_presicion)
    
    for t in tqdm(range(C_upper_rellative.shape[0])):
        x_arr = np.linspace(0, Lattice_dimensions[0]-1, Lattice_dimensions[0])
        y_arr = np.linspace(0, C_upper_rellative.shape[2]-1, C_upper_rellative.shape[2])
        # Create the interpolating function for the current time step
        interp_funk = scipy.interpolate.RegularGridInterpolator((x_arr,y_arr), C_upper_rellative[t, :, :], method='cubic', bounds_error=False, fill_value=None)
        
        
        for i, x in enumerate(x_subgrid):
            # Define a function for which we want to find the root
            def interp_func(y):
                return interp_funk((x, y))
            
            y_mid_interp[t,i] = scipy.optimize.newton(interp_func, y_mid_interp[t-1,i], tol=tol)

    
    return y_mid_interp, x_subgrid

def entropy_production(C_t: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the entropy production over time for the scalar field C.

    Parameters
    ----------
    C_t : jnp.ndarray
        The scalar field over time, shape (t, x, y).

    Returns
    -------
    jnp.ndarray
        The entropy production at each time step S'/(k_b*D).
        S'/(k_b*D) has shape (t,).
    """
    
    # Gradient of C with respect to x and y
    grad_C_t = jnp.gradient(C_t, axis=(1, 2)) 
    
    # Squared magnitude of the gradient at each point
    grad_C_squared = grad_C_t[0]**2 + grad_C_t[1]**2
    

    # Discrete sum over the spatial domain replacing the integral
    # We add a small value to C_t to avoid division by zero. This 
    # is not needed in theory, as the gradient should be zero where C_t is zero, but
    # for numerical stability we add it.
    return jnp.sum(grad_C_squared/(C_t + 1e-10), axis=(1, 2)) 
            
    
    



def initialization_Helmholtz(Lattice_dimensions=jnp.array([200, 200]), delta=5, rho=1.0, U=0.1, c_0 = 1.0):
    u_0y = 10e-5
    k = 2*jnp.pi /( Lattice_dimensions[0]/10)
    X,Y = jnp.meshgrid(jnp.arange(Lattice_dimensions[0]), jnp.arange(Lattice_dimensions[1]), indexing='ij')
    
    u_x = U*(jnp.tanh((Y - 0.25*Lattice_dimensions[0])/delta) - jnp.tanh((Y - 0.75*Lattice_dimensions[0])/delta) - 1)
    u_y = u_0y * (jnp.sin(2*jnp.pi*X/Lattice_dimensions[0]))
    x_hat = jnp.array([1.0, 0.0])
    y_hat = jnp.array([0.0, 1.0])
    u_0 = x_hat[jnp.newaxis, jnp.newaxis, :]*u_x[:, :, jnp.newaxis] + y_hat[jnp.newaxis, jnp.newaxis, :]*u_y[:, :, jnp.newaxis]
    
    
    C_0 = (c_0/2)*(jnp.tanh((Y - 0.25*Lattice_dimensions[1])/delta) - jnp.tanh((Y - 0.75*Lattice_dimensions[1])/delta))
    
    
    w = nu2w(nu=0.01)
    w_d = D2omega_d(D=0.01)
    System = BGK_D2Q9_Diffusion_Advection(lattice_dimensions=Lattice_dimensions, omega=w, omega_d=w_d)
    System.initialize_from_initial_conditions(u=u_0, C_init=C_0, rho=rho)
    return System, u_0, C_0

# Problems
# ----------------------------

def problem_2():
    Lattice_dimensions = jnp.array([250, 250])
    delta = 5
    System, u_0, C_0 = initialization_Helmholtz(Lattice_dimensions=Lattice_dimensions, delta=delta, rho=0.5, U=0.1)
    
    # Too many frames to store all, so save every 100th frame
    u_t, C_t = System.Simulate_BGK(num_iterations=50000, save_interval=250)
    
    vorticity_t = get_vorticity(u_t)

    # animate_u_t(u_t, fps=5, time_step_per_frame=100, filename='velocity_field.gif')
    # animate_C_t(C_t, fps=5, time_step_per_frame=100, filename='concentration_field.gif')
    # animate_vorticity_t(vorticity_t, fps=5, time_step_per_frame=100, filename='vorticity_field.gif')

    # Plots of velocity field at different time steps
    for index in [0, 10, 15, 20]:
        U = jnp.linalg.norm(u_t, axis=3)
    
        quiver_interval = 10    
        fig, ax = plt.subplots()
        X, Y = jnp.meshgrid(jnp.arange(u_t.shape[1]), jnp.arange(u_t.shape[2]), indexing='ij')
        quiver = ax.quiver(X[::quiver_interval, ::quiver_interval], Y[::quiver_interval, ::quiver_interval], u_t[index, ::quiver_interval, ::quiver_interval, 0], u_t[index, ::quiver_interval, ::quiver_interval, 1], color='k')
        Magnitude = ax.imshow(U[index].T[::-1, :], origin='lower', cmap='Reds', alpha=0.5, interpolation='bilinear')
        ax.set_title(f'Velocity Field at t = {index*250}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
        cb = fig.colorbar(Magnitude, ax=ax, label='Velocity Magnitude')
        plt.savefig(f'velocity_field_{index*250}.pdf')
        plt.show()

       
        fig, ax = plt.subplots()
        im = ax.imshow(vorticity_t[index].T[::-1, :], origin='lower', cmap='bwr', interpolation='bilinear', norm=CenteredNorm(vcenter=0))
        ax.set_title(f'Vorticity at time step {index*250}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        cb = fig.colorbar(im, ax=ax, label='Vorticity')   
        str = f'vorticity_field_{index*250}.pdf'
        plt.savefig(str)

        


def problem_4():
    c_0 = 1.0
    Lattice_dimensions = jnp.array([200, 200])
    delta = 5
    U = 0.1
    k = 2*jnp.pi /( Lattice_dimensions[0])
    save_interval = 25
    System, u_0, C_0 = initialization_Helmholtz(Lattice_dimensions=Lattice_dimensions, delta=delta, rho=1, U=U, c_0=c_0)

    u_t, C_t= System.Simulate_BGK(num_iterations=3000, save_interval=save_interval)

    animate_C_t(C_t, fps=10, time_step_per_frame=save_interval)
    y_mid, x_subgrid = interpolate_y_mid(C_t, c_0=c_0, Lattice_dimensions=Lattice_dimensions, sublattice_presicion=5, tol=1e-5)

    

    Initialy_mid = jnp.average(y_mid[0])*jnp.ones_like(y_mid[0]) 
    y_mid_relative = y_mid - Initialy_mid[jnp.newaxis, :]
    

    
    
     # Plot initial and final mid-point

    plt.figure()
    plt.plot(x_subgrid, y_mid_relative[0], label='Initial')

    def update(frame):
        plt.clf()
        plt.plot(x_subgrid, y_mid_relative[frame], label=f'Time step {frame*10}')
        plt.xlabel('X')
        plt.ylabel('Y mid-point of C')
        plt.title('Mid-point of Scalar Field Over Time')
        plt.legend()
        return plt
    ani = animation.FuncAnimation(plt.gcf(), update, frames=range(0, u_t.shape[0]), blit=False, interval=100) # type: ignore
    ani.save('midpoint_evolution.gif', writer='pillow', fps=10)
    plt.show()

    # The interface perturbation is clasified by the amplitude growth over time.

    eta = jnp.max(y_mid_relative, axis=1) - jnp.min(y_mid_relative, axis=1)
    t = jnp.arange(u_t.shape[0])*save_interval  # time steps corresponding to saved frames
    kUt = jnp.copy(t)*k*U  # time steps corresponding to saved frames
    plt.plot(t[5:], jnp.log(eta[5:]), label=r'log(\eta)')
    plt.plot(t[5:], kUt[5:], label=r'Reference: k\Delta U_0 t')
    plt.xlabel('time [iterations]')
    plt.legend()
    plt.ylabel('Amplitude of Interface Perturbation')
    plt.title('Amplitude Growth Over Time')
    plt.savefig('amplitude_growth.pdf')
    plt.show()

    pass

def problem_6():
    Lattice_dimensions = jnp.array([200, 200])
    delta = 1
    System, u_0, C_0 = initialization_Helmholtz(Lattice_dimensions=Lattice_dimensions, delta=delta, rho=1, U=0.1)

    u_t, C_t = System.Simulate_BGK(num_iterations=12500, save_interval=25)



    S_dot = entropy_production(C_t)
    S_t = jnp.cumsum(S_dot)*25  # Cumulative entropy production
    t = jnp.arange(S_dot.shape[0])*25  # time steps corresponding to saved frames

    
    plt.plot(t, S_dot, label="Entropy Production S'/(k_b*D)")
    plt.xlabel('time [iterations]')
    plt.ylabel("Entropy Production S'/(k_b*D)")
    plt.title('Entropy Production Over Time')
    plt.savefig('entropy_production.pdf')
    plt.show()

    plt.plot(t, S_t, label='Cumulative Entropy Production')
    plt.legend()
    plt.xlabel('time [iterations]')
    plt.ylabel("Cumulative Entropy Production S/(k_b*D)")
    plt.title('Cumulative Entropy Production Over Time')
    plt.savefig('cumulative_entropy_production.pdf')
    plt.show()

    

    for index in [80, 220]:
        U = jnp.linalg.norm(u_t, axis=3)
    
        quiver_interval = 10    
        fig, ax = plt.subplots()
        X, Y = jnp.meshgrid(jnp.arange(u_t.shape[1]), jnp.arange(u_t.shape[2]), indexing='ij')
        quiver = ax.quiver(X[::quiver_interval, ::quiver_interval], Y[::quiver_interval, ::quiver_interval], u_t[index, ::quiver_interval, ::quiver_interval, 0], u_t[index, ::quiver_interval, ::quiver_interval, 1], color='k')
        Magnitude = ax.imshow(U[index].T[::-1, :], origin='lower', cmap='Reds', alpha=0.5, interpolation='bilinear')
        ax.set_title(f'Velocity Field at t = {index*25}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
        cb = fig.colorbar(Magnitude, ax=ax, label='Velocity Magnitude')
        plt.savefig(f'velocity_field_{index*25}.pdf')
        plt.show()

    pass

def problem_6_random():
    Lattice_dimensions = jnp.array([200, 200])
    delta = 1
    U = 0.1
    rho = 1.0
    c_0 = 1.0
    u_0y = 10e-3
    


    X,Y = jnp.meshgrid(jnp.arange(Lattice_dimensions[0]), jnp.arange(Lattice_dimensions[1]), indexing='ij')
    
    u_x = U*(jnp.tanh((Y - 0.25*Lattice_dimensions[0])/delta) - jnp.tanh((Y - 0.75*Lattice_dimensions[0])/delta) - 1)
    u_y = u_0y * jnp.array(np.random.uniform(-1, 1, size=(Lattice_dimensions[0], Lattice_dimensions[1])))
    
    
    x_hat = jnp.array([1.0, 0.0])
    y_hat = jnp.array([0.0, 1.0])
    u_0 = x_hat[jnp.newaxis, jnp.newaxis, :]*u_x[:, :, jnp.newaxis] + y_hat[jnp.newaxis, jnp.newaxis, :]*u_y[:, :, jnp.newaxis]
    
    
    C_0 = (c_0/2)*(jnp.tanh((Y - 0.25*Lattice_dimensions[1])/delta) - jnp.tanh((Y - 0.75*Lattice_dimensions[1])/delta))
    
    
    w = nu2w(nu=0.01)
    w_d = D2omega_d(D=0.01)
    System = BGK_D2Q9_Diffusion_Advection(lattice_dimensions=Lattice_dimensions, omega=w, omega_d=w_d)
    System.initialize_from_initial_conditions(u=u_0, C_init=C_0, rho=rho)
    u_t, C_t = System.Simulate_BGK(num_iterations=12500, save_interval=125)
    
    animate_u_t(u_t, fps=5, time_step_per_frame=125)
    
    t = jnp.arange(u_t.shape[0])*125  # time steps corresponding to saved frames
    S_dot = entropy_production(C_t) 
    S_t = jnp.cumsum(S_dot)*125  # Cumulative entropy production
    

    
    plt.plot(t, S_dot, label="Entropy Production S'/(k_b*D)")
    plt.xlabel('time [iterations]')
    plt.ylabel("Entropy Production S'/(k_b*D)")
    plt.title('Entropy Production Over Time')
    plt.savefig('entropy_production_random.pdf')
    plt.show()

    plt.plot(t, S_t, label='Cumulative Entropy Production')
    plt.legend()
    plt.xlabel('time [iterations]')
    plt.ylabel("Cumulative Entropy Production S/(k_b*D)")
    plt.title('Cumulative Entropy Production Over Time')
    plt.savefig('cumulative_entropy_production_random.pdf')
    plt.show()

    pass




# Main script
if __name__ == "__main__":
    problem_2()
    problem_4() 
    
    problem_6()
    #problem_6_random()

    
    pass
 