import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def get_phasor(amplitude, frequency, duration, fs, phase_angle_degrees):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    radian_frequency = 2 * np.pi * frequency
    phase = radian_frequency * t + np.radians(phase_angle_degrees)
    return amplitude * np.exp(1j * phase)

def animate_phasor(phasor, amplitude, duration):

    """
    Function to animate a phasor and its conjugate in the complex plane with real and imaginary axes.

    Parameters:
        phasor (np.ndarray): Array of complex numbers representing the phasor.
        amplitude (float): Maximum amplitude of the phasor.
        duration (float): Total duration of the animation in seconds.
    """
    # Create the figure
    fig = plt.figure(figsize=(12, 5))

    # Time-domain plot
    ax1 = fig.add_subplot(1, 2, 1)
    t = np.linspace(0, duration, len(phasor), endpoint=False)
    ax1.plot(t, np.real(phasor), label='Real Component')
    ax1.plot(t, np.imag(phasor), label='Imaginary Component', linestyle='dashed')
    ax1.set_title('Waveform in Time Domain', pad=30)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    ax1.legend()

    # Polar plot
    ax2 = fig.add_subplot(1, 2, 2, polar=True)
    ax2.set_title('Animating Complex Numbers and Conjugates', pad=30)
    ax2.set_rlim(0, amplitude * 1.5)  # Adjust the amplitude range as needed

    # Add real and imaginary axes
    ax2.plot([0, np.pi], [0, amplitude * 1.5], color='black', linestyle='--', linewidth=0.8)  # Real axis
    ax2.plot([0, 2 * np.pi], [0, amplitude * 1.5], color='black', linestyle='--', linewidth=0.8)  # Real axis
    ax2.plot([np.pi / 2, 3 * np.pi / 2], [0, amplitude * 1.5], color='black', linestyle='--', linewidth=0.8)  # Imaginary axis
    ax2.plot([np.pi / 2, np.pi / 2], [0, amplitude * 1.5], color='black', linestyle='--', linewidth=0.8)  # Imaginary axis


    # Initialize line objects for the animation
    line_phasor, = ax2.plot([], [], marker='.', label='Phasor')
    line_conjugate, = ax2.plot([], [], marker='.', label='Conjugate')

    ax2.legend(loc='upper right')

    # Function to initialize the plot
    def init():
        line_phasor.set_data([], [])
        line_conjugate.set_data([], [])
        return line_phasor, line_conjugate

    # Function to update the plot for each frame
    def update(frame):
        z = phasor[frame]
        z_conj = np.conj(z)

        # Phasor
        r = np.abs(z)
        theta = np.angle(z)

        # Conjugate
        r_conj = np.abs(z_conj)
        theta_conj = np.angle(z_conj)

        line_phasor.set_data([0, theta], [0, r])
        line_conjugate.set_data([0, theta_conj], [0, r_conj])

        return line_phasor, line_conjugate

    # Create the animation
    total_frames = len(phasor)  # Total number of frames
    interval = (duration * 1000) / total_frames  # Interval in milliseconds
    return FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True, repeat=False)

def get_fft_data(phasor, nfft):
    
    # Calculate number of full sections
    sections = len(phasor) // nfft

    # Create strided view of the signal
    strided_signal = np.lib.stride_tricks.as_strided(
        phasor,
        shape=(sections, nfft),
        strides=(phasor.strides[0] * nfft, phasor.strides[0]),
        writeable=False
    )

    # Perform FFT along the second axis
    return np.fft.fft(strided_signal, n=nfft, axis=1)

def plot_average_fft(fft_data, complex_part):
    avg_fft_data = np.mean(fft_data, axis=0)
    plt.figure()
    plt.plot(avg_fft_data, label=f'Average {complex_part}')
    plt.xlabel('FFT Bin')
    plt.ylabel(f'{complex_part} in Frequency Domain')
    plt.legend()
    plt.show()

def exponential_fourier_series(signal, T, n_harmonics, t_reconstruct):
    """
    Calculate the exponential Fourier series coefficients and reconstruct the signal.
    
    Parameters:
        signal (np.ndarray): The input signal sampled over one period.
        T (float): The period of the signal.
        n_harmonics (int): Number of harmonics to compute on either side of zero (e.g., -N to N).
        t_reconstruct (np.ndarray): Time points for signal reconstruction.
    
    Returns:
        c_n (np.ndarray): Fourier series coefficients.
        reconstructed_signal (np.ndarray): The reconstructed signal at t_reconstruct.
    """
    # Discretize time for one period
    N = len(signal)  # Number of samples in the signal
    dt = T / N       # Sampling interval
    t = np.linspace(0, T, N, endpoint=False)
    
    # Fundamental angular frequency
    omega_0 = 2 * np.pi / T
    
    # Calculate Fourier coefficients
    c_n = np.zeros(2 * n_harmonics + 1, dtype=np.complex128)
    for n in range(-n_harmonics, n_harmonics + 1):
        c_n[n + n_harmonics] = (1 / T) * np.sum(signal * np.exp(-1j * n * omega_0 * t) * dt)
    
    # Reconstruct the signal
    reconstructed_signal = np.zeros_like(t_reconstruct, dtype=np.complex128)
    for n in range(-n_harmonics, n_harmonics + 1):
        reconstructed_signal += c_n[n + n_harmonics] * np.exp(1j * n * omega_0 * t_reconstruct)
    
    return c_n, reconstructed_signal.real

def get_fourier_series_approximation(f, x, L, r=None, period_start=None, verbose=True):
    """
    Compute the Fourier series approximation of a periodic function.

    Parameters:
        f (np.ndarray): Signal samples (must be periodic with period L).
        x (np.ndarray): Sample points (must be uniformly spaced).
        L (float): Period of the function.
        r (int, optional): Number of harmonics. If None, set to Nyquist limit.
        period_start (float, optional): Start of the period in x. If None, auto-detects.
        verbose (bool): If True, print Fourier coefficients during calculation.

    Returns:
        fFS (np.ndarray): Fourier series approximation over all x.
    """
    x = np.asarray(x)
    f = np.asarray(f)

    # Check for uniform sampling
    dxs = np.diff(x)
    if not np.allclose(dxs, dxs[0], rtol=1e-6, atol=1e-10):
        raise ValueError("x must be uniformly sampled.")

    dx = dxs[0]

    # Auto-detect period start if not provided
    if period_start is None:
        x0 = x[0]
        period_mask = (x >= x0) & (x < x0 + L)
        if np.sum(period_mask) < 2:
            raise ValueError("Not enough samples for one period.")
        x_period = x[period_mask]
        f_period = f[period_mask]
    else:
        period_mask = (x >= period_start) & (x < period_start + L)
        if np.sum(period_mask) < 2:
            raise ValueError("Not enough samples for one period at the specified start.")
        x_period = x[period_mask]
        f_period = f[period_mask]

    dx = x_period[1] - x_period[0]

    # Set number of harmonics if not provided
    if r is None:
        r = int(L / (2 * dx)) - 1

    if verbose:
        print(f"Using r = {r} harmonics")

    # DC term
    A0 = np.sum(f_period) * dx / L
    fFS = A0 * np.ones_like(f)

    if verbose:
        print(f"A0 (DC term) = {A0:.5f}")

    # Harmonics
    for k in range(1, r + 1):
        Ak = np.sum(f_period * np.cos(2 * np.pi * k * x_period / L)) * dx * 2 / L
        Bk = np.sum(f_period * np.sin(2 * np.pi * k * x_period / L)) * dx * 2 / L
        fFS += Ak * np.cos(2 * np.pi * k * x / L) + Bk * np.sin(2 * np.pi * k * x / L)

        # Print selectively
        if verbose:
            if k <= 10 or k % 50 == 0 or k == r:
                print(f"k={k:3d}: Ak = {Ak:+.5f}, Bk = {Bk:+.5f}")

    return fFS
