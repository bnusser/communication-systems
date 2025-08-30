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

# def get_fft_data(phasor, nfft):
    
#     sections = len(phasor) // nfft

#     strided_signal = np.lib.stride_tricks.as_strided(
#         phasor,
#         shape=(sections, nfft),
#         strides=(phasor.strides[0] * nfft, phasor.strides[0])
#     ).astype(np.complex128)

#     return np.fft.fft(strided_signal, n=nfft, axis=1)

def get_fft_data(phasor, nfft):
    
    # Calculate number of full sections
    sections = len(phasor) // nfft

    # Create strided view of the signal
    strided_signal = np.lib.stride_tricks.as_strided(
        phasor,
        shape=(sections, nfft),
        strides=(phasor.strides[0] * nfft, phasor.strides[0])
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
