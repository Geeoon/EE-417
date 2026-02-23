import numpy as np

def truncate_add_noise_passband(signal, SNR_dB):
    """
    Python equivalent of the MATLAB function truncate_add_noise_passband.

    Parameters
    ----------
    signal : np.ndarray (complex or real)
        Input signal.
    SNR_dB : float
        Desired SNR in dB (measured).

    Returns
    -------
    noisySignal : np.ndarray
        Signal with added complex AWGN.
    """

    # Normalize signal
    signal = signal / np.max(np.abs(signal))

    # Truncate to 1e6 samples
    N = int(1e6)
    signal = signal[:N]

    # Reference signal (complex ones)
    referenceSignal = np.ones(N, dtype=np.complex128) + 1j * np.ones(N)
    referenceSignal /= np.max(np.abs(referenceSignal))

    # --- AWGN with "measured" power ---
    signal_power = np.mean(np.abs(referenceSignal)**2)
    snr_linear = 10**(SNR_dB / 10)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(N) + 1j * np.random.randn(N)
    )

    added_noise = noise  # awgn(reference) - reference
    noisySignal = signal + added_noise

    return noisySignal
