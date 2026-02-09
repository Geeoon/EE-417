import numpy as np

def truncate_add_noise_real(signal, snr):
    """
    Truncates, normalizes, and adds Gaussian noise to a signal.
    
    Parameters:
    signal (ndarray): The input signal array.
    snr (float): Target Signal-to-Noise Ratio in dB.
    
    Returns:
    ndarray: The processed noisy signal.
    """
    # 1. Normalize the signal by its maximum absolute value
    signal = signal / np.max(np.abs(signal))
    
    # 2. Truncate to the first 1,000,000 samples
    # (Using 1e6 as an integer for indexing)
    signal = signal[:int(1e6)]
    
    # 3. Create a reference signal of ones (as in the MATLAB code)
    # The reference is effectively already normalized since max(abs(ones)) is 1.
    ref_signal = np.ones(int(1e6))
    
    # 4. Generate Additive White Gaussian Noise (AWGN)
    # This replicates MATLAB's awgn(referenceSignal, SNR, "measured") logic
    # Calculate required noise power based on signal power and target SNR
    ref_power = np.mean(ref_signal**2)  # For ones, this is 1.0
    snr_linear = 10**(snr / 10.0)
    noise_power = ref_power / snr_linear
    
    # Generate Gaussian noise with calculated variance (noise_power)
    added_noise = np.random.normal(0, np.sqrt(noise_power), len(ref_signal))
    
    # 5. Add noise to the normalized/truncated signal
    noisy_signal = added_noise + signal
    
    return noisy_signal