import numpy as np
import pandas as pd
import mne
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class MuseEEGProcessor:
    def __init__(self, 
                 sampling_rate=256,  # Typical Muse sampling rate
                 low_cutoff=1.0,     # Low-frequency cutoff for bandpass filter
                 high_cutoff=50.0,   # High-frequency cutoff for bandpass filter
                 notch_freq=50.0):   # Notch frequency for power line interference
        """
        Initialize EEG data processor for Muse headset
        
        Parameters:
        -----------
        sampling_rate : int, optional (default=256)
            Sampling rate of the EEG data
        low_cutoff : float, optional (default=1.0)
            Low-frequency cutoff for bandpass filter (Hz)
        high_cutoff : float, optional (default=50.0)
            High-frequency cutoff for bandpass filter (Hz)
        notch_freq : float, optional (default=50.0)
            Frequency to filter out power line interference (Hz)
        """
        self.sampling_rate = sampling_rate
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.notch_freq = notch_freq
        
        # Muse electrode channels
        self.channels = ['TP9', 'AF7', 'AF8', 'TP10']
    
    def preprocess_data(self, raw_eeg_data):
        """
        Preprocess raw EEG data
        
        Parameters:
        -----------
        raw_eeg_data : numpy.ndarray
            Raw EEG data with shape (n_channels, n_samples)
        
        Returns:
        --------
        numpy.ndarray
            Preprocessed EEG data
        """
        # Validate input
        if not isinstance(raw_eeg_data, np.ndarray):
            raise ValueError("Input must be a NumPy array")
        
        # Create MNE Raw object
        info = mne.create_info(
            ch_names=self.channels, 
            sfreq=self.sampling_rate, 
            ch_types='eeg'
        )
        raw = mne.io.RawArray(raw_eeg_data, info)
        
        # Apply preprocessing steps
        
        # 1. Notch filter to remove power line interference
        raw = raw.notch_filter(freqs=self.notch_freq)
        
        # 2. Bandpass filter
        raw = raw.filter(
            l_freq=self.low_cutoff, 
            h_freq=self.high_cutoff
        )
        
        # 3. Common Average Reference (CAR)
        raw = raw.set_eeg_reference('average')
        
        # 4. Artifact removal (basic example using variance threshold)
        # This is a simple method and might need to be replaced 
        # with more sophisticated artifact rejection techniques
        epochs = mne.make_fixed_length_epochs(raw, duration=1.0)
        reject = dict(eeg=200e-6)  # 200 µV threshold
        epochs.drop_bad(reject=reject)
        
        # Convert back to numpy array
        preprocessed_data = epochs.get_data()
        
        return preprocessed_data
    
    def extract_features(self, preprocessed_data):
        """
        Extract features from preprocessed EEG data
        
        Parameters:
        -----------
        preprocessed_data : numpy.ndarray
            Preprocessed EEG data
        
        Returns:
        --------
        numpy.ndarray
            Extracted features
        """
        # Calculate frequency domain features
        features = []
        for epoch in preprocessed_data:
            epoch_features = []
            for channel in epoch:
                # Compute Power Spectral Density (PSD)
                f, psd = signal.welch(channel, fs=self.sampling_rate)
                
                # Extract features for different frequency bands
                delta = np.mean(psd[(f >= 1) & (f < 4)])    # Delta: 1-4 Hz
                theta = np.mean(psd[(f >= 4) & (f < 8)])    # Theta: 4-8 Hz
                alpha = np.mean(psd[(f >= 8) & (f < 13)])   # Alpha: 8-13 Hz
                beta = np.mean(psd[(f >= 13) & (f < 30)])   # Beta: 13-30 Hz
                gamma = np.mean(psd[(f >= 30) & (f < 50)])  # Gamma: 30-50 Hz
                
                epoch_features.extend([delta, theta, alpha, beta, gamma])
            
            features.append(epoch_features)
        
        return np.array(features)
    
    def prepare_for_ml(self, features):
        """
        Prepare features for machine learning model
        
        Parameters:
        -----------
        features : numpy.ndarray
            Extracted features
        
        Returns:
        --------
        numpy.ndarray
            Scaled features ready for ML model
        """
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        return scaled_features
    
    def visualize_preprocessing(self, raw_data, preprocessed_data):
        """
        Visualize raw and preprocessed EEG data
        
        Parameters:
        -----------
        raw_data : numpy.ndarray
            Original EEG data
        preprocessed_data : numpy.ndarray
            Preprocessed EEG data
        """
        # Plot raw vs preprocessed data for each channel
        fig, axs = plt.subplots(len(self.channels), 2, figsize=(15, 10))
        fig.suptitle('Raw vs Preprocessed EEG Data')
        
        for i, channel in enumerate(self.channels):
            # Raw data
            axs[i, 0].plot(raw_data[i, :500])
            axs[i, 0].set_title(f'Raw {channel}')
            
            # Preprocessed data (first epoch)
            axs[i, 1].plot(preprocessed_data[0, i, :])
            axs[i, 1].set_title(f'Preprocessed {channel}')
        
        plt.tight_layout()
        plt.show()

# Example usage
def main():
    # Simulated EEG data (replace with actual Muse data acquisition)
    np.random.seed(42)
    raw_eeg_data = np.random.randn(4, 1000)  # 4 channels, 1000 samples
    
    # Initialize processor
    processor = MuseEEGProcessor()
    
    try:
        # Preprocess data
        preprocessed_data = processor.preprocess_data(raw_eeg_data)
        
        # Visualize preprocessing
        processor.visualize_preprocessing(raw_eeg_data, preprocessed_data)
        
        # Extract features
        features = processor.extract_features(preprocessed_data)
        
        # Prepare for ML model
        ml_ready_features = processor.prepare_for_ml(features)
        
        print("Features shape:", ml_ready_features.shape)
        print("Features prepared for machine learning model")
    
    except Exception as e:
        print(f"An error occurred during EEG processing: {e}")

if __name__ == "__main__":
    main()

# Note: This script requires additional libraries:
# pip install numpy pandas mne scipy matplotlib scikit-learn

"""