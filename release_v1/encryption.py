"""
Audio Encryption for MMSE Assessment
Implements AES-256 encryption for audio files.
"""

import os
import logging
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Optional


class AudioEncryption:
    """AES-256 encryption for audio files with key management."""
    
    def __init__(self, key: Optional[bytes] = None):
        """Initialize with optional key. If no key provided, generates new one."""
        self.key = key if key else AESGCM.generate_key(bit_length=256)
        self.aesgcm = AESGCM(self.key)
    
    @classmethod
    def from_password(cls, password: str, salt: bytes) -> 'AudioEncryption':
        """Create encryption instance from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits
            salt=salt,
            iterations=100000,  # NIST recommended minimum
        )
        key = kdf.derive(password.encode())
        return cls(key)
    
    def encrypt_file(self, input_path: str, output_path: str) -> bool:
        """Encrypt audio file using AES-256-GCM."""
        try:
            # Read input file
            with open(input_path, "rb") as f:
                data = f.read()
            
            # Generate random nonce (12 bytes for GCM)
            nonce = os.urandom(12)
            
            # Encrypt data
            ciphertext = self.aesgcm.encrypt(nonce, data, None)
            
            # Write nonce + ciphertext to output file
            with open(output_path, "wb") as f:
                f.write(nonce + ciphertext)
            
            logging.info(f"Successfully encrypted {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Encryption failed for {input_path}: {e}")
            return False
    
    def decrypt_file(self, input_path: str, output_path: str) -> bool:
        """Decrypt audio file."""
        try:
            # Read encrypted file
            with open(input_path, "rb") as f:
                encrypted_data = f.read()
            
            # Extract nonce and ciphertext
            nonce = encrypted_data[:12]
            ciphertext = encrypted_data[12:]
            
            # Decrypt data
            plaintext = self.aesgcm.decrypt(nonce, ciphertext, None)
            
            # Write decrypted data
            with open(output_path, "wb") as f:
                f.write(plaintext)
            
            logging.info(f"Successfully decrypted {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Decryption failed for {input_path}: {e}")
            return False
    
    def save_key(self, key_path: str) -> bool:
        """Save encryption key to file."""
        try:
            with open(key_path, "wb") as f:
                f.write(self.key)
            
            # Set restrictive permissions (owner read/write only)
            os.chmod(key_path, 0o600)
            
            logging.info(f"Encryption key saved to {key_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save key: {e}")
            return False
    
    @classmethod
    def load_key(cls, key_path: str) -> Optional['AudioEncryption']:
        """Load encryption key from file."""
        try:
            with open(key_path, "rb") as f:
                key = f.read()
            
            logging.info(f"Encryption key loaded from {key_path}")
            return cls(key)
            
        except Exception as e:
            logging.error(f"Failed to load key: {e}")
            return None
    
    def encrypt_dataset_audio(self, dataset_csv: str, audio_dir: str, 
                            encrypted_dir: str, key_file: str) -> bool:
        """Encrypt all audio files referenced in dataset."""
        import pandas as pd
        
        try:
            # Create encrypted directory
            Path(encrypted_dir).mkdir(parents=True, exist_ok=True)
            
            # Load dataset
            df = pd.read_csv(dataset_csv)
            
            if 'audio_path' not in df.columns:
                logging.error("Dataset CSV missing 'audio_path' column")
                return False
            
            success_count = 0
            total_count = len(df)
            
            for idx, row in df.iterrows():
                audio_path = row['audio_path']
                
                # Construct full path
                if not os.path.isabs(audio_path):
                    full_audio_path = os.path.join(audio_dir, audio_path)
                else:
                    full_audio_path = audio_path
                
                if not os.path.exists(full_audio_path):
                    logging.warning(f"Audio file not found: {full_audio_path}")
                    continue
                
                # Generate encrypted filename
                base_name = os.path.basename(audio_path)
                name, ext = os.path.splitext(base_name)
                encrypted_filename = f"{name}_encrypted{ext}.enc"
                encrypted_path = os.path.join(encrypted_dir, encrypted_filename)
                
                # Encrypt file
                if self.encrypt_file(full_audio_path, encrypted_path):
                    success_count += 1
            
            # Save encryption key
            if self.save_key(key_file):
                logging.info(f"Encrypted {success_count}/{total_count} audio files")
                logging.info(f"Encryption key saved to {key_file}")
                return True
            else:
                return False
                
        except Exception as e:
            logging.error(f"Dataset encryption failed: {e}")
            return False
    
    def get_key_info(self) -> dict:
        """Get information about the encryption key."""
        return {
            'algorithm': 'AES-256-GCM',
            'key_length_bits': len(self.key) * 8,
            'key_length_bytes': len(self.key),
            'key_hex': self.key.hex()[:16] + '...',  # Show only first 8 bytes for security
        }


def demonstrate_encryption():
    """Demonstrate encryption functionality."""
    # Generate new encryption instance
    encryptor = AudioEncryption()
    
    # Example usage
    print("Encryption Key Info:", encryptor.get_key_info())
    
    # Save key for later use
    encryptor.save_key("encryption_key.bin")
    
    # Load key from file
    loaded_encryptor = AudioEncryption.load_key("encryption_key.bin")
    
    if loaded_encryptor:
        print("Key successfully loaded from file")
    
    return encryptor


if __name__ == "__main__":
    # Demo
    encryptor = demonstrate_encryption()
