import base64
import json
import os
import urllib.parse
from logging import getLogger
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

logger = getLogger("aetherforge.sync.crypto")

class SyncCrypto:
    """
    Handles End-to-End Encryption (E2EE) for AetherForge multi-node syncing.
    Uses AES-256-GCM for authenticated encryption.
    """
    
    @staticmethod
    def generate_key() -> str:
        """Generates a secure, random 256-bit AES key, encoded as base64."""
        key_bytes = AESGCM.generate_key(bit_length=256)
        return base64.urlsafe_b64encode(key_bytes).decode('utf-8')

    @staticmethod
    def encrypt_payload(payload: dict[str, Any], key_b64: str) -> str:
        """
        Encrypts a JSON dictionary into a base64 string using AES-GCM.
        Generates a 96-bit (12-byte) random nonce per encryption.
        """
        key = base64.urlsafe_b64decode(key_b64)
        aesgcm = AESGCM(key)
        
        nonce = os.urandom(12)
        plaintext = json.dumps(payload).encode('utf-8')
        
        # Authenticated encryption
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)
        
        # Prepend nonce to ciphertext for the decryptor and base64 encode
        combined = nonce + ciphertext
        return base64.urlsafe_b64encode(combined).decode('utf-8')

    @staticmethod
    def decrypt_payload(encrypted_b64: str, key_b64: str) -> dict[str, Any]:
        """
        Decrypts a base64 payload back into a dictionary using AES-GCM.
        Throws ValueError on tampered data or wrong key (InvalidTag).
        """
        key = base64.urlsafe_b64decode(key_b64)
        aesgcm = AESGCM(key)
        
        combined = base64.urlsafe_b64decode(encrypted_b64)
        nonce = combined[:12]
        ciphertext = combined[12:]
        
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
            return json.loads(plaintext.decode('utf-8'))
        except InvalidTag:
            logger.error("E2EE Decryption failed: Invalid tag (wrong key or tampered data)")
            raise ValueError("Decryption failed due to invalid tag or key.")
        except Exception as e:
            logger.error(f"E2EE Decryption error: {e}")
            raise ValueError(f"Decryption error: {e}")

    @staticmethod
    def create_pairing_uri(ip: str, port: int, node_id: str, key_b64: str) -> str:
        """
        Constructs the AetherForge specific URI format for QR codes.
        Format: aetherforge://sync?ip=...&port=...&node=...&key=...
        """
        params = {
            "ip": ip,
            "port": str(port),
            "node": node_id,
            "key": key_b64
        }
        query = urllib.parse.urlencode(params)
        return f"aetherforge://sync?{query}"

    @staticmethod
    def parse_pairing_uri(uri: str) -> dict[str, str | int]:
        """
        Parses a scanned QR URI back into connection parameters.
        Throws ValueError if format is invalid.
        """
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "aetherforge" or parsed.netloc != "sync":
            raise ValueError("Invalid pairing URI schema")
            
        qs = urllib.parse.parse_qs(parsed.query)
        try:
            return {
                "ip": qs["ip"][0],
                "port": int(qs["port"][0]),
                "node_id": qs["node"][0],
                "key": qs["key"][0]
            }
        except KeyError as e:
            raise ValueError(f"Missing required parameter in URI: {e}")
