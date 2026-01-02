import os
import jwt
import json
import requests
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

# Initialize security scheme
security = HTTPBearer()

# --- CONFIGURATION ---
CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")

@lru_cache()
def get_jwks():
    """
    Fetches and caches the JSON Web Key Set (JWKS) from Clerk.
    """
    if not CLERK_JWKS_URL:
        print("⚠️ WARNING: CLERK_JWKS_URL not set in .env. Auth may fail.")
        
    try:
        url = CLERK_JWKS_URL or "https://api.clerk.com/v1/jwks"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Failed to fetch JWKS keys: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Authentication Error")

def verify_clerk_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """
    Verifies the JWT token sent in the Authorization header.
    Returns the user's claims (ID, email, etc.) if valid.
    """
    token = credentials.credentials
    
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        
        jwks = get_jwks()
        
        rsa_key = {}
        for key in jwks["keys"]:
            if key["kid"] == kid:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
                break
        
        if not rsa_key:
            raise HTTPException(status_code=401, detail="Invalid token signature (Key ID not found)")

        payload = jwt.decode(
            token,
            jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(rsa_key)),
            algorithms=["RS256"],
            options={"verify_aud": False} 
        )
        
        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Please login again.")
    except jwt.JWTClaimsError:
        raise HTTPException(status_code=401, detail="Invalid token claims.")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Could not validate credentials")