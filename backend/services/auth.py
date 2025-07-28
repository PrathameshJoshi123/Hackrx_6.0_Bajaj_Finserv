from fastapi import Header, HTTPException
import os
from dotenv import load_dotenv
load_dotenv()

def verify_token(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format")
    
    token = authorization.split(" ")[1]

    if token != os.getenv("AUTH_TOKEN"):
        raise HTTPException(status_code=403, detail="Invalid token")
    
    return token