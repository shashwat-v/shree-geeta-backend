from fastapi import FastAPI, HTTPException, Header, Depends, status, Response, Cookie, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from db import engine, SessionLocal, Base
from models import User
from auth_utils import (
    create_access_token, verify_password, hash_password, 
    decode_access_token, get_current_user, create_refresh_token
)
from pydantic import BaseModel, ValidationError
from fastapi.security import HTTPBearer
from schemas import UserCreate, UserLogin, Token, AskRequest
from services.rag import answer_question, rag_chain
from sse_starlette import EventSourceResponse
import logging
from datetime import datetime
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Failed to create database tables: {e}")
    raise

app = FastAPI(
    title="Shree Geeta API",
    description="Production-ready API for Bhagavad Gita application",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:*",
        "http://10.0.2.2:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Response: {request.method} {request.url.path} "
                f"Status: {response.status_code} "
                f"Duration: {process_time:.3f}s")
    return response

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error", "errors": exc.errors()}
    )

@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    logger.error(f"Database error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Database error occurred"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

def get_db():
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error"
        )
    finally:
        db.close()

@app.get("/", tags=["Health"])
async def read_root():
    return {
        "status": "healthy",
        "message": "Shree Geeta API is running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    try:
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection failed"
        )
    
    return {
        "status": "healthy",
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"],
    summary="Register a new user"
)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    try:
        existing_email = db.query(User).filter(User.email == user.email).first()
        if existing_email:
            logger.warning(f"Registration attempt with existing email: {user.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already exists!"
            )
        
        existing_username = db.query(User).filter(User.username == user.username).first()
        if existing_username:
            logger.warning(f"Registration attempt with existing username: {user.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken!"
            )
        
        new_user = User(
            name=user.name,
            email=user.email,
            username=user.username,
            hashed_password=hash_password(user.password)
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"User registered successfully: {new_user.username}")
        return {
            "msg": "User created successfully!",
            "username": new_user.username,
            "id": new_user.id
        }
    
    except HTTPException:
        raise
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Database integrity error during registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists!"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error during registration: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@app.post(
    "/login",
    response_model=Token,
    tags=["Authentication"],
    summary="Login user"
)
async def login_user(
    login_data: UserLogin,
    response: Response,
    db: Session = Depends(get_db)
):
    try:
        user = db.query(User).filter(User.email == login_data.email).first()
        if not user:
            logger.warning(f"Login attempt with non-existent email: {login_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        if not verify_password(login_data.password, user.hashed_password):
            logger.warning(f"Failed login attempt for user: {user.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        access_token = create_access_token(data={"sub": user.username})
        refresh_token = create_refresh_token(data={"sub": user.username})
        
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=True,
            samesite="lax",
            secure=False,
            max_age=30 * 24 * 60 * 60
        )
        
        logger.info(f"User logged in successfully: {user.username}")
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "username": user.username
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during login: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@app.post("/refresh_token", tags=["Authentication"], summary="Refresh access token")
async def refresh_token(refresh_token: str = Cookie(None)):
    try:
        if refresh_token is None:
            logger.warning("Refresh token attempt without cookie")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No refresh token provided"
            )
        
        payload = decode_access_token(refresh_token)
        if payload is None:
            logger.warning("Invalid refresh token received")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        username = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        new_access_token = create_access_token(data={"sub": username})
        
        logger.info(f"Access token refreshed for user: {username}")
        return {
            "access_token": new_access_token,
            "token_type": "bearer"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during token refresh: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@app.post("/logout", tags=["Authentication"], summary="Logout user")
async def logout_user(
    response: Response,
    user: str = Depends(get_current_user)
):
    try:
        response.delete_cookie(key="refresh_token")
        logger.info(f"User logged out: {user}")
        return {"msg": "Logged out successfully"}
    except Exception as e:
        logger.error(f"Error during logout: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@app.get("/me", tags=["User"], summary="Get current user profile")
async def get_current_user_profile(
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        db_user = db.query(User).filter(User.username == user).first()
        if not db_user:
            logger.error(f"User not found in database: {user}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "id": db_user.id,
            "name": db_user.name,
            "email": db_user.email,
            "username": db_user.username
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user profile: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch profile"
        )

@app.post("/ask_stream", tags=["AI Chat"], summary="Ask question with streaming response")
async def ask_stream(
    body: AskRequest,
    user: str = Depends(get_current_user)
):
    try:
        question = body.question
        if not question or not question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        logger.info(f"User {user} asked: {question[:50]}...")
        
        async def event_generator():
            try:
                for chunk in rag_chain.stream(question):
                    if chunk.content:
                        yield {"data": chunk.content}
                        await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in RAG streaming: {e}", exc_info=True)
                yield {"data": "[Error: Failed to generate response]"}
        
        return EventSourceResponse(event_generator())
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask_stream: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process question"
        )

@app.get("/debug/users", tags=["Debug"], include_in_schema=False)
async def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return {
        "count": len(users),
        "users": [
            {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "name": user.name
            }
            for user in users
        ]
    }

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Shree Geeta API...")
    logger.info("API documentation available at: /docs")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Shree Geeta API...")
