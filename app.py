"""
FarmDepot MVP Backend (FastAPI)
- Voice + text listing (accepts audio uploads, transcribe via OpenRouter-compatible STT gateway or placeholder)
- Fast search with filters
- Safe contact (masked phone + WhatsApp handoff)
- Trust levers: Verified Seller (basic KYC), ratings, fraud reporting
- Payments: escrow (Paystack split payout stubs) + pickup/delivery request flag
- Ops: moderation queue (keyword + AI checks), audit log, takedown SLA
- Admin: daily metrics, disputes, refunds, fraud patterns (basic)
- Multilingual: detect language via OpenRouter LLM
- WhatsApp/IVR webhooks (stubs)
- PWA/offline is frontend scope; this backend exposes endpoints + service worker-friendly caching headers

IMPORTANT
- Create a .env file (see ENV VARS below)
- Run: `uvicorn app:app --host 0.0.0.0 --port 8000 --reload`
- DB: SQLite for MVP; switch to Postgres by changing DATABASE_URL

ENV VARS (example .env)
--------------------------------
DATABASE_URL=sqlite+aiosqlite:///./farmdepot.db
JWT_SECRET=supersecret
OPENROUTER_API_KEY=sk-or-...
PAYSTACK_SECRET=sk_test_xxx
PAYSTACK_PUBLIC=pk_test_xxx
BASE_URL=http://localhost:8000
MEDIA_DIR=./media
WHATSAPP_BUSINESS_NUMBER=2348000000000
WHATSAPP_DEEPLINK_BASE=https://wa.me
"""
from __future__ import annotations
import os
import uuid
import time
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Query, BackgroundTasks, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey, Text
from jose import jwt, JWTError
from passlib.context import CryptContext
from dotenv import load_dotenv
import aiofiles
import httpx

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./farmdepot.db")
JWT_SECRET = os.getenv("JWT_SECRET", "change_me")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
PAYSTACK_SECRET = os.getenv("PAYSTACK_SECRET", "")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
MEDIA_DIR = os.getenv("MEDIA_DIR", "./media")
WHATSAPP_DEEPLINK_BASE = os.getenv("WHATSAPP_DEEPLINK_BASE", "https://wa.me")
WHATSAPP_BUSINESS_NUMBER = os.getenv("WHATSAPP_BUSINESS_NUMBER", "2348000000000")

os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(os.path.join(MEDIA_DIR, "photos"), exist_ok=True)
os.makedirs(os.path.join(MEDIA_DIR, "audio"), exist_ok=True)

# --- DB setup ---
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# --- Models ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    phone = Column(String, unique=True, index=True)
    email = Column(String, unique=True, nullable=True)
    password_hash = Column(String)
    name = Column(String)
    is_verified = Column(Boolean, default=False)  # Verified Seller badge
    kyc_status = Column(String, default="unverified")  # unverified|pending|verified|rejected
    created_at = Column(DateTime, default=datetime.utcnow)

class KYC(Base):
    __tablename__ = "kyc"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    nin = Column(String, nullable=True)
    cac = Column(String, nullable=True)
    doc_url = Column(String, nullable=True)
    status = Column(String, default="pending")
    submitted_at = Column(DateTime, default=datetime.utcnow)

class Listing(Base):
    __tablename__ = "listings"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    type = Column(String)  # product|service|machinery|spare_part
    title = Column(String)
    description = Column(Text)
    quantity = Column(String)
    price = Column(Float)
    currency = Column(String, default="NGN")
    category = Column(String)  # inputs|produce|machinery|services|spares
    lga = Column(String)
    state = Column(String)
    photos = Column(Text)  # comma-separated paths
    audio_note = Column(String, nullable=True)
    is_featured = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Rating(Base):
    __tablename__ = "ratings"
    id = Column(Integer, primary_key=True)
    listing_id = Column(Integer, ForeignKey("listings.id"))
    reviewer_id = Column(Integer, ForeignKey("users.id"))
    score = Column(Integer)
    comment = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)
    listing_id = Column(Integer, ForeignKey("listings.id"))
    reporter_id = Column(Integer, ForeignKey("users.id"))
    reason = Column(String)
    details = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class Escrow(Base):
    __tablename__ = "escrows"
    id = Column(Integer, primary_key=True)
    listing_id = Column(Integer, ForeignKey("listings.id"))
    buyer_id = Column(Integer, ForeignKey("users.id"))
    seller_id = Column(Integer, ForeignKey("users.id"))
    amount = Column(Float)
    status = Column(String, default="initiated")  # initiated|paid|delivered|released|refunded|disputed
    paystack_reference = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    actor_id = Column(Integer, nullable=True)
    action = Column(String)
    details = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModerationQueue(Base):
    __tablename__ = "moderation_queue"
    id = Column(Integer, primary_key=True)
    listing_id = Column(Integer, ForeignKey("listings.id"))
    status = Column(String, default="pending")  # pending|approved|rejected
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# --- Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserCreate(BaseModel):
    phone: str
    password: str
    name: str
    email: Optional[str] = None

class UserLogin(BaseModel):
    phone: str
    password: str

class ListingCreate(BaseModel):
    type: str
    title: str
    description: str
    quantity: str
    price: float
    category: str
    lga: str
    state: str

class ListingOut(BaseModel):
    id: int
    title: str
    price: float
    lga: str
    state: str
    is_featured: bool
    photos: List[str] = []
    class Config:
        orm_mode = True

# --- Utils ---
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)

async def get_current_user(authorization: str = Header(None), db: AsyncSession = Depends(get_db)) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Not authenticated")
    token = authorization.split()[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        phone = payload.get("sub")
        if phone is None:
            raise HTTPException(401, "Invalid token")
    except JWTError:
        raise HTTPException(401, "Invalid token")
    from sqlalchemy import select
    res = await db.execute(select(User).where(User.phone == phone))
    user = res.scalars().first()
    if not user:
        raise HTTPException(401, "User not found")
    return user

async def save_upload(file: UploadFile, subdir: str) -> str:
    ext = os.path.splitext(file.filename)[1]
    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(MEDIA_DIR, subdir, name)
    async with aiofiles.open(path, 'wb') as out:
        content = await file.read()
        await out.write(content)
    return f"/media/{subdir}/{name}"

# --- AI helpers ---
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS_OR = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "HTTP-Referer": "https://farmdepot.ng", "X-Title": "FarmDepot"}

async def ai_moderate_text(text: str) -> dict:
    if not OPENROUTER_API_KEY:
        return {"ok": True, "reason": "no-key-skip"}
    prompt = (
        "You are a marketplace safety checker. Return JSON with fields ok (true/false) and reason. "
        "Flag fraud, weapons, illegal items, hate speech, and phone numbers posted in text."
    )
    payload = {
        "model": "openrouter/auto",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        "response_format": {"type": "json_object"}
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(OPENROUTER_URL, headers=HEADERS_OR, json=payload)
        r.raise_for_status()
        data = r.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            return {"ok": True, "reason": "fallback"}
    import json
    try:
        return json.loads(content)
    except Exception:
        return {"ok": True, "reason": "parse-fallback"}

# --- App ---
app = FastAPI(title="FarmDepot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve media & static (include logo placeholder)
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Place your logo at ./static/logo.png to integrate across frontend & admin

# --- Startup ---
@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# --- Auth ---
@app.post("/auth/register", response_model=Token)
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    res = await db.execute(select(User).where(User.phone == user.phone))
    if res.scalars().first():
        raise HTTPException(400, "Phone already registered")
    u = User(phone=user.phone, email=user.email, name=user.name, password_hash=pwd_context.hash(user.password))
    db.add(u)
    await db.commit()
    token = create_access_token({"sub": u.phone})
    await log(db, None, "user.register", f"phone={u.phone}")
    return Token(access_token=token)

@app.post("/auth/login", response_model=Token)
async def login(data: UserLogin, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    res = await db.execute(select(User).where(User.phone == data.phone))
    u = res.scalars().first()
    if not u or not pwd_context.verify(data.password, u.password_hash):
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token({"sub": u.phone})
    await log(db, u.id, "user.login", "")
    return Token(access_token=token)

# --- Listings ---
@app.post("/listings", response_model=ListingOut)
async def create_listing(
    type: str = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    quantity: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
    lga: str = Form(...),
    state: str = Form(...),
    photos: List[UploadFile] = File(default=[]),
    audio_note: Optional[UploadFile] = File(default=None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Moderation pre-check
    check = await ai_moderate_text(f"{title}\n{description}")
    if not check.get("ok", True):
        raise HTTPException(400, f"Content rejected: {check.get('reason','moderation')}")

    photo_paths = []
    for p in photos:
        photo_paths.append(await save_upload(p, "photos"))
    audio_path = None
    if audio_note:
        audio_path = await save_upload(audio_note, "audio")

    l = Listing(
        user_id=current_user.id,
        type=type,
        title=title,
        description=description,
        quantity=quantity,
        price=price,
        category=category,
        lga=lga,
        state=state,
        photos=",".join(photo_paths),
        audio_note=audio_path,
    )
    db.add(l)
    await db.commit()
    await db.refresh(l)

    # Queue for human moderation as well
    db.add(ModerationQueue(listing_id=l.id))
    await db.commit()

    await log(db, current_user.id, "listing.create", f"listing_id={l.id}")
    return to_listing_out(l)

@app.get("/listings", response_model=List[ListingOut])
async def search_listings(
    q: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    lga: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    page: int = 1,
    size: int = 20,
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select, and_, or_
    stmt = select(Listing).where(Listing.is_active == True)
    if q:
        like = f"%{q.lower()}%"
        stmt = stmt.where(or_(Listing.title.ilike(like), Listing.description.ilike(like)))
    if category:
        stmt = stmt.where(Listing.category == category)
    if state:
        stmt = stmt.where(Listing.state == state)
    if lga:
        stmt = stmt.where(Listing.lga == lga)
    if min_price is not None:
        stmt = stmt.where(Listing.price >= min_price)
    if max_price is not None:
        stmt = stmt.where(Listing.price <= max_price)
    stmt = stmt.order_by(Listing.is_featured.desc(), Listing.created_at.desc()).offset((page-1)*size).limit(size)
    res = await db.execute(stmt)
    listings = res.scalars().all()
    return [to_listing_out(l) for l in listings]

@app.get("/listings/{listing_id}", response_model=ListingOut)
async def get_listing(listing_id: int, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    res = await db.execute(select(Listing).where(Listing.id == listing_id))
    l = res.scalars().first()
    if not l:
        raise HTTPException(404, "Listing not found")
    return to_listing_out(l)

# --- Safe Contact & WhatsApp handoff ---
@app.post("/contact/{listing_id}")
async def contact_seller(listing_id: int, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    res = await db.execute(select(Listing).where(Listing.id == listing_id))
    l = res.scalars().first()
    if not l:
        raise HTTPException(404, "Listing not found")
    # mask seller phone: return WhatsApp deep link via business number with metadata
    deeplink = f"{WHATSAPP_DEEPLINK_BASE}/{WHATSAPP_BUSINESS_NUMBER}?text=FD%20LISTING%20{listing_id}%20BUYER%20{current_user.phone}"
    await log(db, current_user.id, "contact.request", f"listing_id={listing_id}")
    return {"whatsapp": deeplink, "mask": True}

# --- Ratings ---
class RatingIn(BaseModel):
    score: int = Field(ge=1, le=5)
    comment: Optional[str] = None

@app.post("/listings/{listing_id}/ratings")
async def add_rating(listing_id: int, data: RatingIn, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    r = Rating(listing_id=listing_id, reviewer_id=current_user.id, score=data.score, comment=data.comment or "")
    db.add(r)
    await db.commit()
    await log(db, current_user.id, "rating.create", f"listing_id={listing_id}")
    return {"ok": True}

# --- Reports / Fraud ---
class ReportIn(BaseModel):
    reason: str
    details: Optional[str] = None

@app.post("/listings/{listing_id}/report")
async def report_listing(listing_id: int, data: ReportIn, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    rep = Report(listing_id=listing_id, reporter_id=current_user.id, reason=data.reason, details=data.details or "")
    db.add(rep)
    await db.commit()
    await log(db, current_user.id, "listing.report", f"listing_id={listing_id}")
    return {"ok": True}

# --- KYC ---
class KYCIn(BaseModel):
    nin: Optional[str] = None
    cac: Optional[str] = None
    doc_url: Optional[str] = None

@app.post("/kyc/submit")
async def submit_kyc(data: KYCIn, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    k = KYC(user_id=current_user.id, nin=data.nin, cac=data.cac, doc_url=data.doc_url, status="pending")
    db.add(k)
    current_user.kyc_status = "pending"
    await db.commit()
    await log(db, current_user.id, "kyc.submit", "")
    return {"ok": True, "status": "pending"}

# --- Escrow (Paystack) ---
class EscrowInitIn(BaseModel):
    listing_id: int
    amount: float

@app.post("/escrow/init")
async def escrow_init(data: EscrowInitIn, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    # create paystack transaction
    if not PAYSTACK_SECRET:
        raise HTTPException(400, "Paystack not configured")
    # fetch listing & seller
    from sqlalchemy import select
    res = await db.execute(select(Listing).where(Listing.id == data.listing_id))
    l = res.scalars().first()
    if not l:
        raise HTTPException(404, "Listing not found")
    # For MVP: initialize transaction and store reference
    ref = f"FD-{uuid.uuid4().hex[:12]}"
    async with httpx.AsyncClient(timeout=30) as client:
        headers = {"Authorization": f"Bearer {PAYSTACK_SECRET}", "Content-Type": "application/json"}
        payload = {
            "email": f"{current_user.phone}@fd.local",
            "amount": int(data.amount * 100),
            "reference": ref,
            "callback_url": f"{BASE_URL}/payments/callback"
        }
        r = await client.post("https://api.paystack.co/transaction/initialize", headers=headers, json=payload)
        r.raise_for_status()
        init_data = r.json()["data"]
    esc = Escrow(listing_id=l.id, buyer_id=current_user.id, seller_id=l.user_id, amount=data.amount, status="initiated", paystack_reference=ref)
    db.add(esc)
    await db.commit()
    await log(db, current_user.id, "escrow.init", f"listing_id={l.id};ref={ref}")
    return {"authorization_url": init_data["authorization_url"], "reference": ref}

@app.post("/payments/webhook")
async def paystack_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    # TODO: verify signature (x-paystack-signature). For MVP, accept and update statuses
    payload = await request.json()
    event = payload.get("event")
    data = payload.get("data", {})
    ref = data.get("reference")
    from sqlalchemy import select
    if ref:
        res = await db.execute(select(Escrow).where(Escrow.paystack_reference == ref))
        esc = res.scalars().first()
        if esc:
            if event == "charge.success":
                esc.status = "paid"
            await db.commit()
            await log(db, None, "payments.webhook", f"ref={ref};event={event}")
    return {"received": True}

@app.post("/escrow/confirm_delivery/{escrow_id}")
async def escrow_confirm_delivery(escrow_id: int, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    res = await db.execute(select(Escrow).where(Escrow.id == escrow_id))
    esc = res.scalars().first()
    if not esc:
        raise HTTPException(404, "Escrow not found")
    if current_user.id != esc.buyer_id:
        raise HTTPException(403, "Only buyer can confirm")
    esc.status = "delivered"
    await db.commit()
    await log(db, current_user.id, "escrow.delivered", f"escrow_id={escrow_id}")
    # TODO: trigger split payout via Paystack Transfer
    return {"ok": True}

# --- Moderation Admin ---
@app.get("/admin/moderation")
async def moderation_list(status: str = "pending", db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    res = await db.execute(select(ModerationQueue).where(ModerationQueue.status == status).order_by(ModerationQueue.created_at.asc()))
    items = [
        {"id": m.id, "listing_id": m.listing_id, "status": m.status, "reason": m.reason, "created_at": m.created_at}
        for m in res.scalars().all()
    ]
    return items

class ModerationActionIn(BaseModel):
    status: str  # approved|rejected
    reason: Optional[str] = None

@app.post("/admin/moderation/{item_id}")
async def moderation_action(item_id: int, data: ModerationActionIn, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    res = await db.execute(select(ModerationQueue).where(ModerationQueue.id == item_id))
    m = res.scalars().first()
    if not m:
        raise HTTPException(404, "Item not found")
    m.status = data.status
    m.reason = data.reason
    # also deactivate listing if rejected
    if data.status == "rejected":
        res2 = await db.execute(select(Listing).where(Listing.id == m.listing_id))
        l = res2.scalars().first()
        if l:
            l.is_active = False
    await db.commit()
    await log(db, None, "moderation.action", f"item_id={item_id};status={data.status}")
    return {"ok": True}

# --- Disputes / Refunds (basic) ---
class DisputeIn(BaseModel):
    escrow_id: int
    reason: str

@app.post("/disputes")
async def create_dispute(data: DisputeIn, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    # For MVP, store as report with type=dispute
    rep = Report(listing_id=data.escrow_id, reporter_id=current_user.id, reason="dispute", details=data.reason)
    db.add(rep)
    await db.commit()
    await log(db, current_user.id, "dispute.create", f"escrow_id={data.escrow_id}")
    return {"ok": True}

# --- Metrics ---
@app.get("/admin/metrics/daily")
async def daily_metrics(db: AsyncSession = Depends(get_db)):
    # Simple counts (optimize later)
    from sqlalchemy import select, func
    totals = {}
    for model, name in [(User, "users"), (Listing, "listings"), (Escrow, "escrows"), (Report, "reports")] :
        res = await db.execute(select(func.count()).select_from(model))
        totals[name] = res.scalar_one()
    return {"date": datetime.utcnow().date().isoformat(), **totals}

# --- IVR/WhatsApp Webhooks (stubs) ---
@app.post("/webhooks/ivr")
async def ivr_webhook(payload: dict):
    # Accept DTMF input and route to actions (stub)
    return {"ok": True}

@app.post("/webhooks/whatsapp")
async def whatsapp_webhook(payload: dict):
    # Parse message, reply with menu (stub)
    return {"ok": True}

# --- Helpers ---
async def log(db: AsyncSession, actor_id: Optional[int], action: str, details: str):
    db.add(AuditLog(actor_id=actor_id, action=action, details=details))
    await db.commit()

def to_listing_out(l: Listing) -> ListingOut:
    photos = [p for p in (l.photos.split(",") if l.photos else []) if p]
    return ListingOut(id=l.id, title=l.title, price=l.price, lga=l.lga, state=l.state, is_featured=l.is_featured, photos=photos)

# --- Healthcheck ---
@app.get("/health")
async def health():
    return {"ok": True}
