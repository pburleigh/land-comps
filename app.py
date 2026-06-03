from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Optional, Dict, Any, Tuple
import math
import re
import requests
from datetime import datetime

from homeharvest import scrape_property

app = FastAPI()

Status = Literal["SOLD", "FOR_SALE"]


# -----------------------------
# Models
# -----------------------------

class Subject(BaseModel):
    lat: float
    lng: float
    acres: float
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # 2-letter, per your data
    zip: Optional[str] = None


class Filters(BaseModel):
    radius_miles: float = 10
    months_back: int = 12
    include_statuses: List[Status] = ["SOLD", "FOR_SALE"]
    max_candidates: int = 100
    return_top: int = 25
    acres_ratio_min: float = 0.5
    acres_ratio_max: float = 2.0


class LandCompsRequest(BaseModel):
    subject: Subject
    filters: Filters


class Comp(BaseModel):
    status: Status
    price: float
    date: Optional[str] = None  # YYYY-MM-DD

    acres: Optional[float] = None
    lot_sqft: Optional[float] = None
    price_per_acre: Optional[float] = None
    price_per_sqft: Optional[float] = None

    lat: Optional[float] = None
    lng: Optional[float] = None
    distance_miles: Optional[float] = None

    mls_id: Optional[str] = None
    address: Optional[str] = None
    url: Optional[str] = None
    source: str = "homeharvest"


class Summary(BaseModel):
    radius_miles: float
    months_back: int
    returned: int
    median_price_per_acre: Optional[float] = None
    median_price_per_sqft: Optional[float] = None


class LandCompsResponse(BaseModel):
    subject: Subject
    summary: Summary
    comps: List[Comp]


# -----------------------------
# Helper functions
# -----------------------------

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.7613
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def median(values: List[Optional[float]]) -> Optional[float]:
    vals = sorted([v for v in values if v is not None])
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 == 1 else (vals[mid - 1] + vals[mid]) / 2


def has_street_number(street: Optional[str]) -> bool:
    if not street:
        return False
    return re.search(r"\d", street) is not None


def is_usable_address(subject: Subject) -> bool:
    # Usable if:
    # - state exists
    # - at least one of city/zip exists
    # - street has a number (avoids "Main St" only)
    if not subject.state:
        return False
    if not (subject.city or subject.zip):
        return False
    if not has_street_number(subject.street):
        return False
    return True


def format_full_address(subject: Subject) -> str:
    street = (subject.street or "").strip()
    city = (subject.city or "").strip()
    state = (subject.state or "").strip()
    zip_code = (subject.zip or "").strip()

    parts: List[str] = []
    if street:
        parts.append(street)

    tail_parts: List[str] = []
    if city:
        tail_parts.append(f"{city},")
    if state:
        tail_parts.append(state)
    if zip_code:
        tail_parts.append(zip_code)

    tail = " ".join(tail_parts).strip().replace(" ,", ",")
    if tail:
        parts.append(tail)

    return ", ".join(parts) if parts else ""


def reverse_geocode_zip_or_city_state(lat: float, lng: float) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"format": "jsonv2", "lat": lat, "lon": lng, "zoom": 14, "addressdetails": 1}
    headers = {"User-Agent": "land-comps-service/1.0 (contact: your-email@example.com)"}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    addr = data.get("address", {}) or {}

    zip_code = addr.get("postcode")
    city = addr.get("city") or addr.get("town") or addr.get("village")
    state = addr.get("state")
    return zip_code, city, state


def build_location_string(subject: Subject) -> str:
    # 1) usable full address
    if is_usable_address(subject):
        return format_full_address(subject)

    # 2) zip from DB
    if subject.zip:
        return subject.zip.strip()

    # 3) reverse geocode -> zip first, else city/state
    zip_code, city, state = reverse_geocode_zip_or_city_state(subject.lat, subject.lng)
    if zip_code:
        return zip_code
    if city and state:
        return f"{city}, {state}"
    if state:
        return state

    # last resort
    return f"{subject.lat}, {subject.lng}"


def parse_date_to_yyyy_mm_dd(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, str):
        m = re.match(r"^(\d{4}-\d{2}-\d{2})", value)
        return m.group(1) if m else None

    if isinstance(value, datetime):
        return value.date().isoformat()

    return None


def normalize_status(raw: Dict[str, Any]) -> Status:
    # Prefer explicit hints
    s = (raw.get("status") or raw.get("mls_status") or raw.get("_fetch_listing_type") or "").lower()
    if "sold" in s:
        return "SOLD"
    if "for_sale" in s or "sale" in s or "active" in s or "listed" in s:
        return "FOR_SALE"
    # default to FOR_SALE if unclear
    return "FOR_SALE"


def normalize_comp(raw: Dict[str, Any], subject: Subject) -> Optional[Comp]:
    status = normalize_status(raw)

    if status == "SOLD":
        price = raw.get("sold_price") or raw.get("last_sold_price")
        dt = raw.get("last_sold_date")
    else:
        price = raw.get("list_price") or raw.get("list_price_min") or raw.get("list_price_max")
        dt = raw.get("list_date")

    if price is None:
        return None

    lot_sqft = raw.get("lot_sqft")
    lat = raw.get("latitude")
    lng = raw.get("longitude")

    url = raw.get("property_url") or raw.get("permalink")
    mls_id = raw.get("mls_id")

    address = raw.get("formatted_address")
    if not address:
        street = raw.get("street")
        city = raw.get("city")
        state = raw.get("state")
        zip_code = raw.get("zip_code")
        addr_parts = [p for p in [street, city, state, zip_code] if p]
        address = ", ".join(addr_parts) if addr_parts else None

    comp = Comp(
        status=status,
        price=float(price),
        date=parse_date_to_yyyy_mm_dd(dt),
        lot_sqft=float(lot_sqft) if lot_sqft else None,
        lat=float(lat) if lat is not None else None,
        lng=float(lng) if lng is not None else None,
        mls_id=str(mls_id) if mls_id is not None else None,
        address=address,
        url=url,
    )

    # Derived fields
    if comp.lot_sqft and comp.lot_sqft > 0:
        comp.acres = comp.lot_sqft / 43560.0
        comp.price_per_sqft = comp.price / comp.lot_sqft
        comp.price_per_acre = comp.price / comp.acres if comp.acres else None

    if comp.lat is not None and comp.lng is not None:
        comp.distance_miles = haversine_miles(subject.lat, subject.lng, comp.lat, comp.lng)

    return comp


# -----------------------------
# Endpoint
# -----------------------------

@app.post("/land-comps", response_model=LandCompsResponse)
def land_comps(req: LandCompsRequest):
    subject = req.subject
    filters = req.filters

    location_string = build_location_string(subject)
    past_days = int(filters.months_back * 30.4)  # approx months -> days

    # HomeHarvest expects listing_type as a single string in the installed version.
    properties_sold = scrape_property(
        location=location_string,
        listing_type="sold",
        property_type=["land", "farm"],
        past_days=past_days,
        limit=filters.max_candidates,
    )

    properties_for_sale = scrape_property(
        location=location_string,
        listing_type="for_sale",
        property_type=["land", "farm"],
        past_days=past_days,
        limit=filters.max_candidates,
    )

    rows_sold = (
        properties_sold.to_dict(orient="records")
        if hasattr(properties_sold, "to_dict")
        else list(properties_sold)
    )
    rows_for_sale = (
        properties_for_sale.to_dict(orient="records")
        if hasattr(properties_for_sale, "to_dict")
        else list(properties_for_sale)
    )

    for r in rows_sold:
        if isinstance(r, dict):
            r["_fetch_listing_type"] = "sold"
    for r in rows_for_sale:
        if isinstance(r, dict):
            r["_fetch_listing_type"] = "for_sale"

    rows = rows_sold + rows_for_sale

    comps: List[Comp] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue

        comp = normalize_comp(raw, subject)
        if comp is None:
            continue

        # Only include requested statuses
        if comp.status not in filters.include_statuses:
            continue

        # Distance filter (only if known)
        if comp.distance_miles is not None and comp.distance_miles > filters.radius_miles:
            continue

        # Acres similarity filter (only if comp acres known)
        if comp.acres is not None and subject.acres:
            if comp.acres < filters.acres_ratio_min * subject.acres:
                continue
            if comp.acres > filters.acres_ratio_max * subject.acres:
                continue

        comps.append(comp)

    # Ranking: prefer SOLD slightly
    def score(c: Comp) -> float:
        dist = c.distance_miles if c.distance_miles is not None else 9999.0
        acres_diff = abs((c.acres if c.acres is not None else subject.acres) - subject.acres)
        status_penalty = 0.0 if c.status == "SOLD" else 1.0
        return dist + 0.2 * acres_diff + status_penalty

    comps_sorted = sorted(comps, key=score)[: filters.return_top]

    summary = Summary(
        radius_miles=filters.radius_miles,
        months_back=filters.months_back,
        returned=len(comps_sorted),
        median_price_per_acre=median([c.price_per_acre for c in comps_sorted]),
        median_price_per_sqft=median([c.price_per_sqft for c in comps_sorted]),
    )

    return LandCompsResponse(subject=subject, summary=summary, comps=comps_sorted)
