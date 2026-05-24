"""
Timezone helper locked to Asia/Bangkok

This module ensures that all datetimes and dates generated across servers
consistently use the Asia/Bangkok local timezone, preventing mismatches
when deployed to servers in other regions.
"""
from datetime import date, datetime
from zoneinfo import ZoneInfo

BANGKOK_TZ = ZoneInfo("Asia/Bangkok")


def get_local_now() -> datetime:
    """ดึงเวลาปัจจุบันพร้อม timezone Asia/Bangkok"""
    return datetime.now(BANGKOK_TZ)


def get_local_today() -> date:
    """ดึงวันที่ปฏิทินปัจจุบันใน timezone Asia/Bangkok"""
    return get_local_now().date()
