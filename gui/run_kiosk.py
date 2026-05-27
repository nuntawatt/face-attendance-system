# ruff: noqa: E402
"""
Face Attendance Kiosk — OpenCV Desktop Application

คำสั่ง: R=ลงทะเบียน  I=เช็คอิน  O=เช็คเอาท์  Q/ESC=ปิด
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import asyncio, math, time, uuid  # noqa: E401
from datetime import datetime

import cv2
import numpy as np

from app.ai.engine import crop_and_encode_face, face_engine
from app.ai.recognition import embedding_index
from app.core.config import settings
from app.database.session import async_session_factory
from app.models.employee import Employee
from app.models.face_embedding import FaceEmbedding
from app.repositories.employee import EmployeeRepository
from app.repositories.face_embedding import FaceEmbeddingRepository
from app.services.embedding_cache_service import EmbeddingCacheService
from app.services.minio_service import minio_service

# ── Color Palette (BGR) ──────────────────────────────────────────────
C_ACCENT     = (220, 200, 20)    # Teal สีหลัก
C_ACCENT_DIM = (120, 110, 10)    # Teal หม่น
C_SUCCESS    = (90, 215, 80)     # เขียว (เช็คอิน)
C_WARNING    = (30, 175, 255)    # ส้ม (เช็คเอาท์)
C_DANGER     = (70, 75, 235)     # แดง (ไม่พบ)
C_INFO       = (230, 165, 55)    # ฟ้า (กำลังสแกน)
C_WHITE      = (240, 240, 240)
C_GRAY       = (155, 155, 155)
C_DIM        = (90, 90, 90)
C_BG         = (28, 25, 22)

# ── Constants ────────────────────────────────────────────────────────
MIN_DET_SCORE       = 0.35
MIN_REGISTER_SCORE  = 0.5
CHECKOUT_WAIT_SEC   = 600        # 10 นาที
CAMERA_W, CAMERA_H  = 1280, 720
HEADER_H, FOOTER_H  = 62, 42
MSG_DURATION         = {True: 4.0, False: 3.0}   # register=4s, force=3s
WINDOW_NAME          = "Face Attendance System"

# สถานะ → (สี, ข้อความ)
STATUS_MAP = {
    "checked_in":  (C_SUCCESS, "CHECKED IN"),
    "checked_out": (C_WARNING, "CHECKED OUT"),
    "unknown":     (C_DANGER,  "NOT REGISTERED"),
}
_DEFAULT_STATUS = (C_INFO, "SCANNING")

_vignette_cache: dict = {}

# ── Drawing Primitives ───────────────────────────────────────────────

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_AA   = cv2.LINE_AA


def _blend_rect(frame, x1, y1, x2, y2, color, alpha):
    """สี่เหลี่ยมโปร่งแสง ROI-based (เร็วกว่า full-frame copy)"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        return
    roi = frame[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(roi, color, dtype=np.uint8),
                    alpha, roi, 1 - alpha, 0, frame[y1:y2, x1:x2])


def _scale_color(color, factor):
    """ปรับความสว่างของสี"""
    return tuple(min(255, int(c * factor)) for c in color)


def _put(img, text, pos, scale=0.55, color=C_WHITE, thick=1):
    """วาดข้อความ, return ความกว้าง px"""
    cv2.putText(img, text, pos, _FONT, scale, color, thick, _AA)
    return cv2.getTextSize(text, _FONT, scale, thick)[0][0]


def _tw(text, scale=0.55, thick=1):
    """คำนวณความกว้างข้อความ"""
    return cv2.getTextSize(text, _FONT, scale, thick)[0][0]


def _sep(frame, x, y1, y2):
    """เส้นแบ่ง vertical (pipe separator)"""
    cv2.line(frame, (x, y1), (x, y2), C_DIM, 1, _AA)


def _vignette(h, w, strength=0.35):
    """Vignette mask — cache ไว้ คำนวณครั้งเดียว"""
    if (h, w) not in _vignette_cache:
        Y, X = np.linspace(-1, 1, h).reshape(-1, 1), np.linspace(-1, 1, w).reshape(1, -1)
        mask = (np.clip(np.sqrt(X*X + Y*Y) - 0.4, 0, 1) * strength * 255).astype(np.uint8)
        _vignette_cache[(h, w)] = np.stack([mask]*3, axis=-1)
    return _vignette_cache[(h, w)]


# ── UI Components ────────────────────────────────────────────────────

def draw_header(frame, fps, n_reg, n_det):
    """Header bar — live dot + ชื่อระบบ + stats"""
    h, w = frame.shape[:2]
    cy = HEADER_H // 2
    _blend_rect(frame, 0, 0, w, HEADER_H, C_BG, 0.88)
    cv2.line(frame, (0, HEADER_H), (w, HEADER_H), C_ACCENT_DIM, 1, _AA)

    # ไฟกะพริบ live indicator
    p = math.sin(time.monotonic() * 3) * 0.5 + 0.5
    cv2.circle(frame, (22, cy), 5, _scale_color(C_SUCCESS, 0.5 + p*0.5), -1, _AA)
    _put(frame, "FACE ATTENDANCE", (38, cy + 6), 0.65, C_ACCENT, 2)

    # สถิติ — วาดจากขวาไปซ้าย
    items = [
        (datetime.now().strftime("%H:%M:%S"), C_WHITE, 0.55, 2),
        (f"Faces: {n_det}", C_SUCCESS if n_det else C_DIM, 0.48, 1),
        (f"Registered: {n_reg}", C_GRAY, 0.48, 1),
        (f"FPS: {fps:.0f}", C_SUCCESS if fps >= 10 else C_WARNING if fps >= 5 else C_DANGER, 0.48, 1),
    ]
    x = w - 20
    for i, (txt, col, sc, th) in enumerate(items):
        tw = _tw(txt, sc, th)
        x -= tw
        _put(frame, txt, (x, cy + 6), sc, col, th)
        x -= 32
        if i < len(items) - 1:
            _sep(frame, x + 16, 16, HEADER_H - 16)


def draw_instructions(frame):
    """แถบคำสั่ง — key badges"""
    h, w = frame.shape[:2]
    bar_y = h - FOOTER_H
    _blend_rect(frame, 0, bar_y, w, h, C_BG, 0.88)
    cv2.line(frame, (0, bar_y), (w, bar_y), C_ACCENT_DIM, 1, _AA)

    y = h - FOOTER_H // 2 + 4
    x = 20
    badges = [("R", "Register", C_ACCENT), ("I", "Check-In", C_SUCCESS),
              ("O", "Check-Out", C_WARNING), ("Q", "Quit", C_DANGER)]
    for key, label, kc in badges:
        # กรอบ badge
        kw = _tw(key, 0.45)
        bw = kw + 14
        cv2.rectangle(frame, (x, y - 18), (x + bw, y + 4), kc, 1, _AA)
        _put(frame, key, (x + 7, y), 0.45, kc)
        lw = _put(frame, label, (x + bw + 8, y), 0.45, C_GRAY)
        x += bw + lw + 38


def draw_event_log(frame, events):
    """Activity log — การ์ดพร้อม status dot"""
    h, w = frame.shape[:2]
    if not events:
        return
    evts = events[-3:]
    row_h, hdr_h = 30, 28
    ph = hdr_h + len(evts) * row_h + 12
    py = h - FOOTER_H - ph - 6
    px1, px2 = 12, 520

    _blend_rect(frame, px1, py, px2, py + ph, C_BG, 0.82)
    cv2.line(frame, (px1, py), (px2, py), C_ACCENT_DIM, 1, _AA)
    cv2.line(frame, (px1, py), (px1, py + ph), C_ACCENT, 2, _AA)
    _put(frame, "ACTIVITY LOG", (px1+14, py+20), 0.4, C_ACCENT)
    cv2.line(frame, (px1+14, py+hdr_h), (px2-14, py+hdr_h), C_DIM, 1, _AA)

    for i, evt in enumerate(evts):
        y = py + hdr_h + (i+1) * row_h
        is_in = evt["type"] == "check_in"
        dc = C_SUCCESS if is_in else C_WARNING
        cv2.circle(frame, (px1+24, y-5), 4, dc, -1, _AA)
        _put(frame, "IN" if is_in else "OUT", (px1+38, y), 0.45, dc)
        _put(frame, evt["name"], (px1+82, y), 0.45, C_WHITE)
        _put(frame, evt["time"], (px1+300, y), 0.40, C_GRAY)
        _put(frame, f"{evt['score']:.0%}", (px1+420, y), 0.40, C_DIM)


def _draw_corners(frame, x1, y1, x2, y2, color, pulse=1.0):
    """Corner-style bounding box พร้อม glow"""
    cl = max(25, min(x2-x1, y2-y1) // 3)
    t = max(2, int(2 + pulse * 2))

    pts = [((x1,y1),(x1+cl,y1),(x1,y1+cl)), ((x2,y1),(x2-cl,y1),(x2,y1+cl)),
           ((x1,y2),(x1+cl,y2),(x1,y2-cl)), ((x2,y2),(x2-cl,y2),(x2,y2-cl))]

    # Glow + เส้นหลัก (2 passes)
    for thick, col in [(t+4, _scale_color(color, 0.4)), (t, color)]:
        for c, he, ve in pts:
            cv2.line(frame, c, he, col, thick, _AA)
            cv2.line(frame, c, ve, col, thick, _AA)

    cv2.rectangle(frame, (x1,y1), (x2,y2), _scale_color(color, 0.33), 1, _AA)


def _draw_scan_line(frame, x1, y1, x2, y2, color):
    """เส้นสแกนเคลื่อนที่ขึ้น-ลง (animation)"""
    prog = math.sin(time.monotonic() * 3) * 0.5 + 0.5
    sy = int(y1 + (y2 - y1) * prog)
    cv2.line(frame, (x1+4, sy), (x2-4, sy), color, 1, _AA)
    fade = _scale_color(color, 0.25)
    for off in range(1, 6):
        for dy in (sy - off, sy + off):
            if y1 < dy < y2:
                cv2.line(frame, (x1+4, dy), (x2-4, dy), fade, 1, _AA)


def draw_face_overlay(frame, bbox, name, confidence, status, det_score):
    """Overlay รอบใบหน้า — HUD style"""
    x1, y1, x2, y2 = (int(c) for c in bbox)
    box_color, status_text = STATUS_MAP.get(status, _DEFAULT_STATUS)
    pulse = math.sin(time.monotonic() * 2.5) * 0.3 + 0.7

    _draw_corners(frame, x1, y1, x2, y2, box_color, pulse)
    if status == "scanning":
        _draw_scan_line(frame, x1, y1, x2, y2, C_INFO)

    # การ์ดข้อมูลด้านบน bbox
    ch = 62 if name else 32
    cy1 = max(HEADER_H + 4, y1 - ch - 10)
    cx1, cx2 = x1 - 2, max(x2 + 2, x1 + 218)
    _blend_rect(frame, cx1, cy1, cx2, cy1+ch, C_BG, 0.80)
    cv2.line(frame, (cx1, cy1+2), (cx1, cy1+ch-2), box_color, 3, _AA)

    if name:
        _put(frame, name.upper(), (cx1+12, cy1+22), 0.60, C_WHITE, 2)
        dy = cy1 + 40
        cv2.circle(frame, (cx1+18, dy), 4, box_color, -1, _AA)
        _put(frame, status_text, (cx1+30, dy+5), 0.40, box_color)
        # Confidence bar
        bx, bw = cx1+160, min(80, cx2 - cx1 - 210)
        if bw > 20:
            cv2.rectangle(frame, (bx, dy-3), (bx+bw, dy+1), (60,55,50), -1, _AA)
            cv2.rectangle(frame, (bx, dy-3), (bx+max(1, int(bw*confidence)), dy+1), box_color, -1, _AA)
        _put(frame, f"{confidence:.0%}", (bx+bw+8, dy+5), 0.40, C_WHITE)
    else:
        dy = cy1 + ch // 2
        cv2.circle(frame, (cx1+14, dy), 4, box_color, -1, _AA)
        _put(frame, status_text, (cx1+26, dy+5), 0.45, box_color)

    _put(frame, f"det {det_score:.2f}", (x2-70, y2+18), 0.35, C_DIM)


def draw_toast(frame, message):
    """Floating toast notification กลางจอ"""
    h, w = frame.shape[:2]
    if "Registered" in message or "Check-In" in message:
        ac = C_SUCCESS
    elif "Check-Out" in message:
        ac = C_WARNING
    else:
        ac = C_DANGER

    tw = _tw(message, 0.65, 2)
    tw2, th = tw + 60, 50
    tx1, ty1 = (w - tw2) // 2, h // 2 - 25
    _blend_rect(frame, tx1, ty1, tx1+tw2, ty1+th, C_BG, 0.92)
    cv2.line(frame, (tx1, ty1), (tx1+tw2, ty1), ac, 2, _AA)
    cv2.line(frame, (tx1, ty1), (tx1, ty1+th), ac, 2, _AA)
    cv2.circle(frame, (tx1+22, ty1+th//2), 5, ac, -1, _AA)
    _put(frame, message, (tx1+38, ty1+th//2+6), 0.60, C_WHITE, 2)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_event(etype, name, score, emp_id):
    return {"type": etype, "name": name, "time": datetime.now().strftime("%H:%M:%S"),
            "score": score, "emp_id": emp_id}


# ── Registration Flow ────────────────────────────────────────────────

async def register_face_flow(frame, engine, session_factory):
    """ลงทะเบียนใบหน้า: detect → ตรวจซ้ำ → สร้าง employee → บันทึก embedding"""
    faces = await engine.analyze_frame(frame)
    if not faces:
        return False, "No face detected. Please look at the camera.", None
    if len(faces) > 1:
        return False, "Multiple faces detected. Only 1 face allowed.", None

    face = faces[0]
    if face.det_score < MIN_REGISTER_SCORE:
        return False, f"Face too blurry (score: {face.det_score:.2f}). Move closer.", None

    emb_bytes = face.embedding.tobytes()

    # ดักซ้ำ — ตรวจว่าใบหน้านี้เคยลงทะเบียนแล้วหรือยัง
    if embedding_index.size > 0:
        match = await embedding_index.find_match(face.embedding)
        if match and match.is_confident:
            async with session_factory() as s:
                emp = await EmployeeRepository(s).get_by_id(match.employee_id)
                if emp:
                    return False, f"Already Registered: {emp.full_name} ({emp.employee_code})", None
                return False, "Face already registered in system.", None

    # สร้าง employee + บันทึก
    async with session_factory() as s:
        repo = EmployeeRepository(s)

        # หารหัสถัดไปที่ว่าง
        emp_code = "EMP-001"
        for i in range(1, 1000):
            code = f"EMP-{i:03d}"
            if not await repo.get_by_employee_code(code):
                emp_code = code
                break

        # ขอชื่อจาก terminal
        print("\n" + "=" * 40)
        print(f"  Employee Code: {emp_code}")
        suffix = emp_code.split("-")[-1]
        default = f"Test Employee {suffix}"
        name = input(f"  Enter name (Enter='{default}'): ").strip() or default
        print("=" * 40)

        existing = await repo.get_by_employee_code(emp_code)
        emp = existing or Employee(employee_code=emp_code, full_name=name, department="General", position="Staff")
        if not existing:
            s.add(emp)
            await s.flush()

        # Crop → MinIO → Embedding
        crop = crop_and_encode_face(frame, face.bbox)
        img_url = await minio_service.upload_image_async(crop, f"{uuid.uuid4()}.jpg")
        await FaceEmbeddingRepository(s).upsert(FaceEmbedding(
            employee_id=emp.id, embedding_vector=emb_bytes,
            model_version=f"{settings.face_model_pack}_v1",
            image_quality_score=face.quality_score, image_url=img_url))
        await repo.set_face_registered(emp.id, registered=True)
        await s.commit()
        await EmbeddingCacheService(s).rebuild_index()

    return True, f"Registered: {name} ({emp_code})", str(emp.id)


# ── Key Handlers ─────────────────────────────────────────────────────

async def _handle_register(frame, emp_names):
    """ปุ่ม R — ลงทะเบียนใบหน้า"""
    print("\n[REGISTER] Capturing face...")
    ok, msg, eid = await register_face_flow(frame, face_engine, async_session_factory)
    if ok and eid:
        async with async_session_factory() as s:
            emp_names.clear()
            for e in await EmployeeRepository(s).get_active_employees():
                emp_names[str(e.id)] = e.full_name
        print(f"[OK] {msg}")
    else:
        print(f"[FAIL] {msg}")
    return msg, time.monotonic() + 4.0


def _handle_force(emp_id, name, conf, emp_status, events, action):
    """
    ปุ่ม I/O — บังคับเช็คอิน/เช็คเอาท์
    action: "check_in" หรือ "check_out"
    """
    if not emp_id:
        tag = "check-in" if action == "check_in" else "check-out"
        print(f"[FAIL] No active face recognized to force {tag}.")
        return "No recognized face!", time.monotonic() + 3.0

    now = time.monotonic()
    if action == "check_in":
        emp_status[emp_id] = {"check_in_time": now, "status": "checked_in"}
    else:
        emp_status[emp_id] = {"check_in_time": now - CHECKOUT_WAIT_SEC - 1, "status": "checked_out"}

    # ลบ event ประเภทเดียวกันของคนเดิมออก แล้วเพิ่มใหม่
    events[:] = [e for e in events if not (e["emp_id"] == emp_id and e["type"] == action)]
    evt = _make_event(action, name, conf, emp_id)
    events.append(evt)

    tag = "CHECK-IN" if action == "check_in" else "CHECK-OUT"
    print(f"[FORCE {tag}] {name} @ {evt['time']}")
    label = f"Forced {'Check-In' if action == 'check_in' else 'Check-Out'}: {name}"
    return label, time.monotonic() + 3.0


# ── Recognition Logic ────────────────────────────────────────────────

async def _recognize(face, emp_names, emp_status, events):
    """จับคู่ใบหน้า + จัดการเช็คอิน/เช็คเอาท์อัตโนมัติ → (name, emp_id, conf, status)"""
    if embedding_index.size == 0:
        return None, None, 0.0, "unknown"

    match = await embedding_index.find_match(face.embedding)
    if not match:
        return None, None, 0.0, "unknown"
    if not match.is_confident:
        return None, None, match.similarity, "scanning"

    eid = str(match.employee_id)
    name = emp_names.get(eid, f"ID:{eid[:8]}")
    conf = match.similarity
    now = time.monotonic()

    # ยังไม่เคยเช็คอิน → เช็คอินอัตโนมัติ
    if eid not in emp_status:
        emp_status[eid] = {"check_in_time": now, "status": "checked_in"}
        evt = _make_event("check_in", name, conf, eid)
        events.append(evt)
        print(f"[CHECK-IN] {name} [{conf:.0%}] @ {evt['time']}")
        return name, eid, conf, "checked_in"

    st = emp_status[eid]
    if st["status"] == "checked_in" and now - st["check_in_time"] >= CHECKOUT_WAIT_SEC:
        # ผ่าน 10 นาที → เช็คเอาท์อัตโนมัติ
        st["status"] = "checked_out"
        evt = _make_event("check_out", name, conf, eid)
        events.append(evt)
        print(f"[CHECK-OUT] {name} [{conf:.0%}] @ {evt['time']}")
        return name, eid, conf, "checked_out"

    return name, eid, conf, st["status"]


# ── Main Loop ────────────────────────────────────────────────────────

async def main():
    # โหลด AI + embedding index + รายชื่อพนักงาน
    print("[*] Loading AI engine...")
    face_engine.load()
    print("[OK] AI engine ready")

    print("[*] Loading face embeddings...")
    async with async_session_factory() as s:
        await EmbeddingCacheService(s).rebuild_index()
    print(f"[OK] {embedding_index.size} registered faces loaded")

    emp_names: dict[str, str] = {}
    async with async_session_factory() as s:
        for e in await EmployeeRepository(s).get_active_employees():
            emp_names[str(e.id)] = e.full_name

    if not embedding_index.size:
        print("\n[!] No faces registered yet — press 'R' to register.\n")

    # เปิดกล้อง
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    events: list[dict] = []
    emp_status: dict[str, dict] = {}
    fps, fc, ft = 0.0, 0, time.monotonic()
    msg, msg_until = "", 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAMERA_W, CAMERA_H)
    print(f"\n{'='*50}\nFACE ATTENDANCE KIOSK\n[R] Register  [I] Check-In  [O] Check-Out  [Q] Quit\n{'='*50}\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue
            frame = cv2.flip(frame, 1)
            cv2.subtract(frame, _vignette(*frame.shape[:2]), frame)

            # FPS
            fc += 1
            el = time.monotonic() - ft
            if el >= 1.0:
                fps, fc, ft = fc / el, 0, time.monotonic()

            # ตรวจจับ + จับคู่ใบหน้า
            faces = await face_engine.analyze_frame(frame)
            good = [f for f in faces if f.det_score >= MIN_DET_SCORE]
            active_id = active_name = None
            active_conf = 0.0

            for face in good:
                name, eid, conf, status = await _recognize(face, emp_names, emp_status, events)
                if eid and not active_id:
                    active_id, active_name, active_conf = eid, name, conf
                draw_face_overlay(frame, face.bbox, name, conf, status, face.det_score)

            # วาด UI
            draw_header(frame, fps, embedding_index.size, len(good))
            draw_instructions(frame)
            draw_event_log(frame, events)
            if msg and time.monotonic() < msg_until:
                draw_toast(frame, msg)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("r"), ord("R")):
                msg, msg_until = await _handle_register(frame, emp_names)
            elif key in (ord("i"), ord("I")):
                msg, msg_until = _handle_force(active_id, active_name, active_conf,
                                               emp_status, events, "check_in")
            elif key in (ord("o"), ord("O")):
                msg, msg_until = _handle_force(active_id, active_name, active_conf,
                                               emp_status, events, "check_out")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[OK] Kiosk closed.")


if __name__ == "__main__":
    asyncio.run(main())
