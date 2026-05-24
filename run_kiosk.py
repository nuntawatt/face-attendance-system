"""
Face Attendance Kiosk OpenCV Desktop Application

แอปจะเปิดหน้าต่าง OpenCV แสดงภาพจากกล้อง Webcam แบบ real-time
พร้อมวาด bounding box, ชื่อพนักงาน, ค่าความมั่นใจ และสถานะเข้า-ออกงาน

คำสั่ง:
    r = ลงทะเบียนใบหน้าที่เห็นอยู่ตอนนี้
    q/ESC = ปิดโปรแกรม
"""

import asyncio
import time
from datetime import datetime, timezone
from uuid import UUID

import cv2
import numpy as np

# สีที่ใช้ในหน้าจอ (BGR format สำหรับ OpenCV)
COLOR_GREEN = (80, 220, 100)
COLOR_BLUE = (230, 160, 50)
COLOR_RED = (60, 60, 220)
COLOR_ORANGE = (40, 165, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (160, 160, 160)
COLOR_PANEL_BG = (40, 35, 30)
COLOR_ACCENT = (255, 200, 50)
COLOR_CYAN = (230, 200, 0)

# Threshold ต่ำสำหรับ demo (ให้แสดง bbox ได้ง่าย)
MIN_DET_SCORE = 0.35
RECOGNITION_COOLDOWN = 5.0
CHECKOUT_WAIT_SECONDS = 10 * 60  # รอ 10 นาทีก่อนเช็คเอาท์ได้


def draw_label(img, text, pos, font_scale=0.6, color=COLOR_WHITE, thickness=1):
    """วาดข้อความ"""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_face_overlay(frame, bbox, name, confidence, status, det_score):
    """วาด overlay รอบใบหน้า"""
    x1, y1, x2, y2 = [int(c) for c in bbox]
    w = x2 - x1
    h = y2 - y1

    if status == "checked_in":
        box_color = COLOR_GREEN
        status_text = "CHECKED IN"
    elif status == "checked_out":
        box_color = COLOR_ORANGE
        status_text = "CHECKED OUT"
    elif status == "unknown":
        box_color = COLOR_RED
        status_text = "NOT REGISTERED"
    else:
        box_color = COLOR_BLUE
        status_text = "SCANNING..."

    # Corner-style bounding box
    corner_len = max(20, min(w, h) // 4)
    t = 3
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, t)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, t)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, t)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, t)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, t)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, t)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)

    # Info panel ด้านบน bbox
    panel_h = 60 if name else 35
    panel_y1 = max(0, y1 - panel_h - 8)
    panel_y2 = y1 - 4

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 - 1, panel_y1), (x2 + 1, panel_y2), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    if name:
        draw_label(frame, name, (x1 + 6, panel_y1 + 20), font_scale=0.65, color=COLOR_WHITE, thickness=2)
        info = f"{status_text}  |  {confidence:.0%}"
        draw_label(frame, info, (x1 + 6, panel_y1 + 45), font_scale=0.5, color=box_color)
    else:
        draw_label(frame, status_text, (x1 + 6, panel_y1 + 22), font_scale=0.55, color=box_color)

    # det score มุมล่าง
    draw_label(frame, f"det:{det_score:.2f}", (x2 - 80, y2 + 18), font_scale=0.4, color=COLOR_GRAY)


def draw_header(frame, fps, num_registered, num_detected):
    """Header bar ด้านบน"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    draw_label(frame, "FACE ATTENDANCE SYSTEM", (16, 34), font_scale=0.7, color=COLOR_ACCENT, thickness=2)

    now = datetime.now().strftime("%H:%M:%S")
    status = f"FPS: {fps:.0f}  |  Registered: {num_registered}  |  Faces: {num_detected}  |  {now}"
    draw_label(frame, status, (w - 520, 34), font_scale=0.5, color=COLOR_GRAY)

    cv2.line(frame, (0, 52), (w, 52), COLOR_ACCENT, 1)


def draw_instructions(frame):
    """คำสั่งด้านล่าง"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 36), (w, h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    draw_label(frame, "[R] Register Face    [Q/ESC] Quit", (16, h - 12), font_scale=0.5, color=COLOR_CYAN)


def draw_event_log(frame, events):
    """Event log ด้านล่าง"""
    h, w = frame.shape[:2]
    if not events:
        return

    panel_h = min(len(events) * 28 + 20, 120)
    panel_y1 = h - 36 - panel_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y1), (w, h - 36), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, panel_y1), (w, panel_y1), COLOR_ACCENT, 1)

    draw_label(frame, "RECENT EVENTS", (16, panel_y1 + 18), font_scale=0.45, color=COLOR_ACCENT)

    for i, evt in enumerate(events[-3:]):
        y = panel_y1 + 20 + (i + 1) * 26
        color = COLOR_GREEN if evt["type"] == "check_in" else COLOR_ORANGE
        icon = "[IN]" if evt["type"] == "check_in" else "[OUT]"
        text = f"{icon}  {evt['name']}  --  {evt['time']}  ({evt['score']:.0%})"
        draw_label(frame, text, (20, y), font_scale=0.48, color=color)


def draw_register_mode(frame, step_text):
    """Overlay ขณะลงทะเบียน"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # กรอบหน้า
    cx, cy = w // 2, h // 2
    rw, rh = 200, 260
    cv2.ellipse(frame, (cx, cy), (rw, rh), 0, 0, 360, COLOR_CYAN, 2)

    draw_label(frame, step_text, (cx - 180, cy + rh + 40), font_scale=0.7, color=COLOR_CYAN, thickness=2)


async def register_face_flow(frame, face_engine, session_factory):
    """
    ลงทะเบียนใบหน้าจาก frame ปัจจุบัน
    Returns: (success: bool, message: str, employee_id: str|None)
    """
    # ตรวจจับใบหน้าในเฟรม
    faces = await face_engine.analyze_frame(frame)
    if not faces:
        return False, "No face detected. Please look at the camera.", None
    if len(faces) > 1:
        return False, "Multiple faces detected. Only 1 face allowed.", None

    face = faces[0]
    if face.det_score < 0.5:
        return False, f"Face too blurry (score: {face.det_score:.2f}). Move closer.", None

    embedding_bytes = face.embedding.tobytes()

    # ขอชื่อจาก terminal
    print("\n" + "=" * 40)
    emp_code = input("  Enter employee code (e.g. EMP-001): ").strip()
    full_name = input("  Enter full name: ").strip()
    print("=" * 40)

    if not emp_code or not full_name:
        return False, "Registration cancelled empty input.", None

    from app.models.employee import Employee
    from app.models.face_embedding import FaceEmbedding
    from app.repositories.employee import EmployeeRepository
    from app.repositories.face_embedding import FaceEmbeddingRepository
    from app.services.embedding_cache_service import EmbeddingCacheService
    from app.ai.recognition import embedding_index
    from app.core.config import settings

    async with session_factory() as session:
        emp_repo = EmployeeRepository(session)

        # เช็คซ้ำ
        existing = await emp_repo.get_by_employee_code(emp_code)
        if existing:
            # ถ้ามีอยู่แล้ว -> update embedding
            emp = existing
        else:
            # สร้างพนักงานใหม่
            emp = Employee(
                employee_code=emp_code,
                full_name=full_name,
                department="General",
                position="Staff",
            )
            session.add(emp)
            await session.flush()

        # Crop ใบหน้า
        x1, y1, x2, y2 = face.bbox
        h, w = frame.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        face_crop = frame[y1:y2, x1:x2]

        # แปลงเป็น jpeg bytes
        import cv2
        _, buffer = cv2.imencode(".jpg", face_crop)
        crop_bytes = buffer.tobytes()

        # อัพโหลดขึ้น MinIO ด้วย UUID filename
        import uuid
        image_filename = f"{uuid.uuid4()}.jpg"
        from app.services.minio_service import minio_service
        image_url = await asyncio.to_thread(
            minio_service.upload_image, crop_bytes, image_filename
        )

        # Upsert embedding
        emb_repo = FaceEmbeddingRepository(session)
        emb_record = FaceEmbedding(
            employee_id=emp.id,
            embedding_vector=embedding_bytes,
            model_version=f"{settings.face_model_pack}_v1",
            image_quality_score=face.quality_score,
            image_url=image_url,
        )
        await emb_repo.upsert(emb_record)
        await emp_repo.set_face_registered(emp.id, registered=True)
        await session.commit()

        # Rebuild index
        cache = EmbeddingCacheService(session)
        await cache.rebuild_index()

    return True, f"Registered: {full_name} ({emp_code})", str(emp.id)


async def main():
    """Main loop ของ Kiosk Application"""

    # 1. โหลด AI Engine
    print("[*] Loading AI engine...")
    from app.ai.engine import face_engine
    from app.ai.recognition import embedding_index
    from app.database.session import async_session_factory
    from app.services.embedding_cache_service import EmbeddingCacheService
    from app.repositories.employee import EmployeeRepository

    face_engine.load()
    print("[OK] AI engine ready")

    # 2. โหลด Embedding Index
    print("[*] Loading face embeddings...")
    async with async_session_factory() as session:
        cache = EmbeddingCacheService(session)
        await cache.rebuild_index()
    print(f"[OK] {embedding_index.size} registered faces loaded")

    # 3. โหลดข้อมูลพนักงาน
    employee_names: dict[str, str] = {}
    async with async_session_factory() as session:
        repo = EmployeeRepository(session)
        employees = await repo.get_active_employees()
        for emp in employees:
            employee_names[str(emp.id)] = emp.full_name

    if embedding_index.size == 0:
        print("\n[!] No faces registered yet!")
        print("[!] Press 'R' in the kiosk window to register your face.\n")

    # 4. เปิดกล้อง
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # State
    recent_events: list[dict] = []
    # เก็บสถานะการเช็คอิน {emp_id: {"check_in_time": 1234.5, "status": "checked_in"}}
    employee_status: dict[str, dict] = {}
    fps = 0.0
    frame_count = 0
    fps_timer = time.monotonic()
    register_mode = False
    register_msg = ""
    register_msg_until = 0.0

    window_name = "Face Attendance System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\n" + "=" * 50)
    print("FACE ATTENDANCE KIOSK")
    print("[R] Register [Q/ESC] Quit")
    print("=" * 50 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)

            # FPS
            frame_count += 1
            elapsed = time.monotonic() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.monotonic()

            # Face Detection
            faces = await face_engine.analyze_frame(frame)
            good_faces = [f for f in faces if f.det_score >= MIN_DET_SCORE]
            num_detected = len(good_faces)

            # Recognition + Drawing
            for face in good_faces:
                name = None
                confidence = 0.0
                status = "unknown"

                if embedding_index.size > 0:
                    match = await embedding_index.find_match(face.embedding)

                    if match and match.is_confident:
                        emp_id = str(match.employee_id)
                        name = employee_names.get(emp_id, f"ID:{emp_id[:8]}")
                        confidence = match.similarity
                        now = time.monotonic()

                        if emp_id not in employee_status:
                            # 1. เช็คอินครั้งแรก
                            employee_status[emp_id] = {
                                "check_in_time": now,
                                "status": "checked_in"
                            }
                            time_str = datetime.now().strftime("%H:%M:%S")
                            recent_events.append({
                                "type": "check_in",
                                "name": name,
                                "time": time_str,
                                "score": confidence,
                                "emp_id": emp_id,
                            })
                            print(f"[CHECK-IN] {name} [{confidence:.0%}] @ {time_str}")
                            status = "checked_in"
                        else:
                            # 2. เคยเช็คอินไปแล้ว
                            emp_state = employee_status[emp_id]
                            
                            if emp_state["status"] == "checked_in":
                                # ตรวจสอบว่าผ่านไป 10 นาที (600 วินาที) หรือยัง
                                if now - emp_state["check_in_time"] >= CHECKOUT_WAIT_SECONDS:
                                    # เช็คเอาท์ได้
                                    emp_state["status"] = "checked_out"
                                    time_str = datetime.now().strftime("%H:%M:%S")
                                    recent_events.append({
                                        "type": "check_out",
                                        "name": name,
                                        "time": time_str,
                                        "score": confidence,
                                        "emp_id": emp_id,
                                    })
                                    print(f"[CHECK-OUT] {name} [{confidence:.0%}] @ {time_str}")
                                    status = "checked_out"
                                else:
                                    # ยังไม่ถึง 10 นาที -> สถานะยังเป็น check_in อยู่ ไม่ต้องบันทึกซ้ำ
                                    status = "checked_in"
                            else:
                                # ถ้าเป็น checked_out แล้ว ก็ให้เป็น checked_out ต่อไป ไม่บันทึกซ้ำ
                                status = "checked_out"
                    elif match:
                        status = "scanning"
                        confidence = match.similarity

                draw_face_overlay(frame, face.bbox, name, confidence, status, face.det_score)

            # Draw UI
            draw_header(frame, fps, embedding_index.size, num_detected)
            draw_instructions(frame)
            draw_event_log(frame, recent_events)

            # Register message overlay
            if register_msg and time.monotonic() < register_msg_until:
                h_f, w_f = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h_f // 2 - 30), (w_f, h_f // 2 + 30), COLOR_PANEL_BG, -1)
                cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
                color = COLOR_GREEN if "Registered" in register_msg else COLOR_RED
                draw_label(frame, register_msg, (w_f // 2 - 300, h_f // 2 + 8),
                           font_scale=0.7, color=color, thickness=2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r') or key == ord('R'):
                # Register Mode
                print("\n[REGISTER] Capturing face...")
                # ถ่ายภาพ frame ปัจจุบัน (ก่อน flip ไม่ต้อง เพราะ flip แล้ว)
                success, msg, emp_id = await register_face_flow(frame, face_engine, async_session_factory)
                register_msg = msg
                register_msg_until = time.monotonic() + 4.0

                if success and emp_id:
                    # reload employee names
                    async with async_session_factory() as session:
                        repo = EmployeeRepository(session)
                        employees = await repo.get_active_employees()
                        employee_names.clear()
                        for emp in employees:
                            employee_names[str(emp.id)] = emp.full_name
                    print(f"[OK] {msg}")
                else:
                    print(f"[FAIL] {msg}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[OK] Kiosk closed.")


if __name__ == "__main__":
    asyncio.run(main())
