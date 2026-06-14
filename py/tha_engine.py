"""
THA4 ONNX Engine for Super Agent Party
Server-side rendering: ONNX model inference -> JPEG frames -> WebSocket streaming
"""
import asyncio
import io
import json
import logging
import math
import os
import sys
import time
import uuid
import zipfile
import numpy as np
import onnxruntime as ort
import simplejpeg
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# 1. sRGB -> Linear 颜色空间转换
# ------------------------------------------------------------
def _srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


# ------------------------------------------------------------
# 2. 情感 -> 45维姿态参数映射表
# ------------------------------------------------------------
EMOTION_POSE_MAP: Dict[str, np.ndarray] = {}

def _make_pose(indices: list, scale: float = 1.0) -> np.ndarray:
    arr = np.zeros(45, dtype=np.float32)
    for i in indices:
        arr[i] = scale
    return arr

EMOTION_POSE_MAP["happy"] = _make_pose(
    [8, 9, 14, 15, 34, 35],  # eyebrow_happy(L/R) + eye_happy_wink(L/R) + mouth_raised_corner(L/R)
    1.0
)
EMOTION_POSE_MAP["sad"] = _make_pose(
    [0, 1, 6, 7, 32, 33],  # eyebrow_troubled(L/R) + eyebrow_raised(L/R) + mouth_lowered_corner(L/R)
    0.8
)
EMOTION_POSE_MAP["angry"] = _make_pose(
    [2, 3, 4, 5, 20, 21],  # eyebrow_angry(L/R) + eyebrow_lowered(L/R) + eye_unimpressed(L/R)
    0.9
)
EMOTION_POSE_MAP["surprised"] = _make_pose(
    [6, 7, 16, 17, 26, 30],  # eyebrow_raised(L/R) + eye_surprised(L/R) + mouth_aaa + mouth_ooo
    0.7
)
EMOTION_POSE_MAP["relaxed"] = _make_pose(
    [18, 19],  # eye_relaxed(L/R) — close eyes
    1.0
)

# neutral 清零
EMOTION_POSE_MAP["neutral"] = np.zeros(45, dtype=np.float32)

# ------------------------------------------------------------
# 2b. 动作 -> 45维参数映射表（一次性动画，叠加在基础姿态之上）
# ------------------------------------------------------------
#   type "oscillate": sin 振荡动画
#   type "hold":      保持在目标值一段时间后回弹
THA_MOTIONS: Dict[str, dict] = {
    "nod": {
        "params": {39: 0.55},       # head_x 上下点头
        "type": "oscillate",
        "duration": 1.4,
        "frequency": 2.8,
    },
    "shakeHead": {
        "params": {40: 0.5},        # head_y 左右摇头
        "type": "oscillate",
        "duration": 1.0,
        "frequency": 3.5,
    },
    "tiltHead": {
        "params": {41: 0.55},       # neck_z 歪头
        "type": "oscillate",
        "duration": 1.2,
        "frequency": 2.2,
    },
    "bow": {
        "params": {43: 0.6},        # body_z 身体前倾（鞠躬）
        "type": "hold",
        "duration": 1.5,
    },
    "sway": {
        "params": {42: 0.45},       # body_y 身体左右摇摆
        "type": "oscillate",
        "duration": 1.5,
        "frequency": 2.0,
    },
    "lookAround": {
        "params": {38: 0.7, 37: 0.4},  # 眼珠转动 + 微微抬头
        "type": "oscillate",
        "duration": 2.0,
        "frequency": 1.5,
    },
}


# ------------------------------------------------------------
# 3. THAPoseGenerator — 空闲动画 + 情感/口型混合
# ------------------------------------------------------------
class THAPoseGenerator:
    def __init__(self):
        self.t = 0.0
        self.last = time.perf_counter()
        self._pbr = np.random.random() * math.pi * 2
        self._phx = np.random.random() * math.pi * 2
        self._phy = np.random.random() * math.pi * 2
        self._bsb = 0.7 + np.random.random() * 0.3
        self.next_blink = 2.0 + np.random.random() * 4.0
        self.blink_state = 0
        self.blink_timer = 0.0
        self.blink_dur = 0.06
        self.blink_hold = 0.08
        self.mx = 0.0
        self.my = 0.0
        self._mouse_x = 0.0
        self._mouse_y = 0.0

        # 情感混合
        self._emotion_pose = np.zeros(45, dtype=np.float32)
        self._emotion_target = np.zeros(45, dtype=np.float32)
        self._emotion_smooth = 4.0

        # 口型
        self._mouth_amplitude = 0.0
        self._mouth_target = 0.0
        # 👇 【物理延迟归零算法】：将数值拉到 120.0，使后端没有一丁点平滑粘性
        # 前端算出来的 8Hz 高频振幅会被毫无保留、极其敏捷地百分之百执行！
        self._mouth_smooth = 120.0

        # 动作（一次性动画）
        self._motion_name = None
        self._motion_timer = 0.0
        self._motion_data = {}

    def _rb(self):
        return 2.0 + np.random.random() * 4.0

    def set_emotion(self, emotion_name: str):
        """设置情感，平滑过渡到对应姿态"""
        self._emotion_target = EMOTION_POSE_MAP.get(emotion_name, EMOTION_POSE_MAP["neutral"]).copy()

    def set_mouth(self, amplitude: float):
        """设置口型幅度 0.0-1.0"""
        self._mouth_target = max(0.0, min(1.0, float(amplitude)))

    def set_mouse(self, x: float, y: float):
        """设置鼠标位置"""
        self._mouse_x = float(x)
        self._mouse_y = float(y)

    def set_motion(self, motion_name: str):
        """触发一次性动作动画"""
        if motion_name in THA_MOTIONS:
            self._motion_name = motion_name
            self._motion_timer = 0.0
            self._motion_data = THA_MOTIONS[motion_name]

    def clear_motion(self):
        """立即终止当前动作"""
        self._motion_name = None

    def step(self) -> np.ndarray:
        now = time.perf_counter()
        dt = now - self.last
        self.last = now
        self.t += dt

        # 平滑情感
        alpha = min(dt * self._emotion_smooth, 1.0)
        self._emotion_pose += (self._emotion_target - self._emotion_pose) * alpha

        # 平滑口型
        alpha_m = min(dt * self._mouth_smooth, 1.0)
        self._mouth_amplitude += (self._mouth_target - self._mouth_amplitude) * alpha_m

        p = np.zeros(45, dtype=np.float32)

        # breathing (idx 44)
        p[44] = 0.8 * abs(math.sin(self.t * self._bsb + self._pbr))

        # head idle
        idle_hx = 0.32 * math.sin(self.t * 1.1 + self._phx)  # 左右晃头幅度加大
        idle_hy = 0.22 * math.sin(self.t * 1.3 + self._phy)  # 上下点头幅度加大
        idle_nk = 0.14 * math.sin(self.t * 0.55)             # 歪头幅度加大
        idle_ix = 0.18 * math.sin(self.t * 0.45 + self._phy) # 眼珠左右转动更灵活
        idle_iy = 0.12 * math.sin(self.t * 0.55 + self._phx) # 眼珠上下看幅度加大

        # smooth mouse
        self.mx += (self._mouse_x - self.mx) * min(dt * 8.0, 1.0)
        self.my += (self._mouse_y - self.my) * min(dt * 8.0, 1.0)
        mx, my = self.mx, self.my

        # body follows mouse
        p[42] = -mx * 0.45
        p[43] = my * 0.35
        p[39] = idle_hx - my * 1.10
        p[40] = idle_hy - mx * 0.90
        p[41] = idle_nk
        p[37] = idle_ix - my * 0.85
        p[38] = idle_iy - mx * 0.95

        # blinking — 闭眼状态下暂停眨眼
        eyes_closed_by_emotion = max(self._emotion_target[18], self._emotion_target[19]) > 0.3
        if not eyes_closed_by_emotion:
            self.blink_timer += dt
            if self.blink_state == 0:
                if self.blink_timer >= self.next_blink:
                    self.blink_state = 1
                    self.blink_timer = 0.0
                    self.next_blink = self._rb()
            elif self.blink_state == 1:
                v = min(self.blink_timer / self.blink_dur, 1.0)
                p[18] = p[19] = v
                if v >= 1.0:
                    self.blink_state = 2
                    self.blink_timer = 0.0
            elif self.blink_state == 2:
                p[18] = p[19] = 1.0
                if self.blink_timer >= self.blink_hold:
                    self.blink_state = 3
                    self.blink_timer = 0.0
            elif self.blink_state == 3:
                v = 1.0 - min(self.blink_timer / self.blink_dur, 1.0)
                p[18] = p[19] = v
                if v <= 0.0:
                    self.blink_state = 0
                    self.blink_timer = 0.0
        else:
            self.blink_state = 0
            self.blink_timer = 0.0
        p[26] = 0.0

        # 混合情感姿态
        p += self._emotion_pose

        # 混合口型
        p += self._mouth_amplitude * _get_mouth_pose()

        # 混合动作（一次性动画，叠加覆盖）
        if self._motion_name:
            self._motion_timer += dt
            dur = self._motion_data.get("duration", 1.5)
            progress = min(self._motion_timer / dur, 1.0)
            mtype = self._motion_data.get("type", "oscillate")
            freq = self._motion_data.get("frequency", 2.0)

            if mtype == "oscillate":
                # sin 振荡 + 淡入淡出包络
                envelope = math.sin(progress * math.pi)
                wave = math.sin(self._motion_timer * freq * 2.0 * math.pi)
                motion_val = envelope * wave
            else:  # hold
                # 缓入 -> 保持 -> 缓出
                if progress < 0.2:
                    motion_val = progress / 0.2
                elif progress < 0.7:
                    motion_val = 1.0
                else:
                    motion_val = 1.0 - (progress - 0.7) / 0.3

            for idx, scale in self._motion_data.get("params", {}).items():
                p[idx] += motion_val * scale

            if progress >= 1.0:
                self._motion_name = None

        return p


_MOUTH_POSE = None
def _get_mouth_pose() -> np.ndarray:
    """口型姿态模版, 懒初始化"""
    global _MOUTH_POSE
    if _MOUTH_POSE is None:
        _MOUTH_POSE = np.zeros(45, dtype=np.float32)
        for i in [26]:
            _MOUTH_POSE[i] = 1.0
    return _MOUTH_POSE.copy()


# ------------------------------------------------------------
# 4. THAEngine — ONNX 模型加载 & 渲染
# ------------------------------------------------------------
class THAEngine:
    def __init__(self, model_path: str, character_path: str):
        self.session: Optional[ort.InferenceSession] = None
        self.image_np: Optional[np.ndarray] = None
        self._loaded = False
        self.model_path = model_path
        self.character_path = character_path
        
        # 🌟 优化：预分配不变量，避免循环重复分配内存带来的 GC 压力
        self.green_bg = np.array([0.0, 255.0, 0.0], dtype=np.float32).reshape(3, 1, 1)

    def load(self):
        """加载 ONNX 模型和角色纹理"""
        if self._loaded:
            return
            
        # 获取当前环境中所有可用的 ONNX Provider
        available_providers = ort.get_available_providers()
        
        # 按照我们期望的性能优先级排序
        preferred_order = [
            "CUDAExecutionProvider",    # Linux/Windows (Nvidia GPU)
            "DmlExecutionProvider",     # Windows (DirectX: AMD/Intel/Nvidia)
            "CoreMLExecutionProvider",  # macOS (Apple Silicon)
            "CPUExecutionProvider"      # Fallback
        ]
        
        # 筛选出当前系统实际支持的 Providers
        providers = [p for p in preferred_order if p in available_providers]
        
        if not providers:
            providers = ["CPUExecutionProvider"]

        try:
            # 尝试使用最高优先级的硬件加速加载模型
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception as e:
            logger.warning(f"[THA] 硬件加速加载失败，尝试强制回退到 CPU... 错误信息: {e}")
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            
        active_provider = self.session.get_providers()[0]
        
        print(f"\n🚀 [THA] ===============================================")
        print(f"🚀 [THA] 检测到前端加载请求，2D 引擎成功初始化!")
        print(f"🚀 [THA] 模型文件: {os.path.basename(self.model_path)}")
        print(f"🚀 [THA] 激活的硬件加速后端: \033[1;32m{active_provider}\033[0m")
        print(f"🚀 [THA] ===============================================\n")
        
        logger.info(f"[THA] Active Provider: {active_provider}")
        logger.info(f"[THA] Model: {self.model_path}")

        img = np.array(Image.open(self.character_path).convert("RGBA"), dtype=np.float32) / 255.0
        img[:, :, :3] = _srgb_to_linear(img[:, :, :3])
        img[:, :, :3] *= img[:, :, 3:4]
        img = img * 2.0 - 1.0
        self.image_np = np.expand_dims(img.transpose(2, 0, 1), 0).astype(np.float32)
        self._loaded = True

    def render(self, pose: np.ndarray) -> bytes:
        """渲染一帧, 返回 JPEG bytes"""
        if not self._loaded:
            self.load()

        p = pose.reshape(1, 45).astype(np.float32)
        out = self.session.run(None, {"image": self.image_np, "pose": p})[0]
        img_data = out[0]  # (C, 512, 512)  CHW

        C = img_data.shape[0]

        if C == 4:
            # ── RGBA 模型：需合成为绿幕 ──
            rgb = img_data[:3, :, :]
            alpha = img_data[3, :, :]

            if img_data.dtype == np.uint8:
                alpha_f = alpha.astype(np.float32)[np.newaxis, :, :] / 255.0
                result = rgb.astype(np.float32) * alpha_f \
                         + self.green_bg * (1.0 - alpha_f)
                result = np.clip(result, 0, 255).astype(np.uint8)
            else:
                r_min, a_min = np.min(rgb), np.min(alpha)
                if r_min < -0.1:
                    rgb = (rgb + 1.0) * 127.5
                else:
                    rgb = rgb * 255.0
                if a_min < -0.1:
                    alpha = (alpha + 1.0) * 0.5
                alpha = alpha[np.newaxis, :, :]
                result = rgb * alpha + self.green_bg * (1.0 - alpha)
                result = np.clip(result, 0, 255).astype(np.uint8)

        elif C == 3:
            # ── 3 通道绿幕模型（model.onnx 默认走这里）──
            if img_data.dtype == np.uint8:
                result = img_data
            else:
                if np.min(img_data) < -0.1:
                    result = np.clip((img_data + 1.0) * 127.5, 0, 255).astype(np.uint8)
                else:
                    result = np.clip(img_data * 255.0, 0, 255).astype(np.uint8)

        else:
            raise RuntimeError(f"Unsupported channel count: {C}")

        rgb_out = np.ascontiguousarray(result.transpose(1, 2, 0))
        return simplejpeg.encode_jpeg(rgb_out, quality=75, colorspace='RGB')

    @property
    def loaded(self) -> bool:
        return self._loaded


# ------------------------------------------------------------
# 5. THAModelManager — 模型文件管理
# ------------------------------------------------------------
class THAModelManager:
    def __init__(self, default_dir: str, user_upload_dir: str):
        self.default_dir = default_dir
        self.user_upload_dir = user_upload_dir

    def scan_default_models(self, base_url: str = "") -> list:
        return self._scan_models(self.default_dir, "default", base_url)

    def scan_user_models(self, base_url: str = "") -> list:
        return self._scan_models(self.user_upload_dir, "user", base_url)

    def _scan_models(self, directory: str, model_type: str, base_url: str = "") -> list:
        models = []
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            return models

        for entry in os.listdir(directory):
            entry_path = os.path.join(directory, entry)
            if os.path.isdir(entry_path):
                onnx_path = os.path.join(entry_path, "model.onnx")
                char_path = os.path.join(entry_path, "character.png")
                if os.path.exists(onnx_path) and os.path.exists(char_path):
                    models.append({
                        "id": entry,
                        "name": entry,
                        "modelPath": os.path.join(entry_path, "model.onnx"),
                        "charPath": os.path.join(entry_path, "character.png"),
                        "type": model_type
                    })
        models.sort(key=lambda x: x["name"])
        return models

    def install_zip(self, zip_data: bytes, display_name: str) -> Tuple[bool, str, dict]:
        """安装用户上传的zip包, 解压到 user_upload_dir/{display_name}/"""
        safe_name = display_name.strip().replace(" ", "_")
        if not safe_name:
            safe_name = f"model_{uuid.uuid4().hex[:8]}"

        target_dir = os.path.join(self.user_upload_dir, safe_name)
        if os.path.exists(target_dir):
            import shutil
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
                names = zf.namelist()
                onnx_found = False
                png_found = False
                for name in names:
                    basename = os.path.basename(name)
                    if not basename:
                        continue
                    if basename.lower().endswith('.onnx'):
                        zf.extract(name, target_dir)
                        actual_onnx = os.path.join(target_dir, name)
                        if os.path.basename(actual_onnx) != "model.onnx":
                            dest = os.path.join(target_dir, "model.onnx")
                            os.rename(actual_onnx, dest)
                        onnx_found = True
                    elif basename.lower().endswith('.png'):
                        zf.extract(name, target_dir)
                        actual_png = os.path.join(target_dir, name)
                        if os.path.basename(actual_png) != "character.png":
                            dest = os.path.join(target_dir, "character.png")
                            os.rename(actual_png, dest)
                        png_found = True

                if not onnx_found or not png_found:
                    import shutil
                    shutil.rmtree(target_dir)
                    return False, "ZIP包中缺少 model.onnx 或 character.png", {}

            return True, "安装成功", {
                "id": safe_name,
                "name": display_name,
                "type": "user"
            }
        except zipfile.BadZipFile:
            import shutil
            shutil.rmtree(target_dir)
            return False, "无效的ZIP文件", {}
        except Exception as e:
            import shutil
            shutil.rmtree(target_dir)
            return False, f"安装失败: {str(e)}", {}

    def delete_model(self, model_id: str) -> bool:
        target_dir = os.path.join(self.user_upload_dir, model_id)
        if os.path.exists(target_dir) and target_dir.startswith(os.path.abspath(self.user_upload_dir)):
            import shutil
            shutil.rmtree(target_dir)
            return True
        return False


# ------------------------------------------------------------
# 6. 全局引擎缓存
# ------------------------------------------------------------
_engine_cache: Dict[str, THAEngine] = {}


def get_engine(model_path: str, character_path: str) -> THAEngine:
    cache_key = f"{model_path}::{character_path}"
    if cache_key not in _engine_cache:
        # 只建立外壳实例，绝对不在主线程里调用耗时的 engine.load()
        engine = THAEngine(model_path, character_path)
        _engine_cache[cache_key] = engine
    return _engine_cache[cache_key]


def delete_engine_cache_item(model_path: str, character_path: str):
    cache_key = f"{model_path}::{character_path}"
    if cache_key in _engine_cache:
        del _engine_cache[cache_key]


def clear_engine_cache():
    _engine_cache.clear()