"""
Validador de anuncios
"""

from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class AdValidator:
    """Valida especificaciones de anuncios.

    AVISO: los límites de ``PLATFORM_SPECS`` son **provisionales**. No provienen de
    documentación oficial de plataforma y deben validarse contra las specs vigentes
    antes de usarse como rechazo duro. Se marcan explícitamente para que ningún
    agente los trate como verdad.
    """

    SPECS_PROVISIONAL = True

    PLATFORM_SPECS = {
        "instagram": {
            "min_duration": 1,
            "max_duration": 120,
            "recommended_duration": 15,
            "min_resolution": (640, 480),
            "aspect_ratios": ["1:1", "9:16", "4:5"],
            "min_fps": 24,
            "critical_first_seconds": 3,
        },
        "tiktok": {
            "min_duration": 1,
            # Corregido: el valor anterior (10) rechazaba anuncios estándar de 15-60 s.
            "max_duration": 60,
            "recommended_duration": 9,
            "min_resolution": (720, 720),
            "aspect_ratios": ["9:16"],
            "min_fps": 30,
            "critical_first_seconds": 2,
        },
        "facebook": {
            "min_duration": 1,
            "max_duration": 600,
            "recommended_duration": 30,
            "min_resolution": (640, 480),
            "aspect_ratios": ["1:1", "4:5", "16:9"],
            "min_fps": 24,
            "critical_first_seconds": 5,
        },
        "youtube": {
            "min_duration": 1,
            "max_duration": 600,
            "recommended_duration": 30,
            "min_resolution": (1280, 720),
            "aspect_ratios": ["16:9"],
            "min_fps": 24,
            "critical_first_seconds": 3,
        },
    }

    @classmethod
    def validate_for_platform(
        cls, ad_info: dict, platform: str
    ) -> Tuple[bool, List[str]]:
        """
        Valida si un anuncio cumple specs de plataforma

        Args:
            ad_info: Info del anuncio (de AdLoader)
            platform: Plataforma objetivo

        Returns:
            (es_válido, lista_de_issues)
        """
        issues = []
        platform_lower = platform.lower()

        if platform_lower not in cls.PLATFORM_SPECS:
            return False, [f"Plataforma no reconocida: {platform}"]

        spec = cls.PLATFORM_SPECS[platform_lower]

        # Validar duración
        if "video" in ad_info:
            duration = ad_info["video"]["duration"]
            if duration < spec["min_duration"]:
                issues.append(
                    f"Duración mínima: {spec['min_duration']}s (actual: {duration:.1f}s)"
                )
            if duration > spec["max_duration"]:
                issues.append(
                    f"Duración máxima: {spec['max_duration']}s (actual: {duration:.1f}s)"
                )

            # Validar resolución
            width = ad_info["video"]["width"]
            height = ad_info["video"]["height"]
            min_res = spec["min_resolution"]

            if width < min_res[0] or height < min_res[1]:
                issues.append(
                    f"Resolución mínima: {min_res[0]}x{min_res[1]} (actual: {width}x{height})"
                )

            # Validar FPS
            fps = ad_info["video"]["fps"]
            if fps < spec["min_fps"]:
                issues.append(
                    f"FPS mínimo: {spec['min_fps']} (actual: {fps})"
                )

        is_valid = len(issues) == 0

        if is_valid:
            logger.info(f"✓ Anuncio validado para {platform}")
        else:
            logger.warning(f"❌ Validación fallida para {platform}:")
            for issue in issues:
                logger.warning(f"   - {issue}")

        return is_valid, issues

    @classmethod
    def get_recommendations(cls, platform: str) -> Dict:
        """
        Obtiene recomendaciones para una plataforma

        Args:
            platform: Plataforma objetivo

        Returns:
            Dict con recomendaciones
        """
        platform_lower = platform.lower()

        if platform_lower not in cls.PLATFORM_SPECS:
            return {}

        spec = cls.PLATFORM_SPECS[platform_lower]

        return {
            "platform": platform,
            "recommended_duration": spec["recommended_duration"],
            "optimal_resolution": "x".join(
                map(str, spec["min_resolution"])
            ),
            "aspect_ratios": spec["aspect_ratios"],
            "critical_period_seconds": spec["critical_first_seconds"],
            "tips": cls._get_platform_tips(platform_lower),
        }

    @staticmethod
    def _get_platform_tips(platform: str) -> List[str]:
        """Tips específicos por plataforma"""
        tips = {
            "instagram": [
                "Hook fuerte en los primeros 3 segundos",
                "Usa captions (mucho usuario sin volumen)",
                "Colores vibrantes atraen mejor",
                "Llama a acción clara",
            ],
            "tiktok": [
                "Hook ULTRArápido (2s máx)",
                "Transiciones rápidas",
                "Trending sounds",
                "Vertical video obligatorio (9:16)",
            ],
            "facebook": [
                "Empieza sin volumen (auto-mute)",
                "Captions imprescindibles",
                "Relatable content funciona mejor",
                "Duración flexible (1-2 min óptimo)",
            ],
            "youtube": [
                "Hook en primeros 5 segundos",
                "Thumbnail atractivo (importante)",
                "Puede ser más largo (hasta 10 min)",
                "Calls-to-action a lo largo del video",
            ],
        }

        return tips.get(platform, [])

    @classmethod
    def check_multimodal_sync(
        cls, ad_info: dict, max_time_delta: float = 1.0
    ) -> Tuple[bool, str]:
        """
        Verifica sincronización entre video y audio

        Args:
            ad_info: Info del anuncio
            max_time_delta: Máxima diferencia de duración aceptable (segundos)

        Returns:
            (está_sincronizado, mensaje)
        """
        if "video" not in ad_info or "audio" not in ad_info:
            return True, "Multimodal check skipped (not all modalities present)"

        video_duration = ad_info["video"]["duration"]
        audio_duration = ad_info["audio"]["duration"]

        delta = abs(video_duration - audio_duration)

        if delta <= max_time_delta:
            return True, f"✓ Sincronizado (delta: {delta:.2f}s)"
        else:
            return False, f"❌ Desincronizado (delta: {delta:.2f}s > {max_time_delta}s)"
