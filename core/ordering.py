"""
Convenciones de ordenamiento de vértices de TRIBE v2.

TRIBE v2 predice sobre la malla cortical **fsaverage5**. El proyector de superficies
de Meta (`TribeSurfaceProjector.apply`) apila hemisferios con ``np.vstack([left, right])``
sobre arrays de forma ``(n_vertices_hemi, n_timesteps)``.

Por tanto, en un array de forma ``(n_timesteps, n_vertices)``:

    axis=1  ==  [ hemisferio izquierdo (0..10241) | hemisferio derecho (0..10241) ]

Esto es lo contrario de un orden "intercalado" o "por red". Cualquier ROI que se
construya con slices contiguos arbitrarios (p. ej. ``slice(0, 2000)``) NO corresponde
a ninguna región anatómica: es simplemente el primer 20% del hemisferio izquierdo en
el orden de la malla, que atraviesa múltiples lóbulos y redes.

Los ROIs reales se construyen con máscaras booleanas derivadas de un atlas anotado
sobre la MISMA malla (ver ``service/metrics/roi.py``).
"""

FSAAVERAGE5_VERTICES_PER_HEMI = 10242
FSAAVERAGE5_VERTICES = 2 * FSAAVERAGE5_VERTICES_PER_HEMI  # 20484

TR_SECONDS = 1.0
"""TRIBE v2 predice a 1 TR = 1 segundo (confirmado en el notebook oficial)."""

HEMODYNAMIC_OFFSET_SECONDS = 5.0
"""Retardo hemodinámico que el checkpoint **ya compensa** al construir su objetivo.

El config del checkpoint (``cache/checkpoint/facebook__tribev2/config.yaml``) declara
``offset: 5.0`` en el extractor de fMRI. Ese ``offset`` re-declara la serie de BOLD
empezando en ``-offset``, así que la ventana de estímulo ``[s, s+D]`` se aparea con las
muestras de BOLD crudo ``[s+5, s+5+D]``: el modelo aprende ``estímulo(t) -> BOLD(t+5)``,
que es la respuesta al estímulo del segundo ``t``.

Consecuencia: este número es un desplazamiento **ya aplicado**. NO hay que restarlo al
leer ``preds``. Ver :data:`ALIGNMENT_CONVENTION`.
"""

ALIGNMENT_CONVENTION = "stimulus-aligned"
"""``preds[k]`` es la respuesta al segundo ``k`` del estímulo (no BOLD del instante k).

Verificado empíricamente el 2026-09-13 con ``scripts/validate_alignment.py`` sobre 4
anuncios reales y 12 escalones "mutear desde m". El borde de la respuesta sigue al onset
del estímulo con ``edge = -1.48 + 1.023·m`` (residuo RMS 0.45 TR). Calibrando el mismo
estimador sobre las dos hipótesis simuladas con la HRF canónica, devuelve α = 0.0 TR para
"alineado al estímulo" y α = +5.0 TR para "BOLD crudo": la α observada queda a 1.5 TR de
la primera y a 6.5 TR de la segunda. Método independiente (envelope tracking entre la
envolvente RMS de 1 Hz y la ROI más variable): lag mediano +0.5 TR, r mediano 0.56.

Precisión real de la localización absoluta: ≈1.5 s (el borde del escalón aparece ~1.5 TR
antes de ``m``; causa no determinada, candidato: la ventana de contexto del extractor de
audio, que ve audio posterior al TR predicho). Es un sesgo de ~1.5 s sobre una rejilla de
1 TR — no afecta a la conclusión, que separa 0 s de 5 s.
"""


def split_hemispheres(data):
    """Divide un array con axis final/penúltimo de 20484 en (izquierdo, derecho).

    Acepta forma ``(n_timesteps, 20484)`` o ``(20484, n_timesteps)`` y detecta el eje
    de vértices por tamaño. Devuelve siempre tuplas con el eje de vértices al final.
    """
    import numpy as np

    arr = np.asarray(data)

    if arr.shape[-1] == FSAAVERAGE5_VERTICES:
        left = arr[..., :FSAAVERAGE5_VERTICES_PER_HEMI]
        right = arr[..., FSAAVERAGE5_VERTICES_PER_HEMI:]
        return left, right

    if arr.ndim >= 2 and arr.shape[-2] == FSAAVERAGE5_VERTICES:
        arr = np.moveaxis(arr, -2, -1)
        left = arr[..., :FSAAVERAGE5_VERTICES_PER_HEMI]
        right = arr[..., FSAAVERAGE5_VERTICES_PER_HEMI:]
        return left, right

    raise ValueError(
        f"Se esperaba un eje de {FSAAVERAGE5_VERTICES} vértices en {arr.shape}; "
        "TRIBE v2 predice sobre fsaverage5."
    )


def join_hemispheres(left, right):
    """Reconstruye el orden [izq | der] a partir de dos arrays de 10242 vértices."""
    import numpy as np

    left = np.asarray(left)
    right = np.asarray(right)

    if left.shape[-1] != FSAAVERAGE5_VERTICES_PER_HEMI:
        raise ValueError(
            f"Hemisferio izquierdo debe tener {FSAAVERAGE5_VERTICES_PER_HEMI} "
            f"vértices, tiene {left.shape[-1]}"
        )
    if right.shape[-1] != FSAAVERAGE5_VERTICES_PER_HEMI:
        raise ValueError(
            f"Hemisferio derecho debe tener {FSAAVERAGE5_VERTICES_PER_HEMI} "
            f"vértices, tiene {right.shape[-1]}"
        )

    return np.concatenate([left, right], axis=-1)
