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
"""Las predicciones se desplazan 5 s al pasado para compensar el retraso hemodinámico.

La convención exacta (si el array devuelto ya está alineado al onset del estímulo)
debe fijarse empíricamente en Fase 0.5 contra el ejemplo del notebook oficial.
NO asumir alineación hasta verificarlo.
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
