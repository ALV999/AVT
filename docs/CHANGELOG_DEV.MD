# Changelog de Desarrollo - AudioBrain

## [0.1.1] - 2026-03-31

### Implementado - FASE 1: Modelo Transformer (COMPLETADO)

#### Componentes Creados en esta sesión

**Modelo Principal (`audiobrain/model/core.py`)**

- `core.py` - `AudioBrainCore`: Modelo principal que integra ProjectionHead + SoundscapeTransformer
  - Input: `[batch, seq_len, 2048]` (embeddings de PANNs)
  - Output: `[batch, seq_len, 512]` (espacio latente contextualizado)
  - Métodos: `forward()`, `encode()`, `get_config()`, `count_parameters()`, `__repr__()`
  - Auto-configuración de dispositivo (CUDA/MPS/CPU)
  - Validación de dimensiones en tiempo de ejecución

#### Especificaciones de AudioBrainCore

- Integra ProjectionHead y SoundscapeTransformer en un solo modelo
- Configurable mediante BrainConfig dataclass
- Incluye método `encode()` para claridad semántica
- Método `count_parameters()` para inspección del modelo
- Representación string detallada con `__repr__()`

#### Archivos Creados/Modificados

| Archivo                          | Líneas | Descripción                        |
| -------------------------------- | ------ | ---------------------------------- |
| `audiobrain/model/core.py`       | 150    | Modelo principal AudioBrainCore    |

#### Validaciones Incluidas

- Validación de dimensiones de entrada (2048 -> 512)
- Auto-detección y movimiento al dispositivo correcto
- Aplicación automática de seed para reproducibilidad
- Conteo de parámetros entrenables

#### Estado del Proyecto

✅ **FASE 1 COMPLETADA**: Todos los componentes del Transformer están implementados:
- `BrainConfig` - Configuración completa
- `ProjectionHead` - Capa de proyección
- `SoundscapeTransformer` - Encoder Transformer
- `AudioBrainCore` - Modelo principal integrado

---

## [0.1.0] - 2026-03-31

### Implementado - FASE 1: Modelo Transformer

#### Componentes Creados

**Estructura del Módulo (`audiobrain/`)**

- `__init__.py` - Paquete principal con exports
- `model/__init__.py` - Subpaquete del modelo

**Configuración (`audiobrain/model/`)**

- `config.py` - Dataclass `BrainConfig` con hiperparámetros

**Capas del Modelo**

- `projection.py` - `ProjectionHead` (Linear -> LayerNorm -> Dropout)
- `transformer.py` - `SoundscapeTransformer` (nn.TransformerEncoder)

#### Especificaciones

**ProjectionHead:**

- Input: `[batch, 2048]`
- Output: `[batch, 512]`
- Arqu.: Linear(2048, 512) -> LayerNorm -> Dropout(0.1)

**SoundscapeTransformer:**

- Input: `[batch, seq_len, 512]`
- Output: `[batch, seq_len, 512]`
- d_model=512, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1
- batch_first=True

#### Archivos Creados

| Archivo                           | Líneas | Descripción               |
| --------------------------------- | ------ | ------------------------- |
| `audiobrain/__init__.py`          | 15     | Paquete principal         |
| `audiobrain/model/__init__.py`    | 8      | Subpaquete modelo         |
| `audiobrain/model/config.py`      | 87     | Configuración BrainConfig |
| `audiobrain/model/projection.py`  | 89     | Capa ProjectionHead       |
| `audiobrain/model/transformer.py` | 100    | Encoder Transformer       |

#### Validaciones Incluidas

- Verificación de divisibilidad: `nhead` debe dividir `d_model`
- Rango de dropout: `[0, 1)`
- Auto-detección de device: CUDA -> MPS -> CPU
- Validación de dimensiones en `forward()`
- Gradient clipping (`max_norm=1.0`) configurado en `BrainConfig`

#### Próximos Pasos

1. Implementar `AudioBrainCore` (core.py)
2. Tests unitarios (`audiobrain/tests/`)
3. Integración con PANNs (Fase 2)

---

## Próximas Fases

- **Fase 2:** Pipeline de Audio (PANNs + carga WAV)
- **Fase 3:** Sintesis por Mosaicing (k-NN + crossfade)
- **Fase 4:** Visualizacion ASCII 63x63
- **Fase 5:** Integracion y CLI
