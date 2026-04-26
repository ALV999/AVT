# Changelog de Desarrollo - AudioBrain

## [0.1.2] - 2026-03-31

### Implementado - FASE 2: Integración PANNs (COMPLETADO)

#### Componentes Creados en esta sesión

**Extractor de Características (`audiobrain/model/feature_extractor.py`)**

- `PANNsFeatureExtractor`: Clase principal para extracción de features de audio
  - Input: Audio waveform crudo o path a archivo .wav/.mp3
  - Output: Embeddings de 2048-dim usando PANNs CNN14
  - Métodos principales:
    - `preprocess_audio()`: Carga, resamplea y normaliza audio
    - `extract_features()`: Extrae embeddings en chunks de 1 segundo
    - `extract_for_transformer()`: Formatea output para AudioBrainCore [1, 63, 2048]
  - Auto-detección de dispositivo (CUDA/MPS/CPU)
  - Fallback a implementación manual si no hay modelos pre-entrenados

**Pipeline de Procesamiento (`audiobrain/model/pipeline.py`)**

- `AudioProcessingPipeline`: Conecta PANNs con AudioBrainCore
  - Integra feature extraction + transformer encoding
  - Métodos:
    - `process_audio()`: Procesa un archivo individual
    - `process_batch()`: Procesa múltiples archivos en lote
    - `get_model_info()`: Retorna información del modelo
  - Output: Tuple (raw_embeddings [1, 63, 2048], latent_vectors [1, 63, 512])

**Actualización de Módulo (`audiobrain/model/__init__.py`)**

- Exporta `PANNsFeatureExtractor` junto con componentes existentes
- Actualiza `__all__` para incluir nuevo extractor

#### Especificaciones Técnicas

**PANNs Feature Extractor:**
- Sample rate: 32kHz (configurable)
- Embedding dimension: 2048 (PANNs CNN14)
- Chunk duration: 1.0 segundos por embedding
- Hop length: 320 samples (~10ms)
- Modelos soportados: HuggingFace transformers + fallback manual

**Audio Processing Pipeline:**
- Secuencia completa: Raw Audio → PANNs → [2048-dim] → AudioBrainCore → [512-dim]
- Target sequence length: 63 (para visualización 63x63)
- Batch processing support para múltiples archivos
- Auto-interpolación/padding para alcanzar seq_len=63

#### Archivos Creados/Modificados

| Archivo                              | Líneas | Descripción                          |
| ------------------------------------ | ------ | ------------------------------------ |
| `audiobrain/model/feature_extractor.py` | 258    | Extractor PANNs completo             |
| `audiobrain/model/pipeline.py`       | 201    | Pipeline de procesamiento integrado  |
| `audiobrain/model/__init__.py`       | +2     | Exporta PANNsFeatureExtractor        |

#### Flujo de Datos Completado

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│ Raw Audio   │ ──→ │ PANNs CNN14  │ ──→ │ AudioBrainCore  │ ──→ │ Latent Vectors│
│ .wav/.mp3   │     │ [2048-dim]   │     │ Transformer     │     │ [512-dim]    │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
      ↓                    ↓                      ↓                     ↓
  [samples]          [seq, 2048]            [1, seq, 512]        [1, 63, 512]
```

#### Estado del Proyecto

✅ **FASE 2 COMPLETADA**: Pipeline completo implementado:
- `PANNsFeatureExtractor` - Extracción de características desde audio crudo
- `AudioProcessingPipeline` - Integración completa PANNs → Transformer
- Conexión funcional entre módulos Fase 1 y Fase 2

⚠️ **Nota**: La creación de audio real ocurrirá en FASE 3 (Síntesis por Mosaicing)

---

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
