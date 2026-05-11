# Changelog de Desarrollo - AudioBrain

## [Unreleased] - 2026-05-11

### Documentación

- Aclarada la licencia del repositorio como Apache-2.0 (ver `LICENSE`).
- Reconciliados nombres de componentes y parámetros por defecto con el código actual.
- Marcado `CHANGELOG_DEV.MD` (raíz) como resumen en inglés; este documento queda como fuente detallada.

### Notas de Parámetros (Estado Actual)

- `AudioMosaicSynthesizer` (sintetizador): `segment_duration=1.0`, `sample_rate=22050`, `crossfade_duration=0.1`
- `AudioGenerationPipeline` (pipeline): `segment_duration=0.1`, `sample_rate=32000`, `crossfade_duration=0.02`

## [0.1.3] - 2026-04-26

### Implementado - FASE 3: Síntesis de Audio por Mosaicing (COMPLETADO)

#### Componentes Creados en esta sesión

**Sintetizador de Audio (`audiobrain/model/synthesizer.py`)**

- `AudioMosaicSynthesizer`: Clase principal para síntesis de audio mediante mosaicing
  - Input: Vectores latentes [batch, seq_len, 512] del Transformer
  - Output: Waveform de audio sintetizado
  - Métodos principales:
    - `build_database()`: Construye base de datos de segmentos de audio con sus latents
    - `synthesize()`: Genera audio usando k-NN search en espacio latent
    - `save_audio()`: Guarda audio sintetizado a archivo .wav
    - `_crossfade_concatenate()`: Une segmentos con crossfading suave
  - Parámetros configurables:
    - segment_duration: 1.0s por defecto (el pipeline usa 0.1s por defecto)
    - crossfade_duration: 0.1s por defecto (el pipeline usa 0.02s por defecto)
    - k_neighbors: 5 mejores matches para variedad
  - Ventana de crossfade: coseno elevado para transiciones naturales

**Pipeline de Generación Completo (`audiobrain/model/generation_pipeline.py`)**

- `AudioGenerationPipeline`: Interfaz unificada para generación completa de audio
  - Combina: PANNs → AudioBrainCore → Mosaic Synthesis
  - Métodos:
    - `build_database()`: Prepara base de datos de síntesis
    - `generate()`: Genera audio desde archivo fuente
    - `generate_and_save()`: Genera y guarda directamente a archivo
    - `generate_batch()`: Procesa múltiples archivos
    - `get_info()`: Información completa del pipeline
  - Flujo simplificado para usuario final

**Actualización de Módulo (`audiobrain/model/__init__.py`)**

- Exporta `AudioMosaicSynthesizer` y `AudioGenerationPipeline`
- Actualiza `__all__` para incluir todos los componentes

#### Especificaciones Técnicas

**Audio Mosaic Synthesizer:**
- Segment duration: 1.0s por defecto (pipeline: 0.1s)
- Sample rate: 22050 por defecto (pipeline: 32kHz)
- Crossfade: 0.1s por defecto (pipeline: 0.02s) con ventana de coseno elevado
- k-NN search: Encuentra segmentos similares en espacio latent 512-dim
- Estrategia: Selecciona aleatoriamente entre top-k para variedad

**Audio Generation Pipeline:**
- Secuencia completa: Raw Audio → PANNs [2048-dim] → AudioBrainCore [512-dim] → Mosaic Synthesis → Audio Output
- Database building: Extrae segmentos y calcula latents automáticamente
- Batch processing: Soporte para múltiples archivos fuente
- Auto-guardado: Opción para guardar resultados directamente

#### Archivos Creados/Modificados

| Archivo                              | Líneas | Descripción                          |
| ------------------------------------ | ------ | ------------------------------------ |
| `audiobrain/model/synthesizer.py`    | 288    | Sintetizador por mosaicing completo  |
| `audiobrain/model/generation_pipeline.py` | 247    | Pipeline de generación unificado       |
| `audiobrain/model/__init__.py`       | +3     | Exporta nuevos componentes           |

#### Flujo de Datos Completado

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│ Source      │ ──→ │ PANNs CNN14  │ ──→ │ AudioBrainCore  │ ──→ │ Mosaic       │ ──→ │ Generated   │
│ Audio       │     │ [2048-dim]   │     │ Transformer     │     │ Synthesizer  │     │ Audio       │
│ .wav/.mp3   │     │              │     │                 │     │ (k-NN)       │     │ .wav        │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘     └─────────────┘
     ↓                    ↓                      ↓                     ↓                    ↓
 [samples]          [seq, 2048]            [1, seq, 512]         [segments]         [audio waveform]
```

**Proceso de Síntesis:**
1. Para cada vector latent objetivo [512-dim]
2. Buscar k vecinos más cercanos en database de latents
3. Seleccionar uno aleatoriamente del top-k
4. Obtener segmento de audio correspondiente
5. Aplicar crossfade entre segmentos consecutivos
6. Retornar waveform completo

#### Estado del Proyecto

✅ **FASE 3 COMPLETADA**: Sistema completo de generación de audio implementado:
- `AudioMosaicSynthesizer` - Síntesis por mosaicing con k-NN
- `AudioGenerationPipeline` - Pipeline unificado end-to-end
- Conexión funcional completa: Audio → Features → Transformer → Audio

🎵 **Ahora el sistema puede crear audio real** desde archivos fuente usando:
```python
from audiobrain import AudioGenerationPipeline

# Inicializar con base de datos
pipeline = AudioGenerationPipeline(
    database_files=['samples/*.wav'],
    max_database_segments=500
)

# Generar nuevo audio
audio, latents = pipeline.generate('source.wav')

# Guardar resultado
pipeline.generate_and_save('source.wav', 'output.wav')
```

---

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
