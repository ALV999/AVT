# A/VT - Prompt Maestro de Desarrollo

# Proyecto de Tesis: Audio/Visual Transformer

## 1. CONTEXTO GENERAL DEL PROYECTO

A/VT es un sistema generativo de paisajes sonoros que crea composiciones
originales a partir de samples de audio proporcionados por el usuario,
acompañadas de una visualización estructural en matriz ASCII de 63x63 caracteres.

Objetivos del Sistema:

1. Generar nuevas composiciones sonoras mediante reordenamiento inteligente de samples
2. Visualizar la estructura sonora en tiempo real mediante matriz ASCII codificada
3. Operar en entorno local con recursos moderados

Arquitectura Completa (4 Modulos):

- Modulo 1: Extraccion de Features (PANNs) - Pendiente
- Modulo 2: Modelo Transformer (CORE) - EN PROGRESO
- Modulo 3: Sintesis por Mosaicing (k-NN) - Pendiente
- Modulo 4: Visualizacion ASCII 63x63 - Pendiente

## 2. FASE ACTUAL: FASE 1 - IMPLEMENTACION DEL MODELO

Objetivo de Esta Fase:
Implementar el nucleo del modelo Transformer que aprende secuencias de embeddings
y genera nuevas trayectorias en el espacio latente de 512 dimensiones.

Alcance de Esta Fase - INCLUIDO:

- ProjectionHead (2048 a 512)
- SoundscapeTransformer
- AudioBrainCore (clase principal)
- Entrenamiento (train_step)
- Generacion autoregresiva

Alcance de Esta Fase - NO INCLUIDO:

- Carga de archivos de audio
- Sintesis de audio final
- Busqueda k-NN
- Visualizacion ASCII
- Integracion con PANNs real

## 3. ESPECIFICACION DE INPUTS Y OUTPUTS

Etapa 1: Proyeccion (ProjectionHead)
Input: embedding_panns (torch.Tensor, shape: [batch, 2048])
Output: embedding_proyectado (torch.Tensor, shape: [batch, 512])

Etapa 2: Secuenciacion (SoundscapeTransformer)
Input: secuencia_embeddings (torch.Tensor, shape: [batch, seq_len, 512])
Output: secuencia_contextual (torch.Tensor, shape: [batch, seq_len, 512])

Etapa 3: Generacion (AudioBrainCore)
Input: start_embedding (torch.Tensor, shape: [1, 1, 512])
Output: generated_sequence (torch.Tensor, shape: [1, length, 512])

Resumen de Dimensiones:

- batch_size: 8-32 (entrenamiento), 1 (inferencia)
- seq_len: Numero de segmentos de 0.5s (ej. 60 = 30s de audio)
- 2048: Dimension original de embeddings PANNs
- 512: Dimension del espacio latente del Transformer

## 4. ARQUITECTURA TECNICA DETALLADA

ProjectionHead:

- Funcion: Adaptar salida de PANNs al espacio del Transformer
- Capas: Linear(2048, 512) -> LayerNorm -> Dropout(0.1)
- Input: [batch, 2048]
- Output: [batch, 512]

SoundscapeTransformer:

- Arquitectura: nn.TransformerEncoder
- d_model: 512
- nhead: 8
- num_layers: 2
- dim_feedforward: 2048
- dropout: 0.1
- batch_first: True
- Input: [batch, seq_len, 512]
- Output: [batch, seq_len, 512]

AudioBrainCore (Clase Principal):
Metodos requeridos:

- **init**(config: BrainConfig): Inicializa capas y device
- forward(embeddings, mask) -> Tensor: Forward pass completo
- train_step(batch, optimizer, criterion) -> float: Un paso de entrenamiento
- generate(start_emb, length, temperature) -> Tensor: Generacion autoregresiva
- save_checkpoint(path) -> None: Guarda pesos del modelo
- load_checkpoint(path) -> None: Carga pesos del modelo

## 5. ESPECIFICACIONES TECNICAS

Framework y Librerias:

- PyTorch: 2.0+
- Python: 3.9+
- Librerias permitidas: torch, numpy, dataclasses
- Librerias NO permitidas en este modulo: librosa, soundfile, panns_inference

Requisitos de Codigo:

- Type hinting: Obligatorio en todas las funciones
- Documentacion: Docstrings en formato Google style
- Dispositivo: Manejo automatico de CUDA/MPS/CPU
- Estabilidad: Gradient clipping (max_norm=1.0) en train_step
- Validacion: Asserts o raise ValueError para dimensiones incorrectas

Configuracion (BrainConfig):

- d_model: int = 512
- nhead: int = 8
- num_layers: int = 2
- dim_feedforward: int = 2048
- dropout: float = 0.1
- learning_rate: float = 0.001
- device: str = 'cuda' si disponible, sino 'cpu'
- max_seq_len: int = 512

## 6. ESTRUCTURA DE ARCHIVOS

audiobrain/
**init**.py
model/
**init**.py
config.py (BrainConfig dataclass)
projection.py (ProjectionHead)
transformer.py (SoundscapeTransformer)
core.py (AudioBrainCore)
docs/
PROMPT_MASTER.md (este archivo)
CONTEXT.md (estado del proyecto)
CHANGELOG_DEV.md (registro de cambios)
tests/
test_projection.py
test_transformer.py
test_core.py

## 7. EJEMPLO DE USO ESPERADO

import torch
from audiobrain.model import AudioBrainCore, BrainConfig

config = BrainConfig(
d_model=512,
nhead=8,
num_layers=2,
learning_rate=0.001,
device='mps' if torch.backends.mps.is_available() else 'cpu'
)

model = AudioBrainCore(config)
model = model.to(config.device)

batch_size = 4
seq_len = 60
dummy_input = torch.randn(batch_size, seq_len, 2048).to(config.device)

output = model(dummy_input)

# output.shape debe ser (batch_size, seq_len, 512)

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.MSELoss()
loss = model.train_step(dummy_input, optimizer, criterion)

start_emb = torch.randn(1, 1, 512).to(config.device)
generated = model.generate(start_emb, length=60, temperature=0.8)

# generated.shape debe ser (1, 60, 512)

## 8. INSTRUCCIONES PARA EL ASISTENTE DE CODIGO

Al Iniciar Cada Sesion:

1. Leer este archivo PROMPT_MASTER.md completo
2. Leer docs/CONTEXT.md para estado actual del proyecto
3. Leer docs/CHANGELOG_DEV.md para ver ultima tarea completada

Al Generar Codigo:

1. Priorizar claridad sobre optimizacion prematura
2. Incluir validacion de dimensiones en todos los forward pass
3. Incluir comentarios que referencien este documento
4. Mantener codigo modular para facilitar pruebas unitarias

Al Completar una Tarea:

1. Actualizar docs/CHANGELOG_DEV.md con tarea, archivos, descripcion
2. Actualizar docs/CONTEXT.md si el estado del proyecto cambio

Restricciones Estrictas:

- NO modificar archivos fuera de audiobrain/model/ en esta fase
- NO importar librerias de audio (librosa, soundfile, panns_inference)
- NO implementar k-NN, sintesis o visualizacion (fases posteriores)
- NO cambiar hiperparametros arquitectonicos sin consultar

## 9. PROXIMAS FASES (Referencia)

Fase 1: Modelo Transformer - EN PROGRESO
Fase 2: Pipeline de Audio (PANNs + carga WAV) - Pendiente
Fase 3: Sintesis por Mosaicing (k-NN + crossfade) - Pendiente
Fase 4: Visualizacion ASCII 63x63 - Pendiente
Fase 5: Integracion y CLI - Pendiente

FIN DEL PROMPT MAESTRO
