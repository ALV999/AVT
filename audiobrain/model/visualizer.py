"""
Artistic Visualization Module — latent spectrogram with ASCII Chladni overlay.

Two complementary views derived entirely from the ML model's latent vectors:
  1. Latent Spectrogram: activation grid where (t, dim) = learned audio feature
  2. Chladni Nodal Lines: circular plate interference patterns from latent modes
"""

from __future__ import annotations

import base64, io, os, sys, struct, zlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── Color support ──

_colorama_available = False
try:
    import colorama; colorama.init(); _colorama_available = True
except ImportError: pass

def _supports_color() -> bool:
    if not _colorama_available:
        if os.name == 'nt':
            try:
                import ctypes; k=ctypes.windll.kernel32; m=ctypes.c_ulong()
                k.GetConsoleMode(k.GetStdHandle(-11), ctypes.byref(m))
                return bool(m.value & 0x0004)
            except: return False
        if not sys.stdout.isatty(): return False
        t = os.environ.get('TERM',''); return t not in ('dumb','')
    return True

def _ansi_rgb(r, g, b, ch): return f"\033[38;2;{r};{g};{b}m{ch}\033[0m"
def _plain_char(r, g, b, ch): return ch
_color_fn = _ansi_rgb if _supports_color() else _plain_char

# ── Character ramps ──

CHAR_RAMPS: dict[str, str] = {
    "dots":      "  ··∘∘●●●•◦◆◇◈◉◊○◎●",
    "ascii":     "  .:;-=+*#%@★★★★",
    "blocks":    "  ░░▒▒▓▓█████",
    "braille":   "  ⠁⠃⠇⠏⠟⠿⣿",
    "binary":    " 0101010101",
    "shades":    "  ░░▒░▒▓▒▓██▓█",
    "minimal":   "  .·:·::▪▪▪",
    "hexdots":   "  ⠄⠆⠇⠏⠿⣿⣿",
    "cubes":     "  ░░▒▒▓▓████▓▓",
    "arrows":    "  ·›»►▸▶▼▲◆",
    "sparks":    "  ·˙∴⋯⋰⋱⚡✨",
    "pixels":    "  ░░▒▒▓▓██████",
    "cistercian":"  ⠀⠁⠃⠇⠏⠟⠿⣿⣿",
    "morse":     "  ·-▪▬▬▪",
    "lcd":       "  ▓███▓",
}

# ── Color schemes ──

def _ocean(v):
    v=max(0.,min(1.,v))
    if v<.33: t=v/.33; return (0,0,int(128*t))
    elif v<.66: t=(v-.33)/.33; return (0,int(200*t),128+int(127*t))
    else: t=(v-.66)/.34; return (int(200*t),200+int(55*t),255)

def _interp_forest(v):
    v=max(0.,min(1.,v))
    if v<.5: t=v/.5; return (0,int(180*t),0)
    else: t=(v-.5)/.5; return (int(255*t),180+int(75*t),int(255*t))

def _interp_sunset(v):
    v=max(0.,min(1.,v))
    if v<.33: t=v/.33; return (int(128*(1-t)+200*t),0,int(128*(1-t)))
    elif v<.66: t=(v-.33)/.33; return (200+int(55*t),int(100*t),0)
    else: t=(v-.66)/.34; return (255,100+int(155*t),int(200*t))

def _interp_aurora(v):
    v=max(0.,min(1.,v))
    if v<.33: t=v/.33; return (0,int(128+127*t),int(128+127*(1-t)))
    elif v<.66: t=(v-.33)/.33; return (int(128*t),int(255*(1-t)),int(128*t))
    else: t=(v-.66)/.34; return (128+int(127*t),int(128*t),int(128+127*(1-t)))

def _interp_mono(v): g=int(max(0.,min(1.,v))*255); return (g,g,g)

def _interp_neon(v):
    v=max(0.,min(1.,v))
    if v<.25: t=v/.25; return (int(255*t),0,255)
    elif v<.5: t=(v-.25)/.25; return (0,int(255*t),255)
    elif v<.75: t=(v-.5)/.25; return (0,255,int(255*(1-t)))
    else: t=(v-.75)/.25; return (int(255*t),255,0)

def _interp_dusk(v):
    v=max(0.,min(1.,v))
    if v<.5: t=v/.5; return (int(40+60*t),0,int(80+80*t))
    else: t=(v-.5)/.5; return (int(100+155*t),int(40*t),int(160-80*t))

def _interp_ember(v):
    v=max(0.,min(1.,v))
    if v<.33: t=v/.33; return (int(60+40*t),0,int(10*t))
    elif v<.66: t=(v-.33)/.33; return (int(100+155*t),int(80*t),0)
    else: t=(v-.66)/.34; return (255,int(80+175*t),int(100*t))

def _interp_glacier(v):
    v=max(0.,min(1.,v))
    if v<.5: t=v/.5; return (int(180*t),int(200*t),int(200+55*t))
    else: t=(v-.5)/.5; return (int(180+75*t),int(200+55*t),255)

def _heat(v):
    v=max(0.,min(1.,v))
    if v<.33: t=v/.33; return (int(255*t),0,0)
    elif v<.66: t=(v-.33)/.33; return (255,int(255*t),0)
    else: t=(v-.66)/.34; return (255,255,int(255*t))

COLOR_SCHEMES: dict[str, callable] = {
    "heat": _heat, "ocean": _ocean, "forest": _interp_forest,
    "sunset": _interp_sunset, "aurora": _interp_aurora, "mono": _interp_mono,
    "neon": _interp_neon, "dusk": _interp_dusk, "ember": _interp_ember,
    "glacier": _interp_glacier,
}

# ── Latent Spectrogram ──

class LatentSpectrogram:
    def __init__(self, grid_size=64): self.grid_size = grid_size

    def compute(self, latents):
        if latents.dim() == 3: latents = latents[0]
        latents = latents.cpu().numpy().astype(np.float32)
        n_steps, n_dims = latents.shape; gs = self.grid_size
        if n_steps != gs:
            xo = np.linspace(0,1,n_steps); xn = np.linspace(0,1,gs)
            li = np.zeros((gs,n_dims),dtype=np.float32)
            for d in range(n_dims): li[:,d] = np.interp(xn,xo,latents[:,d])
            latents = li
        fd = min(n_dims,gs); spec = latents[:,:fd]
        if fd < gs:
            pad = np.zeros((gs,gs-fd),dtype=np.float32); spec = np.concatenate([spec,pad],axis=1)
        result = np.zeros_like(spec)
        for col in range(gs):
            cd = spec[:,col]; lo,hi = np.percentile(cd,5), np.percentile(cd,95)
            rng = hi-lo
            result[:,col] = np.clip((cd-lo)/rng,0,1) if rng>1e-8 else .5
        return result.astype(np.float32)

class RealSpectrogram:
    def __init__(self, gs=64): self.gs=gs
    def compute(self, wf, sr):
        gs=self.gs
        if len(wf)==0: return np.zeros((gs,gs),dtype=np.float32)
        nperseg=gs*2; hop=max(1,len(wf)//gs); n_frames=min(gs,max(1,(len(wf)-nperseg)//hop+1))
        if n_frames<2: return np.zeros((gs,gs),dtype=np.float32)
        w=np.hanning(nperseg); spec=np.zeros((gs//2+1,gs),dtype=np.float32)
        for i in range(n_frames):
            s=i*hop; e=min(s+nperseg,len(wf))
            if e-s<nperseg//2: continue
            fr=np.zeros(nperseg,dtype=np.float32); sg_=wf[s:e]; fr[:len(sg_)]=sg_*w[:len(sg_)]
            fft=np.abs(np.fft.rfft(fr)); col=int(i*gs/n_frames)
            if col>=gs: col=gs-1
            bins=min(len(fft),gs//2+1); spec[:bins,col]+=fft[:bins]
        spec=np.flipud(spec[:gs//2+1,:gs])
        if spec.shape[0]>gs: spec=spec[:gs,:]
        elif spec.shape[0]<gs: spec=np.concatenate([np.zeros((gs-spec.shape[0],gs),dtype=np.float32),spec],axis=0)
        if spec.shape[1]>gs: spec=spec[:,:gs]
        elif spec.shape[1]<gs: spec=np.concatenate([spec,np.zeros((gs,gs-spec.shape[1]),dtype=np.float32)],axis=1)
        spec=np.log1p(spec); vmin,vmax=spec.min(),spec.max()
        return ((spec-vmin)/(vmax-vmin)).astype(np.float32) if vmax-vmin>1e-8 else np.full((gs,gs),.5,dtype=np.float32)

# ── Chladni Oscilloscope ──

class ChladniOscilloscope:
    def __init__(self, gs=64, nm=128):
        self.gs=gs; self.nm=min(nm,128)
        self._modes = [(i%16, i//16+1) for i in range(self.nm)]
        y,x=np.indices((gs,gs)); cx=(gs-1)/2.; cy=(gs-1)/2.
        self._r=np.clip(np.sqrt(((x-cx)/cx)**2+((y-cy)/cy)**2)/np.sqrt(2),0,1)
        self._t=np.arctan2(y-cy,x-cx); self._last_field=None

    def compute(self, latents):
        if latents.dim()==3: latents=latents[0]
        latents=latents.cpu().numpy().astype(np.float32)
        nt,nd=latents.shape; gs=self.gs
        nmp=min(self.nm,nd//2)
        ma=np.zeros(nmp,dtype=np.float32); ap=np.zeros(nmp,dtype=np.float32)
        ps=abs(int(np.sum(latents[0,:min(8,nd)]))+hash(tuple(latents[0,:min(4,nd)].astype(float))))%100000
        np.random.seed(ps); po=np.random.randn(nmp).astype(np.float32)*.3
        for i in range(nmp):
            ad=4*min(i,nd//4-1); pd=ad+1
            ma[i]=np.max(np.abs(latents[:,ad])) if ad<nd else np.random.randn()*.1
            ap[i]=(np.mean(latents[:,pd])+po[i])*np.pi
        si=np.argsort(ma)[::-1]; tn=max(8,nmp//3); ai=si[:tn]
        cm=np.ones_like(self._r); ctr=self._r<.15; cm[ctr]=.2+.8*(self._r[ctr]/.15)
        field=np.zeros((gs,gs),dtype=np.float32)
        for idx in ai:
            amp=ma[idx]
            if amp<1e-4: continue
            ph=ap[idx]; m,n=self._modes[idx]; w=1/(1+.3*n)
            rd=np.sin(np.pi*n*self._r); ag=np.cos(m*self._t+ph); field+=amp*w*rd*ag
        field*=cm; self._last_field=field.copy()
        lm,th=self._detect_lines(field)
        return lm,th

    def _detect_lines(self, field):
        gs=self.gs; sf=np.sign(field)
        gy=np.zeros_like(field); gx=np.zeros_like(field)
        gy[1:-1]=field[2:]-field[:-2]; gx[:,1:-1]=field[:,2:]-field[:,:-2]
        gm=np.sqrt(gy**2+gx**2); gn=np.zeros_like(gm)
        for col in range(gs):
            cg=gm[:,col]; hi=np.percentile(cg,95)
            gn[:,col]=np.clip(cg/hi,0,1) if hi>1e-8 else .5
        rl=np.zeros((gs,gs),dtype=bool)
        he=sf[:,:-1]!=sf[:,1:]; rl[:,:-1]|=he; rl[:,1:]|=he
        ve=sf[:-1,:]!=sf[1:,:]; rl[:-1,:]|=ve; rl[1:,:]|=ve
        de=sf[:-1,:-1]!=sf[1:,1:]; rl[:-1,:-1]|=de; rl[1:,1:]|=de
        d2=sf[:-1,1:]!=sf[1:,:-1]; rl[:-1,1:]|=d2; rl[1:,:-1]|=d2
        lines=rl.copy()
        th=np.zeros((gs,gs),dtype=np.float32)
        th[lines]=np.clip(1-gn[lines],.05,1); th[lines]=th[lines]**.4
        return lines, th.astype(np.float32)

# ── Visualizer ──

class AudioBrainVisualizer:
    def __init__(self, grid_size=64, chars="dots", colors="heat", chladni_chars="lines", chladni_color="white", force_color=False):
        self.grid_size=grid_size
        self.chars=CHAR_RAMPS.get(chars,CHAR_RAMPS["ascii"])
        self.colors=COLOR_SCHEMES.get(colors,_heat)
        self.chladni_chars=chladni_chars; self.chladni_color=chladni_color
        self._use_color=force_color or _supports_color()
        self.spectrogram=LatentSpectrogram(grid_size)
        self.oscilloscope=ChladniOscilloscope(grid_size)
        self.real_spec=RealSpectrogram(gs=64)

    def compute_views(self, latents):
        lm,th=self.oscilloscope.compute(latents); cf=self.oscilloscope._last_field
        return {"spectrogram":self.spectrogram.compute(latents),"chladni_lines":lm,"chladni_thickness":th,"chladni_field":cf}

    def _build_char_grid(self, field, overlay_mask=None):
        gs=field.shape[0]; nc=len(self.chars); cells=[]
        for y in range(gs):
            for x in range(gs):
                v=float(field[y,x])
                if overlay_mask is not None and overlay_mask[y,x]:
                    ch=self.chars[-1]
                    cells.append(f'<span class="ch">{ch}</span>')
                else:
                    idx=min(int(v*(nc-1)),nc-1); ch=self.chars[idx]
                    r,g,b=self.colors(v)
                    cells.append(f'<span style="color:rgb({r},{g},{b})">{ch}</span>')
        return "\n".join(cells)

    def build_html(self, latents, audio_data=None, title="AudioBrain", metadata=None, waveform=None, sample_rate=32000):
        views=self.compute_views(latents); gs=self.grid_size
        spec=views["spectrogram"]; lines=views["chladni_lines"]
        latent_grid=self._build_char_grid(spec, overlay_mask=lines)
        real_grid=""
        if waveform is not None and len(waveform)>0:
            rf=self.real_spec.compute(waveform,sample_rate)
            real_grid=self._build_char_grid(rf)
        ns=latents.shape[1] if latents.dim()==3 else latents.shape[0]; nd=latents.shape[-1]
        meta_rows=""
        if metadata:
            for k,v in metadata.items():
                meta_rows+=f'<div class="kv-row"><span class="kv-key">{k}</span><span class="kv-val">{v}</span></div>'
        else: meta_rows+=f'<div class="kv-row"><span class="kv-key">file</span><span class="kv-val">{title}</span></div>'
        b64=base64.b64encode(audio_data).decode("utf-8") if audio_data else ""

        # ── SVG micrographics ──
        _radar='<rect x="6" y="0" width="1" height="1" fill="var(--dim)" opacity="0.5"/>' \
               '<rect x="6" y="1" width="1" height="1" fill="var(--dim)" opacity="0.5"/>' \
               '<rect x="6" y="2" width="1" height="1" fill="var(--dim)" opacity="0.5"/>' \
               '<rect x="6" y="3" width="1" height="1" fill="var(--dim)" opacity="0.5"/>' \
               '<rect x="7" y="0" width="1" height="1" fill="var(--dim)" opacity="0.5"/>' \
               '<rect x="7" y="3" width="1" height="1" fill="var(--dim)" opacity="0.5"/>' \
               '<rect x="8" y="0" width="1" height="1" fill="var(--dim)" opacity="0.5"/>' \
               '<rect x="8" y="3" width="1" height="1" fill="var(--dim)" opacity="0.5"/>' \
               '<rect x="9" y="0" width="1" height="1" fill="var(--dim)" opacity="0.5"/>' \
               '<rect x="9" y="1" width="1" height="1" fill="var(--dim)" opacity="0.5"/>' \
               '<rect x="9" y="2" width="1" height="1" fill="var(--dim)" opacity="0.5"/>' \
               '<rect x="9" y="3" width="1" height="1" fill="var(--dim)" opacity="0.5"/>' \
               '<rect x="5" y="0" width="1" height="1" fill="var(--dim)" opacity="0.8"/>' \
               '<rect x="5" y="1" width="1" height="1" fill="var(--dim)" opacity="0.8"/>' \
               '<rect x="5" y="2" width="1" height="1" fill="var(--dim)" opacity="0.8"/>' \
               '<rect x="5" y="3" width="1" height="1" fill="var(--dim)" opacity="0.8"/>' \
               '<rect x="10" y="0" width="1" height="1" fill="var(--dim)" opacity="0.8"/>' \
               '<rect x="10" y="1" width="1" height="1" fill="var(--dim)" opacity="0.8"/>' \
               '<rect x="10" y="2" width="1" height="1" fill="var(--dim)" opacity="0.8"/>' \
               '<rect x="10" y="3" width="1" height="1" fill="var(--dim)" opacity="0.8"/>' \
               '<rect x="0" y="1" width="16" height="1" fill="var(--dim)" opacity="0.6"/>' \
               '<rect x="0" y="2" width="16" height="1" fill="var(--dim)" opacity="0.6"/>'
        _blocks='<rect x="2" y="0" width="1" height="1" fill="var(--dim)" opacity="0.5"/>' \
                '<rect x="5" y="0" width="1" height="1" fill="var(--dim)" opacity="0.7"/>' \
                '<rect x="8" y="0" width="1" height="1" fill="var(--dim)" opacity="0.6"/>' \
                '<rect x="9" y="0" width="1" height="1" fill="var(--dim)" opacity="0.4"/>' \
                '<rect x="13" y="0" width="1" height="1" fill="var(--dim)" opacity="0.6"/>' \
                '<rect x="15" y="0" width="1" height="1" fill="var(--dim)" opacity="0.3"/>' \
                '<rect x="0" y="1" width="1" height="1" fill="var(--dim)" opacity="0.3"/>' \
                '<rect x="2" y="1" width="1" height="1" fill="var(--dim)" opacity="0.4"/>' \
                '<rect x="3" y="1" width="1" height="1" fill="var(--dim)" opacity="0.8"/>' \
                '<rect x="7" y="1" width="1" height="1" fill="var(--dim)" opacity="0.5"/>' \
                '<rect x="11" y="1" width="1" height="1" fill="var(--dim)" opacity="0.7"/>' \
                '<rect x="14" y="1" width="1" height="1" fill="var(--dim)" opacity="0.4"/>' \
                '<rect x="1" y="2" width="1" height="1" fill="var(--dim)" opacity="0.6"/>' \
                '<rect x="4" y="2" width="1" height="1" fill="var(--dim)" opacity="0.3"/>' \
                '<rect x="6" y="2" width="1" height="1" fill="var(--dim)" opacity="0.8"/>' \
                '<rect x="8" y="2" width="1" height="1" fill="var(--dim)" opacity="0.5"/>' \
                '<rect x="10" y="2" width="1" height="1" fill="var(--dim)" opacity="0.7"/>' \
                '<rect x="12" y="2" width="1" height="1" fill="var(--dim)" opacity="0.4"/>' \
                '<rect x="15" y="2" width="1" height="1" fill="var(--dim)" opacity="0.6"/>'
        _wave='<rect x="0" y="2" width="1" height="1" fill="var(--dim)" opacity="0.3"/>' \
              '<rect x="1" y="2" width="1" height="1" fill="var(--dim)" opacity="0.8"/>' \
              '<rect x="2" y="3" width="1" height="1" fill="var(--dim)" opacity="0.9"/>' \
              '<rect x="3" y="3" width="1" height="1" fill="var(--dim)" opacity="0.85"/>' \
              '<rect x="4" y="2" width="1" height="1" fill="var(--dim)" opacity="0.6"/>' \
              '<rect x="5" y="1" width="1" height="1" fill="var(--dim)" opacity="0.6"/>' \
              '<rect x="6" y="0" width="1" height="1" fill="var(--dim)" opacity="0.85"/>' \
              '<rect x="7" y="0" width="1" height="1" fill="var(--dim)" opacity="0.9"/>' \
              '<rect x="8" y="1" width="1" height="1" fill="var(--dim)" opacity="0.75"/>' \
              '<rect x="9" y="1" width="1" height="1" fill="var(--dim)" opacity="0.35"/>' \
              '<rect x="10" y="2" width="1" height="1" fill="var(--dim)" opacity="0.8"/>' \
              '<rect x="11" y="3" width="1" height="1" fill="var(--dim)" opacity="0.9"/>' \
              '<rect x="12" y="3" width="1" height="1" fill="var(--dim)" opacity="0.85"/>' \
              '<rect x="13" y="2" width="1" height="1" fill="var(--dim)" opacity="0.55"/>' \
              '<rect x="14" y="1" width="1" height="1" fill="var(--dim)" opacity="0.6"/>' \
              '<rect x="15" y="0" width="1" height="1" fill="var(--dim)" opacity="0.85"/>'

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>AVT — {title}</title><style>
@import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
*{{margin:0;padding:0;box-sizing:border-box}}
:root{{--bg:#000000;--fg:#bbbbbb;--dim:#555555;--accent:#ffffff;--border:#333333;--panel:#0a0a0a;--btn-bg:#0a0a0a;--btn-fg:#bbbbbb;--btn-bd:#333333;--btn-hv:#ffffff;--wf:rgba(255,255,255,.55);--pb:#0a0a0a;--pg:#222222;--pff:#bbbbbb}}
[data-theme="light"]{{--bg:#eeeeee;--fg:#333333;--dim:#999999;--accent:#000000;--border:#bbbbbb;--panel:#f5f5f5;--btn-bg:#f5f5f5;--btn-fg:#333333;--btn-bd:#bbbbbb;--btn-hv:#000000;--wf:rgba(0,0,0,.5);--pb:#f5f5f5;--pg:#dddddd;--pff:#333333}}
html,body{{height:100%}}
body{{background:var(--bg);color:var(--fg);font-family:'VT323','Courier New',Courier,monospace;font-size:16px;line-height:1.2}}
.container{{display:flex;height:100vh;gap:1px;background:var(--border)}}
.sidebar{{width:260px;min-width:260px;background:var(--panel);display:flex;flex-direction:column;padding:8px;gap:4px;overflow:hidden}}
.sidebar-header{{text-align:center;border-bottom:1px solid var(--border);padding-bottom:6px;flex-shrink:0}}
.sidebar-header h1{{font-size:20px;font-weight:400;color:var(--accent);letter-spacing:.3em;text-transform:uppercase;margin:0}}
.sidebar-header .sub{{font-size:11px;color:var(--dim);margin-top:1px}}
.section{{border:1px solid var(--border);padding:4px 6px;flex-shrink:0}}
.section-title{{font-size:11px;color:var(--accent);text-transform:uppercase;letter-spacing:.2em;border-bottom:1px solid var(--border);padding-bottom:1px;margin-bottom:2px}}
.kv-row{{display:flex;justify-content:space-between;margin:0}}
.kv-key{{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.1em}}
.kv-val{{color:var(--fg);font-size:12px;text-align:right}}
.explain{{font-size:11px;color:var(--dim);line-height:1.2}}
.explain b{{color:var(--fg);font-weight:normal;display:block;font-size:12px;margin-top:4px;margin-bottom:0}}
.micrograph{{flex-shrink:0;height:12px;opacity:.6;overflow:hidden}}
.micrograph svg{{display:block;width:100%;height:100%}}
.btn{{background:var(--btn-bg);color:var(--btn-fg);border:1px solid var(--btn-bd);padding:4px 10px;cursor:pointer;font-size:12px;font-family:inherit;text-transform:uppercase;letter-spacing:.15em;margin-top:auto;flex-shrink:0}}
.btn:hover{{background:var(--btn-hv);color:var(--bg);border-color:var(--accent)}}
.main{{flex:1;background:var(--bg);display:flex;flex-direction:column;min-width:0}}
.main-scroll{{flex:1;overflow-y:auto;padding:8px 12px 12px 12px;display:flex;flex-direction:column;gap:10px}}
.main-sticky{{flex-shrink:0;position:sticky;top:0;z-index:10;background:var(--bg);padding:12px 12px 8px 12px;display:flex;flex-direction:column;gap:10px;border-bottom:1px solid var(--border);margin-bottom:4px}}
.main-header{{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--border);padding-bottom:4px;flex-shrink:0}}
.main-header h2{{font-size:24px;font-weight:400;color:var(--accent);text-transform:uppercase;letter-spacing:.2em;margin:0}}
.main-header .meta{{font-size:14px;color:var(--dim);text-align:right}}
.controls{{display:flex;align-items:center;gap:10px;flex-shrink:0}}
.play-btn{{background:var(--btn-bg);color:var(--btn-fg);border:1px solid var(--btn-bd);padding:4px 14px;cursor:pointer;font-size:18px;font-family:inherit;min-width:52px;text-align:center}}
.play-btn:hover{{background:var(--btn-hv);color:var(--bg);border-color:var(--accent)}}
.time{{font-size:16px;color:var(--fg);min-width:48px;font-variant-numeric:tabular-nums;text-align:right}}
.progress{{flex:1;height:2px;background:var(--border);cursor:pointer;position:relative}}
.progress-fill{{height:100%;background:var(--fg);width:0%;transition:width .1s linear}}
.viz-panel{{border:1px solid var(--border);padding:8px 10px;flex-shrink:0;position:relative;overflow:hidden}}
.viz-panel.real{{overflow:hidden}}
.viz-label{{font-size:14px;color:var(--accent);text-transform:uppercase;letter-spacing:.2em;margin-bottom:6px}}
.viz-label .pre{{color:var(--accent);opacity:.6}}
.viz-row{{display:flex;gap:10px;flex-shrink:0}}
.viz-row .viz-panel{{flex:1;min-width:0;height:100px;display:flex;flex-direction:column;align-items:center;justify-content:center}}
.waveform-outer{{width:100%;overflow:hidden}}
.waveform{{white-space:pre;font-size:10px;font-family:'VT323','Courier New',monospace;line-height:1;color:var(--dim);overflow:hidden;width:100%;padding:2px 0}}
.legend{{display:flex;justify-content:center;gap:2em;margin:4px 0;font-size:13px;color:var(--dim);flex-shrink:0}}
.legend span{{display:flex;align-items:center;gap:4px}}
.legend .sw{{display:inline-block;width:8px;height:8px;border:1px solid var(--border)}}
.ascii-grid{{display:grid;grid-template-columns:repeat({gs},1fr);grid-template-rows:repeat({gs},1fr);align-items:center;justify-items:center;font-family:'VT323','Courier New',monospace;font-size:14px;line-height:.85;letter-spacing:-1px;user-select:none;width:100%;overflow:hidden}}
.ascii-grid.real{{grid-template-columns:repeat({gs},1fr);font-size:{max(7,11-gs//16)}px;line-height:.4;overflow:hidden}}
.ascii-grid.real span{{line-height:.3;display:inline-block;transform:scaleY(.6)}}
.viz-panel.latent{{overflow:visible;display:flex;flex-direction:column}}
.viz-panel.latent .ascii-grid{{align-self:center;width:100%;max-height:calc(100% - 26px)}}
.ch{{color:var(--accent);font-size:14px;line-height:1}}
.pnl-btn{{background:var(--btn-bg);color:var(--btn-fg);border:1px solid var(--btn-bd);padding:2px 10px;cursor:pointer;font-size:13px;font-family:inherit;text-transform:uppercase;letter-spacing:.1em;position:absolute;top:6px;right:10px;z-index:5}}
.pnl-btn:hover{{background:var(--btn-hv);color:var(--bg);border-color:var(--accent)}}
footer{{color:var(--dim);font-size:13px;text-align:center;padding:4px 0;flex-shrink:0}}
.tooltip{{display:none;position:absolute;z-index:10;background:rgba(2,2,8,.96);color:#ddd;padding:2px 6px;font-size:13px;font-family:'VT323',monospace;pointer-events:none}}
::-webkit-scrollbar{{width:8px;height:8px}}
::-webkit-scrollbar-track{{background:var(--bg);border-left:1px solid var(--border)}}
::-webkit-scrollbar-thumb{{background:var(--border);border:1px solid var(--bg)}}
::-webkit-scrollbar-thumb:hover{{background:var(--dim)}}
*{{scrollbar-width:thin;scrollbar-color:var(--border) var(--bg)}}
@media(max-width:800px){{.container{{flex-direction:column}}.sidebar{{width:100%;min-width:0;height:auto;overflow-y:auto}}.main{{overflow-y:visible}}}}
</style></head><body data-theme="dark"><div class="container">
<div class="sidebar">
<div class="sidebar-header"><h1>A/V/T</h1><div class="sub">audiobrain</div></div>
<div class="micrograph"><svg width="100%" height="100%" viewBox="0 0 16 4" preserveAspectRatio="none">{_radar}</svg></div>
<div class="section"><div class="section-title">Generation</div>{meta_rows}</div>
<div class="micrograph"><svg width="100%" height="100%" viewBox="0 0 16 3" preserveAspectRatio="none">{_blocks}</svg></div>
<div class="section"><div class="section-title">Model</div>
<div class="kv-row"><span class="kv-key">grid</span><span class="kv-val">{gs}&times;{gs}</span></div>
<div class="kv-row"><span class="kv-key">dims</span><span class="kv-val">{nd}</span></div>
<div class="kv-row"><span class="kv-key">seg</span><span class="kv-val">{ns}</span></div>
</div>
<div class="micrograph"><svg width="100%" height="100%" viewBox="0 0 16 4" preserveAspectRatio="none">{_wave}</svg></div>
<div class="section"><div class="section-title">Visualization</div>
<div class="explain">
<b>Latent Spectrogram</b>Each cell = learned dimension activation. Colored by intensity, textured by character density.
<b>Chladni Nodal Lines</b>Dominant resonant modes traced as bright chars at sign-change nodes.
<b>Real Spectrogram</b>STFT frequency analysis &mdash; ML vs. signal.
<b>Live Waveform</b>Real-time oscilloscope.
</div></div>
<div class="section"><div class="section-title">About</div>
<div class="explain"><b>A/VT</b> maps sound to learned features and back. The visualizations ARE the ML product.</div>
</div>
<button class="btn" onclick="toggleTheme()">\u25d0 Toggle Theme</button>
</div>
<div class="main">
<div class="main-sticky">
<div class="main-header"><h2>{title}</h2><div class="meta">{ns} seg &middot; {nd} dims &middot; {gs}&times;{gs}</div></div>
<div class="controls">
<button class="play-btn" id="pb" onclick="tp()">\u25b6</button>
<div class="progress" id="pr" onclick="sk(event)"><div class="progress-fill" id="pf"></div></div>
<span class="time" id="td">0:00</span>
</div>
<audio id="ae" preload="metadata" src="data:audio/wav;base64,{b64}"></audio>
<div class="viz-row">
<div class="viz-panel"><div class="viz-label"><span class="pre">></span> Waveform</div>
<div class="waveform-outer"><pre id="wf-pre" class="waveform"></pre></div></div>
<div class="viz-panel real"><div class="viz-label"><span class="pre">></span> Real Spectrogram</div>
<div class="ascii-grid real">{real_grid if real_grid else latent_grid}</div></div>
</div>
</div>
<div class="main-scroll">
<div class="viz-panel latent" id="latent-viz" onmousemove="st(event)" onmouseleave="ht()">
<div class="viz-label"><span class="pre">></span> Latent + Chladni</div>
<button class="pnl-btn" onclick="downloadPNG()" title="Download as PNG">\u2913 PNG</button>
<div class="ascii-grid" id="latent-grid">{latent_grid}</div>
<div id="tt" class="tooltip"></div>
</div>
<div class="legend">
<span><span class="sw" style="background:var(--dim)"></span>low activation</span>
<span><span class="sw" style="background:var(--accent)"></span>peak activation</span>
<span><span class="sw" style="background:var(--accent);box-shadow:0 0 3px var(--accent)"></span>chladni nodes</span>
</div>
<footer>AV/T v0.5 &middot; {ns} segments</footer>
</div></div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script>
function toggleTheme(){{var b=document.body;b.setAttribute('data-theme',b.getAttribute('data-theme')==='light'?'dark':'light')}}
var gridEl=document.getElementById('latent-grid');var tt=document.getElementById('tt');var GS={gs};
function st(e){{var r=gridEl.getBoundingClientRect();var x=Math.floor((e.clientX-r.left)/r.width*GS);var y=Math.floor((e.clientY-r.top)/r.height*GS);if(x>=0&&x<GS&&y>=0&&y<GS){{tt.style.display='block';tt.style.left=(e.clientX-r.left+10)+'px';tt.style.top=(e.clientY-r.top-16)+'px';tt.textContent='['+x+','+y+']'}}}}
function ht(){{tt.style.display='none'}}
var ae=document.getElementById('ae');var wfPre=document.getElementById('wf-pre');
var pb=document.getElementById('pb');var pr=document.getElementById('pr');
var pf=document.getElementById('pf');var td=document.getElementById('td');
var ac=null,an=null,aid=null;
var rampChars="{''.join(c for c in self.chars if c.strip() or c == ' ')}".replace(/ /g,'');
if(!rampChars)rampChars=' .:-=+*#%@';
function tp(){{if(ae.paused){{ae.play();pb.textContent='\u23f8'}}else{{ae.pause();pb.textContent='\u25b6'}}}}
ae.addEventListener('play',function(){{pb.textContent='\u23f8';
if(!ac){{ac=new(window.AudioContext||window.webkitAudioContext)();var src=ac.createMediaElementSource(ae);an=ac.createAnalyser();an.fftSize=512;src.connect(an);an.connect(ac.destination)}}
if(ac.state==='suspended')ac.resume();draw();}});
ae.addEventListener('pause',function(){{pb.textContent='\u25b6';cancelAnimationFrame(aid)}});
ae.addEventListener('ended',function(){{pb.textContent='\u25b6';cancelAnimationFrame(aid);wfPre.textContent='';pf.style.width='0%';td.textContent='0:00'}});
ae.addEventListener('timeupdate',function(){{if(ae.duration){{var pct=(ae.currentTime/ae.duration)*100;pf.style.width=pct+'%';var m=Math.floor(ae.currentTime/60);var s=Math.floor(ae.currentTime%60);td.textContent=m+':'+(s<10?'0'+s:s)}}}});
function sk(e){{if(!ae.duration)return;var rect=pr.getBoundingClientRect();var pct=(e.clientX-rect.left)/rect.width;ae.currentTime=pct*ae.duration}}
function draw(){{aid=requestAnimationFrame(draw);if(!an)return;var bl=an.frequencyBinCount;var tdata=new Uint8Array(bl);an.getByteTimeDomainData(tdata);
var COLS=80,ROWS=12;var chars=rampChars;var nc=chars.length;
var grid=[];for(var r=0;r<ROWS;r++){{grid[r]=[];for(var c=0;c<COLS;c++)grid[r][c]=' '}}
var step=Math.max(1,Math.floor(bl/COLS));
for(var c=0;c<COLS;c++){{
  var sum=0;var count=0;
  for(var i=c*step;i<Math.min((c+1)*step,bl);i++){{sum+=tdata[i];count++}}
  var avg=count>0?sum/count:128;
  var row=Math.round((avg/255)*(ROWS-1));
  var intensity=Math.abs(avg-128)/128;
  var chIdx=Math.min(nc-1,Math.floor(intensity*(nc-1)));
  grid[ROWS-1-row][c]=chars[chIdx];
}}
var lines=[];
for(var r=0;r<ROWS;r++)lines.push(grid[r].join(''));
wfPre.textContent=lines.join('\\n');
}}
function downloadPNG(){{var el=document.getElementById('latent-grid');if(!el)return;html2canvas(el,{{backgroundColor:getComputedStyle(document.body).getPropertyValue('--bg')}}).then(function(canvas){{var a=document.createElement('a');a.download='avt_latent_chladni.png';a.href=canvas.toDataURL('image/png');a.click()}}).catch(function(e){{console.error('PNG download failed:',e)}})}}
</script></body></html>"""

    def save_html(self, path, latents, audio_data=None, title="AudioBrain", metadata=None, waveform=None, sample_rate=32000):
        Path(path).write_text(self.build_html(latents,audio_data,title,metadata,waveform,sample_rate),encoding="utf-8"); return str(path)


def visualize_latents(latents, grid_size=128, chars="dots", colors="heat", title="AudioBrain", show="overlay"):
    viz=AudioBrainVisualizer(grid_size=grid_size,chars=chars,colors=colors); viz.render_terminal(latents,title=title,show=show)