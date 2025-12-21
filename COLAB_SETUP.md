# 🚀 Google Colab Setup Guide for 4DGaussians-Enhanced

Bu doküman, 4DGaussians-Enhanced projesini Google Colab'de çalıştırmak için gereken tüm adımları ve düzeltmeleri açıklar.

## 📋 Düzeltilen Sorunlar (Fixed Issues)

### 1. PyTorch Sürüm Uyumsuzluğu (PyTorch Version Incompatibility)

**Sorun:** Eski PyTorch 1.13.1 sürümü Colab ve SAM2 ile uyumsuz.

**Çözüm:** 
- `requirements.txt` dosyasından sabit PyTorch sürüm pinleri kaldırıldı
- Colab'in varsayılan PyTorch kurulumu kullanılıyor (2.x)
- `numpy<2.0.0` kısıtlaması eklendi (uyumluluk için)

### 2. C++ Derleme Hataları (C++ Compilation Errors)

**Sorun:** 
- `AT_CHECK` makrosu PyTorch 2.x'te kaldırıldı
- `FLT_MAX` için `#include <cfloat>` eksik
- Ninja olmadan derleme çok yavaş

**Çözüm:** `scripts/colab_setup.py`
- Otomatik patch sistemi: `AT_CHECK` → `TORCH_CHECK` dönüşümü
- Eksik `#include <cfloat>` direktiflerini otomatik ekler
- Ninja build sistemini kurar (daha hızlı derleme)

### 3. SAM2 Paket İsmi Yanlış (SAM2 Package Name Error)

**Sorun:** PyPI'da `segment-anything-2` resmi paketi yok.

**Çözüm:**
- GitHub'dan direkt kurulum: `pip install "git+https://github.com/facebookresearch/sam2.git"`
- `colab_setup.py` script'i otomatik olarak GitHub'dan kurar

### 4. SAM 2.0 vs 2.1 Uyumsuzluğu (SAM 2.0 vs 2.1 Incompatibility)

**Sorun:**
- SAM 2.1 config dosyalarını `configs/sam2.1/` altına taşıdı
- Eski config isimleri (`sam2_hiera_l.yaml`) artık geçersiz
- HuggingFace modelleri SAM 2.0, kod SAM 2.1 bekliyor → weight mismatch

**Çözüm:** `scripts/download_sam2.py`
- SAM 2.1 config path'leri: `sam2.1_hiera_l.yaml`, `sam2.1_hiera_b+.yaml`, vb.
- SAM 2.1 modelleri HuggingFace'den indirilir: `facebook/sam2.1-hiera-large`
- Config dosyaları GitHub'dan otomatik indirilir

### 5. SAM2 Config Dosyası Eksikliği (SAM2 Config File Missing)

**Sorun:** `build_sam2(config_file, checkpoint)` fiziksel `.yaml` dosyası gerektirir. 
pip install config dosyalarını içermiyor.

**Çözüm:** `scripts/download_sam2.py`
- GitHub'dan config dosyalarını indirir
- `checkpoints/configs/` klasörüne kaydeder
- Tüm model boyutlarını destekler (tiny, small, base, large)

### 6. SAM2 Placeholder Kodu (SAM2 Placeholder Code)

**Sorun:** Mevcut kod gerçek SAM2 inference yapmıyordu, sadece beyaz maske üretiyordu.

**Çözüm:** `utils/sam2_utils.py` - Tam yeniden yazım
- `SAM2MaskGenerator` class'ı implementasyonu
- YOLO ile obje tespiti
- SAM2.1 ile segmentasyon
- Gerçek maske üretimi

### 7. YOLO Türkçe Prompt Sorunu (YOLO Turkish Prompt Issue)

**Sorun:** Kullanıcı "insan" yazarsa YOLO "person" döndürdüğü için eşleşmiyor.

**Çözüm:** `utils/sam2_utils.py`
```python
CLASS_ALIASES = {
    "insan": "person",
    "araba": "car",
    "köpek": "dog",
    # ...
}
```

### 8. Google Drive'a Doğrudan Yazma (Direct Drive Writing)

**Sorun:** Training sırasında Drive'a yazmak 2-3x yavaşlatır, bağlantı kopabilir.

**Çözüm:** Notebook Cell 7
- Training lokal diskte yapılır (`/content/output/`)
- Sadece training bittikten sonra Drive'a kopyalanır
- Çok daha hızlı ve stabil

### 9. COLMAP Eksikliği (COLMAP Missing)

**Sorun:** Ham resimlerden (sadece `images/` klasörü) kamera pozları hesaplanamıyor.

**Çözüm:** `scripts/run_colmap.py`
- COLMAP kurulumu (`apt-get install colmap`)
- Feature extraction, matching, mapping pipeline
- 4DGaussians formatına dönüştürme
- Notebook Cell 3: Opsiyonel COLMAP processing

### 10. Veri Hazırlama Eksikliği (Data Preparation Missing)

**Sorun:** Drive'dan veri çekme, unzip, format doğrulama mekanizması yoktu.

**Çözüm:** Notebook Cell 2
- Google Drive mount
- Zip extraction veya klasör kopyalama
- Format algılama (blender, colmap, dynerf, multicam, raw_images)
- Otomatik yapı doğrulama

---

## 📦 Yeni Dosyalar (New Files)

### 1. `scripts/colab_setup.py`
Colab ortamını hazırlar:
- Ninja build sistemi kurulumu
- SAM2 GitHub'dan kurulum
- C++ dosyalarını patch'ler (AT_CHECK → TORCH_CHECK, cfloat includes)

### 2. `scripts/download_sam2.py`
SAM2.1 config ve model indirici:
- Config dosyalarını GitHub'dan indir
- Model checkpoint'lerini HuggingFace'den indir
- Tüm model boyutlarını destekler

### 3. `scripts/run_colmap.py`
COLMAP wrapper scripti:
- Feature extraction, matching, mapping
- 4DGaussians formatına dönüştürme
- GPU acceleration desteği

### 4. `utils/sam2_utils.py` (Yeniden Yazıldı)
Gerçek SAM2.1 + YOLO implementasyonu:
- `SAM2MaskGenerator` class'ı
- YOLO ile obje tespiti
- SAM2.1 ile segmentasyon
- Türkçe class name aliasları

### 5. `notebooks/4DGS_Enhanced.ipynb` (Yeniden Yazıldı)
9-cell Colab workflow:
1. **Installation**: Tüm bağımlılıkları kur
2. **Data Setup**: Drive mount, unzip, format validation
3. **COLMAP Processing**: Opsiyonel - ham resimler için
4. **SAM2 Mask Generation**: YOLO + SAM2.1
5. **Mask Preview**: Interaktif maske önizleme
6. **Training Configuration**: Preset'lerle kolay konfigürasyon
7. **Training**: Lokal disk'te training, sonra Drive'a kopyala
8. **Render**: Video oluştur
9. **Export PLY**: Opsiyonel - frame başına point cloud

---

## 🎯 Kullanım (Usage)

### Hızlı Başlangıç (Quick Start)

1. **Colab'de aç:**
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/semhfe/4DGaussians-Enhanced/blob/master/notebooks/4DGS_Enhanced.ipynb)

2. **GPU'yu aktif et:**
   Runtime > Change runtime type > GPU > A100 (önerilen) veya T4

3. **Cell'leri sırayla çalıştır:**
   - Cell 1: Kurulum (5-10 dakika)
   - Cell 2: Veri hazırlama
   - Cell 3: COLMAP (opsiyonel, ham resimler varsa)
   - Cell 4: Maske oluşturma
   - Cell 5: Maske önizleme
   - Cell 6: Training config
   - Cell 7: Training başlat
   - Cell 8: Render
   - Cell 9: PLY export (opsiyonel)

### Training Presets

| Preset | İterasyonlar | Süre (A100) | Süre (T4) | Kullanım |
|--------|-------------|-------------|-----------|----------|
| `quick_test` | 14,000 | ~30 dk | ~1 saat | Hızlı test |
| `standard` | 30,000 | ~1.5 saat | ~3 saat | Dengeli kalite |
| `high_quality` | 60,000 | ~3-4 saat | ~6-8 saat | En iyi kalite |
| `fast_motion` | 45,000 | ~2 saat | ~4 saat | Dans/aksiyon |

### Veri Formatları

Desteklenen formatlar:
- **Blender/NeRF Synthetic**: `transforms_train.json` içeren klasör
- **COLMAP**: `sparse/` ve `images/` içeren klasör
- **Multi-camera**: `cam01/`, `cam02/`, ... klasörleri
- **Raw Images**: Sadece resimler (COLMAP ile işlenir)

---

## 🐛 Sorun Giderme (Troubleshooting)

### "AT_CHECK not found" hatası
**Çözüm:** Cell 1'de `colab_setup.py` çalıştırıldığından emin olun.

### "SAM2 config not found" hatası
**Çözüm:** Cell 1'de `download_sam2.py` çalıştırıldığından emin olun.

### Maskeler tamamen beyaz/siyah
**Çözüm:** Cell 4'te `DETECTION_PROMPT` ve `CONFIDENCE_THRESHOLD` ayarlayın.

### Training çok yavaş
**Çözüm:** 
- `quick_test` preset'i kullanın
- A100 GPU seçin (Colab Pro gerekli)
- `EVERY_N_FRAMES` değerini artırın (maske için)

### Drive bağlantısı kopuyor
**Çözüm:** Training lokal disk'te yapılıyor, sadece sonunda Drive'a kopyalanıyor. Sorun olmamalı.

---

## 📝 Notlar (Notes)

### SAM2.1 Model Boyutları

| Model | Checkpoint Boyutu | VRAM | Hız | Kalite |
|-------|------------------|------|-----|--------|
| `tiny` | ~38 MB | ~2 GB | En hızlı | En düşük |
| `small` | ~90 MB | ~3 GB | Hızlı | İyi |
| `base` | ~149 MB | ~4 GB | Orta | Çok iyi |
| `large` | ~213 MB | ~6 GB | Yavaş | En iyi |

### Performans İpuçları

1. **GPU seçimi:** A100 > V100 > T4
2. **Training preset:** İlk test için `quick_test`, sonra `standard`
3. **Maske generation:** İlk defa `large` model ile test edin
4. **Veri boyutu:** Çok büyük dataset'ler için `EVERY_N_FRAMES > 1` kullanın

---

## 🤝 Katkıda Bulunma (Contributing)

Bu düzeltmeler şunları içerir:
- ✅ Tüm PyTorch 2.x uyumluluk sorunları
- ✅ SAM2.1 tam implementasyonu
- ✅ COLMAP entegrasyonu
- ✅ Türkçe dil desteği
- ✅ Colab-optimized workflow

Yeni sorun bulursanız veya öneriniz varsa issue açın!

---

## 📚 Referanslar (References)

- [4DGaussians Original](https://github.com/hustvl/4DGaussians)
- [SAM2 GitHub](https://github.com/facebookresearch/sam2)
- [YOLO Ultralytics](https://github.com/ultralytics/ultralytics)
- [COLMAP](https://colmap.github.io/)

---

**Son Güncelleme:** 2024-12-21
