# Bitki Yaprak Hastalik Siniflandirmasi (PlantVillage)

**Yazar:** isim: Omar A.M Issa // Ogrenci Numarasi: 220212901

## Genel Bakis
Bu proje, goruntulerden bitki yaprak hastaliklarini siniflandirmak icin uctan uca bir is akis boru hatti kurar. Veri alma ve on isleme, klasik bir makine ogrenmesi temel cizgisi (HOG + SVM), bir transfer ogrenmesi modeli (ResNet-18) ve karmasiklik matrisleri, precision-recall egrileri ve Grad-CAM gibi gorsel teshislerle degerlendirme adimlarini kapsar. Ana notebook tum akisi calistirir ve cikti artefaktlarini `outputs` altina kaydeder.

## Neden bu proje
Bitki hastaliklari verimi ve kaliteyi dusurur; erken tespit tarim icin kritiktir. Bu proje iki tamamlayici yaklasimi inceler:
- Hizli egitilebilen ve bir mantik kontrolu saglayan hafif, yorumlanabilir bir temel cizgi (elle tasarlanmis ozellikler + SVM).
- Daha zengin gorsel oruntuleri yakalayan ve genellikle daha yuksek dogruluk saglayan bir derin ogrenme modeli (ResNet-18).

Amac, bu yaklasimlari standart bir veri kumesi uzerinde karsilastirmak, degisimleri ve takaslari belgelemek ve tekrarlanabilir bir referans boru hatti sunmaktir.

## Veri kumesi
Bu proje, saglikli ve hastalikli urun yapraklarinin etiketli goruntulerini iceren PlantVillage veri kumesini kullanir. Bu repoda islenmis veri kumesi su ozelliklere sahiptir:
- **24 sinif**
- **Train:** 19,862 goruntu
- **Val:** 4,256 goruntu
- **Test:** 4,257 goruntu
- **Toplam:** 28,375 goruntu

Siniflardan biri `Background_without_leaves` olup, digerleri urun/hastalik veya urun/saglikli kategorileridir.

### Neden bu veri kumesi
- Bitki hastaligi siniflandirmasi icin yaygin kullanilan bir karsilastirma veri kumesidir; sonuclari karsilastirmayi kolaylastirir.
- Etiketler acik ve tutarlidir; bu da model performansini anotasyon gürültusunden ayirmaya yardimci olur.
- Hem klasik ML temel cizgilerini hem de transfer ogrenmesini destekleyecek kadar buyuktur, ama asiri buyuk degildir.

### Nereden edinilir
Boru hatti iki kaynagi destekler:
1) **TensorFlow Datasets (tercih edilen)**  
   Notebook, `plant_village` veya `plant_village/plantvillage` deneyecek ve mevcutsa otomatik indirecektir.

2) **Kaggle (yedek)**  
   Veri kumesi adi: `emmarex/plantdisease`  
   Calistirilan komut:
   ```
   kaggle datasets download -d emmarex/plantdisease

PlantVillage veri kumesini manuel indirdiyseniz, klasoru sinif alt klasorleriyle birlikte `data/raw/` altina yerlestirin.

## Proje yapisi
```
project_leaf_disease/
  notebook/
    leaf_disease_classification.ipynb   # Uctan uca is akis ve deneyler
  src/
    config.py                           # Yollar ve genel hiperparametreler
    data.py                             # Veri kumesi indirme, bolme, yukleyiciler
    features.py                         # HOG + renk histogrami ozellik cikarimi
    models.py                           # SVM boru hatti ve ResNet-18 olusturucu
    train.py                            # Egitim donguleri, taramalar, arama
    eval.py                             # Metrikler, tahminler, PR egrileri
    viz.py                              # Gorsellestirme araclari ve Grad-CAM
    utils_seed.py                       # Tekrarlanabilirlik ve ortam bilgisi
  data/
    raw/                                # Indirilen veri (TFDS veya Kaggle)
    processed/                          # Train/val/test bolunmeleri + splits.json
  outputs/
    figures/                            # Grafikler ve gorsel teshisler
    tables/                             # CSV ozetleri
    models/                             # Kaydedilmis kontrol noktalar
    logs/                               # Opsiyonel loglar
  report/                               # Rapor kaynaklari ve PDF cikti
  requirements.txt                      # Python bagimliliklari
  README.md
```

## Gereksinimler ve kurulum
- Python 3.10+ onerilir
- Jupyter Notebook veya JupyterLab
- Temel kutuphaneler: PyTorch, torchvision, scikit-learn, scikit-image, numpy, pandas, matplotlib, tensorflow-datasets, kaggle
- Tam liste `requirements.txt` icinde

### Kurulum (venv)
Sanal ortam olusturun ve etkinlestirin, sonra bagimliliklari yukleyin:
```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate

Kurulum:
```bash
python -m pip install -r requirements.txt
```



### Yaygin hatalar ve cozumler
- `ModuleNotFoundError`: notebook cekirdegi `.venv` kullanmiyor. Cekirdegi yeniden secin ve yeniden baslatin.
- `Import 'torch' could not be resolved`: CPU guvenli PyTorch kurun: `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
- `Kaggle API error`: `kaggle.json` dosyasini `C:\Users\<user>\.kaggle\kaggle.json` altina koyun veya `KAGGLE_USERNAME` / `KAGGLE_KEY` ayarlayin.

## Calistirma
1) Notebooku acin:
   ```bash
   jupyter notebook notebook/leaf_disease_classification.ipynb
   ```
2) Tum hucreleri yukaridan asagi calistirin. Ciktilar `outputs/` altina kaydedilir.

## Daha hizli SVM aramasi (alt kume)
Klasik HOG+SVM icin hiperparametre aramasi varsayilan olarak egitim verisinin bir alt
kumesi uzerinde calisir (stratified). En iyi ayarlar bulunduktan sonra final model tum
egitim verisiyle yeniden fit edilir.

Alt kume boyutunu `run_hog_svm_search(..., search_subset_size=8000)` ile degistirebilir veya
`search_subset_size=0` vererek kapatabilirsiniz.

## Sinirlamalar ve dezavantajlar
- **Yuksek bellek kullanimi:** HOG + renk histogrami cikarimi on binlerce goruntu icin buyuk ozellik matrisleri olusturur. Bu, RAM'i ciddi sekilde kullanabilir ve dusuk bellekli makinelerde notebooku yavaslatabilir veya dondurebilir.
- **Agir egitim yuku:** ResNet-18 egitimi ve taramalari CPU'da uzun surebilir, CUDA sistemlerinde ise buyuk GPU bellegi kullanarak tepkiselligi etkileyebilir.
- **Disk kullanimi:** Boru hatti goruntuleri `data/processed/` altina kopyalar ve ciktilari `outputs/` altinda saklar; bu da depolama kullanimini artirir.

Notebook yavas hissediyorsa `BATCH_SIZE` degerini dusurun, goruntu boyutunu kucultun veya taramalari atlayin.



## Tekrarlanabilirlik
- Python, NumPy, PyTorch ve DataLoader worker'lari icin sabit tohumlar `src/utils_seed.py` icinde ayarlanmistir.
- Bolunmeler stratified olarak yapilir ve `data/processed/splits.json` altina kaydedilir.
- Ortam bilgileri, notebook basliginda `get_env_info()` ile yazdirilir.

## Cikti artefaktlari
- `outputs/figures/`: sinif dagilimi, ornek gridler, karmasiklik matrisleri, PR egrileri, Grad-CAM, yanlis siniflandirmalar
- `outputs/tables/`: ozet tablolar (CSV)
- `outputs/models/`: kaydedilmis model kontrol noktalar
- `outputs/logs/`: opsiyonel loglar

## Notlar
- Notebook, SVM icin bir hiperparametre taramasi ve DL model icin kucuk bir LR/WD taramasi icerir.
- Ince ayar vs dondurulmus backbone ablation'i, optimizasyon ve ablation gereksinimlerini karsilamak icin eklenmistir.
