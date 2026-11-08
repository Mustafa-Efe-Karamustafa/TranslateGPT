# -*- coding: utf-8 -*-
"""
generate_data.py

opus-100 (en-tr) verisini indirip tek bir dosyaya yazar:

    data/data.txt

Format: her satirda bir cift, ingilizce ile turkce arasinda ':' var

    I got something.:Bir sey buldum.
    Come on, what the hell did I say?:Hadi ama, ne soyledim ki?

Calistirmak icin:
    python generate_data.py
"""

import io
import os
import sys

from datasets import load_dataset

# Windows konsolu cp1254 kullandigi icin Turkce karakterler '?' basiliyor.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ----------------------------------------------------------------------------
# AYARLAR
# ----------------------------------------------------------------------------

DATA_DIR = "data"
CIKTI = "data.txt"
AYIRICI = ":"        # ingilizceden turkceye gecis isareti


# ----------------------------------------------------------------------------
# FILTRE
# ----------------------------------------------------------------------------

def temiz_mi(en, tr):
    """
    Bozuk cifti ayiklar.

    opus-100'un train kismi buyuk olcude OpenSubtitles'tan geliyor ve
    yaklasik %1.2'si hizasi kaymis. Gercek bozuk kayit ornekleri:
        EN: '_'      TR: 'Kendine dikkat et Veronika.'
        EN: 'Mm.'    TR: 'Isaac, yatmak istiyorsan, yatağa gel.'
        EN: '♪ ♪'    TR: 'Game of Thrones, 5. Sezon, 5. Bölüm'
    """
    en, tr = en.strip(), tr.strip()

    # bos ya da tek karakterlik cop
    if len(en) < 2 or len(tr) < 2:
        return False

    # harf hic yoksa (sadece noktalama / nota isareti) at
    if not any(c.isalpha() for c in en) or not any(c.isalpha() for c in tr):
        return False

    # Uzunluk orani kontrolu. Kisa cumlelerde oran dogal olarak dalgalanir
    # ("What?" / "Ne?" gibi), o yuzden sadece uzun ciftlerde bakiyoruz.
    if len(en) >= 10 and len(tr) >= 10:
        oran = len(tr) / len(en)
        if oran < 0.4 or oran > 2.5:
            return False

    return True


def tek_satir_yap(metin):
    """Satir sonlarini ve fazla bosluklari temizler, boylece her cift tek satir kalir."""
    return " ".join(metin.split())


# ----------------------------------------------------------------------------
# ANA AKIS
# ----------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Veri indiriliyor: Helsinki-NLP/opus-100 (en-tr)")
    ds = load_dataset("Helsinki-NLP/opus-100", "en-tr")

    # train + validation + test hepsi tek dosyaya
    bolumler = ["train", "validation", "test"]
    toplam_kaynak = sum(len(ds[b]) for b in bolumler)
    print(f"  {' + '.join(f'{b}={len(ds[b]):,}' for b in bolumler)}  =  {toplam_kaynak:,} cift")

    yol = os.path.join(DATA_DIR, CIKTI)
    yazilan = atilan = ayirici_belirsiz = 0

    print(f"\nYaziliyor: {yol}")
    with open(yol, "w", encoding="utf-8", newline="\n") as f:
        for bolum in bolumler:
            for ornek in ds[bolum]:
                en = ornek["translation"]["en"]
                tr = ornek["translation"]["tr"]

                if not temiz_mi(en, tr):
                    atilan += 1
                    continue

                en = tek_satir_yap(en)
                tr = tek_satir_yap(tr)

                # Ayirici metnin icinde de geciyorsa satir belirsiz olur,
                # cunku split(':') yanlis yerden boler. Sayisini raporluyoruz.
                if AYIRICI in en:
                    ayirici_belirsiz += 1

                f.write(f"{en}{AYIRICI}{tr}\n")
                yazilan += 1

    mb = os.path.getsize(yol) / 1024 / 1024
    print(f"  {yazilan:,} satir yazildi")
    print(f"  {atilan:,} satir atildi (bozuk hizalama)")
    print(f"  dosya boyutu: {mb:.1f} MB")

    if ayirici_belirsiz:
        oran = ayirici_belirsiz / yazilan * 100
        print(f"\n  DIKKAT: {ayirici_belirsiz:,} satirin ({oran:.2f}%) ingilizce kisminda")
        print(f"  '{AYIRICI}' karakteri var (ornek: '6:00.', 'Say: I do not find...').")
        print(f"  Bu satirlarda split('{AYIRICI}', 1) yanlis yerden boler.")

    print("\n--- ilk 5 satir ---")
    with open(yol, encoding="utf-8") as f:
        for _ in range(5):
            print("  " + f.readline().rstrip("\n"))

    print("\nBitti.")


if __name__ == "__main__":
    main()
