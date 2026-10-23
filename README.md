# bicikl_voznja.py 🚴

Program u kojem **jedan veštački neuron** (logistička regresija) uči da odluči da li treba voziti bicikl na osnovu vremenskih uslova.

Neuron ne dobija gotova pravila. Počinje sa nasumičnim težinama, a zatim ih uči iz primera pomoću gradient descent-a.

## Zahtevi

- Python 3.10+ (koristi se `int | None` anotacija)
- [NumPy](https://numpy.org/)

```bash
pip install numpy
```

## Pokretanje

```bash
python bicikl_voznja.py
```

## Kako radi

### Ulazi

Svaki ulaz je vrednost od 0 do 1:

| Ulaz | Značenje |
|------|----------|
| `sunčano` | 1.0 = potpuno sunčano, 0.0 = oblačno |
| `toplo` | 1.0 = vruće, 0.0 = ledeno |
| `pada kiša` | 1.0 = pada kiša, 0.0 = ne pada |

Primer: `[1.0, 0.7, 0.0]` znači sunčano, prilično toplo, bez kiše.

### Formula

```
z      = dot(ulazi, težine) + bias
izlaz  = sigmoid(z) = 1 / (1 + e^(-z))
```

- **Težine** pokazuju koliko je koji ulaz bitan. Pozitivna težina podstiče vožnju, negativna je odvraća.
- **Bias** pomera prag odluke.
- **Sigmoid** pretvara zbir `z` u verovatnoću između 0 i 1.
- Ako je verovatnoća veća od `THRESHOLD` (0.5), odluka je da se vozi bicikl.

### Trening

Neuron se trenira na 14 ručno napisanih primera (`X_train`, `y_train`). U svakoj epohi:

1. Izračuna se predikcija `p` za sve primere.
2. Izračuna se greška `p - y`.
3. Izračunaju se gradijenti po težinama i biasu.
4. Težine i bias se pomere u suprotnom smeru od gradijenta, za korak `LEARNING_RATE`.

Kao mera greške koristi se **binary cross-entropy** (loss). Uz nju se prati i **tačnost**.

## Podešavanja

Nalaze se na vrhu fajla:

| Parametar | Vrednost | Opis |
|-----------|----------|------|
| `SEED` | `42` | Fiksiran seed, pa su rezultati isti pri svakom pokretanju. `None` daje nasumične težine. |
| `LEARNING_RATE` | `0.5` | Veličina koraka pri učenju. |
| `EPOCHS` | `3000` | Broj prolaza kroz ceo skup podataka. |
| `THRESHOLD` | `0.5` | Prag verovatnoće za odluku. |

## Struktura koda

| Deo | Opis |
|-----|------|
| `Neuron.__init__` | Nasumično inicijalizuje težine i bias. |
| `Neuron.sigmoid` | Sigmoid aktivacija (sa `clip` zaštitom od overflow-a). |
| `Neuron.forward` | Vraća verovatnoću za jedan primer ili za ceo skup. |
| `Neuron.loss` | Binary cross-entropy. |
| `Neuron.accuracy` | Procenat tačno pogođenih odluka. |
| `Neuron.train` | Gradient descent petlja, vraća istoriju loss-a. |
| `Neuron.explain` | Ispisuje doprinos svakog ulaza odluci. |
| `X_train`, `y_train` | Podaci za trening. |
| `main()` | Pre treninga, trening, posle treninga, predikcije na novim scenarijima. |

## Primer izlaza

Pre treninga neuron pogađa nasumično:

```
weights: [ 0.548 -0.122  0.717]
bias:    0.395
loss: 0.8996 | tačnost: 42.9%
```

Tokom treninga loss pada, a tačnost raste:

```
Epoha     1 | loss: 0.8416 | tačnost:  42.9%
Epoha   500 | loss: 0.1622 | tačnost: 100.0%
Epoha  1000 | loss: 0.1180 | tačnost: 100.0%
Epoha  3000 | loss: 0.0683 | tačnost: 100.0%
```

Naučene težine pokazuju da kiša snažno obara odluku, a toplo vreme i sunce je podižu:

```
sunčano    težina = +6.364
toplo      težina = +12.086
pada kiša  težina = -10.467
bias       = -8.938
```

Predikcija za sunčan i topao dan, sa objašnjenjem:

```
[Sunčan i topao dan]  ulaz = [1.  0.8 0. ]
  Doprinosi:
    sunčano    ulaz=1.00  doprinos=+6.364
    toplo      ulaz=0.80  doprinos=+9.669
    pada kiša  ulaz=0.00  doprinos=-0.000
    bias                   doprinos=-8.938
  Zbir (z) = +7.094
  Verovatnoća: 99.9%  ->  Odluka: treba voziti bicikl
```

Tačne vrednosti zavise od `SEED`-a i podešavanja.

## Ograničenja

- Trening skup je veoma mali i ručno napisan, pa se tačnost meri na istim primerima na kojima se uči. To pokazuje da neuron uči, ali ne i koliko dobro generalizuje.
- Jedan neuron može da razdvoji samo linearno razdvojive podatke. Za složenije odluke potrebno je više neurona u slojevima (MLP).

## Ideje za proširenje

- Podela podataka na trening i test skup.
- Grafik loss-a kroz epohe (`history` se već vraća iz `train`).
- Dodavanje novih ulaza (vetar, vreme dana, temperatura u °C).
- Prelazak na mali MLP sa skrivenim slojem.