# SingleNeuron – RideBike 🚴

Jednostavan Python program koji pomoću **jednog neurona** (single neuron) odlučuje da li treba voziti bicikl tog dana.

Projekat ima tri verzije koje idu od najosnovnije do najkompletnije: osnovna verzija sa petljom, optimizovana verzija sa `numpy.dot` i proširena verzija sa pravim treningom neurona.

## Zahtevi

- Python 3.10+ (proširena verzija koristi `int | None` anotaciju)
- [NumPy](https://numpy.org/)

Instalacija:

```bash
pip install numpy
```

## Struktura projekta

| Fajl | Opis |
|------|------|
| `Voznja_Bicikla.py` | Testni slučaj (osnovna verzija sa `for` petljom). |
| `Optimizovana_Voznja.py` | Optimizovana verzija. `for` petlja je zamenjena funkcijom `numpy.dot`, koja radi na sličnom principu (težinska suma ulaza). Dodat je i *bias*, pa se output može razlikovati od pokretanja do pokretanja. |
| `neuron_bicikl.py` | Proširena verzija. Neuron je klasa, ima sigmoid aktivaciju, uči težine gradient descent-om i objašnjava svaku odluku. |

## Kako radi

Neuron uzima ulazne parametre (vremenski uslovi), množi ih težinama, sabira rezultat i dodaje bias. Na osnovu dobijene vrednosti odlučuje da li je dan pogodan za vožnju bicikla.

```
izlaz = aktivacija( dot(ulazi, težine) + bias )
```

### Ulazi

Svaki ulaz je vrednost od 0 do 1:

| Ulaz | Značenje |
|------|----------|
| `sunčano` | 1.0 = potpuno sunčano, 0.0 = oblačno |
| `toplo` | 1.0 = vruće, 0.0 = ledeno |
| `pada kiša` | 1.0 = pada kiša, 0.0 = ne pada |

Primer: `[1.0, 0.7, 0.0]` znači sunčano, prilično toplo, bez kiše.

### Težine i bias

- **Težine** govore koliko je koji ulaz bitan. Pozitivna težina podstiče vožnju, negativna je odvraća (npr. kiša).
- **Bias** pomera prag odluke. Bez njega neuron bi uvek odlučivao samo na osnovu ulaza.

### Aktivacija

U osnovnim verzijama odluka je prag: ako je `output > 0` vozi se bicikl. U proširenoj verziji koristi se **sigmoid**, pa izlaz postaje verovatnoća od 0 do 1:

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Ako je verovatnoća iznad praga (`THRESHOLD = 0.5`), preporuka je da se vozi.

## Proširena verzija (`neuron_bicikl.py`)

Šta radi:

1. Nasumično inicijalizuje težine i bias.
2. Trenira neuron **gradient descent-om** na malom skupu od 14 primera.
3. Prati **loss** (binary cross-entropy) i **tačnost** tokom treninga.
4. Testira naučeni model na novim scenarijima.
5. Metodom `explain()` ispisuje doprinos svakog ulaza odluci.

### Trening

U svakoj epohi neuron:

1. Izračuna predikciju `p` za sve primere.
2. Izračuna grešku `p - y` (razlika između predikcije i tačnog odgovora).
3. Izračuna gradijente po težinama i biasu.
4. Pomeri težine i bias u suprotnom smeru od gradijenta, za korak `LEARNING_RATE`.

### Podešavanja

Nalaze se na vrhu fajla:

| Parametar | Vrednost | Opis |
|-----------|----------|------|
| `SEED` | `42` | Fiksiran seed za ponovljive rezultate. `None` daje potpuno nasumične težine. |
| `LEARNING_RATE` | `0.5` | Veličina koraka pri učenju. |
| `EPOCHS` | `3000` | Broj prolaza kroz ceo skup podataka. |
| `THRESHOLD` | `0.5` | Prag verovatnoće za odluku. |

### Primer izlaza

Posle treninga neuron nauči da kiša snažno obara odluku, a toplo vreme i sunce je podižu:

```
sunčano    težina = +6.364
toplo      težina = +12.086
pada kiša  težina = -10.467
bias       = -8.938
Finalna tačnost: 100.0%
```

Predikcija za sunčan i topao dan:

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

Tačne vrednosti zavise od seed-a.

## Pokretanje

```bash
python Voznja_Bicikla.py
python Optimizovana_Voznja.py
python neuron_bicikl.py
```

## Napomena

- U `Optimizovana_Voznja.py` težine i bias se biraju nasumično pri svakom pokretanju, pa rezultat nije uvek isti. To je očekivano ponašanje, jer neuron tu nije treniran.
- U `neuron_bicikl.py` rezultat je ponovljiv zahvaljujući fiksiranom `SEED`-u, a težine se uče iz podataka umesto da ostanu nasumične.
- Jedan neuron može da razdvoji samo podatke koji su linearno razdvojivi. Za složenije odluke potrebno je više neurona raspoređenih u slojeve (MLP).
