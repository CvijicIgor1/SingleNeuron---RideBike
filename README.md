# SingleNeuron – RideBike 🚴

Jednostavan Python program koji pomoću **jednog neurona** (single neuron) odlučuje da li treba voziti bicikl tog dana.

## Zahtevi

- Python 3.x
- [NumPy](https://numpy.org/)

Instalacija:

```bash
pip install numpy
```

## Struktura projekta

| Fajl | Opis |
|------|------|
| `Optimizovana_Voznja.py` | Optimizovana verzija. `for` petlja je zamenjena funkcijom `numpy.dot`, koja radi na sličnom principu (težinska suma ulaza). Dodat je i *bias*, pa se output može razlikovati od pokretanja do pokretanja. |
| `Voznja_Bicikla.py` | Testni slučaj (osnovna verzija sa petljom). |

## Kako radi

Neuron uzima ulazne parametre (npr. vremenski uslovi), množi ih težinama, sabira rezultat i dodaje bias. Na osnovu dobijene vrednosti odlučuje da li je dan pogodan za vožnju bicikla.

```
izlaz = aktivacija( dot(ulazi, težine) + bias )
```

## Pokretanje

```bash
python Voznja_Bicikla.py
python Optimizovana_Voznja.py
```

## Napomena

Pošto se u `Optimizovana_Voznja.py` koristi bias, rezultat nije uvek isti pri svakom pokretanju – to je očekivano ponašanje.
