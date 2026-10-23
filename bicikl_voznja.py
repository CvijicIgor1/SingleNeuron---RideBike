"""
Jedan veštački neuron (logistička regresija) koji odlučuje da li treba voziti bicikl.

Ulazi:  [sunčano, toplo, pada_kiša]  (vrednosti od 0 do 1)
Izlaz:  verovatnoća (0-1) da je dobro vreme za bicikl

Šta ovaj kod radi:
  1. Nasumično inicijalizuje težine i bias (kao tvoja prvobitna verzija)
  2. Koristi sigmoid aktivaciju da izlaz bude verovatnoća
  3. Trenira neuron gradient descent-om na malom skupu primera
  4. Prati loss i tačnost tokom treninga
  5. Objašnjava svaku odluku (doprinos svakog ulaza)
"""

import numpy as np  # https://numpy.org/

# ---------------------------------------------------------------------------
# Podešavanja
# ---------------------------------------------------------------------------
SEED = 42            # fiksiran seed -> isti rezultati pri svakom pokretanju (stavi None za random)
LEARNING_RATE = 0.5  # koliko veliki korak pravimo pri učenju
EPOCHS = 3000        # broj prolaza kroz ceo skup podataka
THRESHOLD = 0.5      # prag za odluku (verovatnoća iznad = vozi bicikl)

FEATURES = ["sunčano", "toplo", "pada kiša"]


# ---------------------------------------------------------------------------
# Neuron
# ---------------------------------------------------------------------------
class Neuron:
    def __init__(self, n_inputs: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(-1, 1, n_inputs)  # težine: koliko je koji ulaz bitan
        self.bias = rng.uniform(-1, 1)               # bias: pomera prag odluke

    @staticmethod
    def sigmoid(z):
        """Spljošti bilo koji broj u opseg (0, 1). clip sprečava overflow."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        """Sirovi zbir (z) -> sigmoid -> verovatnoća. Radi i za jedan primer i za ceo batch."""
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def loss(self, X, y):
        """Binary cross-entropy: koliko je model 'pogrešan' u proseku."""
        p = np.clip(self.forward(X), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def accuracy(self, X, y, threshold=THRESHOLD):
        predictions = (self.forward(X) > threshold).astype(int)
        return np.mean(predictions == y)

    def train(self, X, y, lr=LEARNING_RATE, epochs=EPOCHS, log_every=500):
        """Gradient descent. Za sigmoid + cross-entropy gradijent je jednostavan: (p - y)."""
        n = len(X)
        history = []

        for epoch in range(1, epochs + 1):
            p = self.forward(X)
            error = p - y                      # greška po primeru

            grad_w = np.dot(X.T, error) / n    # gradijent po težinama
            grad_b = np.mean(error)            # gradijent po biasu

            self.weights -= lr * grad_w        # korak u suprotnom smeru od gradijenta
            self.bias -= lr * grad_b

            current_loss = self.loss(X, y)
            history.append(current_loss)

            if epoch == 1 or epoch % log_every == 0:
                acc = self.accuracy(X, y)
                print(f"Epoha {epoch:>5} | loss: {current_loss:.4f} | tačnost: {acc * 100:5.1f}%")

        return history

    def explain(self, x):
        """Pokaži koliko je svaki ulaz doprineo odluci."""
        contributions = x * self.weights
        z = contributions.sum() + self.bias
        print("  Doprinosi:")
        for name, value, contrib in zip(FEATURES, x, contributions):
            print(f"    {name:<10} ulaz={value:.2f}  doprinos={contrib:+.3f}")
        print(f"    {'bias':<10} {'':<10}  doprinos={self.bias:+.3f}")
        print(f"  Zbir (z) = {z:+.3f}")


# ---------------------------------------------------------------------------
# Podaci za trening: [sunčano, toplo, pada_kiša] -> 1 (vozi) / 0 (ne vozi)
# ---------------------------------------------------------------------------
X_train = np.array([
    [1.0, 1.0, 0.0],  # sunce, vruće
    [1.0, 0.7, 0.0],  # sunce, toplo
    [1.0, 0.3, 0.0],  # sunce, malo hladnije
    [1.0, 0.0, 0.0],  # sunce, ali ledeno
    [0.7, 0.9, 0.0],  # skoro sunčano, toplo
    [0.5, 0.8, 0.0],  # promenljivo, toplo
    [0.5, 0.4, 0.0],  # promenljivo, hladnije
    [0.0, 0.9, 0.0],  # oblačno, ali toplo
    [0.0, 0.5, 0.0],  # oblačno, osrednje
    [0.0, 0.2, 0.0],  # oblačno, hladno
    [1.0, 0.8, 1.0],  # sunce + kiša (pljusak)
    [0.5, 0.7, 1.0],  # promenljivo + kiša
    [0.0, 0.6, 1.0],  # oblačno + kiša
    [0.0, 0.1, 1.0],  # hladno + kiša
], dtype=float)

y_train = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    neuron = Neuron(n_inputs=len(FEATURES), seed=SEED)

    print("=" * 60)
    print("PRE TRENINGA (nasumične težine)")
    print("=" * 60)
    print("weights:", np.round(neuron.weights, 3))
    print("bias:   ", round(float(neuron.bias), 3))
    print(f"loss: {neuron.loss(X_train, y_train):.4f} | "
          f"tačnost: {neuron.accuracy(X_train, y_train) * 100:.1f}%\n")

    print("=" * 60)
    print("TRENING")
    print("=" * 60)
    neuron.train(X_train, y_train)

    print("\n" + "=" * 60)
    print("POSLE TRENINGA (naučene težine)")
    print("=" * 60)
    for name, w in zip(FEATURES, neuron.weights):
        print(f"  {name:<10} težina = {w:+.3f}")
    print(f"  {'bias':<10} = {neuron.bias:+.3f}")
    print(f"  Finalna tačnost: {neuron.accuracy(X_train, y_train) * 100:.1f}%")

    # Test na scenarijima koje neuron nikad nije video
    scenarios = {
        "Sunčan i topao dan":      np.array([1.0, 0.8, 0.0]),
        "Oblačno i hladno":        np.array([0.2, 0.2, 0.0]),
        "Pljusak":                 np.array([0.0, 0.5, 1.0]),
        "Tvoj prvobitni primer":   np.array([1.0, 0.7, 0.0]),
        "Sunčano ali ledeno":      np.array([0.9, 0.1, 0.0]),
    }

    print("\n" + "=" * 60)
    print("PREDIKCIJE")
    print("=" * 60)
    for name, x in scenarios.items():
        prob = float(neuron.forward(x))
        decision = "treba voziti bicikl" if prob > THRESHOLD else "ne treba voziti bicikl"
        print(f"\n[{name}]  ulaz = {x}")
        neuron.explain(x)
        print(f"  Verovatnoća: {prob * 100:.1f}%  ->  Odluka: {decision}")


if __name__ == "__main__":
    main()