import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold,train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from pathlib import Path
import joblib
#https://nethackwiki.com/wiki/Comestible

def prepare_data(file_path):
    """
    Prepara il dataset dal formato JSON per il training del modello.
    Legge i dati da un file JSON.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    all_item_names = set()

    for entry in data:
        for item_in_inv in entry["inventory"]:
            all_item_names.add(item_in_inv["item"])

    sorted_item_names = sorted(list(all_item_names))

    for entry in data:
        health_current = entry["current_health"]["current"]
        health_max = entry["current_health"]["max"]
        health_percentage = health_current / health_max if health_max > 0 else 0

        inventory_features = {
            f"inventory_{item_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()}_qty": 0
            for item_name in sorted_item_names}
        for item_in_inv in entry["inventory"]:
            clean_item_name = item_in_inv["item"].replace(' ', '_').replace('(', '').replace(')', '').replace('-',
                                                                                                              '_').lower()
            inventory_features[f"inventory_{clean_item_name}_qty"] = item_in_inv["quantity"]

        features = {
            "health_percentage": health_percentage,
            **inventory_features
        }

        processed_data.append({
            "features": features,
            "optimal_food_choice": entry["optimal_food_choice"]
        })

    df = pd.DataFrame([item["features"] for item in processed_data])
    df['optimal_food_choice'] = [item["optimal_food_choice"] for item in processed_data]

    return df, sorted_item_names


# --- 2. Definizione del Modello ---
class FoodPredictorMLP(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            # Layer Lineare (Dense/Fully Connected): Applica una trasformazione lineare ai dati (moltiplicazione matrice + aggiunta bias).
            # 'input_size' è il numero di feature in ingresso (es. salute, quantità di oggetti nell'inventario).
            # '128' è il numero di neuroni in questo strato, cioè quante "informazioni" o "pattern" può imparare.
            nn.Linear(input_size, 128),

            # Funzione di Attivazione GELU (Gaussian Error Linear Unit): Introduce non linearità.
            # Questo permette alla rete di imparare pattern più complessi rispetto a semplici relazioni lineari.
            # Aiuta il modello a catturare relazioni intricate nei dati.
            nn.GELU(),

            # BatchNorm1d (Batch Normalization 1D): Normalizza gli input di ogni strato per ogni mini-batch.
            # Questo aiuta a stabilizzare il training, permettendo l'uso di learning rate più alti
            # e agendo anche da una forma di regolarizzazione. Rende il training più veloce e stabile.
            nn.BatchNorm1d(128),

            # Dropout: Spegne casualmente una percentuale di neuroni (definita da 'dropout', es. 0.3 = 30%) durante il training.
            # Questo previene l'overfitting, rendendo il modello meno dipendente da specifici neuroni e più robusto.
            nn.Dropout(dropout),

            # Secondo Layer Lineare: Simile al primo, ma prende 128 input e produce 96 output.
            nn.Linear(128, 96),
            nn.GELU(),  # Funzione di attivazione GELU
            nn.BatchNorm1d(96),  # Normalizzazione del batch
            nn.Dropout(dropout),  # Dropout per prevenire overfitting

            # Terzo Layer Lineare: Prende 96 input e produce 64 output.
            nn.Linear(96, 64),
            nn.GELU(),  # Funzione di attivazione GELU
            nn.BatchNorm1d(64),  # Normalizzazione del batch
            nn.Dropout(dropout),  # Dropout per prevenire overfitting

            # Layer Lineare Finale (Output Layer): Prende 64 input e produce 'num_classes' output.
            # 'num_classes' è il numero totale di possibili scelte di cibo (classi) che il modello deve prevedere.
            # Gli output di questo layer sono solitamente passati a una funzione Softmax (implicitamente gestita da CrossEntropyLoss)
            # per ottenere le probabilità di ogni classe.
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# --- Funzioni di salvataggio e caricamento ---

def save_model(model, label_encoder, all_known_items, scaler, base_name="food"):
    """
    Salva lo stato del modello PyTorch, il LabelEncoder di scikit-learn, la lista degli item e lo scaler.
    I file verranno salvati nella stessa directory dello script.
    """
    current_dir = Path(__file__).parent

    model_path = current_dir / f"{base_name}.pt"
    encoder_path = current_dir / f"{base_name}_label_encoder.pkl"
    items_path = current_dir / f"{base_name}_all_known_items.json"
    scaler_path = current_dir / f"{base_name}_scaler.pkl"

    try:
        torch.save(model.state_dict(), model_path)
        joblib.dump(label_encoder, encoder_path)
        with open(items_path, 'w') as f:
            json.dump(all_known_items, f, indent=4)
        joblib.dump(scaler, scaler_path)
        print(f"[INFO] Modello salvato in '{model_path}'")
        print(f"[INFO] LabelEncoder salvato in '{encoder_path}'")
        print(f"[INFO] Lista degli item conosciuti salvata in '{items_path}'")
        print(f"[INFO] Scaler salvato in '{scaler_path}'")
    except Exception as e:
        print(f"[ERROR] Errore durante il salvataggio del modello/encoder/scaler: {e}")


def load_model(input_size, num_classes, base_name="food"):
    """
    Carica lo stato del modello PyTorch, il LabelEncoder di scikit-learn, la lista degli item e lo scaler.
    Cerca i file nella stessa directory dello script.
    """
    current_dir = Path(__file__).parent

    model_path = current_dir / f"{base_name}.pt"
    encoder_path = current_dir / f"{base_name}_label_encoder.pkl"
    items_path = current_dir / f"{base_name}_all_known_items.json"
    scaler_path = current_dir / f"{base_name}_scaler.pkl"

    model = FoodPredictorMLP(input_size, num_classes)
    label_encoder = None
    all_known_items = []
    scaler = None

    if not model_path.exists() or not encoder_path.exists() or not items_path.exists() or not scaler_path.exists():
        print(
            f"[WARN] Uno o più file del modello/encoder/item/scaler non trovati in '{current_dir}'. Addestramento necessario.")
        return None, None, None, None

    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"[INFO] Modello caricato da '{model_path}'")
    except Exception as e:
        print(f"[ERROR] Errore durante il caricamento del modello da '{model_path}': {e}")
        return None, None, None, None

    try:
        label_encoder = joblib.load(encoder_path)
        print(f"[INFO] LabelEncoder caricato da '{encoder_path}'")
    except Exception as e:
        print(f"[ERROR] Errore durante il caricamento del LabelEncoder da '{encoder_path}': {e}")
        return None, None, None, None

    try:
        with open(items_path, 'r') as f:
            all_known_items = json.load(f)
        print(f"[INFO] Lista degli item conosciuti caricata da '{items_path}'")
    except Exception as e:
        print(f"[ERROR] Errore durante il caricamento della lista degli item da '{items_path}': {e}")
        return None, None, None, None

    try:
        scaler = joblib.load(scaler_path)
        print(f"[INFO] Scaler caricato da '{scaler_path}'")
    except Exception as e:
        print(f"[ERROR] Errore durante il caricamento dello Scaler da '{scaler_path}': {e}")
        return None, None, None, None

    return model, label_encoder, all_known_items, scaler


# --- Main Script ---

if __name__ == "__main__":
    df_prepared, all_known_items_for_prep = prepare_data("food.json")

    print("DataFrame preparato (prime 5 righe):")
    print(df_prepared.head())
    print("\nDimensioni del DataFrame:", df_prepared.shape)

    X = df_prepared.drop('optimal_food_choice', axis=1)
    y = df_prepared['optimal_food_choice']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nMapping delle etichette (cibo -> numero):")
    for i, item in enumerate(label_encoder.classes_):
        print(f"{item}: {i}")

    # Scaler fittato sull'intero dataset X prima della cross-validation
    # Lo scaler verrà salvato e usato per la predizione di nuovi dati.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Fitta lo scaler su tutti i dati

    X_tensor_full = torch.tensor(X_scaled, dtype=torch.float32)  # Questo sarà il dataset completo scalato
    y_tensor_full = torch.tensor(y_encoded, dtype=torch.long)

    input_size = X_tensor_full.shape[1]
    num_classes = len(label_encoder.classes_)

    # Separazione di un Test Set completamente indipendente
    # Questo test set NON sarà mai usato durante la cross-validation
    # per una valutazione finale imparziale.
    X_train_val_initial, X_test_final, y_train_val_initial, y_test_final = train_test_split(
        X_tensor_full, y_tensor_full, test_size=0.15, random_state=42
    )
    print(
        f"\nDimensioni del Training + Validation Set iniziale: {X_train_val_initial.shape}, {y_train_val_initial.shape}")
    print(f"Dimensioni del Test Set finale (indipendente): {X_test_final.shape}, {y_test_final.shape}")

    # Inizializza StratifiedKFold sul set di training+validation
    # StratifiedKFold garantisce che ogni fold abbia circa la stessa proporzione di campioni per classe
    n_splits = 5  # Puoi provare con 5 o 10 fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accuracies = []

    # Questo sarà il modello migliore trovato durante la cross-validation
    # Verrà usato come base per il modello finale riaddestrato sull'intero set.
    best_overall_val_accuracy = 0
    best_overall_model_state_dict = None

    print(f"\n--- Inizio Cross-Validation con {n_splits} fold ---")

    for fold, (train_index, val_index) in enumerate(skf.split(X_train_val_initial, y_train_val_initial)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        X_train_fold, X_val_fold = X_train_val_initial[train_index], X_train_val_initial[val_index]
        y_train_fold, y_val_fold = y_train_val_initial[train_index], y_train_val_initial[val_index]

        print(f"Dimensioni del Training Set (Fold {fold + 1}): {X_train_fold.shape}, {y_train_fold.shape}")
        print(f"Dimensioni del Validation Set (Fold {fold + 1}): {X_val_fold.shape}, {y_val_fold.shape}")

        # Inizializza un nuovo modello per ogni fold
        model = FoodPredictorMLP(input_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=0.001)
        num_epochs = 1200

        model.train()
        best_val_accuracy_this_fold = 0
        epochs_no_improve = 0
        patience = 50

        # Percorso temporaneo per salvare il miglior modello di questo fold
        current_dir = Path(__file__).parent
        best_fold_model_path = current_dir / f"food_best_model_fold_{fold + 1}.pt"

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_train_fold)
            loss = criterion(outputs, y_train_fold)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_fold)
                    _, predicted = torch.max(val_outputs.data, 1)
                    correct = (predicted == y_val_fold).sum().item()
                    accuracy = correct / y_val_fold.size(0)

                    # Commentato per ridurre l'output verboso durante il training
                    # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy (Fold {fold+1}): {accuracy:.4f}")

                    if accuracy > best_val_accuracy_this_fold:
                        best_val_accuracy_this_fold = accuracy
                        epochs_no_improve = 0
                        torch.save(model.state_dict(), best_fold_model_path)  # Salva il modello migliore del fold
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve == patience:
                            print(
                                f"Early stopping triggered at epoch {epoch + 1} for Fold {fold + 1}. Best Validation Accuracy: {best_val_accuracy_this_fold:.4f}")
                            break
                model.train()

        # Carica il miglior modello di questo fold per la valutazione finale del fold
        model.load_state_dict(torch.load(best_fold_model_path))
        model.eval()
        with torch.no_grad():
            val_outputs_fold = model(X_val_fold)  # Valuta sul set di validazione del fold
            _, predicted_fold = torch.max(val_outputs_fold.data, 1)
            correct_fold = (predicted_fold == y_val_fold).sum().item()
            accuracy_fold = correct_fold / y_val_fold.size(0)
            fold_accuracies.append(accuracy_fold)
            print(f"Accuratezza Finale sul Validation Set (Fold {fold + 1}): {accuracy_fold:.4f}")

        # Tieni traccia del miglior modello tra tutti i fold
        if best_val_accuracy_this_fold > best_overall_val_accuracy:
            best_overall_val_accuracy = best_val_accuracy_this_fold
            best_overall_model_state_dict = model.state_dict()  # Salva lo stato del modello migliore

    print("\n--- Cross-Validation Completata ---")
    print(f"Accuratezze per ogni fold: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"Accuratezza Media della Cross-Validation: {np.mean(fold_accuracies):.4f}")
    print(f"Deviazione Standard delle Accuratezze: {np.std(fold_accuracies):.4f}")

    # --- Riaddestramento del Modello Finale sull'intero set di Training+Validation ---
    # Questa è la pratica comune dopo la cross-validation: addestrare il modello finale
    # sull'intero set di dati di training (che ora include il validation set iniziale)
    # utilizzando gli iperparametri che si sono dimostrati migliori.
    print("\n[INFO] Riaddestramento del modello finale sull'intero dataset di Training + Validation...")
    final_model = FoodPredictorMLP(input_size, num_classes)

    # Se un modello migliore è stato identificato durante la CV, lo carichiamo come punto di partenza
    if best_overall_model_state_dict:
        final_model.load_state_dict(best_overall_model_state_dict)
        print("[INFO] Caricato il miglior stato del modello dalla Cross-Validation come punto di partenza.")

    final_criterion = nn.CrossEntropyLoss()
    final_optimizer = optim.Adam(final_model.parameters(), lr=0.0008, weight_decay=0.001)

    # Si può scegliere un numero di epoche fisso o reimplementare l'early stopping
    # con una piccola parte del X_train_val_initial come validazione temporanea.
    # Per semplicità, usiamo un numero di epoche basato sulle osservazioni della CV.
    final_num_epochs = 700

    final_model.train()
    for epoch in range(final_num_epochs):
        final_optimizer.zero_grad()
        outputs = final_model(X_train_val_initial)  # Training su tutto il set train+val
        loss = final_criterion(outputs, y_train_val_initial)
        loss.backward()
        final_optimizer.step()
        if (epoch + 1) % 100 == 0 or epoch == final_num_epochs - 1:
            print(f"Final Model Epoch [{epoch + 1}/{final_num_epochs}], Loss: {loss.item():.4f}")

    print("[INFO] Addestramento modello finale completato.")
    save_model(final_model, label_encoder, all_known_items_for_prep, scaler)
    # Imposta il modello finale per le predizioni successive
    model = final_model

    # --- Valutazione del Modello sul Test Set Indipendente ---
    # Questa è la valutazione finale e imparziale delle performance del modello.
    print("\n--- Valutazione sul Test Set Indipendente ---")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_final)
        _, predicted = torch.max(test_outputs.data, 1)
        correct = (predicted == y_test_final).sum().item()
        total = y_test_final.size(0)
        test_accuracy = correct / total
        print(f"Accuratezza sul Test Set Indipendente: {test_accuracy:.4f}")


    # --- Funzione di Predizione per nuovi input ---
    def predict_food_choice(health_current, health_max, inventory_list, encoder_le, all_known_items_from_training,
                            data_scaler):
        """
        Predice la scelta ottimale del cibo per un nuovo scenario.
        :param health_current: Salute attuale del giocatore.
        :param health_max: Salute massima del giocatore.
        :param inventory_list: Lista di dizionari {"item": "nome_item", "quantity": N}.
        :param encoder_le: Il LabelEncoder addestrato per decodificare l'output.
        :param all_known_items_from_training: La lista ordinata di tutti i nomi di item conosciuti dal training.
        :param data_scaler: Lo StandardScaler addestrato.
        """
        model.eval()

        health_percentage = health_current / health_max if health_max > 0 else 0
        input_features = {"health_percentage": health_percentage}

        inventory_features = {
            f"inventory_{item_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()}_qty": 0
            for item_name in all_known_items_from_training}

        for item_in_inv in inventory_list:
            clean_item_name = item_in_inv["item"].replace(' ', '_').replace('(', '').replace(')', '').replace('-',
                                                                                                              '_').lower()
            if f"inventory_{clean_item_name}_qty" in inventory_features:
                inventory_features[f"inventory_{clean_item_name}_qty"] = item_in_inv["quantity"]
            else:
                print(f"[WARN] Item '{item_in_inv['item']}' non visto durante il training. Verrà ignorato.")

        features_ordered_list = [input_features["health_percentage"]]

        # Assicurati che l'ordine delle feature sia lo stesso usato durante il training
        training_feature_names = X.columns.tolist()  # X viene dal DataFrame iniziale

        for col_name in training_feature_names:
            if col_name != 'health_percentage':
                item_key = col_name
                features_ordered_list.append(inventory_features.get(item_key, 0))

        input_array = np.array(features_ordered_list, dtype=np.float32).reshape(1, -1)
        input_scaled_array = data_scaler.transform(input_array)
        input_tensor = torch.tensor(input_scaled_array, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class_idx = torch.argmax(output).item()

        predicted_food_name = encoder_le.inverse_transform([predicted_class_idx])[0]
        return predicted_food_name


    # --- 8. Piccolo Test Set per la Predizione ---
    print("\n--- TEST DI PREDIZIONE SU NUOVI SCENARI ---")

    final_known_items_list = all_known_items_for_prep
    test_scenarios = [
        {
            "health": {"current": 8, "max": 20},
            "inventory": [{"item": "apple", "quantity": 1}, {"item": "food ration", "quantity": 1}],
            "expected": "food ration"
        },
        {
            "health": {"current": 1, "max": 20},
            "inventory": [{"item": "huge chunk of meat", "quantity": 1}, {"item": "apple", "quantity": 10}],
            "expected": "huge chunk of meat"
        },
        {
            "health": {"current": 19, "max": 20},
            "inventory": [{"item": "apple", "quantity": 1}, {"item": "candy bar", "quantity": 2}],
            "expected": "None"
        },
        {
            "health": {"current": 5, "max": 20},
            "inventory": [{"item": "tin of spinach (cursed)", "quantity": 1}],
            "expected": "tin of spinach (cursed)"
        },
        {
            "health": {"current": 15, "max": 20},
            "inventory": [{"item": "None", "quantity": 1}],
            "expected": "None"
        },
        {
            "health": {"current": 10, "max": 20},
            "inventory": [
                {"item": "lembas wafer", "quantity": 1},
                {"item": "food ration", "quantity": 1},
                {"item": "apple", "quantity": 5}
            ],
            "expected": "lembas wafer"
        },
        {
            "health": {"current": 20, "max": 20},
            "inventory": [
                {"item": "apple", "quantity": 5},
                {"item": "candy bar", "quantity": 3}
            ],
            "expected": "None"
        }
    ]

    for i, scenario in enumerate(test_scenarios):
        predicted_food = predict_food_choice(
            scenario["health"]["current"],
            scenario["health"]["max"],
            scenario["inventory"],
            label_encoder,
            final_known_items_list,
            scaler
        )
        print(f"\nScenario {i + 1}:")
        print(f"  Salute: {scenario['health']['current']}/{scenario['health']['max']}")
        print(f"  Inventario: {scenario['inventory']}")
        print(f"  Previsto: {predicted_food}")
        print(f"  Atteso: {scenario['expected']}")
        print(f"  Corretto: {predicted_food == scenario['expected']}")