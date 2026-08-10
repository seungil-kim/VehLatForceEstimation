import json
import pickle
import re
import shutil
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat, savemat
from sklearn.preprocessing import MinMaxScaler

SEED = 42
LOCAL_RUN = False  # True: 로컬 환경(그래프 창 표시), False: 서버 환경(창 비활성화)
EARLYSTOPPING = False

PROJECT_DIR = Path(__file__).resolve().parent
# ZIPSAVE_DIR = Path("/root/kadap/MyDisk")
ZIPSAVE_DIR = Path("/root/kadap/MyDisk")

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
RESULT_DIR = OUTPUT_DIR / "results"
DATAFILE_NAME = "LSTM_Dataset.mat"
SPLITFILE_NAME = "LSTM_ScenarioSplit.mat"
RESULTS_MAT_NAME = "LSTM_Results.mat"
LOSS_MAT_NAME = "Loss_Curve.mat"

def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA >=10.2 결정론적 matmul/RNN을 위해 필요

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def mat_load(data_dir=DATA_DIR, datafile_name=DATAFILE_NAME, splitfile_name=SPLITFILE_NAME):
    core_file = data_dir / datafile_name
    split_file = data_dir / splitfile_name

    if not core_file.is_file():
        raise FileNotFoundError(f"Dataset 파일을 찾지 못했습니다: {core_file}")
    if not split_file.is_file():
        raise FileNotFoundError(f"Scenario split 파일을 찾지 못했습니다: {split_file}")

    core_data = loadmat(core_file)
    dataset = core_data["LSTMDataset"]["Scenario"]

    split_data = loadmat(split_file, squeeze_me=True, struct_as_record=False)
    if "Split" not in split_data:
        raise KeyError(f"{splitfile_name} 내부에서 'Split' 변수를 찾지 못했습니다.")
    split_data = split_data["Split"]

    # dataset은 (1,1) object 배열이므로 [0, 0]으로 언랩해야 시나리오 이름(dtype.names)을 얻을 수 있다.
    scenario_names = list(dataset[0, 0].dtype.names)
    print("Loaded dataset:", dataset.shape, "scenarios:", len(scenario_names))

    return dataset, split_data, scenario_names


def get_split_field(split_struct, field_name, required=True):
    if not hasattr(split_struct, field_name):
        if required:
            raise KeyError(
                f"LSTM_ScenarioSplit.mat의 Split 구조체에 '{field_name}' 필드가 없습니다."
            )
        return []

    raw = getattr(split_struct, field_name)
    arr = np.asarray(raw, dtype=object).ravel()
    out = []

    for item in arr:
        while isinstance(item, np.ndarray) and item.size == 1:
            item = item.reshape(-1)[0]
        if item is None:
            continue
        out.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))

    return out


def validate_splits(split_dict, scenario_name_set):
    for split_name, split_scenarios in split_dict.items():
        missing_scenarios = sorted(set(split_scenarios) - scenario_name_set)
        if missing_scenarios:
            raise ValueError(
                f"[{split_name}] Dataset에 없는 시나리오가 있습니다:\n"
                + "\n".join(missing_scenarios)
            )

    scenario_to_split = {}
    for split_name, split_scenarios in split_dict.items():
        for scenario_name in split_scenarios:
            if scenario_name in scenario_to_split:
                previous_split = scenario_to_split[scenario_name]
                raise ValueError(
                    "Split 간 중복 시나리오가 있습니다:\n"
                    f"{scenario_name}\n"
                    f"- 기존 Split: {previous_split}\n"
                    f"- 중복 Split: {split_name}"
                )
            scenario_to_split[scenario_name] = split_name

    unassigned_scenarios = sorted(scenario_name_set - set(scenario_to_split.keys()))
    return scenario_to_split, unassigned_scenarios


def unwrap_mat_struct(value):
    # scipy.io.loadmat은 MATLAB 중첩 struct를 object 배열로 감싸는데,
    # 감싸는 깊이가 시나리오마다(생성 스크립트/버전에 따라) 다를 수 있어
    # object dtype이면서 원소가 1개인 동안 계속 벗겨낸다.
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.reshape(-1)[0]
    return value


def get_scenario_data(dataset, name):
    # 새 데이터 구조: Scenario.<name>.{Meta, Time, LSTM.{X, Y, ...}}
    scenario = unwrap_mat_struct(dataset[0, 0][name])
    lstm_block = unwrap_mat_struct(scenario["LSTM"])
    X = unwrap_mat_struct(lstm_block["X"]).astype(np.float32)
    Y = unwrap_mat_struct(lstm_block["Y"]).astype(np.float32)
    time = unwrap_mat_struct(scenario["Time"])
    return X, Y, time


def seq_data(x, y, sequence_length):
    x_seq = []
    y_seq = []

    for i in range(len(x) - sequence_length + 1):
        x_seq.append(x[i : i + sequence_length])
        y_seq.append(y[i + sequence_length - 1])

    return np.array(x_seq), np.array(y_seq)


def get_scenario_category(scenario_name):
    # 데이터 불균형 완화를 위한 카테고리 분류.
    # 시나리오 이름에서 mu/v 조건과 Lturn/Rturn 접미사를 떼어내면
    # DLC, SLC, ISO_Steady_State_Circle, Sine_Sweep_Steer 등 조작 유형만 남는다.
    category = re.split(r"_mu\d", scenario_name)[0]
    category = re.sub(r"_(Lturn|Rturn)$", "", category)
    return category


def build_scalers(dataset, train_scenarios):
    X_raw = []
    Y_raw = []
    for name in train_scenarios:
        X, Y, _ = get_scenario_data(dataset, name)
        X_raw.append(X)
        Y_raw.append(Y)

    X_train_raw_all = np.concatenate(X_raw, axis=0)
    Y_train_raw_all = np.concatenate(Y_raw, axis=0)

    x_scaler = MinMaxScaler().fit(X_train_raw_all)
    y_scaler = MinMaxScaler().fit(Y_train_raw_all)
    return x_scaler, y_scaler


def make_sequence_dataset(dataset, scenario_list, x_scaler, y_scaler, sequence_length, dataset_name):
    x_list = []
    y_list = []
    sample_counts = {}  # scenario_name -> 생성된 시퀀스 샘플 수 (카테고리 가중치 계산용)

    for name in scenario_list:
        X, Y, _ = get_scenario_data(dataset, name)
        X_scaled = x_scaler.transform(X)
        Y_scaled = y_scaler.transform(Y)

        x_seq, y_seq = seq_data(X_scaled, Y_scaled, sequence_length)
        x_list.append(x_seq)
        y_list.append(y_seq)
        sample_counts[name] = len(x_seq)

        print(
            f"[{dataset_name}] {name} -> "
            f"X_seq: {x_seq.shape}, Y_seq: {y_seq.shape}"
        )

    X_out = np.concatenate(x_list, axis=0)
    Y_out = np.concatenate(y_list, axis=0)
    return X_out, Y_out, sample_counts


def build_category_sample_weights(scenario_list, sample_counts):
    # 목표: 매 epoch 배치가 "카테고리(조작 유형)"별로 균등하게 뽑히도록
    # 샘플별 가중치를 부여한다. 가중치 = 1 / (카테고리 내 시나리오 수 * 해당 시나리오 샘플 수)
    # -> 카테고리 간 균등 + 카테고리 내 시나리오 간 균등을 동시에 달성.
    categories = {name: get_scenario_category(name) for name in scenario_list}
    scenarios_per_category = {}
    for name, category in categories.items():
        scenarios_per_category[category] = scenarios_per_category.get(category, 0) + 1

    weights = []
    for name in scenario_list:
        category = categories[name]
        n_scenarios_in_category = scenarios_per_category[category]
        n_samples_in_scenario = sample_counts[name]
        weight = 1.0 / (n_scenarios_in_category * n_samples_in_scenario)
        weights.extend([weight] * n_samples_in_scenario)

    return torch.as_tensor(weights, dtype=torch.double)


def create_dataloaders(x_train_seq, y_train_seq, x_val_seq, y_val_seq, batch_size, train_sample_weights=None):
    train_dataset = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
    val_dataset = torch.utils.data.TensorDataset(x_val_seq, y_val_seq)

    if train_sample_weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(
            train_sample_weights, num_samples=len(train_sample_weights), replacement=True
        )
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size)
    return train_loader, val_loader


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(LSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, 2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        return self.fc(out)


def fit_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
              num_epochs, patience, model_path, device):
    train_loss_graph = []
    val_loss_graph = []
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    last_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        epoch_max_grad_norm = 0.0
        for seq, target in train_loader:
            seq = seq.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(seq), target)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            epoch_max_grad_norm = max(epoch_max_grad_norm, grad_norm.item())
            optimizer.step()

            batch_size_now = seq.size(0)
            train_loss_sum += loss.item() * batch_size_now
            train_sample_count += batch_size_now

        train_loss = train_loss_sum / train_sample_count
        train_loss_graph.append(train_loss)

        model.eval()
        val_loss_sum = 0.0
        val_sample_count = 0
        with torch.no_grad():
            for seq, target in val_loader:
                seq = seq.to(device)
                target = target.to(device)
                loss = criterion(model(seq), target)
                batch_size_now = seq.size(0)
                val_loss_sum += loss.item() * batch_size_now
                val_sample_count += batch_size_now

        val_loss = val_loss_sum / val_sample_count
        val_loss_graph.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_loss_graph": train_loss_graph.copy(),
                "val_loss_graph": val_loss_graph.copy(),
            }, model_path)
        else:
            patience_counter += 1

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(
                f"[Epoch {epoch:03d}] Train Loss: {train_loss:.8e} | "
                f"Val Loss: {val_loss:.8e} | Best Val Loss: {best_val_loss:.8e} | "
                f"Patience: {patience_counter}/{patience} | "
                f"Grad Norm: {epoch_max_grad_norm:.4f} | LR: {current_lr:.2e}"
            )

        if EARLYSTOPPING and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            last_epoch = epoch
            break

        last_epoch = epoch

    torch.save({
        "epoch": last_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss_graph[-1],
        "val_loss": val_loss_graph[-1],
    }, MODEL_DIR / "lstm_last_model.pth")

    return train_loss_graph, val_loss_graph, best_epoch, best_val_loss


def save_training_artifacts(x_scaler, y_scaler, run_config):
    with open(MODEL_DIR / "scalers.pkl", "wb") as f:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, f)

    with open(RESULT_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)


def plot_loss(train_loss_graph, val_loss_graph, best_epoch):
    epochs = range(1, len(train_loss_graph) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_loss_graph, label="Train Loss")
    plt.plot(epochs, val_loss_graph, label="Validation Loss")
    plt.axvline(best_epoch + 1, linestyle="--", label=f"Best Epoch = {best_epoch + 1}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(RESULT_DIR / "Loss.png", dpi=300)

    plt.figure(figsize=(9, 5))
    plt.semilogy(train_loss_graph, label="Train Loss")
    plt.semilogy(val_loss_graph, label="Validation Loss")
    plt.axvline(best_epoch + 1, linestyle="--", label=f"Best Epoch = {best_epoch + 1}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.title("LSTM Training and Validation Loss")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "Loss_log.png", dpi=300)


def save_loss_curve_mat(train_loss_graph, val_loss_graph, best_epoch, best_val_loss, output_path):
    epoch = np.arange(1, len(train_loss_graph) + 1, dtype=np.float64).reshape(-1, 1)
    savemat(output_path, {
        "LossCurve": {
            "Epoch": epoch,
            "TrainLoss": np.asarray(train_loss_graph, dtype=np.float64).reshape(-1, 1),
            "ValLoss": np.asarray(val_loss_graph, dtype=np.float64).reshape(-1, 1),
            "BestEpoch": best_epoch + 1,
            "BestValLoss": best_val_loss,
        }
    })
    print(f"Saved loss curve data: {output_path}")


def get_time_array(time_raw):
    time_data = time_raw
    while isinstance(time_data, np.ndarray) and time_data.dtype == object and time_data.size == 1:
        time_data = time_data.flat[0]
    return np.asarray(time_data, dtype=np.float32).squeeze()


def predict_scenario(dataset, scenario_name, model, x_scaler, y_scaler, sequence_length, device):
    X_raw, Y_raw, time_raw = get_scenario_data(dataset, scenario_name)
    time = get_time_array(time_raw)

    if X_raw.ndim != 2 or Y_raw.ndim != 2:
        raise ValueError(f"{scenario_name}: X, Y 데이터는 모두 2차원 행렬이어야 합니다.")
    if Y_raw.shape[1] != 2:
        raise ValueError(f"{scenario_name}: Y는 [Fyf, Fyr] 형태의 N×2 행렬이어야 합니다.")
    if not (len(X_raw) == len(Y_raw) == len(time)):
        raise ValueError(
            f"{scenario_name}: X, Y, Time 길이가 일치하지 않습니다. "
            f"X={len(X_raw)}, Y={len(Y_raw)}, Time={len(time)}"
        )
    if len(X_raw) < sequence_length:
        raise ValueError(f"{scenario_name}: sequence_length={sequence_length}보다 데이터 길이가 짧습니다.")

    X_scaled = x_scaler.transform(X_raw)
    x_seq, _ = seq_data(X_scaled, Y_raw, sequence_length)
    x_tensor = torch.as_tensor(x_seq.astype(np.float32), device=device)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()

    pred_force = y_scaler.inverse_transform(pred_scaled)
    start_idx = sequence_length - 1
    time_aligned = time[start_idx:]
    true_force = Y_raw[start_idx:]

    if not (len(time_aligned) == len(true_force) == len(pred_force)):
        raise RuntimeError(f"{scenario_name}: 예측값/정답/시간축 길이가 일치하지 않습니다.")

    return time_aligned, true_force, pred_force


def evaluate_scenario_list(
    scenario_list,
    split_name,
    metrics_path,
    dataset,
    model,
    x_scaler,
    y_scaler,
    sequence_length,
    device,
    results_accumulator=None,
):
    metric_rows = []
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Lateral Force Estimation Metrics (LSTM vs Reference)\n")
        f.write(f"Split: {split_name}\n")
        f.write("=" * 70 + "\n")

    for scenario_name in scenario_list:
        time_eval, true_force, pred_force = predict_scenario(
            dataset=dataset,
            scenario_name=scenario_name,
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            sequence_length=sequence_length,
            device=device,
        )

        lstm_error_front = pred_force[:, 0] - true_force[:, 0]
        lstm_error_rear = pred_force[:, 1] - true_force[:, 1]

        lstm_rmse_front = np.sqrt(np.mean(lstm_error_front ** 2))
        lstm_rmse_rear = np.sqrt(np.mean(lstm_error_rear ** 2))
        lstm_mae_front = np.mean(np.abs(lstm_error_front))
        lstm_mae_rear = np.mean(np.abs(lstm_error_rear))
        lstm_max_error_front = np.max(np.abs(lstm_error_front))
        lstm_max_error_rear = np.max(np.abs(lstm_error_rear))

        metric_rows.append({
            "Scenario": scenario_name,
            "Category": get_scenario_category(scenario_name),
            "LSTM_Front_RMSE_N": lstm_rmse_front,
            "LSTM_Front_MAE_N": lstm_mae_front,
            "LSTM_Front_MaxError_N": lstm_max_error_front,
            "LSTM_Rear_RMSE_N": lstm_rmse_rear,
            "LSTM_Rear_MAE_N": lstm_mae_rear,
            "LSTM_Rear_MaxError_N": lstm_max_error_rear,
        })

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(f"\nScenario: {scenario_name}\n")
            f.write("-" * 70 + "\n")
            f.write("[LSTM vs Reference]\n")
            f.write(f"Front RMSE      : {lstm_rmse_front:.2f} N\n")
            f.write(f"Front MAE       : {lstm_mae_front:.2f} N\n")
            f.write(f"Front Max Error : {lstm_max_error_front:.2f} N\n")
            f.write(f"Rear RMSE       : {lstm_rmse_rear:.2f} N\n")
            f.write(f"Rear MAE        : {lstm_mae_rear:.2f} N\n")
            f.write(f"Rear Max Error  : {lstm_max_error_rear:.2f} N\n")

        plot_scenario_results(
            split_name,
            scenario_name,
            time_eval,
            true_force,
            pred_force,
        )

        if results_accumulator is not None:
            results_accumulator[scenario_name] = {
                "Split": split_name,
                "Time": np.asarray(time_eval, dtype=np.float64).reshape(-1, 1),
                "Reference": np.asarray(true_force, dtype=np.float64),
                "LSTM_Pred": np.asarray(pred_force, dtype=np.float64),
            }

    metrics_df = pd.DataFrame(metric_rows)
    if not metrics_df.empty:
        mean_columns = [
            "LSTM_Front_RMSE_N",
            "LSTM_Front_MAE_N",
            "LSTM_Rear_RMSE_N",
            "LSTM_Rear_MAE_N",
        ]
        print(f"\n[{split_name}] Mean metrics (overall)")
        print(metrics_df[mean_columns].mean())
        print(f"\n[{split_name}] Mean metrics by category")
        print(metrics_df.groupby("Category")[mean_columns].mean())

    return metrics_df


def plot_scenario_results(split_name, scenario_name, time_eval, true_force, pred_force):
    plt.figure(figsize=(10, 4))
    plt.plot(time_eval, true_force[:, 0], label="Reference (Ground Truth)")
    plt.plot(time_eval, pred_force[:, 0], label="LSTM Prediction")
    plt.xlabel("Time [s]")
    plt.ylabel("Front Lateral Force [N]")
    plt.title(f"{split_name} | {scenario_name} - Front Lateral Force")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{split_name}_{scenario_name}_front_lateral_force.png", dpi=300)
    if LOCAL_RUN:
        plt.show()
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(time_eval, true_force[:, 1], label="Reference (Ground Truth)")
    plt.plot(time_eval, pred_force[:, 1], label="LSTM Prediction")
    plt.xlabel("Time [s]")
    plt.ylabel("Rear Lateral Force [N]")
    plt.title(f"{split_name} | {scenario_name} - Rear Lateral Force")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{split_name}_{scenario_name}_rear_lateral_force.png", dpi=300)
    if LOCAL_RUN:
        plt.show()
    plt.close()

def save_lstm_results_mat(results_accumulator, output_path):
    if not results_accumulator:
        print("저장할 LSTM 추론 결과가 없습니다.")
        return

    # 시나리오 이름을 struct 필드명으로 쓰면 MATLAB 5 포맷의 31자 필드명 제한에 걸리는
    # 시나리오가 있어(예: ISO14791_... 계열), 이름은 데이터 필드(Name)로 넣고
    # 1xN struct array로 저장한다. MATLAB에서는 LSTMResults.Scenario(i).Name 로 조회.
    scenario_names = list(results_accumulator.keys())
    dtype = [("Name", "O"), ("Split", "O"), ("Time", "O"), ("Reference", "O"), ("LSTM_Pred", "O")]
    scenario_array = np.empty((1, len(scenario_names)), dtype=dtype)
    for i, scenario_name in enumerate(scenario_names):
        result = results_accumulator[scenario_name]
        scenario_array[0, i]["Name"] = scenario_name
        scenario_array[0, i]["Split"] = result["Split"]
        scenario_array[0, i]["Time"] = result["Time"]
        scenario_array[0, i]["Reference"] = result["Reference"]
        scenario_array[0, i]["LSTM_Pred"] = result["LSTM_Pred"]

    savemat(output_path, {"LSTMResults": {"Scenario": scenario_array}})
    print(f"Saved LSTM inference results: {output_path}")


def zip_output(sequence_length, batch_size, num_epochs, hidden_size):
    zip_name = ZIPSAVE_DIR / (
        f"lstm_h{hidden_size}_seq{sequence_length:03d}_"
        f"bs{batch_size}_ep{num_epochs}"
    )
    shutil.make_archive(str(zip_name), "zip", root_dir=str(OUTPUT_DIR), base_dir=".")
    print(f"Saved ZIP: {zip_name}.zip")


def main():
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    dataset, split_data, scenario_names = mat_load()
    scenario_name_set = set(scenario_names)

    train_scenarios = get_split_field(split_data, "Train")
    val_scenarios = get_split_field(split_data, "Validation")
    test_scenarios = get_split_field(split_data, "Test")

    split_dict = {
        "Train": train_scenarios,
        "Validation": val_scenarios,
        "Test": test_scenarios,
    }
    _, unassigned_scenarios = validate_splits(split_dict, scenario_name_set)

    print("Train scenarios:", len(train_scenarios))
    print("Validation scenarios:", len(val_scenarios))
    print("Test scenarios:", len(test_scenarios))
    if unassigned_scenarios:
        print("Unassigned scenarios:", unassigned_scenarios)

    x_scaler, y_scaler = build_scalers(dataset, train_scenarios)
    sequence_length = 50
    x_train_seq, y_train_seq, train_sample_counts = make_sequence_dataset(
        dataset, train_scenarios, x_scaler, y_scaler, sequence_length, "Train"
    )
    x_val_seq, y_val_seq, _ = make_sequence_dataset(
        dataset, sorted(val_scenarios), x_scaler, y_scaler, sequence_length, "Validation"
    )

    train_sample_weights = build_category_sample_weights(train_scenarios, train_sample_counts)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    x_train_seq = torch.as_tensor(x_train_seq, dtype=torch.float32, device=device)
    y_train_seq = torch.as_tensor(y_train_seq, dtype=torch.float32, device=device)
    x_val_seq = torch.as_tensor(x_val_seq, dtype=torch.float32, device=device)
    y_val_seq = torch.as_tensor(y_val_seq, dtype=torch.float32, device=device)

    train_loader, val_loader = create_dataloaders(
        x_train_seq, y_train_seq, x_val_seq, y_val_seq, batch_size=128, train_sample_weights=train_sample_weights
    )
    hidden_size = 32
    model = LSTM(
        input_size=x_train_seq.size(2),
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        num_layers=2,
        device=device,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    num_epochs = 400
    early_stopping_patience = 100

    train_loss_graph, val_loss_graph, best_epoch, best_val_loss = fit_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        early_stopping_patience,
        MODEL_DIR / "lstm_best_model.pth",
        device,
    )

    print(f"Best epoch: {best_epoch}, best validation loss: {best_val_loss:.6f}")
    save_training_artifacts(
        x_scaler,
        y_scaler,
        {
            "sequence_length": sequence_length,
            "batch_size": 128,
            "hidden_size": hidden_size,
            "num_layers": 2,
            "learning_rate": 1e-3,
            "num_epochs": num_epochs,
            "train_scenarios": train_scenarios,
            "validation_scenarios": val_scenarios,
            "test_scenarios": test_scenarios,
        },
    )
    plot_loss(train_loss_graph, val_loss_graph, best_epoch)
    save_loss_curve_mat(train_loss_graph, val_loss_graph, best_epoch, best_val_loss, RESULT_DIR / LOSS_MAT_NAME)

    best_checkpoint = torch.load(MODEL_DIR / "lstm_best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    # Validation/Test 모두 UKF와 비교하지 않고, LSTM_Dataset.mat의 reference 횡력과만 비교한다.
    # UKF_Estimator.mat과의 비교는 여기서 저장하는 LSTM_Results.mat을 MATLAB에서 별도로 도시하여 수행한다.
    lstm_results = {}

    validation_metrics_df = evaluate_scenario_list(
        scenario_list=val_scenarios,
        split_name="Validation",
        metrics_path=RESULT_DIR / "validation_metrics.txt",
        dataset=dataset,
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        sequence_length=sequence_length,
        device=device,
        results_accumulator=lstm_results,
    )
    print("Validation metrics summary")
    print(validation_metrics_df)

    test_metrics_df = evaluate_scenario_list(
        scenario_list=test_scenarios,
        split_name="Test",
        metrics_path=RESULT_DIR / "test_metrics.txt",
        dataset=dataset,
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        sequence_length=sequence_length,
        device=device,
        results_accumulator=lstm_results,
    )
    print("Test metrics summary")
    print(test_metrics_df)

    save_lstm_results_mat(lstm_results, RESULT_DIR / RESULTS_MAT_NAME)

    zip_output(sequence_length, batch_size=128, num_epochs=num_epochs, hidden_size=hidden_size)


if __name__ == "__main__":
    main()