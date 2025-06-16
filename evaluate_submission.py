import os
import argparse
import importlib
import importlib.resources as pkg_resources
import pandas as pd
import torch
import torch.nn.functional as F
import json
from torch.hub import download_url_to_file
from sklearn.metrics import accuracy_score
import numpy as np
from collections import defaultdict


# Dataset config
dataset_config = {
    "split_url": "https://github.com/CPJKU/dcase2024_task1_baseline/releases/download/files/",
    "test_split_csv": "test.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DCASE Task 1 submission.")
    parser.add_argument("--submission_name", type=str, required=True)
    parser.add_argument("--submission_index", type=int, required=True)
    parser.add_argument("--dev_set_dir", type=str, required=True)
    parser.add_argument("--eval_set_dir", type=str, required=True)
    return parser.parse_args()


def download_split_file(resource_dir: str, split_name: str) -> str:
    os.makedirs(resource_dir, exist_ok=True)
    split_path = os.path.join(resource_dir, split_name)
    if not os.path.isfile(split_path):
        print(f"Downloading {split_name} to {split_path} ...")
        download_url_to_file(dataset_config["split_url"] + split_name, split_path)
    return split_path


def load_test_split(dataset_dir: str, resource_pkg: str) -> pd.DataFrame:
    meta_csv = os.path.join(dataset_dir, "meta.csv")

    try:
        with pkg_resources.path(resource_pkg, "test.csv") as test_csv_path:
            test_csv_file = str(test_csv_path)
    except FileNotFoundError:
        print("test.csv not found in package resources. Downloading ...")
        resource_dir = os.path.join(os.path.dirname(__file__), resource_pkg.replace('.', '/'), "resources")
        test_csv_file = download_split_file(resource_dir, dataset_config["test_split_csv"])

    df_meta = pd.read_csv(meta_csv, sep="\t")
    df_test = pd.read_csv(test_csv_file, sep="\t").drop(columns=["scene_label"], errors="ignore")
    df_test = df_test.merge(df_meta, on="filename")

    return df_test


def run_evaluation(args):
    # --- Load module ---
    module_path = f"{args.submission_name}.{args.submission_name}_{args.submission_index}"
    print(f"Importing inference module: {module_path}")
    api = importlib.import_module(module_path)

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    # --- Load test data ---
    print("Loading test split ...")
    df_test = load_test_split(args.dev_set_dir, f"{args.submission_name}.resources")
    file_paths = [os.path.join(args.dev_set_dir, fname) for fname in df_test["filename"]]
    device_ids = df_test["source_label"].tolist()
    scene_labels = df_test["scene_label"].tolist()

    print("Running test set predictions ...")
    predictions, class_order = api.predict(
        file_paths=file_paths,
        device_ids=device_ids,
        model_file_path=None,
        use_cuda=use_cuda
    )

    # Map ground truth scene labels to class indices based on class_order
    label_to_idx = {label: idx for idx, label in enumerate(class_order)}
    true_labels = [label_to_idx[label] for label in scene_labels]

    # Compute accuracy
    pred_labels = [pred.argmax().item() for pred in predictions]
    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n✅ Overall Test Accuracy: {acc * 100:.2f}%")

    # === DETAILED ACCURACY ANALYSIS ===
    
    # Class-wise accuracy calculation
    print(f"\n=== CLASS-WISE TEST ACCURACY ===")
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for true_label, pred_label, scene_label in zip(true_labels, pred_labels, scene_labels):
        class_total[scene_label] += 1
        if true_label == pred_label:
            class_correct[scene_label] += 1
    
    class_accuracies = {}
    for class_name in class_order:
        if class_name in class_total:
            class_acc = class_correct[class_name] / class_total[class_name]
            class_accuracies[class_name] = class_acc
            print(f"   {class_name:20s}: {class_acc:.4f} ({class_correct[class_name]:4d}/{class_total[class_name]:4d})")
        else:
            class_accuracies[class_name] = 0.0
            print(f"   {class_name:20s}: 0.0000 (   0/   0)")
    
    # Macro-average accuracy (average of class accuracies)
    macro_avg_acc = np.mean(list(class_accuracies.values()))
    print(f"\n📈 Macro-Average Accuracy: {macro_avg_acc:.4f}")
    
    # Device-wise accuracy calculation
    print(f"\n=== DEVICE-WISE TEST ACCURACY ===")
    device_correct = defaultdict(int)
    device_total = defaultdict(int)
    
    for true_label, pred_label, device_id in zip(true_labels, pred_labels, device_ids):
        device_total[device_id] += 1
        if true_label == pred_label:
            device_correct[device_id] += 1
    
    device_accuracies = {}
    
    # Load model to get device configuration
    model = api.load_model()
    trained_devices = model.config.train_device_ids
    all_devices = model.config.all_device_ids
    unseen_devices = [d for d in all_devices if d not in trained_devices]
    
    # Group devices
    real_devices = ['a', 'b', 'c']  # Real recording devices
    seen_sim_devices = [d for d in trained_devices if d not in real_devices]  # Seen simulated devices
    unseen_sim_devices = [d for d in unseen_devices if d not in real_devices]  # Unseen simulated devices
    
    # Real devices
    if any(d in device_total for d in real_devices):
        print(f"📱 REAL DEVICES:")
        real_correct, real_total = 0, 0
        for device_id in real_devices:
            if device_id in device_total:
                device_acc = device_correct[device_id] / device_total[device_id]
                device_accuracies[device_id] = device_acc
                print(f"   Device {device_id}: {device_acc:.4f} ({device_correct[device_id]:4d}/{device_total[device_id]:4d})")
                real_correct += device_correct[device_id]
                real_total += device_total[device_id]
        if real_total > 0:
            real_avg_acc = real_correct / real_total
            print(f"   📊 REAL devices average: {real_avg_acc:.4f} ({real_correct:4d}/{real_total:4d})")
    
    # Seen simulated devices
    if seen_sim_devices and any(d in device_total for d in seen_sim_devices):
        print(f"\n👁️  SEEN SIMULATED DEVICES:")
        seen_correct, seen_total = 0, 0
        for device_id in seen_sim_devices:
            if device_id in device_total:
                device_acc = device_correct[device_id] / device_total[device_id]
                device_accuracies[device_id] = device_acc
                print(f"   Device {device_id}: {device_acc:.4f} ({device_correct[device_id]:4d}/{device_total[device_id]:4d})")
                seen_correct += device_correct[device_id]
                seen_total += device_total[device_id]
        if seen_total > 0:
            seen_avg_acc = seen_correct / seen_total
            print(f"   👁️  SEEN simulated devices average: {seen_avg_acc:.4f} ({seen_correct:4d}/{seen_total:4d})")
    
    # Unseen simulated devices
    if unseen_sim_devices and any(d in device_total for d in unseen_sim_devices):
        print(f"\n🔍 UNSEEN SIMULATED DEVICES:")
        unseen_correct, unseen_total = 0, 0
        for device_id in unseen_sim_devices:
            if device_id in device_total:
                device_acc = device_correct[device_id] / device_total[device_id]
                device_accuracies[device_id] = device_acc
                print(f"   Device {device_id}: {device_acc:.4f} ({device_correct[device_id]:4d}/{device_total[device_id]:4d})")
                unseen_correct += device_correct[device_id]
                unseen_total += device_total[device_id]
        if unseen_total > 0:
            unseen_avg_acc = unseen_correct / unseen_total
            print(f"   🎯 UNSEEN simulated devices average: {unseen_avg_acc:.4f} ({unseen_correct:4d}/{unseen_total:4d})")
            print(f"\n🔥 KEY METRIC - UNSEEN DEVICE PERFORMANCE: {unseen_avg_acc:.4f}")
    
    # All other devices not in the above categories
    other_devices = [d for d in device_total.keys() if d not in real_devices + seen_sim_devices + unseen_sim_devices]
    if other_devices:
        print(f"\n❓ OTHER DEVICES:")
        for device_id in other_devices:
            device_acc = device_correct[device_id] / device_total[device_id]
            device_accuracies[device_id] = device_acc
            print(f"   Device {device_id}: {device_acc:.4f} ({device_correct[device_id]:4d}/{device_total[device_id]:4d})")

    # === CONFUSION MATRIX ANALYSIS (Optional) ===
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total samples: {len(true_labels)}")
    print(f"Number of classes: {len(class_order)}")
    print(f"Number of devices: {len(set(device_ids))}")
    print(f"Trained devices: {trained_devices}")
    print(f"Unseen devices: {unseen_devices}")

    # === GENERAL MODEL TESTING ===
    print(f"\n" + "="*80)
    print(f"🔄 TESTING WITH GENERAL MODEL ONLY (Original Pre-trained Model)")
    print(f"="*80)
    
    # Run predictions with general model only
    print("Running general model predictions ...")
    general_predictions, general_class_order = api.predict_with_general_model(
        file_paths=file_paths,
        device_ids=device_ids,
        model_file_path=None,
        use_cuda=use_cuda
    )
    
    assert general_class_order == class_order, "Class order mismatch between device-specific and general model"
    
    # Compute general model accuracy
    general_pred_labels = [pred.argmax().item() for pred in general_predictions]
    general_acc = accuracy_score(true_labels, general_pred_labels)
    print(f"\n✅ General Model Overall Test Accuracy: {general_acc * 100:.2f}%")
    
    # === GENERAL MODEL DETAILED ACCURACY ANALYSIS ===
    
    # Class-wise accuracy calculation for general model
    print(f"\n=== GENERAL MODEL CLASS-WISE TEST ACCURACY ===")
    general_class_correct = defaultdict(int)
    general_class_total = defaultdict(int)
    
    for true_label, pred_label, scene_label in zip(true_labels, general_pred_labels, scene_labels):
        general_class_total[scene_label] += 1
        if true_label == pred_label:
            general_class_correct[scene_label] += 1
    
    general_class_accuracies = {}
    for class_name in class_order:
        if class_name in general_class_total:
            class_acc = general_class_correct[class_name] / general_class_total[class_name]
            general_class_accuracies[class_name] = class_acc
            print(f"   {class_name:20s}: {class_acc:.4f} ({general_class_correct[class_name]:4d}/{general_class_total[class_name]:4d})")
        else:
            general_class_accuracies[class_name] = 0.0
            print(f"   {class_name:20s}: 0.0000 (   0/   0)")
    
    # Macro-average accuracy for general model
    general_macro_avg_acc = np.mean(list(general_class_accuracies.values()))
    print(f"\n📈 General Model Macro-Average Accuracy: {general_macro_avg_acc:.4f}")
    
    # Device-wise accuracy calculation for general model
    print(f"\n=== GENERAL MODEL DEVICE-WISE TEST ACCURACY ===")
    general_device_correct = defaultdict(int)
    general_device_total = defaultdict(int)
    
    for true_label, pred_label, device_id in zip(true_labels, general_pred_labels, device_ids):
        general_device_total[device_id] += 1
        if true_label == pred_label:
            general_device_correct[device_id] += 1
    
    general_device_accuracies = {}
    
    # Real devices (general model)
    if any(d in general_device_total for d in real_devices):
        print(f"📱 REAL DEVICES (General Model):")
        general_real_correct, general_real_total = 0, 0
        for device_id in real_devices:
            if device_id in general_device_total:
                device_acc = general_device_correct[device_id] / general_device_total[device_id]
                general_device_accuracies[device_id] = device_acc
                print(f"   Device {device_id}: {device_acc:.4f} ({general_device_correct[device_id]:4d}/{general_device_total[device_id]:4d})")
                general_real_correct += general_device_correct[device_id]
                general_real_total += general_device_total[device_id]
        if general_real_total > 0:
            general_real_avg_acc = general_real_correct / general_real_total
            print(f"   📊 REAL devices average (General): {general_real_avg_acc:.4f} ({general_real_correct:4d}/{general_real_total:4d})")
    
    # Seen simulated devices (general model)
    if seen_sim_devices and any(d in general_device_total for d in seen_sim_devices):
        print(f"\n👁️  SEEN SIMULATED DEVICES (General Model):")
        general_seen_correct, general_seen_total = 0, 0
        for device_id in seen_sim_devices:
            if device_id in general_device_total:
                device_acc = general_device_correct[device_id] / general_device_total[device_id]
                general_device_accuracies[device_id] = device_acc
                print(f"   Device {device_id}: {device_acc:.4f} ({general_device_correct[device_id]:4d}/{general_device_total[device_id]:4d})")
                general_seen_correct += general_device_correct[device_id]
                general_seen_total += general_device_total[device_id]
        if general_seen_total > 0:
            general_seen_avg_acc = general_seen_correct / general_seen_total
            print(f"   👁️  SEEN simulated devices average (General): {general_seen_avg_acc:.4f} ({general_seen_correct:4d}/{general_seen_total:4d})")
    
    # Unseen simulated devices (general model)
    if unseen_sim_devices and any(d in general_device_total for d in unseen_sim_devices):
        print(f"\n🔍 UNSEEN SIMULATED DEVICES (General Model):")
        general_unseen_correct, general_unseen_total = 0, 0
        for device_id in unseen_sim_devices:
            if device_id in general_device_total:
                device_acc = general_device_correct[device_id] / general_device_total[device_id]
                general_device_accuracies[device_id] = device_acc
                print(f"   Device {device_id}: {device_acc:.4f} ({general_device_correct[device_id]:4d}/{general_device_total[device_id]:4d})")
                general_unseen_correct += general_device_correct[device_id]
                general_unseen_total += general_device_total[device_id]
        if general_unseen_total > 0:
            general_unseen_avg_acc = general_unseen_correct / general_unseen_total
            print(f"   🎯 UNSEEN simulated devices average (General): {general_unseen_avg_acc:.4f} ({general_unseen_correct:4d}/{general_unseen_total:4d})")
            print(f"\n🔥 GENERAL MODEL - UNSEEN DEVICE PERFORMANCE: {general_unseen_avg_acc:.4f}")
    
    # All other devices (general model)
    general_other_devices = [d for d in general_device_total.keys() if d not in real_devices + seen_sim_devices + unseen_sim_devices]
    if general_other_devices:
        print(f"\n❓ OTHER DEVICES (General Model):")
        for device_id in general_other_devices:
            device_acc = general_device_correct[device_id] / general_device_total[device_id]
            general_device_accuracies[device_id] = device_acc
            print(f"   Device {device_id}: {device_acc:.4f} ({general_device_correct[device_id]:4d}/{general_device_total[device_id]:4d})")

    # === COMPARISON ANALYSIS ===
    print(f"\n" + "="*80)
    print(f"📊 DEVICE-SPECIFIC vs GENERAL MODEL COMPARISON")
    print(f"="*80)
    
    print(f"\n=== OVERALL ACCURACY COMPARISON ===")
    print(f"Device-Specific Model: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"General Model:         {general_acc:.4f} ({general_acc * 100:.2f}%)")
    print(f"Improvement:           {(acc - general_acc):.4f} ({(acc - general_acc) * 100:+.2f}%)")
    
    print(f"\n=== MACRO-AVERAGE ACCURACY COMPARISON ===")
    print(f"Device-Specific Model: {macro_avg_acc:.4f} ({macro_avg_acc * 100:.2f}%)")
    print(f"General Model:         {general_macro_avg_acc:.4f} ({general_macro_avg_acc * 100:.2f}%)")
    print(f"Improvement:           {(macro_avg_acc - general_macro_avg_acc):.4f} ({(macro_avg_acc - general_macro_avg_acc) * 100:+.2f}%)")
    
    # Device group comparison
    if 'unseen_avg_acc' in locals() and 'general_unseen_avg_acc' in locals():
        print(f"\n=== UNSEEN DEVICE PERFORMANCE COMPARISON ===")
        print(f"Device-Specific Model: {unseen_avg_acc:.4f} ({unseen_avg_acc * 100:.2f}%)")
        print(f"General Model:         {general_unseen_avg_acc:.4f} ({general_unseen_avg_acc * 100:.2f}%)")
        print(f"Improvement:           {(unseen_avg_acc - general_unseen_avg_acc):.4f} ({(unseen_avg_acc - general_unseen_avg_acc) * 100:+.2f}%)")
    
    # --- Load evaluation data ---
    print("\n" + "="*80)
    print("🔄 PROCEEDING TO EVALUATION SET")
    print("="*80)
    print("\nLoading evaluation set ...")
    df_eval = pd.read_csv(os.path.join(args.eval_set_dir, "evaluation_setup", "fold1_test.csv"), sep="\t")
    eval_file_paths = [os.path.join(args.eval_set_dir, fname) for fname in df_eval["filename"]]
    eval_device_ids = df_eval["device_id"].tolist()

    print("Running evaluation set predictions ...")
    eval_predictions, eval_class_order = api.predict(
        file_paths=eval_file_paths,
        device_ids=eval_device_ids,
        model_file_path=None,
        use_cuda=use_cuda
    )

    assert eval_class_order == class_order, "Class order mismatch between test and evaluation prediction"

    # --- Format and save submission ---
    output_dir = os.path.join("predictions", f"{args.submission_name}_{args.submission_index}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving predictions to: {output_dir}/output.csv")
    all_probs = torch.stack(eval_predictions)
    all_probs = F.softmax(all_probs, dim=1)
    predicted_labels = [class_order[i] for i in torch.argmax(all_probs, dim=1)]

    submission = pd.DataFrame({
        "filename": df_eval["filename"],
        "scene_label": predicted_labels
    })
    for i, label in enumerate(class_order):
        submission[label] = all_probs[:, i].tolist()

    submission.to_csv(os.path.join(output_dir, "output.csv"), sep="\t", index=False)

    # --- Save model weights and info ---
    print("Saving model state dict ...")
    
    # Save the multi-device student container state dict
    model_state_dict = {
        'multi_device_student': model.model.state_dict(),
        'device_models': {
            device_id: model.model.get_model_for_device(device_id).state_dict()
            for device_id in model.config.train_device_ids
        },
        'original_model': model.model.get_original_model().state_dict()
    }
    torch.save(model_state_dict, os.path.join(output_dir, "model_state_dict.pt"))

    # Enhanced info with detailed accuracy metrics
    info = {
        # Device-Specific Model Results
        "Device_Specific_Model": {
            "Overall_Test_Accuracy": round(acc * 100, 2),
            "Macro_Average_Accuracy": round(macro_avg_acc * 100, 2),
            "Class_Wise_Accuracy": {k: round(v * 100, 2) for k, v in class_accuracies.items()},
            "Device_Wise_Accuracy": {k: round(v * 100, 2) for k, v in device_accuracies.items()}
        },
        
        # General Model Results
        "General_Model": {
            "Overall_Test_Accuracy": round(general_acc * 100, 2),
            "Macro_Average_Accuracy": round(general_macro_avg_acc * 100, 2),
            "Class_Wise_Accuracy": {k: round(v * 100, 2) for k, v in general_class_accuracies.items()},
            "Device_Wise_Accuracy": {k: round(v * 100, 2) for k, v in general_device_accuracies.items()}
        },
        
        # Comparison Results
        "Model_Comparison": {
            "Overall_Accuracy_Improvement": round((acc - general_acc) * 100, 2),
            "Macro_Average_Accuracy_Improvement": round((macro_avg_acc - general_macro_avg_acc) * 100, 2)
        },
        
        # Model Configuration
        "Model_Type": "Student Device-Specific",
        "Trained_Devices": model.config.train_device_ids,
        "All_Devices": model.config.all_device_ids,
        "Unseen_Devices": [d for d in model.config.all_device_ids if d not in model.config.train_device_ids]
    }
    
    # Add group-wise accuracies for device-specific model if available
    if 'real_avg_acc' in locals():
        info["Device_Specific_Model"]["Real_Devices_Average_Accuracy"] = round(real_avg_acc * 100, 2)
    if 'seen_avg_acc' in locals():
        info["Device_Specific_Model"]["Seen_Simulated_Devices_Average_Accuracy"] = round(seen_avg_acc * 100, 2)
    if 'unseen_avg_acc' in locals():
        info["Device_Specific_Model"]["Unseen_Simulated_Devices_Average_Accuracy"] = round(unseen_avg_acc * 100, 2)
    
    # Add group-wise accuracies for general model if available
    if 'general_real_avg_acc' in locals():
        info["General_Model"]["Real_Devices_Average_Accuracy"] = round(general_real_avg_acc * 100, 2)
    if 'general_seen_avg_acc' in locals():
        info["General_Model"]["Seen_Simulated_Devices_Average_Accuracy"] = round(general_seen_avg_acc * 100, 2)
    if 'general_unseen_avg_acc' in locals():
        info["General_Model"]["Unseen_Simulated_Devices_Average_Accuracy"] = round(general_unseen_avg_acc * 100, 2)
    
    # Add group-wise comparison if available
    if 'unseen_avg_acc' in locals() and 'general_unseen_avg_acc' in locals():
        info["Model_Comparison"]["Unseen_Device_Accuracy_Improvement"] = round((unseen_avg_acc - general_unseen_avg_acc) * 100, 2)
    if 'real_avg_acc' in locals() and 'general_real_avg_acc' in locals():
        info["Model_Comparison"]["Real_Device_Accuracy_Improvement"] = round((real_avg_acc - general_real_avg_acc) * 100, 2)
    if 'seen_avg_acc' in locals() and 'general_seen_avg_acc' in locals():
        info["Model_Comparison"]["Seen_Device_Accuracy_Improvement"] = round((seen_avg_acc - general_seen_avg_acc) * 100, 2)
    
    with open(os.path.join(output_dir, "test_accuracy.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✅ Submission ready: {output_dir}/output.csv")
    print(f"📊 Model info saved: {output_dir}/test_accuracy.json")
    print(f"💾 Model weights saved: {output_dir}/model_state_dict.pt")


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
