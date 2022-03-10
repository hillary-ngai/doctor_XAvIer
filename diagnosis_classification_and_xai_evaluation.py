import pandas as pd
import numpy as np
from numpy import trapz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from captum.attr import IntegratedGradients
from scipy.special import softmax
import shap
import matplotlib.pyplot as plt
from typing import List
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import os


# Map diagnosis
DIAGNOSIS_MAP = {
    'adhd': 0,
    'depression': 1,
    'osteoporosis': 2,
    'influenza': 3,
    'copd': 4,
    'type ii diabetes': 5,
    'other': 6
}
REV_DIAGNOSIS_MAP = {v: k for k, v in DIAGNOSIS_MAP.items()}
EXPLAINABILITY_TYPES = ["integrated_gradients", "shapley"]


class DiagnosisClassifier(nn.Module):
    def __init__(self, num_features = 549, num_classes = 7):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        output = self.fc4(x)

        return output


class DiagnosisDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = X  # [num_samples, num_features]
        self.y = y  # [num_samples]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        diagnosis_label = self.y[index]
        features = self.X[index]

        return features, diagnosis_label


class DiagnosisClassifier(nn.Module):
    def __init__(self, num_features=549, num_classes=7):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        output = self.fc4(x)
        return output


def get_classification_report(dataloader,
                              net,
                              device,
                              train_classification_reports: List,
                              test_classification_reports: List,
                              best_model_path: str,
                              split: str,
                              save_csv=True):
    checkpoint = torch.load(best_model_path)
    print('Achieved smallest testing loss %.3f on epoch %d.' % (checkpoint["test_mean_loss"], checkpoint["epoch"]))
    net.load_state_dict(checkpoint['model_state_dict'])

    predictions = []
    labels = []

    with torch.no_grad():
        for data in dataloader:
            inputs, diagnosis_labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)

            # Get the predicted diagnosis
            predicted_diagnosis = torch.argmax(outputs, dim=1)
            predictions += list(predicted_diagnosis.cpu().numpy())
            labels += list(diagnosis_labels.cpu().numpy())

    report = classification_report(labels, predictions, output_dict=True)
    report = pd.DataFrame(report).transpose().reset_index()

    report['index'] = report['index'].map(lambda x: REV_DIAGNOSIS_MAP[int(x)] if x.isdigit() else x)
    report['test_mean_loss'] = checkpoint["test_mean_loss"]
    report['best_epoch_on_test_set'] = checkpoint['epoch']

    if save_csv:
        report.to_csv("results/" + split + "_results.csv")

    if split == "train":
        train_classification_reports.append(report)
    elif split == "test":
        test_classification_reports.append(report)

    return report


# Explainability
# https://captum.ai/docs/extension/integrated_gradients
def ig_attributions_df(diagnosis: str,
                       X_test,
                       y_test,
                       X_test_baseline,
                       net,
                       device,
                       top_k: int = None,
                       sort_by_attribution_percent=True,
                       ascending=False):
    ig = IntegratedGradients(net)

    input = torch.tensor(X_test[y_test == diagnosis].values, dtype=torch.float32).to(device)
    baselines = torch.tensor(X_test_baseline[y_test == diagnosis].values, dtype=torch.float32).to(device)

    attributions, approximation_error = ig.attribute(
        input, baselines=baselines,
        target=DIAGNOSIS_MAP[diagnosis], return_convergence_delta=True)

    mean_attributions = torch.mean(attributions, axis=0).cpu().numpy()

    attribution_dict = {'feature': X_test.columns,
                        "mean_attribution": mean_attributions}

    df = pd.DataFrame.from_dict(attribution_dict)

    if sort_by_attribution_percent:
        attribution_percent = softmax(abs(mean_attributions))
        df['attribution_percent'] = attribution_percent
        df = df.sort_values(by=["attribution_percent"], ascending=ascending)

    if top_k is None:
        return df

    return df.iloc[:top_k]


def shapley_attributions_df(diagnosis: str,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            net,
                            device,
                            top_k: int = None,
                            sort_by_attribution_percent=True,
                            ascending=False):
    explainer = shap.DeepExplainer(net,
                                   torch.tensor(X_train[y_train == diagnosis].values, dtype=torch.float32).to(device))
    shapley_values = explainer.shap_values(
        torch.tensor(X_test[y_test == diagnosis].values, dtype=torch.float32).to(device))

    mean_shapley_vals = np.mean(shapley_values[DIAGNOSIS_MAP[diagnosis]], axis=0)

    attribution_dict = {'feature': X_test.columns,
                        "mean_attribution": mean_shapley_vals}

    df = pd.DataFrame.from_dict(attribution_dict)

    if sort_by_attribution_percent:
        attribution_percent = softmax(abs(mean_shapley_vals))
        df['attribution_percent'] = attribution_percent
        df = df.sort_values(by=["attribution_percent"], ascending=ascending)

    if top_k is None:
        return df

    return df.iloc[:top_k]


def accuracy_scores_percent_dropped_features(
        diagnosis: str,
        params: dict,
        X_test_baseline,
        X_test,
        y_test,
        net,
        device,
        explainability_type: str,
        percent_dropped_features: List):
    X_test_dropped_features = X_test.copy()

    if explainability_type == "integrated_gradients":
        plot_attributions = ig_attributions_df

    elif explainability_type == "shapley":
        plot_attributions = shapley_attributions_df

    accuracy_scores = []
    for top_k in percent_dropped_features:
        for important_feature in plot_attributions(
                diagnosis=diagnosis,
                **params,
                device=device,
                net=net,
                top_k=top_k)['feature'].values:
            X_test_dropped_features[important_feature] = X_test_baseline[important_feature]

        predictions = []
        diagnosis_idxs = y_test == diagnosis
        labels = y_test[diagnosis_idxs].map(DIAGNOSIS_MAP)

        outputs = net.forward(
            torch.tensor(X_test_dropped_features[diagnosis_idxs].values, dtype=torch.float32).to(device))
        predicted_diagnosis = torch.argmax(outputs, dim=1)
        predictions += list(predicted_diagnosis.cpu().numpy())

        accuracy_scores.append(accuracy_score(labels, predictions))

    return accuracy_scores


def set_explainability_accuracies(
        X_train: np.array,
        y_train: np.array,
        X_test: np.array,
        y_test: np.array,
        X_test_baseline: np.array,
        net,
        device,
        percent_dropped_features: List,
        explainability_accuracies: dict
):
    for diagnosis in DIAGNOSIS_MAP.keys():
        for explainability_type in EXPLAINABILITY_TYPES:
            explainability_params = dict(
                X_test=X_test,
                y_test=y_test
            )
            if explainability_type == "integrated_gradients":
                explainability_params.update(
                    dict(X_test_baseline=X_test_baseline))

            elif explainability_type == "shapley":
                explainability_params.update(
                    dict(X_train=X_train, y_train=y_train))

            xai_and_diagnosis = explainability_type + "_" + diagnosis
            accuracy_scores = accuracy_scores_percent_dropped_features(
                diagnosis,
                explainability_params,
                X_test_baseline,
                X_test,
                y_test,
                net,
                device,
                explainability_type,
                percent_dropped_features)

            # Cumulative sum of explainability accuracies per
            # explainability method and diagnosis
            explainability_accuracies[xai_and_diagnosis].append(accuracy_scores)


def add_explainability_score_dfs(
        X_train: np.array,
        y_train: np.array,
        X_test: np.array,
        y_test: np.array,
        X_test_baseline: np.array,
        net,
        device,
        ig_attributions_dfs: dict,
        shapley_attributions_dfs: dict
):
    for diagnosis in DIAGNOSIS_MAP.keys():
        if diagnosis not in ig_attributions_dfs:
            ig_attributions_dfs[diagnosis] = ig_attributions_df(
                diagnosis,
                X_test,
                y_test,
                X_test_baseline,
                net,
                device,
                sort_by_attribution_percent=False)
        else:
            ig_attributions_dfs[diagnosis]['mean_attribution'] += ig_attributions_df(
                diagnosis,
                X_test,
                y_test,
                X_test_baseline,
                net,
                device,
                sort_by_attribution_percent=False)['mean_attribution']

    for diagnosis in DIAGNOSIS_MAP.keys():
        if diagnosis not in shapley_attributions_dfs:
            shapley_attributions_dfs[diagnosis] = shapley_attributions_df(
                diagnosis,
                X_train,
                y_train,
                X_test,
                y_test,
                net,
                device,
                sort_by_attribution_percent=False)

        else:
            shapley_attributions_dfs[diagnosis]['mean_attribution'] += shapley_attributions_df(
                diagnosis,
                X_train,
                y_train,
                X_test,
                y_test,
                net,
                device,
                sort_by_attribution_percent=False)['mean_attribution']


def train(k_fold_idx: int,
          X_train: np.array,
          X_test: np.array,
          y_train: np.array,
          y_test: np.array,
          X_test_baseline: np.array,
          model_params: dict,
          dataloader_params: dict,
          train_classification_reports: List,
          test_classification_reports: List,
          explainability_accuracies: dict,
          percent_dropped_features: List,
          ig_attributions_dfs: dict,
          shapley_attributions_dfs: dict,
          device):
    # Generators
    train_dataset = DiagnosisDataset(torch.tensor(X_train.values, dtype=torch.float32),
                                     torch.tensor(y_train.map(DIAGNOSIS_MAP).values, dtype=torch.long))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **dataloader_params, shuffle=True)

    test_dataset = DiagnosisDataset(torch.tensor(X_test.values, dtype=torch.float32),
                                    torch.tensor(y_test.map(DIAGNOSIS_MAP).values, dtype=torch.long))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, **dataloader_params, shuffle=False)

    best_model_path = "k_fold_" + str(k_fold_idx) + "_best_model.pt"
    net = DiagnosisClassifier(num_features=model_params["num_features"],
                              num_classes=model_params["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    min_loss = None

    # Create TensorBoard object
    writer = SummaryWriter("logs")
    writer.flush()

    for epoch in range(model_params["n_epochs"]):  # loop over the dataset multiple times

        train_losses = []
        test_losses = []

        for i, data in enumerate(train_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero out the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            train_losses.append(loss.item())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Log the running average batch loss on the training data
        writer.add_scalar('Train Average Loss per Epoch',
                          np.mean(train_losses),
                          epoch
                          )

        for i, data in enumerate(test_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())

        # Log the running average batch loss on the testing data
        writer.add_scalar('Test Average Loss per Batch',
                          np.mean(test_losses),
                          epoch
                          )

        if min_loss is None or np.mean(test_losses) < min_loss:
            min_loss = np.mean(test_losses)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_mean_loss': np.mean(test_losses)
            }, best_model_path)

        print('epoch: %d, train loss: %.3f' % (epoch, np.mean(train_losses)))
        print('epoch: %d, test loss: %.3f\n' % (epoch, np.mean(test_losses)))

    # Close the writer
    writer.close()
    print('Finished Training')

    # Train diagnosis classification results
    get_classification_report(train_dataloader,
                              net,
                              device,
                              train_classification_reports,
                              test_classification_reports,
                              best_model_path,
                              split="train")

    # Test diagnosis classification results
    get_classification_report(test_dataloader,
                              net,
                              device,
                              train_classification_reports,
                              test_classification_reports,
                              best_model_path,
                              split="test")

    # Add explainability score dataframes
    add_explainability_score_dfs(
        X_train,
        y_train,
        X_test,
        y_test,
        X_test_baseline,
        net,
        device,
        ig_attributions_dfs,
        shapley_attributions_dfs)

    # Sum all the k-fold accuracies for plotting
    set_explainability_accuracies(
        X_train,
        y_train,
        X_test,
        y_test,
        X_test_baseline,
        net,
        device,
        percent_dropped_features,
        explainability_accuracies)


def k_fold_classification_report(classification_reports: List, split: str,
                                 save_csv=True):
    '''Takes the mean of the classification reports.'''

    mean_classification_report = None
    index = None
    for classification_report in classification_reports:
        if mean_classification_report is None:
            mean_classification_report = classification_report.drop(columns=["index"])
            index = classification_report["index"]

        else:
            mean_classification_report += classification_report.drop(columns=["index"])

    mean_classification_report /= len(classification_reports)
    mean_classification_report.insert(0, "index", index)

    if save_csv:
        mean_classification_report.to_csv("results/" +
                                          split + "_mean_classification_report.csv")

    return mean_classification_report


def save_xai_diagnosis_feature_attribution_dfs(
        attributions_dfs: dict, explainability_type: str, n_splits: int = 5):
    for diagnosis, df in attributions_dfs.items():
        # Divide by K to calculate the mean feature attribution scores
        # across the folds
        df['mean_attribution'] = df['mean_attribution'] / n_splits

        # Sort the features by attribution percent
        df['attribution_percent'] = softmax(abs(df['mean_attribution']))
        df = df.sort_values(by=["attribution_percent"], ascending=False)

        # TODO: for debugging purposes
        attributions_dfs[diagnosis] = df

        # Save dataframe
        filename = explainability_type + "_" + diagnosis + "_feature_attribution_df.csv"
        df.to_csv("results/" + filename)


def xai_results_dfs(percent_dropped_features, explainability_accuracies: dict, save_to_csv=True):
    #  Feature Attribution Dropping (FAD) curves for each diagnosis
    xai_acc_std_dfs = defaultdict(pd.DataFrame)

    # Divide by K to calculate the mean accuracy across the folds
    for xai_and_diagnosis, accuracies in explainability_accuracies.items():
        diagnosis = xai_and_diagnosis.split('_')[-1]
        xai = xai_and_diagnosis[:len(xai_and_diagnosis) - len(diagnosis) - 1]

        accuracies = np.array(accuracies)
        mean_accuracy = np.mean(accuracies, axis=0)
        accuracy_std = np.std(accuracies, axis=0)

        xai_acc_std_dfs[diagnosis]["percent_dropped_features"] = percent_dropped_features
        xai_acc_std_dfs[diagnosis][xai + "_acc"] = mean_accuracy
        xai_acc_std_dfs[diagnosis][xai + "_acc_std"] = accuracy_std

    # Save explainability accuracies and standard devations dataframes
    if save_to_csv:
        for diagnosis, df in xai_acc_std_dfs.items():
            df.to_csv("results/xai_results_df_" + diagnosis + ".csv")

    return xai_acc_std_dfs


def _save_diagnosis_accuracy_plot(diagnosis: str,
                                  percent_dropped_features: List,
                                  explainability_accuracies: dict):
    plt.figure()
    for i, explainability_type in enumerate(EXPLAINABILITY_TYPES):
        if i:
            ls = "dotted"
        else:
            ls = "solid"

        accuracies = explainability_accuracies[explainability_type + "_" + diagnosis]

        # Calculate the area under the curve
        dx = percent_dropped_features[1] - percent_dropped_features[0]
        area_under_curve = trapz(accuracies, dx=dx)

        label = explainability_type.replace("_", " ").title()
        label += "\n(AUC = {:.2f})".format(area_under_curve)

        plt.plot(percent_dropped_features,
                 accuracies,
                 marker="o",
                 ls=ls,
                 label=label)

    plt.legend(title="Explainability Method", loc="upper right",
               bbox_to_anchor=(1.45, 1))
    plt.xlabel("Percent of Features Dropped \nin Order of Explainability Score")
    plt.ylabel("Mean Accuracy")
    plt.title(diagnosis.title().replace("Ii", "II") + " Diagnosis K-Fold CV Accuracy \nper Percent of Features Dropped")
    plt.savefig("results/ " + diagnosis.replace(" ", "_") + "_diagnosis_accuracy_per_percent_of_features_dropped.png",
                bbox_inches="tight")


# Diagnosis K-Fold Cross-Validation Accuracy per Percent of Features Dropped Plots
def save_diagnosis_accuracy_plots(xai_acc_std_dfs,
                                  dpi: int = 1200,
                                  font_size: str = "x-small",
                                  axis_font_size: str = "xx-small",
                                  auc_x_lim: int = 20):
    # Save AUC dictionary
    auc_dict = defaultdict(list)

    num_cols = 4
    fig, axs = plt.subplots(2, num_cols, sharex=True, sharey=True)
    fig.dpi = dpi

    for i, diagnosis in REV_DIAGNOSIS_MAP.items():
        if i >= 3:
            i += 1

        row = i // num_cols
        col = i % num_cols

        percent_dropped_features = xai_acc_std_dfs[diagnosis]["percent_dropped_features"]
        dx = percent_dropped_features[1] - percent_dropped_features[0]

        for j, explainability_type in enumerate(EXPLAINABILITY_TYPES):
            if j:
                ls = "dotted"
            else:
                ls = "solid"

            accuracies = xai_acc_std_dfs[diagnosis][explainability_type + "_acc"]
            label = explainability_type.replace("_", "\n").title()

            # Calculate the area under the curve and save to dictionary
            acc_until_xlim = [
                acc for acc, percent in zip(accuracies, percent_dropped_features)
                if percent <= auc_x_lim]
            area_under_curve = trapz(acc_until_xlim, dx=dx)
            worst_case_auc = max(acc_until_xlim) * auc_x_lim
            normalized_area_under_curve = area_under_curve / worst_case_auc
            auc_dict[diagnosis].append(normalized_area_under_curve)

            # Plot subplot
            axs[row, col].plot(percent_dropped_features,
                               accuracies,
                               marker="o",
                               markersize=2,
                               ls=ls,
                               label=label)

            # Set title for each subplot
            if diagnosis == "copd" or diagnosis == "adhd":
                plot_title = diagnosis.upper()
            else:
                plot_title = diagnosis.title().replace("Ii", "II")

            axs[row, col].set_title(plot_title, fontsize=font_size)

    # Common x-label
    fig.text(0.5, 0.035,
             "Percent of Features Dropped in Order of Explainability Score",
             ha="center",
             fontsize=font_size)

    # Common y-label
    fig.text(0.05, 0.5,
             "Mean Accuracy",
             va='center',
             rotation='vertical',
             fontsize=font_size)

    # Create height space between subplots
    fig.subplots_adjust(hspace=0.3)

    for i, ax in enumerate(axs.flat):
        # Hide x labels and tick labels for top plots and y ticks for right plots
        ax.label_outer()

        # Set xticks and yticks
        ax.set_xticks(np.linspace(0, 100, 5 + 1))
        ax.tick_params(axis='x', labelsize=axis_font_size)

        ax.set_yticks(np.linspace(0, 1, 5 + 1))
        ax.tick_params(axis='y', labelsize=axis_font_size)

    plt.legend(title="  Explainability\n    Method  ",
               loc="upper right",
               borderpad=1.7,
               bbox_to_anchor=(1.08, 2.35),
               framealpha=1,
               labelspacing=2.9,
               title_fontsize=font_size,
               fontsize=font_size)

    # Save plot to PNG file
    plt.savefig("results/diagnosis_accuracy_per_percent_of_features_dropped.png",
                bbox_inches="tight")

    # Save AUC dataframe
    auc_df = pd.DataFrame.from_dict(auc_dict)
    auc_df = auc_df.rename(index={i: xai for i, xai in enumerate(EXPLAINABILITY_TYPES)})
    auc_df.to_csv("results/FAD_AUC_results_until_xlim_of_" + str(auc_x_lim) + ".csv")


def main():
    df = pd.read_pickle('med-dialogue-data/encoded_disease_classification_df.pkl')

    # Create baseline dataframe
    baseline_df = df.copy()
    continuous_features = ['age', 'height', 'weight', 'diastolic_bp', 'systolic_bp']
    binary_features = ['gender_male', 'currently_smokes_True']
    not_categorical_cols = ['dataset', 'filename', 'diagnosis'] + binary_features + continuous_features

    # Set continuous features to the mean
    for feature in continuous_features:
        baseline_df[feature] = np.mean(baseline_df[feature])

    # Set binary features to 0.5
    for feature in binary_features:
        baseline_df[feature] = 0.5

    # Set all other columns to zero
    for feature in baseline_df.columns:
        if feature not in not_categorical_cols:
            baseline_df[feature] = 0

    baseline_df = baseline_df.drop(columns=["filename", "diagnosis", "dataset"])

    # Create results directory
    results_dir = "results"

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    y = df['diagnosis']
    X = df.drop(columns=["filename", "diagnosis", "dataset"])

    # Dataloader Parameters
    dataloader_params = {"batch_size": 32,
                         "num_workers": 1}
    model_params = {
        "n_epochs": 1,
        "num_features": X.shape[1],
        "num_classes": 7
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # K-Fold Cross-Validation
    n_splits = 2
    skf = StratifiedKFold(n_splits=n_splits)

    train_classification_reports = []
    test_classification_reports = []

    ig_attributions_dfs = defaultdict()
    shapley_attributions_dfs = defaultdict()

    percent_dropped_features = list(
        range(0, 100 + 1, 5))
    explainability_accuracies = defaultdict(list)

    for k_fold_idx, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        print("K-Fold Index: %d \n" % k_fold_idx)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_test_baseline = baseline_df.iloc[test_index]

        train(k_fold_idx,
              X_train,
              X_test,
              y_train,
              y_test,
              X_test_baseline,
              model_params,
              dataloader_params,
              train_classification_reports,
              test_classification_reports,
              explainability_accuracies,
              percent_dropped_features,
              ig_attributions_dfs,
              shapley_attributions_dfs,
              device)

    k_fold_classification_report(train_classification_reports, split="train")
    k_fold_classification_report(test_classification_reports, split="test")

    # Divide by K to calculate the mean feature attribution scores across the folds
    for explainability_type in EXPLAINABILITY_TYPES:
        if explainability_type == "integrated_gradients":
            attributions_dfs = ig_attributions_dfs
        elif explainability_type == "shapley":
            attributions_dfs = shapley_attributions_dfs

        save_xai_diagnosis_feature_attribution_dfs(
            attributions_dfs, explainability_type)

    # Save explainability results for each diagnosis
    xai_acc_std_dfs = xai_results_dfs(percent_dropped_features, explainability_accuracies)

    # Plot and save diagnosis k-fold CV acuracies per percent of features dropped in
    # order of importance
    save_diagnosis_accuracy_plots(xai_acc_std_dfs, dpi=600)


if __name__ == "__main__":
    main()
